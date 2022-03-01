"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import numpy as np
import torch
import dgl
import datetime
import re
import random
import networkx as nx
from matplotlib import pyplot as plt

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]

    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else:
            print('[' + t + '] ' + str(line))

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print_log('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2],triplet[1]])
        #adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size,tables_id):
    """Sample edges by neighborhool expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])
    tables_included=[]

    for i in range(0, sample_size):
        found = False
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0


        probabilities = (weights) / np.sum(weights)

        if i==0:
            chosen_vertex=random.choice(tables_id)

        retry=0

        if i>0:
            if random.random()>0.90:
                while not found and retry<50:
                    retry+=1
                    chosen_vertex=random.choice(np.where(seen==True)[0])
                    chosen_adj_list = adj_list[chosen_vertex]
                    np.random.shuffle(chosen_adj_list)

                    for chosen_edge in chosen_adj_list:
                        edge_number = chosen_edge[0]
                        other_vertex = chosen_edge[1]
                        if other_vertex in tables_id:
                            found=True
                            break
                    if found:

                        if not picked[edge_number]:

                            edges[i] = edge_number
                            other_vertex = chosen_edge[1]
                            picked[edge_number] = True
                            sample_counts[chosen_vertex] -= 1
                            #sample_counts[other_vertex] -= 1
                            seen[other_vertex] = True
                            tables_included.append(other_vertex)
                        else:
                            found=False


        if found:
            continue
        if i>0:
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]


        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]
        other_vertex = chosen_edge[1]

        while picked[edge_number]:
            # chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            # chosen_edge = chosen_adj_list[chosen_edge]
            # edge_number = chosen_edge[0]
            # other_vertex = chosen_edge[1]

            chosen_vertex = random.choice(np.where(seen == True)[0])
            chosen_adj_list = adj_list[chosen_vertex]
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]
            other_vertex = chosen_edge[1]

        seen[chosen_vertex] = True
        edges[i] = edge_number
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        #sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    print('nb tables included is {}'.format(len(set(tables_included))))

    return edges

def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate,tables_id, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size,tables_id)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()

    # my_graph = nx.Graph()
    # edges_to_draw = list(set(list(zip(dst, src, rel))))
    # edges_to_draw = sorted(edges_to_draw)
    # # my_graph.add_edges_from(edges_to_draw[:10])
    #
    # for item in edges_to_draw:
    #     my_graph.add_edge(item[1], item[0], weight=item[2]*10)
    # pos = nx.spring_layout(my_graph)
    # labels = nx.get_edge_attributes(my_graph, 'weight')
    # plt.figure()
    # nx.draw(my_graph, pos, edge_color='black', width=1, linewidths=1, arrows=True,
    #         node_size=100, node_color='red', alpha=0.9,
    #         labels={node: node for node in my_graph.nodes()})
    # nx.draw_networkx_edge_labels(my_graph, pos, edge_labels=labels, font_color='red')
    # plt.axis('off')
    # plt.show()



    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                       negative_rate)

    #samples, labels = negative_relations(relabeled_edges, len(uniq_v),
    #                                    negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    #g, rel, norm,_ = build_graph_from_triplets_modified(len(uniq_v), num_rels,
    #                                     (src, rel, dst))

    g, rel, norm=build_graph_directly(len(uniq_v), (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm


def build_graph_directly(num_nodes, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    #edges = sorted(zip(dst, src, rel))
    #dst, src, rel = np.array(edges).transpose()


    inverse_mapping=[11,12,13,3,4,5,6,7,8,20,21,0,1,2,3,4,5,6,7,8,9,10]
    rel2=np.array([inverse_mapping[i] for i in rel])
    rel = np.concatenate((rel, rel2))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    edges = list(set(list(zip(dst, src, rel))))
    edges = sorted(edges)

    #dst, src, rel = np.array(list(set(edges))).transpose()
    dst, src, rel = np.array(edges).transpose()


    # my_graph = nx.Graph()
    #
    # for item in edges:
    #     my_graph.add_edge(item[1], item[0], weight=str(item[2]))
    # pos = nx.spring_layout(my_graph)
    # labels = nx.get_edge_attributes(my_graph, 'weight')
    # plt.figure()
    # nx.draw(my_graph, pos, edge_color='black', width=1, linewidths=1,arrows=True,
    #         node_size=500, node_color='pink', alpha=0.9,
    #         labels={node: node for node in my_graph.nodes()})
    # nx.draw_networkx_edge_labels(my_graph, pos, edge_labels=labels, font_color='red')
    # plt.axis('off')
    # plt.show()







    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm

def build_graph_from_triplets_modified(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    #rel = np.concatenate((rel, rel + num_rels))
    #rel = np.array([i - num_rels if i in [12, 13, 14, 15, 16, 17] else i for i in rel])
    #rel = np.array([i - num_rels if i in [9,10,11] else i for i in rel])
    #rel= np.array([i - num_rels if i in [12,13,14,15,16,17] else i for i in rel])
    #rel = np.array([i - num_rels if i in [14, 15, 16, 17,18,19] else i for i in rel])

    inverse_mapping = [11, 12, 13, 3, 4, 5, 6, 7, 8, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rel2 = np.array([inverse_mapping[i] for i in rel])
    rel = np.concatenate((rel, rel2))

    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    train_data=np.concatenate([src.reshape(len(src),-1),rel.reshape(len(rel),-1),dst.reshape(len(dst),-1)],axis=1)
    return g, rel, norm,train_data

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets_modified(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def negative_relations(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    negative_relation=[]
    irrelevant, somehowrelevant, relevant=6,7,8

    for i in range(size_of_batch):
        curr=pos_samples[i]
        if curr[1]==irrelevant:
            negative_relation.append([curr[0],somehowrelevant,curr[2]])
            negative_relation.append([curr[0], relevant, curr[2]])

        elif curr[1]==somehowrelevant:

            negative_relation.append([curr[0], irrelevant, curr[2]])
            negative_relation.append([curr[0], relevant, curr[2]])

        elif curr[1] == relevant:

            negative_relation.append([curr[0], somehowrelevant, curr[2]])
            negative_relation.append([curr[0], irrelevant, curr[2]])

    negative_relation_labels = np.zeros(len(negative_relation), dtype=np.float32)
    negative_relation=np.array(negative_relation)


    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    labels=np.concatenate((labels,negative_relation_labels))

    return np.concatenate((pos_samples, neg_samples,negative_relation)), labels


#######################################################################
#
# Utility function for evaluations
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# TODO (lingfan): implement filtered metrics
# return MRR (raw), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()
