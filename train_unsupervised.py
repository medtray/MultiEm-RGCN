"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR
  probably because the model only uses one GNN layer so messages are propagated
  among immediate neighbors. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data
from dgl.nn.pytorch import RelGraphConv
import subprocess

from model import BaseRGCN
import os

import utils


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)


class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(2*num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)

        #unified embedding
        # x = embed.unsqueeze(0)
        # y = x
        # x_norm = (x ** 2).sum(2).view(x.shape[0], x.shape[1], 1)
        # y_t = y.permute(0, 2, 1).contiguous()
        # y_norm = (y ** 2).sum(2).view(y.shape[0], 1, y.shape[1])
        # dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        # dist[dist != dist] = 0  # replace nan values with 0
        # pairwise_dist = torch.clamp(dist, 0.0, np.inf).squeeze(0)
        # sum_pairwise=torch.sum(pairwise_dist)

        # positive_triples=triplets[torch.nonzero(labels==1).squeeze()]
        # pdist = nn.PairwiseDistance(p=2)
        #
        # def loss_term(rel_id,weight):
        #     a = torch.nonzero(positive_triples[:, 1] == rel_id)
        #     bb = positive_triples[a.squeeze()]
        #     sum_pairwise=0
        #     if (len(bb.shape) == 1):
        #         bb = bb.unsqueeze(0)
        #     if (len(bb.shape) > 0):
        #         embed1 = embed[bb[:, 0]]
        #         embed2 = embed[bb[:, 2]]
        #         output = pdist(embed1, embed2)
        #         sum_pairwise = weight * torch.sum(output)
        #
        #     return sum_pairwise
        #
        sum_pairwise=0
        # epsilon=0.00001
        # for i in range(3):
        #     sum_pairwise+=loss_term(i, i+epsilon)
        #
        # for i in range(11,14):
        #     sum_pairwise += loss_term(i, i-11+epsilon)
        #
        # sum_pairwise += loss_term(9, 2+epsilon)
        # sum_pairwise += loss_term(20, 2+epsilon)

        return predict_loss + self.reg_param * reg_loss+0.0*sum_pairwise


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def _read_dictionary_test(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[0]] = line[1]
    return d


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


class RGCNLinkDataset(object):

    def __init__(self, name):
        self.name = name
        self.dir = './'
        self.dir = os.path.join(self.dir, self.name)

    def load(self):
        entity_path = os.path.join(self.dir, 'entities.dict')
        relation_path = os.path.join(self.dir, 'relations.dict')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def get_relation_score(embedding, w, a, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        target = b[batch_start: batch_end]

        relevance_relations = [6,7,8]
        # relevance_relations = [1, 2, 3]
        scores = []
        for rel_relation in relevance_relations:
            relation = rel_relation * torch.ones(target.shape[0]).type(torch.int64)
            s = embedding[batch_a]
            r = w[relation]
            o = embedding[target]
            scores.append(torch.sum(s * r * o, dim=1))

        final_scores = torch.cat([score.view(-1, 1) for score in scores], dim=1)
        labels = torch.argmax(final_scores, dim=1)

        ranks.append(labels)
    return torch.cat(ranks)


def get_relevance_relation_score(embedding, w, a, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        target = b[batch_start: batch_end]

        rel_relation = 7
        # relevance_relations = [1, 2, 3]

        relation = rel_relation * torch.ones(target.shape[0]).type(torch.int64)
        s = embedding[batch_a]
        r = w[relation]
        o = embedding[target]
        ranks.append(torch.sum(s * r * o, dim=1))


    return torch.cat(ranks)


def calculate_ndcg(output_file, ndcg_file):
    # batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    batcmd = "./trec_eval -m map " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    map = float(res[2])

    batcmd = "./trec_eval -m recip_rank " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    mrr = float(res[2])

    batcmd = "./trec_eval -m ndcg_cut.5 " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg, map, mrr


def main(args):

    args.device = 'cuda:' + str(args.gpu)

    # load graph data
    data = RGCNLinkDataset(args.dataset)
    dir_base = data.dir
    data.load()
    num_nodes = data.num_nodes
    train_data = data.train
    # valid_data = data.valid
    # test_data = data.test
    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)


    # validation and testing triplets
    # valid_data = torch.LongTensor(valid_data)
    # test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm,train_data = utils.build_test_graph(
        num_nodes, num_rels, train_data)




    # test_deg = test_graph.in_degrees(
    #     range(test_graph.number_of_nodes())).float().view(-1, 1)
    # test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    # test_rel = torch.from_numpy(test_rel)
    # test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = os.path.join(args.dataset,'model_state_modified.pth')

    def load_checkpoint_for_eval(model, filename):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename,map_location=args.device)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model

    model = load_checkpoint_for_eval(model, model_state_file)
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")
    tables_id_file = os.path.join(args.dataset, 'tables_id.npy')
    tables_id = np.load(tables_id_file)

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,tables_id,
                args.edge_sampler)



        print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))


        optimizer.zero_grad()

        if epoch % args.evaluate_every == 0:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       model_state_file)


        # # validation
        # if epoch % args.evaluate_every == 0:
        #     # perform validation on CPU because full graph is too large
        #     if use_cuda:
        #         model.cpu()
        #     model.eval()
        #     print("start eval")
        #     embed = model(test_graph, test_node_id, test_rel, test_norm)
        #     mrr = utils.calc_mrr(embed, model.w_relation, valid_data,
        #                          hits=[1, 3, 10], eval_bz=args.eval_batch_size)
        #     # save best model
        #     if mrr < best_mrr:
        #         if epoch >= args.n_epochs:
        #             break
        #     else:
        #         best_mrr = mrr
        #         torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
        #                    model_state_file)
        #     if use_cuda:
        #         model.cuda()

        if epoch >= args.n_epochs:
            break

    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
               model_state_file)

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    # print("\nstart testing:")
    # # use best model checkpoint
    # checkpoint = torch.load(model_state_file)
    # if use_cuda:
    #     model.cpu()  # test on CPU
    # model.eval()
    # model.load_state_dict(checkpoint['state_dict'])
    # print("Using best epoch: {}".format(checkpoint['epoch']))
    # embed = model(test_graph, test_node_id, test_rel, test_norm)
    #
    # utils.calc_mrr(embed, model.w_relation, test_data,
    #                hits=[1, 3, 10], eval_bz=args.eval_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=100,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=10,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="number of minimum training epochs")
    parser.add_argument("--dataset", type=str, default='wikiTables',
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=2000,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=5000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=100,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
                        help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(args)
    main(args)
