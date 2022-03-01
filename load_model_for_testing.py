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
        self.w_relation = nn.Parameter(torch.Tensor(1*num_rels, h_dim))
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
        return predict_loss + self.reg_param * reg_loss


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

        rel_relation = 9
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

def load_checkpoint_for_eval(model, filename,device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def main(args):
    # load graph data
    data = RGCNLinkDataset(args.dataset)
    dir_base = data.dir
    data.load()
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
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
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)

    model_state_file = './wikiTables/model_state.pth'
    model=load_checkpoint_for_eval(model, model_state_file)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    embed = model(test_graph, test_node_id, test_rel, test_norm)
    s = test_data[:, 0]
    r = test_data[:, 1]
    o = test_data[:, 2]
    test_size = test_data.shape[0]
    # ranks1 = get_relation_score(embed, model.w_relation, s, o, test_size, batch_size=args.eval_batch_size)
    # ranks2 = get_relation_score(embed, model.w_relation, o, s, test_size, batch_size=args.eval_batch_size)
    # ranks_ = (ranks1.type(torch.float) + ranks2.type(torch.float)) / 2

    ranks = get_relevance_relation_score(embed, model.w_relation, s, o, test_size, batch_size=args.eval_batch_size)

    entity_path = os.path.join(dir_base, 'entities.dict')
    entities_dict = _read_dictionary_test(entity_path)
    test_score_file = os.path.join(dir_base, 'scores.txt')
    split_id = 1
    output_qrels_test = './wikiTables/qrels_test' + str(split_id) + '.txt'

    with open(test_score_file, 'w') as f:
        for i, (s, _, o) in enumerate(test_data):
            qq = entities_dict[str(o.tolist())]
            tt = entities_dict[str(s.tolist())]
            qq = qq.split('_')[0]
            row = qq + '\t' + 'Q0' + '\t' + tt + '\t' + '0' + '\t' + str(ranks[i].tolist()) + '\trow'
            if i == test_data.shape[0] - 1:
                f.write(row)
            else:
                f.write(row + '\n')

    test_ndcg, test_map, test_mrr = calculate_ndcg(test_score_file, output_qrels_test)
    print('ndcg={}'.format(test_ndcg))
    print('map={}'.format(test_map))
    print('mrr={}'.format(test_mrr))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=50,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=10,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="number of minimum training epochs")
    parser.add_argument("--dataset", type=str, default='wikiTables',
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=5000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=0,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(args)
    main(args)