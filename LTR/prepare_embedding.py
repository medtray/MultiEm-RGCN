import sys
import os
cwd = os.getcwd()
from pathlib import Path
path = Path(cwd)
sys.path.append(os.path.join(path.parent.absolute()))
from collections import Counter
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from utils_ import *
from sklearn import preprocessing
from dgl.nn.pytorch import RelGraphConv
from load_model_for_testing import *
parser = argparse.ArgumentParser(description='Compute embeddings for MultiEm-RGCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--device', type=int, default=0)

parser.add_argument("--n-hidden", type=int, default=100,
                        help="number of hidden units")

parser.add_argument("--n-bases", type=int, default=10,
                    help="number of weight blocks for each relation")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")

parser.add_argument("--dataset", type=str, default='wikiTables',
                    help="dataset to use")
parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")


args = parser.parse_args()
#torch.cuda.set_device(args.device)
#args.device='cuda:'+str(args.device)
args.use_cuda=False
cwd=os.getcwd()
parent_path=Path(cwd).parent
args.parent_path=parent_path

data = RGCNLinkDataset(args.dataset)
data.dir = os.path.join(args.parent_path, args.dataset)
dir_base = data.dir
print(dir_base)


data.load()
num_nodes = data.num_nodes
train_data = data.train
num_rels = data.num_rels

# check cuda
use_cuda = args.use_cuda
if use_cuda:
    torch.cuda.set_device(args.device)

# create model
model = LinkPredict(num_nodes,
                    args.n_hidden,
                    num_rels,
                    num_bases=args.n_bases,
                    num_hidden_layers=args.n_layers,
                    dropout=args.dropout,
                    use_cuda=use_cuda,
                    reg_param=args.regularization)


# build test graph
test_graph, test_rel, test_norm,_ = utils.build_test_graph(
    num_nodes, num_rels, train_data)

#model_state_file = './wikiTables/model_state.pth'
model_state_file = os.path.join(dir_base, 'model_state.pth')
model = load_checkpoint_for_eval(model, model_state_file,args.device)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
wv={}

model.cpu()
model.eval()
print('start calculating embedding')
embed = model(test_graph, test_node_id, test_rel, test_norm)
print('finish calculating embedding')
nb_tokens,dim=embed.shape
entities_file = os.path.join(dir_base, 'entities.dict')
entities=open(entities_file,'r')
lines=entities.readlines()
with tqdm(total=len(lines)) as pbar0:
    for line in lines:
        line=line.strip()
        line=line.split('\t')
        wv[line[1]]=embed[int(line[0])].tolist()
        #print(line)
        pbar0.update(1)

wv['unk']=torch.tensor(list(np.random.rand(dim))).tolist()
wv['.'] =torch.tensor(list(np.random.rand(dim))).tolist()
wv[','] = torch.tensor(list(np.random.rand(dim))).tolist()

word_to_index = {}
index_to_word = []

for i,key in enumerate(wv.keys()):
    word_to_index[key]=i
    index_to_word.append(key)

torch.save(embed,os.path.join(dir_base, 'wv.pt'))

print('start saving')

np.save(os.path.join(dir_base, 'wv.npy'),wv)
print('finish saving')
np.save(os.path.join(dir_base, 'word_to_index.npy'),word_to_index)
np.save(os.path.join(dir_base, 'index_to_word.npy'),index_to_word)
