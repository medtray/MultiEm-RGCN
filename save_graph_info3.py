import numpy as np
import json
import os
from prepare_wikiTables_rdfs4 import PrepareGraph
import pandas as pd

def read_file_for_nfcg(file,delimiter):
    text_file = open(file, "r")
    lines = text_file.readlines()
    list_lines = []

    for line in lines:
        # print(line)
        line = line[0:len(line) - 1]
        aa = line.split(delimiter)
        list_lines.append(aa)
    inter = np.array(list_lines)

    return inter

def read_queries(file,delimiter):
    text_file = open(file, "r")
    lines = text_file.readlines()
    list_lines = []

    for line in lines:
        # print(line)
        line = line[0:len(line) - 1]
        aa = line.split(delimiter)
        query=[aa[0],' '.join(aa[1:])]

        list_lines.append(query)
    inter = np.array(list_lines)

    return inter

base='./wikiTables'
tables_id_file = os.path.join(base, 'tables_id.npy')
data=read_file_for_nfcg(os.path.join(base,'qrels.txt'),'\t')
queries=read_queries(os.path.join(base,'queries_wiki'),' ')
path = os.path.join(base,'data_fields_with_values.json')
with open(path) as f:
    dt = json.load(f)

index_to_use=[]
for i,row in enumerate(data):
    if row[2] in dt:
        index_to_use.append(i)

data=data[index_to_use]

GraphCons=PrepareGraph(queries,dt)

GraphCons.add_links_docs(data,tables_id_file)

entities_file=os.path.join(base,'entities.dict')
relations_file=os.path.join(base,'relations.dict')
train_file=os.path.join(base,'train.txt')
test_file=os.path.join(base,'test.txt')
valid_file=os.path.join(base,'valid.txt')

sep='\t'
with open(relations_file,'w') as rf:
    relations_mapping=''
    for i,relation in enumerate(GraphCons.relations):
        relations_mapping+=str(i)+sep+relation+'\n'

    relations_mapping=relations_mapping[:-1]
    rf.write(relations_mapping)
    rf.close()

with open(entities_file,'w') as ef:
    entities_mapping=''

    for (k,v) in GraphCons.entities.items():
        entities_mapping+=str(v)+sep+k+'\n'

    entities_mapping = entities_mapping[:-1]
    ef.write(entities_mapping)
    ef.close()

with open(train_file,'w') as tf:
    tf.write(GraphCons.triples)
    tf.close()

with open(test_file,'w') as tef:
    tef.write(GraphCons.test_triples)
    tef.close()

with open(valid_file,'w') as vf:
    vf.write(GraphCons.test_triples)
    vf.close()



