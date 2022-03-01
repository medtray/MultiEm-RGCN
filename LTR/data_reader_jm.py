from collections import Counter
import sys
import os
cwd = os.getcwd()
from pathlib import Path
path = Path(cwd)
sys.path.append(os.path.join(path.parent.absolute(),'nordlys'))

from nordlys.core.retrieval.elastic import Elastic
from nordlys.core.retrieval.elastic_cache import ElasticCache
#from nordlys.config import PLOGGER

from nordlys.core.retrieval.scorer import *
from nltk.corpus import wordnet as wn


import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from sklearn import preprocessing
from dgl.nn.pytorch import RelGraphConv

sys.path.append("/home/mohamedt/R-GCN-Graph")
from utils_ import *
from load_model_for_testing import *

es = ElasticCache("new_data_index")


def load_pretrained_wv(path):
    wv = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.split(' ')
            wv[items[0]] = torch.DoubleTensor([float(a) for a in items[1:]])
    return wv

def pad_or_crop(field,max_tokens,to_add):
    if len(field)>max_tokens:
        field=field[0:max_tokens]
    if len(field)<max_tokens:
        for i in range(max_tokens-len(field)):
            field+=to_add
    return field


def pad_or_crop_with_rep(field,max_tokens,dictt,field_type):

    final=[]

    for f in field:
        if f in dictt.keys():
            final.append(f)
        else:
            final.append('unk')

    if len(final)>max_tokens:
        final=final[0:max_tokens]
    if len(final)<max_tokens:
        inter=final.copy()
        if len(inter)==0:
            #print('here empty')
            if field_type in ['description','attributes']:
                inter=[',']
            else:
                inter = ['.']
        j=0
        for i in range(max_tokens-len(final)):
            final.append(inter[j%len(inter)])
            j+=1

    return final

def encode_field(field,dictt,field_type):
    vect_ = []
    for qu in field:
        try:
            vect_ += [dictt[qu]]
        except:

            if field_type in ['description','attributes']:
                vect_ += [dictt[',']]
            else:
                vect_ += [dictt['.']]

    return vect_


def get_www_all_features(feature_file):
    ids_left = []
    ids_right = []
    features = []
    labels  = []
    f_f = open(feature_file,'r')
    line = f_f.readline()
    for line in f_f:
        seps = line.strip().split(',')
        qid = seps[0]
        tid = seps[2]
        ids_left.append(qid)
        ids_right.append(tid)
        rel = seps[-1]
        labels.append(int(rel))
        '''
        if int(rel) > 0:
            labels.append(1)
        else:
            labels.append(0)
        '''
        #feat_range=np.arange(3,25)
        q_doc_f = np.array([float(each) for each in seps[3:-1]])
        #q_doc_f = np.array([float(each) for num,each in enumerate(seps) if num in feat_range or num==41])
        #feat_range = np.arange(25, 41)
        #q_doc_f = np.array([float(each) for num, each in enumerate(seps) if num in feat_range])
        features.append(q_doc_f)


    df = pd.DataFrame({
        'id_left': ids_left,
        'id_right': ids_right,
        'features': features,
        'label': labels
    })
    return df


def synset_translation(word):
    nb_tran=0
    synsets = wn.synsets(word.strip())
    translated_token=[syn._name for syn in synsets]

    stack = synsets
    while stack and nb_tran<20:
        el = stack.pop(0)
        hypers = el.hypernyms()
        stack += hypers
        translated_token += [hyper._name for hyper in hypers]
        nb_tran+=1

    return translated_token



class DataAndQueryJM(Dataset):
    def __init__(self,file_name,wv,word_to_index,index_to_word,output_file,args):

        data = RGCNLinkDataset(args.dataset)
        data.dir = os.path.join(args.parent_path, args.dataset)
        dir_base = data.dir

        if wv and word_to_index and index_to_word:
            self.wv=wv
            self.word_to_index=word_to_index
            self.index_to_word=index_to_word
        else:
            self.word_to_index = {}
            self.index_to_word = []
            self.wv=np.load(os.path.join(dir_base,'wv.npy'),allow_pickle=True)
            self.wv=self.wv[()]
            self.word_to_index=np.load(os.path.join(dir_base,'word_to_index.npy'),allow_pickle=True)
            self.word_to_index=self.word_to_index[()]
            self.index_to_word=list(np.load(os.path.join(dir_base,'index_to_word.npy')))

        labels = []
        all_vector_tables = []

        max_tokens_query_QTE = 150
        all_desc_QTE = []
        all_query_QTE = []


        max_tokens_desc_w = 20
        max_tokens_att_w = 10
        max_tokens_query_w = 6
        all_desc_w = []
        all_att_w = []
        all_values_w = []
        all_query_w = []


        max_tokens_desc_wn = 50
        max_tokens_att_wn = 50
        max_tokens_query_wn = 30
        all_desc_wn = []
        all_att_wn = []
        all_values_wn = []
        all_query_wn = []



        path = os.path.join(dir_base, 'data_fields_with_values.json')

        with open(path) as f:
            dt = json.load(f)

        data_csv = pd.read_csv(os.path.join(dir_base, 'features2.csv'))

        test_data = data_csv['table_id']
        query = data_csv['query']

        text_file = open(file_name, "r")
        # text_file = open("ranking_results/train.txt", "r")
        lines = text_file.readlines()

        queries_id = []
        list_lines = []

        for line in lines:
            # print(line)
            line = line[0:len(line) - 1]
            aa = line.split('\t')
            queries_id += [aa[0]]
            list_lines.append(aa)

        queries_id = [int(i) for i in queries_id]

        qq = np.sort(list(set(queries_id)))

        test_data = list(test_data)

        to_save = []
        all_query_labels = []
        all_semantic = []
        normalize = True

        all_df = get_www_all_features(os.path.join(dir_base, 'features2.csv'))

        for q in qq:
            # print(q)
            # if q>2:
            #     break
            indexes = [i for i, x in enumerate(queries_id) if x == q]
            indices = data_csv[data_csv['query_id'] == q].index.tolist()
            # print(indexes)

            inter = np.array(list_lines)[indexes]

            test_query = list(query[indices])[0]

            query_tokens = test_query

            result = es.search(query_tokens, 'desc_att', 10000)
            query_tokens = list(result.keys())[:max_tokens_query_QTE]
            query_ = pad_or_crop_with_rep(query_tokens, max_tokens_query_QTE, self.word_to_index, 'query')
            vector_query_QT = [encode_field(query_, self.word_to_index, 'query')]

            query_tokens = preprocess(test_query, 'description')
            query_ = pad_or_crop_with_rep(query_tokens, max_tokens_query_w, self.word_to_index, 'query')
            vector_query_w = [encode_field(query_, self.word_to_index, 'query')]

            QT = []
            for token in query_tokens:
                trans = synset_translation(token)
                QT += trans

            query_ = pad_or_crop_with_rep(QT, max_tokens_query_wn, self.word_to_index, 'query')
            vector_query_wn = [encode_field(query_, self.word_to_index, 'query')]

            for item in inter:
                if item[2] in test_data:
                    all_query_labels.append(q)
                    rel = float(
                        data_csv[((data_csv['query_id'] == q) & (data_csv['table_id'] == item[2]))].iloc[0]['rel'])

                    el = all_df.loc[(all_df['id_left'] == str(q)) & (all_df['id_right'] == item[2])]
                    el = el['features']
                    all_semantic.append(list(el.values)[0])

                    all_desc_QTE.append([self.word_to_index[item[2]]])
                    all_query_QTE.append([vector_query_QT])

                    table = dt[item[2]]

                    pgTitle_feat = table['pgTitle']
                    if len(pgTitle_feat) > 0:
                        pgTitle_feat = pgTitle_feat.split(' ')
                    else:
                        pgTitle_feat = []

                    secondTitle_feat = table['secondTitle']
                    if len(secondTitle_feat) > 0:
                        secondTitle_feat = secondTitle_feat.split(' ')
                    else:
                        secondTitle_feat = []

                    caption_feat = table['caption']
                    if len(caption_feat) > 0:
                        caption_feat = caption_feat.split(' ')
                    else:
                        caption_feat = []

                    description = pgTitle_feat + secondTitle_feat + caption_feat

                    original_attributes = table['attributes']
                    if len(original_attributes) > 0:
                        original_attributes = original_attributes.split(' ')
                    else:
                        original_attributes = []

                    values = table['data']
                    if len(original_attributes) > 0:
                        values = values.split(' ')
                    else:
                        values = []

                    description = pad_or_crop_with_rep(description, max_tokens_desc_w, self.word_to_index, 'description')
                    original_attributes = pad_or_crop_with_rep(original_attributes, max_tokens_att_w, self.word_to_index,
                                                               'attributes')
                    values = pad_or_crop_with_rep(values, max_tokens_desc_w, self.word_to_index, 'description')

                    vector_desc = [encode_field(description, self.word_to_index, 'attributes')]
                    vector_att = [encode_field(original_attributes, self.word_to_index, 'description')]
                    vector_values = [encode_field(values, self.word_to_index, 'description')]



                    all_desc_w.append([vector_desc])
                    all_att_w.append([vector_att])
                    all_query_w.append([vector_query_w])
                    all_values_w.append([vector_values])

                    des_t = []
                    for token in description:
                        trans = synset_translation(token)
                        des_t += trans

                    description = pad_or_crop_with_rep(des_t, max_tokens_desc_wn, self.word_to_index, 'description')

                    att_t = []
                    for token in original_attributes:
                        trans = synset_translation(token)
                        att_t += trans

                    original_attributes = pad_or_crop_with_rep(att_t, max_tokens_att_wn, self.word_to_index,
                                                               'attributes')
                    val_t = []
                    for token in values:
                        trans = synset_translation(token)
                        val_t += trans

                    values = pad_or_crop_with_rep(val_t, max_tokens_desc_wn, self.word_to_index, 'description')

                    vector_desc = [encode_field(description, self.word_to_index, 'attributes')]
                    vector_att = [encode_field(original_attributes, self.word_to_index, 'description')]
                    vector_values = [encode_field(values, self.word_to_index, 'description')]


                    all_desc_wn.append([vector_desc])
                    all_att_wn.append([vector_att])
                    all_query_wn.append([vector_query_wn])
                    all_values_wn.append([vector_values])


                    labels.append(rel)
                    to_save.append(item)

        all_semantic = np.stack(all_semantic, axis=0)
        if normalize:
            # scaler = preprocessing.StandardScaler()
            # all_semantic = scaler.fit_transform(all_semantic)
            all_semantic = preprocessing.normalize(all_semantic)

        self.all_desc_QTE = torch.tensor(all_desc_QTE)
        self.all_query_QTE = torch.tensor(all_query_QTE)

        self.all_desc_w = torch.tensor(all_desc_w)
        self.all_att_w = torch.tensor(all_att_w)
        self.all_query_w = torch.tensor(all_query_w)

        self.all_desc_wn = torch.tensor(all_desc_wn)
        self.all_att_wn = torch.tensor(all_att_wn)
        self.all_query_wn = torch.tensor(all_query_wn)


        self.all_query_labels = all_query_labels
        self.all_semantic = all_semantic
        self.all_semantic = torch.tensor(self.all_semantic)

        self.labels = labels
        inter = np.array(to_save)
        np.savetxt(output_file, inter, fmt="%s", delimiter='\t')

    def __getitem__(self, t):
        """
            return: the t-th (center, context) word pair and their co-occurrence frequency.
        """
        ## Your codes go here
        return self.all_desc_w[t],self.all_att_w[t], self.all_query_w[t], self.all_desc_wn[t], self.all_att_wn[t], \
               self.all_query_wn[t], self.all_desc_QTE[t], self.all_query_QTE[t],\
               self.labels[t], self.all_semantic[t]

    def __len__(self):
        """
            return: the total number of (center, context) word pairs.
        """
        ## Your codes go here
        return len(self.all_desc_w)
