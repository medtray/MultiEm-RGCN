import numpy as np
import json
import os
from math import log
import scipy.sparse as sp
from utils_ import preprocess
from random import shuffle
import pandas as pd
from tqdm import tqdm
import random
from scipy.spatial.distance import cosine
from utils import loadWord2Vec,clean_str

from nltk.corpus import wordnet as wn
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from concurrent.futures import ThreadPoolExecutor,wait
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

nb_threads=6
executor = ThreadPoolExecutor(nb_threads)


def prepare_table(test_table):
    attributes = test_table['title']
    pgTitle = test_table['pgTitle']
    secondTitle = test_table['secondTitle']
    caption = test_table['caption']
    data = test_table['data']

    pgTitle_feat = preprocess(pgTitle, 'description')
    secondTitle_feat = preprocess(secondTitle, 'description')
    caption_feat = preprocess(caption, 'description')
    description = pgTitle_feat + secondTitle_feat + caption_feat

    data_csv = pd.DataFrame(data, columns=attributes)
    attributes = list(data_csv)

    inter_att = ' '.join(attributes)
    att_tokens_inter = preprocess(inter_att, 'attribute')

    if len(att_tokens_inter) == 0:
        data_csv = data_csv.transpose()
        # vec_att = np.array(attributes).reshape(-1, 1)
        data_csv_array = np.array(data_csv)
        # data_csv_array = np.concatenate([vec_att, data_csv_array], axis=1)
        if data_csv_array.size > 0:
            attributes = data_csv_array[0, :]
            data_csv = pd.DataFrame(data_csv_array, columns=attributes)

            data_csv = data_csv.drop([0], axis=0).reset_index(drop=True)
        else:
            data_csv = data_csv.transpose()

    all_att_tokens = []
    for att in attributes:
        att_tokens = preprocess(att, 'attribute')
        all_att_tokens += att_tokens

    original_attributes = all_att_tokens
    values = data_csv.values

    data = ' '.join(y for x in values for y in x)
    data_tokens = preprocess(data, 'description')

    return description,original_attributes,data_tokens

class PrepareGraph:

    def __init__(self, queries, dt):
        self.queries = queries
        self.dt = dt
        self.entities = {}
        self.counter = -1
        self.triples = ''
        self.test_triples = ''
        self.delimiter = '\t'
        # self.relations=['has_value','has_attribute','has_description','has_term','not_relevant_to','somehow_relevant_to','relevant_to']
        #self.relations = ['has_tfidf0', 'has_tfidf1', 'has_tfidf2', 'has_pmi0', 'has_pmi1', 'has_pmi2']
        #self.relations = ['has_tfidf0', 'has_tfidf1', 'has_tfidf2', 'has_pmi0', 'has_pmi1', 'has_pmi2',
         #                 'has_cosine0','has_cosine1','has_cosine2','has_wordnet0','has_wordnet1',
         #                 'has_wordnet2']
        self.relations = ['has_tfidf0', 'has_tfidf1', 'has_tfidf2', 'has_pmi0', 'has_pmi1', 'has_pmi2',
                          'has_cosine0', 'has_cosine1', 'has_cosine2', 'has_syn', 'has_hyper']

        #self.relations = ['has_tfidf0', 'has_tfidf1', 'has_tfidf2', 'has_pmi0', 'has_pmi1', 'has_pmi2', 'has_term',
        #                  'has_description', 'has_attribute',
        #                  'not_relevant_to',
        #                  'somehow_relevant_to', 'relevant_to']
        # self.relations = ['has_term', 'not_relevant_to',
        # 'somehow_relevant_to', 'relevant_to']

    def add_to_entities(self, tokens):
        for tok in tokens:
            if tok not in self.entities:
                self.counter += 1
                self.entities[tok] = self.counter



    def add_links_docs(self, data,tables_id_file):
        shuffle_doc_words_list_tables = []
        tables_collection = set()
        add_to_test=0.95
        tables_id = []
        collection_to_ind_tables = []
        for j, line in enumerate(data):
            #print(j)
            table = line[2]
            if j>500:
               break

            if table in tables_collection:
                continue

            table_data = self.dt[table]


            pgTitle_feat = table_data['pgTitle']
            secondTitle_feat = table_data['secondTitle']
            caption_feat = table_data['caption']

            if len(pgTitle_feat) > 0:
                pgTitle_feat = pgTitle_feat.split(' ')
            else:
                pgTitle_feat = []

            if len(secondTitle_feat) > 0:
                secondTitle_feat = secondTitle_feat.split(' ')
            else:
                secondTitle_feat = []

            if len(caption_feat) > 0:
                caption_feat = caption_feat.split(' ')
            else:
                caption_feat = []

            description = pgTitle_feat + secondTitle_feat + caption_feat

            original_attributes = table_data['attributes']
            if len(original_attributes) > 0:
                original_attributes = original_attributes.split(' ')
            else:
                original_attributes = []

            values = table_data['data']
            if len(values) > 0:
                values = values.split(' ')
            else:
                values = []

            if table not in self.entities:
                self.counter += 1
                self.entities[table] = self.counter

            tables_id.append(self.counter)

            self.add_to_entities(description)
            self.add_to_entities(original_attributes)

            text_table = description + original_attributes
            shuffle_doc_words_list_tables.append(' '.join(text_table))
            tables_collection.add(table)
            collection_to_ind_tables.append(table)


        shuffle_doc_words_list_tables_tfid = shuffle_doc_words_list_tables.copy()

        add_wikiTables_path=os.path.join('wikiTables','wikiPreprocessed.json')
        with open(add_wikiTables_path) as f:
            add_wikiTables = json.load(f)

        list_of_categories = list(add_wikiTables.keys())
        nb_files = len(list_of_categories)
        mylist = list(range(nb_files))
        shuffle(mylist)
        list_of_categories = np.array(list_of_categories)[mylist]
        list_of_categories=[]
        nb_files_in_training = nb_files

        with tqdm(total=nb_files_in_training) as pbar0:
            for jj, category in enumerate(list_of_categories):
                #print(jj)

                dt = add_wikiTables[category]
                list_of_tables = list(dt.keys())
                nb_tables = len(list_of_tables)
                mylist_tables = list(range(nb_tables))
                shuffle(mylist_tables)
                list_of_tables = np.array(list_of_tables)[mylist_tables]
                nb_tables_in_training_file = nb_tables

                for tab_id,table_name in enumerate(list_of_tables):
                    if tab_id>=nb_tables_in_training_file:
                        break
                    test_table = dt[table_name]
                    if table_name in self.dt:
                        continue

                    pgTitle_feat = test_table['pgTitle']
                    secondTitle_feat = test_table['secondTitle']
                    caption_feat = test_table['caption']
                    if len(pgTitle_feat) > 0:
                        pgTitle_feat = pgTitle_feat.split(' ')
                    else:
                        pgTitle_feat = []

                    if len(secondTitle_feat) > 0:
                        secondTitle_feat = secondTitle_feat.split(' ')
                    else:
                        secondTitle_feat = []

                    if len(caption_feat) > 0:
                        caption_feat = caption_feat.split(' ')
                    else:
                        caption_feat = []


                    description = pgTitle_feat + secondTitle_feat + caption_feat

                    original_attributes = test_table['attributes']

                    if len(original_attributes) > 0:
                        original_attributes = original_attributes.split(' ')
                    else:
                        original_attributes = []


                    text_table = description + original_attributes
                    shuffle_doc_words_list_tables_tfid.append(' '.join(text_table))

                    if random.random()>0.99999:
                        if table_name not in self.entities:
                            self.counter += 1
                            self.entities[table_name] = self.counter
                        tables_id.append(self.counter)

                        self.add_to_entities(description)
                        self.add_to_entities(original_attributes)

                        text_table = description + original_attributes
                        shuffle_doc_words_list_tables.append(' '.join(text_table))
                        tables_collection.add(table_name)
                        collection_to_ind_tables.append(table_name)



                pbar0.update(1)

        np.save(tables_id_file, tables_id)

        def buil_vocab(shuffle_doc_words_list):

            # build vocab
            word_freq = {}
            word_set = set()
            for doc_words in shuffle_doc_words_list:
                words = doc_words.split()
                for word in words:
                    word_set.add(word)
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
            vocab = list(word_set)
            word_id_map = {}
            for i,w in enumerate(vocab):
                word_id_map[w] = i
            return vocab,word_id_map

        tables_vocab,word_id_map_tables = buil_vocab(shuffle_doc_words_list_tables)
        tables_vocab_size = len(tables_vocab)

        def prepare_for_word_doc_freq(shuffle_doc_words_list):

            word_doc_list = {}
            for i in range(len(shuffle_doc_words_list)):
                doc_words = shuffle_doc_words_list[i]
                words = doc_words.split()
                appeared = set()
                for word in words:
                    if word in appeared:
                        continue
                    if word in word_doc_list:
                        doc_list = word_doc_list[word]
                        doc_list.append(i)
                        word_doc_list[word] = doc_list
                    else:
                        word_doc_list[word] = [i]
                    appeared.add(word)

            word_doc_freq = {}
            for word, doc_list in word_doc_list.items():
                word_doc_freq[word] = len(doc_list)

            return word_doc_freq

        word_doc_freq_tables = prepare_for_word_doc_freq(shuffle_doc_words_list_tables_tfid)

        def calculate_wordnet_syn_triples(vocab,vocab_size):

            hyper_dict={}

            with tqdm(total=vocab_size) as pbar4:
                for word in vocab:
                    synsets = wn.synsets(clean_str(word.strip()))
                    for syn in synsets:
                        triple = word + self.delimiter + 'has_syn' + self.delimiter + syn._name + '\n'
                        if syn._name not in self.entities:
                            self.counter += 1
                            self.entities[syn._name] = self.counter
                        self.triples += triple
                        if random.random() > add_to_test:
                            self.test_triples += triple

                    stack=synsets
                    while stack:
                        el=stack.pop()
                        hypers=el.hypernyms()
                        stack+=hypers
                        for hyper in hypers:
                            if el._name not in hyper_dict:
                                triple = el._name + self.delimiter + 'has_hyper' + self.delimiter + hyper._name + '\n'
                                if el._name not in self.entities:
                                    self.counter += 1
                                    self.entities[el._name] = self.counter

                                if hyper._name not in self.entities:
                                    self.counter += 1
                                    self.entities[hyper._name] = self.counter

                                self.triples += triple
                                if random.random() > add_to_test:
                                    self.test_triples += triple

                                hyper_dict[el._name]=[hyper._name]


                            else:
                                val=hyper_dict[el._name]
                                if hyper._name not in val:
                                    triple = el._name + self.delimiter + 'has_hyper' + self.delimiter + hyper._name + '\n'
                                    if el._name not in self.entities:
                                        self.counter += 1
                                        self.entities[el._name] = self.counter

                                    if hyper._name not in self.entities:
                                        self.counter += 1
                                        self.entities[hyper._name] = self.counter

                                    self.triples += triple
                                    if random.random() > add_to_test:
                                        self.test_triples += triple

                                    hyper_dict[el._name].append(hyper._name)

                    pbar4.update(1)
            pass



        # def calculate_wordnet_syn(vocab,vocab_size):
        #
        #     tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english')
        #     definitions = []
        #     with tqdm(total=vocab_size) as pbar4:
        #         for word in vocab:
        #             word = word.strip()
        #             synsets = wn.synsets(clean_str(word))
        #             hypers = []
        #             word_defs = []
        #             for synset in synsets:
        #                 syn_def = synset.definition()
        #                 word_defs.append(syn_def)
        #                 hypers += synset.hypernyms()
        #
        #             stack = hypers.copy()
        #             all_hypers = []
        #             while stack:
        #                 el = stack.pop()
        #                 all_hypers.append(el)
        #                 hypers = el.hypernyms()
        #                 stack += hypers
        #
        #             all_hypers = list(set(all_hypers))
        #
        #             for hyper in all_hypers:
        #                 syn_def = hyper.definition()
        #                 word_defs.append(syn_def)
        #
        #             word_des = ' '.join(word_defs)
        #             if word_des == '':
        #                 word_des = '<PAD>'
        #             definitions.append(word_des)
        #
        #             pbar4.update(1)
        #
        #     #string = '\n'.join(definitions)
        #
        #     tfidf_matrix = tfidf_vec.fit_transform(definitions)
        #     tfidf_matrix_array = tfidf_matrix.toarray()
        #     #print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))
        #
        #     word_vectors = []
        #
        #     for i in range(len(vocab)):
        #         word = vocab[i]
        #         vector = tfidf_matrix_array[i]
        #         str_vector = []
        #         for j in range(len(vector)):
        #             str_vector.append(str(vector[j]))
        #         temp = ' '.join(str_vector)
        #         word_vector = word + ' ' + temp
        #         word_vectors.append(word_vector)
        #
        #     string = '\n'.join(word_vectors)
        #
        #     dataset = 'wikiTables'
        #
        #     f = open('./' + dataset + '/word_vectors_wordnet.txt', 'w')
        #     f.write(string)
        #     f.close()
        #
        #     word_vector_file = './' + dataset + '/word_vectors_wordnet.txt'
        #     _, _, word_vector_map = loadWord2Vec(word_vector_file)
        #
        #     manager = multiprocessing.Manager()
        #     return_dict = manager.dict()
        #
        #     start_indices=[]
        #     end_indices=[]
        #     step=int(len(vocab)/nb_threads)
        #     for ii in range(nb_threads):
        #         start_indices.append(ii*step)
        #         if ii==nb_threads-1:
        #             end_indices.append(len(vocab))
        #         else:
        #             end_indices.append((ii+1) * step)
        #
        #     print(start_indices)
        #     print(end_indices)
        #
        #     def func(start_index, end_index, thread_index, return_dict):
        #         row = []
        #         col = []
        #         weight = []
        #
        #         with tqdm(total=end_index - start_index) as pbar4:
        #             for i in range(start_index, end_index):
        #                 for j in range(i, vocab_size):
        #                     vector_i = np.array(word_vector_map[vocab[i]])
        #                     vector_j = np.array(word_vector_map[vocab[j]])
        #                     similarity = 1.0 - cosine(vector_i, vector_j)
        #                     if similarity > 0.5:
        #                         # print(vocab[i], vocab[j], similarity)
        #                         row.append(i)
        #                         col.append(j)
        #                         weight.append(similarity)
        #
        #                 pbar4.update(1)
        #
        #         return_dict[thread_index] = {0: row, 1: col, 2: weight}
        #
        #
        #     processes=[]
        #
        #     for thread_index in range(nb_threads):
        #         p = multiprocessing.Process(target=func, args=(start_indices[thread_index],end_indices[thread_index],thread_index,return_dict,))
        #         processes.append(p)
        #         p.start()
        #
        #     for process in processes:
        #         process.join()
        #
        #
        #     final_rows=[]
        #     final_cols=[]
        #     final_weights=[]
        #     for i in range(nb_threads):
        #         final_rows+=return_dict[i][0]
        #         final_cols+=return_dict[i][1]
        #         final_weights+=return_dict[i][2]
        #
        #     wordnet_between_words = sp.csr_matrix(
        #         (final_weights, (final_rows, final_cols)), shape=(vocab_size, vocab_size))
        #
        #     return wordnet_between_words
        #
        #
        # def wordnet_to_triple(wordnet_between_words, vocab, nbins):
        #
        #     entity_index, word_index = wordnet_between_words.nonzero()
        #     pmi_values = wordnet_between_words.data
        #
        #     arr = np.copy(pmi_values)
        #
        #     mean = np.mean(pmi_values, axis=0)
        #     sd = np.std(pmi_values, axis=0)
        #
        #     final_list = [x for x in arr if (x > mean - 2 * sd)]
        #     final_list = [x for x in final_list if (x < mean + 2 * sd)]
        #
        #     start = np.min(final_list)
        #     end = np.max(final_list)
        #     step = (end - start) / nbins
        #
        #     bins = []
        #     for i in range(nbins - 1):
        #         bins.append(start + step * (i + 1))
        #     inds = np.digitize(pmi_values, bins)
        #
        #     manager = multiprocessing.Manager()
        #     return_dict = manager.dict()
        #
        #     start_indices = []
        #     end_indices = []
        #     step = int(len(entity_index) / nb_threads)
        #     for ii in range(nb_threads):
        #         start_indices.append(ii * step)
        #         if ii == nb_threads - 1:
        #             end_indices.append(len(entity_index))
        #         else:
        #             end_indices.append((ii + 1) * step)
        #
        #     print(start_indices)
        #     print(end_indices)
        #
        #     def func(start_index, end_index, thread_index,return_dict):
        #         triples=''
        #         test_triples=''
        #         with tqdm(total=end_index - start_index) as pbar4:
        #             for k in range(start_index,end_index):
        #                 entity = vocab[entity_index[k]]
        #                 wordd = vocab[word_index[k]]
        #                 if k==end_index-1:
        #                     triple = entity + self.delimiter + 'has_wordnet' + str(
        #                         inds[k]) + self.delimiter + wordd
        #                 else:
        #                     triple = entity + self.delimiter + 'has_wordnet' + str(
        #                         inds[k]) + self.delimiter + wordd + '\n'
        #
        #                 # if inds[k]==2:
        #                 #   triple += entity + self.delimiter + 'has_pmi' + str(1) + self.delimiter + wordd + '\n'
        #                 triples += triple
        #                 if k == end_index - 1:
        #                     test_triples += triple
        #                 else:
        #                     if random.random() > add_to_test:
        #                         test_triples += triple
        #
        #                 pbar4.update(1)
        #
        #         return_dict[thread_index] ={0:triples,1:test_triples}
        #
        #     processes = []
        #
        #     for thread_index in range(nb_threads):
        #         p = multiprocessing.Process(target=func, args=(
        #         start_indices[thread_index], end_indices[thread_index], thread_index,return_dict,))
        #         processes.append(p)
        #         p.start()
        #
        #     for process in processes:
        #         process.join()
        #
        #     for i in range(nb_threads):
        #         if i==nb_threads-1:
        #             self.triples += return_dict[i][0]
        #             print(len(self.triples))
        #             self.test_triples += return_dict[i][1]
        #         else:
        #             self.triples += return_dict[i][0] + '\n'
        #             print(len(self.triples))
        #             self.test_triples += return_dict[i][1] + '\n'



        def calculate_cosine_similarity(vocab, vocab_size):

            #word_vector_file = '/home/mohamedt/ARCI/glove.6B.50d.txt'
            word_vector_file = '/home/mohamed/PycharmProjects/glove.6B/glove.6B.50d.txt'
            _,_, word_vector_map = loadWord2Vec(word_vector_file)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            start_indices = []
            end_indices = []
            step = int(len(vocab) / nb_threads)
            for ii in range(nb_threads):
                start_indices.append(ii * step)
                if ii == nb_threads - 1:
                    end_indices.append(len(vocab))
                else:
                    end_indices.append((ii + 1) * step)

            print(start_indices)
            print(end_indices)

            def func(start_index, end_index, thread_index, return_dict):
                row = []
                col = []
                weight = []

                with tqdm(total=end_index-start_index) as pbar4:
                    for i in range(start_index,end_index):
                        for j in range(i, vocab_size):
                            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                                vector_i = np.array(word_vector_map[vocab[i]])
                                vector_j = np.array(word_vector_map[vocab[j]])
                                similarity = 1.0 - cosine(vector_i, vector_j)
                                if similarity > 0.5:
                                    # print(vocab[i], vocab[j], similarity)
                                    row.append(i)
                                    col.append(j)
                                    weight.append(similarity)

                        pbar4.update(1)

                return_dict[thread_index] = {0: row, 1: col, 2: weight}

            processes = []

            for thread_index in range(nb_threads):
                p = multiprocessing.Process(target=func, args=(
                start_indices[thread_index], end_indices[thread_index], thread_index, return_dict,))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()

            final_rows = []
            final_cols = []
            final_weights = []
            for i in range(nb_threads):
                final_rows += return_dict[i][0]
                final_cols += return_dict[i][1]
                final_weights += return_dict[i][2]

            del manager

            cosine_between_words = sp.csr_matrix(
                (final_weights, (final_rows, final_cols)), shape=(vocab_size, vocab_size))

            return cosine_between_words



        def cosine_to_triple(cosine_between_words, vocab, nbins):

            entity_index, word_index = cosine_between_words.nonzero()
            pmi_values = cosine_between_words.data

            arr = np.copy(pmi_values)

            mean = np.mean(pmi_values, axis=0)
            sd = np.std(pmi_values, axis=0)

            final_list = [x for x in arr if (x > mean - 2 * sd)]
            final_list = [x for x in final_list if (x < mean + 2 * sd)]

            start = np.min(final_list)
            end = np.max(final_list)
            step = (end - start) / nbins

            bins = []
            for i in range(nbins - 1):
                bins.append(start + step * (i + 1))
            inds = np.digitize(pmi_values, bins)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            start_indices = []
            end_indices = []
            step = int(len(entity_index) / nb_threads)
            for ii in range(nb_threads):
                start_indices.append(ii * step)
                if ii == nb_threads - 1:
                    end_indices.append(len(entity_index))
                else:
                    end_indices.append((ii + 1) * step)

            print(start_indices)
            print(end_indices)

            def func(start_index, end_index, thread_index, return_dict):
                triples = ''
                test_triples = ''
                nb_samples=0
                with tqdm(total=end_index - start_index) as pbar4:
                    for k in range(start_index, end_index):
                        nb_samples+=1
                        entity = vocab[entity_index[k]]
                        wordd = vocab[word_index[k]]
                        if k == end_index - 1:
                            triple = entity + self.delimiter + 'has_cosine' + str(
                                inds[k]) + self.delimiter + wordd
                        else:
                            triple = entity + self.delimiter + 'has_cosine' + str(
                                inds[k]) + self.delimiter + wordd + '\n'

                        # if inds[k]==2:
                        #   triple += entity + self.delimiter + 'has_pmi' + str(1) + self.delimiter + wordd + '\n'
                        triples += triple
                        if k == end_index - 1:
                            test_triples += triple
                        else:
                            if random.random() > add_to_test:
                                test_triples += triple

                        pbar4.update(1)

                return_dict[thread_index] = {0: triples, 1: test_triples,2:nb_samples}

            processes = []

            for thread_index in range(nb_threads):
                p = multiprocessing.Process(target=func, args=(
                    start_indices[thread_index], end_indices[thread_index], thread_index, return_dict,))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()

            ll=0
            for i in range(nb_threads):
                ll+=len(return_dict[i][0].split('\n'))
                self.triples += return_dict[i][0] + '\n'
                #print(len(self.triples))
                self.test_triples += return_dict[i][1] + '\n'

            #print(ll)




        def calcualte_pmi(shuffle_doc_words_list, word_id_map, vocab, vocab_size):

            # word co-occurence with context windows
            window_size = 20
            windows = []

            for doc_words in shuffle_doc_words_list:
                words = doc_words.split()
                length = len(words)
                if length <= window_size:
                    windows.append(words)
                else:
                    # print(length, length - window_size + 1)
                    for j in range(length - window_size + 1):
                        window = words[j: j + window_size]
                        windows.append(window)
                        # print(window)

            word_window_freq = {}
            with tqdm(total=len(windows)) as pbar:
                for window in windows:
                    appeared = set()
                    for i in range(len(window)):
                        if window[i] in appeared:
                            continue
                        if window[i] in word_window_freq:
                            word_window_freq[window[i]] += 1
                        else:
                            word_window_freq[window[i]] = 1
                        appeared.add(window[i])
                    pbar.update(1)

            word_pair_count = {}
            with tqdm(total=len(windows)) as pbar2:

                for kk,window in enumerate(windows):
                    # if kk==58:
                    #     print('here')
                    window.sort()
                    for i in range(1, len(window)):
                        for j in range(0, i):
                            word_i = window[i]
                            word_j = window[j]
                            if word_i not in word_id_map or word_j not in word_id_map:
                                continue
                            word_i_id = word_id_map[word_i]

                            word_j_id = word_id_map[word_j]
                            if word_i_id == word_j_id:
                                continue
                            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                            if word_pair_str in word_pair_count:
                                word_pair_count[word_pair_str] += 1
                            else:
                                word_pair_count[word_pair_str] = 1
                            # two orders
                            # word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                            # if word_pair_str in word_pair_count:
                            #    word_pair_count[word_pair_str] += 1
                            # else:
                            #    word_pair_count[word_pair_str] = 1

                    pbar2.update(1)

            row = []
            col = []
            weight = []

            # pmi as weights

            num_window = len(windows)

            with tqdm(total=len(word_pair_count)) as pbar3:
                for key in word_pair_count:
                    temp = key.split(',')
                    i = int(temp[0])
                    j = int(temp[1])
                    count = word_pair_count[key]
                    word_freq_i = word_window_freq[vocab[i]]
                    word_freq_j = word_window_freq[vocab[j]]
                    pmi = log((1.0 * count / num_window) /
                              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
                    # if pmi is None:
                    #     print('pmiiiiiiiii')
                    #pmi=10*random.random()+10
                    if pmi <= 0:
                        pbar3.update(1)
                        #print('pmi({},{})=0'.format(vocab[i],vocab[j]))
                        continue
                    row.append(i)
                    col.append(j)
                    weight.append(pmi)
                    pbar3.update(1)

            pmi_between_words = sp.csr_matrix(
                (weight, (row, col)), shape=(vocab_size, vocab_size))

            return pmi_between_words




        def pmi_to_triple(pmi_between_words, vocab, nbins):

            entity_index, word_index = pmi_between_words.nonzero()
            pmi_values = pmi_between_words.data

            arr = np.copy(pmi_values)

            mean = np.mean(pmi_values, axis=0)
            sd = np.std(pmi_values, axis=0)

            final_list = [x for x in arr if (x > mean - 2 * sd)]
            final_list = [x for x in final_list if (x < mean + 2 * sd)]

            start = np.min(final_list)
            end = np.max(final_list)
            step = (end - start) / nbins

            bins = []
            for i in range(nbins - 1):
                bins.append(start + step * (i + 1))
            inds = np.digitize(pmi_values, bins)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            start_indices = []
            end_indices = []
            step = int(len(entity_index) / nb_threads)
            for ii in range(nb_threads):
                start_indices.append(ii * step)
                if ii == nb_threads - 1:
                    end_indices.append(len(entity_index))
                else:
                    end_indices.append((ii + 1) * step)

            print(start_indices)
            print(end_indices)

            def func(start_index, end_index, thread_index, return_dict):
                triples = ''
                test_triples = ''
                with tqdm(total=end_index - start_index) as pbar4:
                    for k in range(start_index, end_index):
                        entity = vocab[entity_index[k]]
                        wordd = vocab[word_index[k]]
                        if k == end_index - 1:
                            triple = entity + self.delimiter + 'has_pmi' + str(
                                inds[k]) + self.delimiter + wordd
                        else:
                            triple = entity + self.delimiter + 'has_pmi' + str(
                                inds[k]) + self.delimiter + wordd + '\n'

                        # if inds[k]==2:
                        #   triple += entity + self.delimiter + 'has_pmi' + str(1) + self.delimiter + wordd + '\n'
                        triples += triple
                        if k == end_index - 1:
                            test_triples += triple
                        else:
                            if random.random() > add_to_test:
                                test_triples += triple

                        pbar4.update(1)

                return_dict[thread_index] = {0: triples, 1: test_triples}

            processes = []

            for thread_index in range(nb_threads):
                p = multiprocessing.Process(target=func, args=(
                    start_indices[thread_index], end_indices[thread_index], thread_index, return_dict,))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()


            for i in range(nb_threads):
                self.triples += return_dict[i][0] + '\n'
                #print(len(self.triples))
                self.test_triples += return_dict[i][1] + '\n'


        # doc word frequency

        def tf_idf_calculation(shuffle_doc_words_list, word_id_map, word_doc_freq, vocab, vocab_size):
            doc_word_freq = {}

            with tqdm(total=len(shuffle_doc_words_list)) as pbar5:
                for doc_id in range(len(shuffle_doc_words_list)):
                    doc_words = shuffle_doc_words_list[doc_id]
                    words = doc_words.split()
                    for word in words:
                        word_id = word_id_map[word]
                        doc_word_str = str(doc_id) + ',' + str(word_id)
                        if doc_word_str in doc_word_freq:
                            doc_word_freq[doc_word_str] += 1
                        else:
                            doc_word_freq[doc_word_str] = 1

                    pbar5.update(1)

            row = []
            col = []
            weight = []

            with tqdm(total=len(shuffle_doc_words_list)) as pbar6:
                for i in range(len(shuffle_doc_words_list)):
                    doc_words = shuffle_doc_words_list[i]
                    words = doc_words.split()
                    doc_word_set = set()
                    for word in words:
                        if word in doc_word_set:
                            continue
                        j = word_id_map[word]
                        key = str(i) + ',' + str(j)
                        freq = doc_word_freq[key]

                        row.append(i)
                        col.append(j)
                        idf = log(1.0 * len(shuffle_doc_words_list_tables_tfid) /
                                  word_doc_freq[vocab[j]])
                        weight.append(freq * idf)
                        doc_word_set.add(word)

                    pbar6.update(1)

            tf_idf = sp.csr_matrix(
                (weight, (row, col)), shape=(len(shuffle_doc_words_list), vocab_size))

            return tf_idf



        def tf_idf_to_triple(tf_idf, collection_to_ind, vocab, nbins_tfidf):

            entity_index, word_index = tf_idf.nonzero()
            tfidf_values = tf_idf.data
            arr = np.copy(tfidf_values)

            mean = np.mean(tfidf_values, axis=0)
            sd = np.std(tfidf_values, axis=0)

            final_list = [x for x in arr if (x > mean - 2 * sd)]
            final_list = [x for x in final_list if (x < mean + 2 * sd)]

            start = np.min(final_list)
            end = np.max(final_list)
            step = (end - start) / nbins_tfidf

            bins = []
            for i in range(nbins_tfidf - 1):
                bins.append(start + step * (i + 1))
            inds = np.digitize(tfidf_values, bins)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            start_indices = []
            end_indices = []
            step = int(len(entity_index) / nb_threads)
            for ii in range(nb_threads):
                start_indices.append(ii * step)
                if ii == nb_threads - 1:
                    end_indices.append(len(entity_index))
                else:
                    end_indices.append((ii + 1) * step)

            print(start_indices)
            print(end_indices)

            def func(start_index, end_index, thread_index, return_dict):
                triples = ''
                test_triples = ''
                with tqdm(total=end_index - start_index) as pbar4:
                    for k in range(start_index, end_index):
                        entity = collection_to_ind[entity_index[k]]
                        wordd = vocab[word_index[k]]
                        if k == end_index - 1:
                            triple = entity + self.delimiter + 'has_tfidf' + str(
                                inds[k]) + self.delimiter + wordd
                        else:
                            triple = entity + self.delimiter + 'has_tfidf' + str(
                                inds[k]) + self.delimiter + wordd + '\n'

                        # if inds[k]==2:
                        #   triple += entity + self.delimiter + 'has_pmi' + str(1) + self.delimiter + wordd + '\n'
                        triples += triple
                        if k == end_index - 1:
                            test_triples += triple
                        else:
                            if random.random() > add_to_test:
                                test_triples += triple

                        pbar4.update(1)

                return_dict[thread_index] = {0: triples, 1: test_triples}

            processes = []

            for thread_index in range(nb_threads):
                p = multiprocessing.Process(target=func, args=(
                    start_indices[thread_index], end_indices[thread_index], thread_index, return_dict,))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()

            for i in range(nb_threads):
                if i == nb_threads - 1:
                    self.triples += return_dict[i][0]
                    # print(len(self.triples))
                    self.test_triples += return_dict[i][1]
                else:
                    self.triples += return_dict[i][0] + '\n'
                    # print(len(self.triples))
                    self.test_triples += return_dict[i][1] + '\n'

        # all_vocab = tables_vocab
        all_vocab_size = len(tables_vocab)
        #word_id_map_all_voab = word_id_map_tables

        print('>>>>>start pmi between words')
        # pmi_between_words = calcualte_pmi(all_doc_words_list, word_id_map_all_voab, all_vocab, all_vocab_size)
        pmi_between_words = calcualte_pmi(shuffle_doc_words_list_tables_tfid, word_id_map_tables, tables_vocab,
                                          all_vocab_size)
        print('>>>>>start pmi to triples')
        nbins_pmi = 3
        pmi_to_triple(pmi_between_words, tables_vocab, nbins_pmi)

        print('>>>>>start wordnet syns triples')
        #calculate_wordnet_syn_triples(tables_vocab, all_vocab_size)

        # wordnet_between_words=calculate_wordnet_syn(all_vocab[:1000], 1000)
        # nbins_wordnet = 3
        # wordnet_to_triple(wordnet_between_words, all_vocab, nbins_wordnet)

        print('>>>>>start cosine similarity between words')
        cosine_between_words = calculate_cosine_similarity(tables_vocab, all_vocab_size)
        nbins_cosine = 3
        print('>>>>>start cosine similarity to triples')
        cosine_to_triple(cosine_between_words, tables_vocab, nbins_cosine)

        print('>>>>>start tf-idf calculation')
        tf_idf_tables = tf_idf_calculation(shuffle_doc_words_list_tables, word_id_map_tables, word_doc_freq_tables,
                                           tables_vocab, tables_vocab_size)
        nbins_tfidf = 3
        print('>>>>>start tf-idf to triples')
        # can be time consuming
        tf_idf_to_triple(tf_idf_tables, collection_to_ind_tables, tables_vocab, nbins_tfidf)



