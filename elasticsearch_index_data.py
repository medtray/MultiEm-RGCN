from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

import os

import json
from random import shuffle
from collections import Counter

from os import path
import sys
sys.path.append(path.abspath('nordlys'))

from nordlys.core.retrieval.elastic import Elastic
import pandas as pd
from utils import *

from tqdm import tqdm as tqdm
index_name = "new_data_index"

mappings = {
    "attributes": Elastic.analyzed_field(),
    "description": Elastic.analyzed_field(),
    "data": Elastic.analyzed_field(),
    "desc_att": Elastic.analyzed_field(),
    "desc_att_data": Elastic.analyzed_field()
}

docs={}

path='wikiTables/data_fields_with_values.json'

with open(path) as f:
    dt = json.load(f)

with tqdm(total=len(dt)) as pbar:
    for ii,table_name in enumerate(dt):

        test_table = dt[table_name]
        #print(table_name)

        attributes = test_table['attributes']
        description = test_table['pgTitle']+' '+test_table['secondTitle']+' '+test_table['secondTitle']
        data = test_table['data']

        #description = preprocess(description, 'description')
        #description = ' '.join(description)

        #attributes = preprocess(attributes, 'attribute')
        #attributes = ' '.join(attributes)

        data = ' '.join(y for x in data for y in x)
        if table_name not in docs:
            docs[table_name] = {}
            docs[table_name]['attributes'] = attributes
            docs[table_name]['description'] = description
            docs[table_name]['data'] = data

            docs[table_name]['desc_att'] = description+' '+attributes
            docs[table_name]['desc_att_data'] = description+' '+attributes+' '+data

        pbar.update(1)


print('total number of tables is ', len(dt))






elastic = Elastic(index_name)
elastic.create_index(mappings, model='BM25',force=True)
elastic.add_docs_bulk(docs)
print("index has been built")

