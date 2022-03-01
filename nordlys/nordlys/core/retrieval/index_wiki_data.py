from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

import os

import json

from collections import Counter

import pandas as pd
from utils import *

data_folder='tables_redi2_1'

data_csv=pd.read_csv('features2.csv')
#attributes = list(data_csv)

test_data=data_csv['table_id']
query=data_csv['query']
relevance=data_csv['rel']