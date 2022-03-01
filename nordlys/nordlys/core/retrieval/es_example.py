
from elasticsearch import Elasticsearch
es=Elasticsearch()

body={
  'author': 'John Doe',
  'blog': 'Learning Elasticsearch',
  'title': 'Using Python with Elasticsearch',
  'tags': ['python', 'elasticsearch', 'tips'],
}

es.index(index='test_index', doc_type='post', id=1, body=body)

tokens = es.indices.analyze(index='test_index', body={'text': 'author er tet +56 t h + 484'})

ss={
    "aggs" : {
        "grades_stats" : { "stats" : { "field" : "author" } }
    }
}

aa=es.search(index='test_index',body=ss)

print('done')
