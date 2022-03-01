"""
Toy Indexer
===========

Toy indexing example for testing purposes.

:Authors: Krisztian Balog, Faegheh Hasibi
"""
from nordlys.core.retrieval.elastic import Elastic
from nordlys.core.retrieval.elastic_cache import ElasticCache
from nordlys.core.retrieval.scorer import *
import math

def main():
    index_name = "toy_index"

    mappings = {
        "title": Elastic.analyzed_field(),
        "content": Elastic.analyzed_field(),
    }

    docs = {
        1: {"title": "Rap God pp m m m m m m m",
            "content": "gonna, gonna, Look, I was gonna go easy on you and not to hurt your feelings cc"
            },
        2: {"title": "Lose Yourself",
            "content": "Yo, if you cc could just, for one minute Or one split second in time, forget everything Everything that bothers you, or your problems Everything, and follow me"
            },
        3: {"title": "Love Way",
            "content": "Just gonna stand there and watch me burn But that's alright, because I like the way it hurts you you you you cc"
            },
        4: {"title": "Monster",
            "content": ["you gonna gonna I'm friends with the monster", "That's under my bed Get along with the voices inside of my head cc"]
            },
        5: {"title": "Beautiful",
            "content": "d"
            }
    }

    elastic = Elastic(index_name)
    elastic.create_index(mappings, model='BM25',force=True)
    elastic.add_docs_bulk(docs)
    print("index has been built")

    es = Elastic(index_name)
    feature_used='title'
    test_query='cc'
    result = es.search(test_query, feature_used)
    print(result)

    params = {"fields": feature_used}
    score = ScorerLM(es, test_query, params).score_doc(5)
    print(score)

    len_C_f = es.coll_length(feature_used)
    tf_t_C_f = es.coll_term_freq(test_query, feature_used)
    nb_docs=es.doc_count(feature_used)
    term_freq=es.doc_freq(test_query,feature_used)
    stat=es.avg_len(feature_used)
    ss=es.get_field_stats(feature_used)

    print(stat)

    idf = math.log((nb_docs + 1.0) / (term_freq + 1.0)) + 1.0

    print(len_C_f)
    print(tf_t_C_f)
    print(nb_docs)
    print(term_freq)
    print(idf)




if __name__ == "__main__":
    main()
