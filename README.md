# Relational Graph Embeddings for Table Retrieval

This repository contains source code for the [`MultiEm-RGCN` model](https://ieeexplore.ieee.org/abstract/document/9378239), a two-phased table retrieval
method which uses graph embeddings pretrained on a large table corpus. In `phase I`, we propose a new knowledge graph that incorporates both dataset-dependent and dataset-agnostic knowledge from table corpus. External semantic and lexical resources are used for edges and nodes leading to an heterogeneous graph. Multiple types of embeddings are learned simultaneously from our proposed knowledge graph using `R-GCN` with the link prediction pre-training task. This leads to an embedding for each node in the knowledge graph (word,`WordNet`, and table nodes). In `phase II`, `R-GCN` embeddings are incorporated into an LTR architecture that combines multiple embeddings from our heterogeneous graph to solve the table retrieval task.

## Installation

First, install the conda environment `multiemrgcn` with supporting libraries.

```bash
conda create --name multiemrgcn python=3.7
conda activate multiemrgcn
pip install -r requirements.txt
```

## Data

We use `WikiTables` collection for evaluation.[`WikiTables` corpus](http://websail-fe.cs.northwestern.edu/TabEL/tables.json.gz) contains over 1.6𝑀 tables that are extracted from Wikipedia. Each table has five indexable fields: table caption, attributes (column headings), data rows, page title, and section title. In addition, each table contains statistics: number of columns, number of rows, and set of numerical columns of the table. We use the same queries that were used by [Zhang and Balog](https://github.com/iai-group/www2018-table), where every query-table pair is evaluated using three numbers: 0 means “irrelevant”, 1 means “partially relevant” and 2 means “relevant”. This collection has 60 queries with ground-truth relevance judgments. 

## Training with MultiEm-RGCN

Pre-trained graph embeddings and preprocessed data could be downloaded from this [Google Drive shared folder](https://drive.google.com/file/d/1cytoEki7nPompD0P4RQG199FvajUeWsP/view?usp=sharing). Please uncompress the zip folder and add it inside the code folder.

To train the embedding from scratch for `phase I` (preprocessed data is needed):
```bash
python save_graph_info3.py
python train_unsupervised.py
```
after training, the next step is to save the embeddings for all nodes in the graph:
```bash
cd LTR
python prepare_embedding.py
```
After obtaining the graph embeddings, the `phase II` consists of using the graph embeddings for table search:

the first step is to run [elasticsearch-5.5.3](https://www.elastic.co/downloads/past-releases/elasticsearch-5-5-3) that is needed to compute the `pseudo-query` from the table corpus. Then, the table corpus should be indexed using `elasticsearch` (the preprocced data is needed):
```bash
python elasticsearch_index_data.py
```
The elasticsearch interface code is from [nordlys](https://github.com/iai-group/nordlys).

The second step is to train `MultiEm-RGCN` `phase II` with the default parameters:
```bash
cd LTR
python kfolds_joint_embedding.py
```

## Reference

If you plan to use `MultiEm-RGCN` in your project, please consider citing [our paper](https://ieeexplore.ieee.org/abstract/document/9378239):

```bash
@INPROCEEDINGS{trabelsi20bigdata,
  author={Trabelsi, Mohamed and Chen, Zhiyu and Davison, Brian D. and Heflin, Jeff},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)}, 
  title={Relational Graph Embeddings for Table Retrieval}, 
  year={2020},
  volume={},
  number={},
  pages={3005-3014},
  doi={10.1109/BigData50022.2020.9378239}}
```
 ## Contact
  
  if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu


