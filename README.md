# OptMatch

This repository is the implementation for the submission "*OptMatch: An Efficient and Generic Neural
Network-assisted Subgraph Matching Approach*".

## Enviroments

Execute `env.sh` to prepare the environment.
```bash
cd src
. ./env.sh
```

## Training
Before starting traing, please run the online filter which is implemented in C++ and helps fast filter candidates. The following is an example command for Yeast data graph.
```bash
./OnlineFilter.out -d ../data/yeast/data.graph
```
Then run the `train.py` to train models.
```bash
python train.py --dataname [name_of_data_graph] --nnode [number_of_query_graph_nodes] --qsize [number_of_distinct_labels]
```
The following is an example command for the Yeast data graph, where the query graphs each consist of 8 nodes, and they have 71 discinct labels.
```bash
python train.py --dataname yeast --nnode 8 --qsize 71 
```
## Matching
After training, please use `matcher.py` to use OptMatch.

For exact matching, OptMatch retrieves matching status of the search program `SubgraphSearching.out` implemeted in C++ via shared memory and returns the recommanded strategy. 

Taking the query `query_00001.graph` as an example, please run the following commands in two separate terminals:
```bash
./SubgraphSearching.out -d ../data/yeast/data.graph -q ../data/yeast/queries_8/queries/query_00000.graph 
```
```bash
python matcher.py --graph-path ../data_orig/yeast/data.graph --query-path ../data/yeast/queries_8/queries/query_00000.graph
```

For approximate matching, please pass `--neural` parameter to `matcher.py`. Please note that, `SubgraphSearching.out` is not needed for approximate matching.

```bash
python matcher.py --graph-path ../data_orig/yeast/data.graph --query-path ../data/yeast/queries_8/queries/query_00000.graph --neural
```
