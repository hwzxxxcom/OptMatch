import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

from models import NeoGNN, LinkPredictor
from utils import init_seed
import numpy as np
import scipy.sparse as ssp
import os
import pickle
import torch_sparse
import networkx as nx
import warnings
from utils import AA
from torch_sparse import SparseTensor
import pandas as pd
from torch_scatter import scatter_add
import time
from utilss import *

class NagyData():
    def  __init__(self, q, g):
        self.gmap = dict()
        self.qmap = dict()
        self.nodes = []
        edges = []
        for i in g.nodes:
            self.gmap[i] = len(self.nodes)
            self.nodes.append(len(self.nodes))
        for i in q.nodes:
            self.qmap[i] = len(self.nodes)
            self.nodes.append(len(self.nodes))
        for u, v in g.edges:
            edges.append((self.gmap[u], self.gmap[v]))
        for u, v in q.edges:
            edges.append((self.qmap[u], self.qmap[v]))
        self.num_nodes = len(self.nodes)
        row, col = torch.tensor(edges).T
        edge_weight = torch.ones(row.shape[0], dtype=float)
        self.adj_t = torch_sparse.tensor.SparseTensor(row = row, col = col, 
                                                        value = torch.ones(col.shape), 
                                                        sparse_sizes = (len(self.nodes), len(self.nodes))).to_symmetric()
        row, col, _ = self.adj_t.coo()
        edge_index = torch.stack([row,col])
        edge_weight = torch.ones(edge_index.size(1), dtype=float)
        self.A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                    shape=(len(self.nodes), len(self.nodes)))
    def map_qgv(self, qv, gv):
        return torch.tensor([self.qmap[qv], self.gmap[gv]])

def samp_neighbor(g, center):
    results = [center]
    for i in range(2):
        new_nodes = sum([list(g.neighbors(selected_node)) for selected_node in results], start=[])
        results += list(set(new_nodes) - set(results))
    return (nx2nx(g.subgraph(results)))

def assign(query, graph, order, matches, selected, 
           candidates, reverse_order, sgldepth, dpth, k):
    u = order[dpth]
    #print(candidates)
    cur_candidates = []
    for v in candidates[u]:
        if v in selected:
            continue
        valid = True
        for nu in query.neighbors(u):
            #print(nu, reverse_order[nu], dpth)
            if reverse_order[nu] >= dpth: continue
            if (v, matches[reverse_order[nu]]) not in graph.edges: 
                valid = False
                break
        if valid: cur_candidates.append(v)
    #print(cur_candidates)
    if dpth  < 0: 
        raise NotImplementedError
    else:
        orig_k = k
        k = k if dpth < sgldepth else 1
        k = min(k, len(cur_candidates))
        if not cur_candidates: 
            return 
        data = NagyData(query, graph)
        dist = model(torch.stack([data.map_qgv(u, cand) for cand in cur_candidates]).T, 
              data, data.A, predictor, torch.ones([len(data.nodes), args.hidden_channels]))[2].reshape(-1)
        tobeselected = [cur_candidates[int(i)] for i in 
                                        dist.topk(k, largest = True).indices]
        for cur_selected in tobeselected:
            matches.append(cur_selected)
            assert len(matches) == dpth + 1
            selected.add(cur_selected)
            if len(matches) == len(order):
                return matches
            res = assign(query, graph, order, matches, selected, 
                   candidates, reverse_order, sgldepth, dpth + 1, orig_k)
            if res: return res
            del matches[-1]
            selected.remove(cur_selected)

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--batch_size', type=int, default= 2 * 1024)
    parser.add_argument('--gnn_batch_size', type=int, default= 64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataname', type=str, default='wordnet')


    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=128) 
    parser.add_argument('--g_phi_dim', type=int, default=128) 

    parser.add_argument('--gnn', type=str, default='NeoGNN')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--beta', type=float, default=0.1)

    parser.add_argument('--query-path', default='', type=str)
    parser.add_argument('--graph-path', default='', type=str)
    parser.add_argument('--prefix', default='/home/nagy/TrackCountPredict/data', type=str)
    parser.add_argument('--nnode', default=5, type=int)
    parser.add_argument('--queryno', default=0, type= int)
    parser.add_argument('--k', default=1, type=int)

    args = parser.parse_args()
    print(args)
    args.dataset = args.dataname

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    device = torch.device(device)

    model = NeoGNN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, args=args).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.mlp_num_layers, args.dropout).to(device)
    model.load_state_dict(torch.load('saved_models/%s_model.pth' % args.dataname))
    predictor.load_state_dict(torch.load('saved_models/%s_predictor.pth' % args.dataname))

    if args.graph_path:
        graph = load_grf_nx(os.path.abspath(args.graph_path))
        ng = graph.number_of_nodes()
    else:
        graphpath = os.path.abspath(args.prefix.rstrip('/') + '/' + args.dataname + '/data.graph')
        graph = load_grf_nx(graphpath)
    stime = time.time()
    if args.query_path:
        querypath = os.path.abspath(args.query_path)
    else:
        querypath = args.prefix.rstrip('/') + '/' + args.dataset + '/' + 'queries_%d/queries/query_%05d.graph' % (args.nnode, args.queryno)
        querypath = os.path.abspath(querypath)
    query = load_grf_nx(querypath)

    order, candidates = cpp_GQL(args.nnode, querypath, args.dataset)
    
    print(order, len(candidates))
    reverse_order = dict()
    for i in range(len(order)):
        reverse_order[order[i]] = i

    matches = []
    selected = set()
    result = assign(query, graph, order, matches, selected, candidates, reverse_order, 4 if args.nnode != 24 else 6, 0, args.k)

    print(time.time() - stime, 's')
    print(result)
# g = nx.Graph()
# g.add_node(0)
# g.add_node(1)
# g.add_node(2)
# g.add_node(3)
# g.add_edge(0,1)
# g.add_edge(1,2)
# g.add_edge(1,3)
# g.add_edge(2,3)

# data =NagyData(g,g)
# a = model(data.map_qgv(0, 1), data, data.A, predictor, torch.ones([len(data.nodes), args.hidden_channels]))
# b = model(torch.stack([data.map_qgv(0, 1), data.map_qgv(1, 0), data.map_qgv(1, 1), data.map_qgv(0, 1), data.map_qgv(1, 0), data.map_qgv(1, 1)]).T, data, data.A, predictor, torch.ones([len(data.nodes), args.hidden_channels]))
# print(a[2], b[2])