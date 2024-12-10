import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
# from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable
from utilss import *

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    
    parser.add_argument('--query-path', default='', type=str)
    parser.add_argument('--graph-path', default='', type=str)
    parser.add_argument('--prefix', default='/home/nagy/TrackCountPredict/data', type=str)
    parser.add_argument('--nnode', default=5, type=int)
    parser.add_argument('--queryno', default=0, type= int)
    parser.add_argument('--k', default=1, type=int)
    
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    return args


import scipy.sparse as ssp
import torch_sparse
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
        h = model(torch.ones([len(data.nodes)], dtype = torch.long), data.adj_t)
        cnprobs = []
        edge = torch.stack([data.map_qgv(u, cand) for cand in cur_candidates]).T
        dist = predictor.multidomainforward(h, data.adj_t, edge, cndropprobs=cnprobs).detach().reshape(-1)
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

if __name__ == '__main__':
    args = parseargs()
    print(args, flush=True)

    if args.dataset in ['yeast', 'wordnet', 'dblp']:
        evaluator = Evaluator(name=f'ogbl-ddi')

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    ret = []

    model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                args.gnndp, args.ln, args.res, data.max_x,
                args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
    predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                        args.predp, args.preedp, args.lnnn).to(device)
    model.load_state_dict(torch.load(f"gmodel/{args.dataset}_model.pt", map_location="cpu"), strict=False)
    predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_predictor.pt", map_location="cpu"), strict=False)

    if args.graph_path:
        graph = load_grf_nx(os.path.abspath(args.graph_path))
        ng = graph.number_of_nodes()
    else:
        graphpath = os.path.abspath(args.prefix.rstrip('/') + '/' + args.dataset + '/data.graph')
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

