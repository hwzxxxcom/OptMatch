import argparse
import os
import models
import igraph as ig
import torch

from models import *
from dataset import *
from utils import *
from functools import partial
from copy import deepcopy

from torch.utils.data import DataLoader
import utilss 

def process_model_config(config):
    model_config = deepcopy(config)

    # for reversed edges:
    # the number of edges becomes double
    # the number of edge labels becomes double
    if config["add_rev"]:
        model_config["max_nge"] *= 2
        model_config["max_ngel"] *= 2
        model_config["max_npe"] *= 2
        model_config["max_npel"] *= 2

    if config["convert_dual"]:
        max_ngv = model_config["max_ngv"]
        max_npv = model_config["max_npv"]
        avg_gd = math.ceil(model_config["max_nge"] / model_config["max_ngv"])
        avg_pd = math.ceil(model_config["max_npe"] / model_config["max_npv"])

        model_config["max_ngv"] = model_config["max_nge"]
        model_config["max_nge"] = (avg_gd * avg_gd) * max_ngv // 2 - max_ngv
        model_config["max_npv"] = model_config["max_npe"]
        model_config["max_npe"] = (avg_pd * avg_pd) * max_npv // 2 - max_npv
        model_config["max_ngvl"] = model_config["max_ngel"]
        model_config["max_ngel"] = model_config["max_ngvl"]
        model_config["max_npvl"] = model_config["max_npel"]
        model_config["max_npel"] = model_config["max_npvl"]

    return model_config

def build_model(config, **kw):
    if config["rep_net"] == "CNN":
        model = CNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RNN":
        model = RNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "TXL":
        model = TransformerXL(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RGCN":
        model = RGCN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RGIN":
        model = RGIN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "CompGCN":
        model = CompGCN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "DMPNN":
        model = DMPNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "LRP":
        model = LRP(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "DMPLRP":
        model = DMPLRP(pred_return_weights=config["match_weights"], **config, **kw)
    return model

def load_model(path, **kw):
    model = build_model(process_model_config(config), **kw)
    model.load_state_dict(
        th.load(
            path,
            map_location=th.device("cpu")
        )
    )
    return model

import networkx as nx
def nx2graph(graph: nx.Graph):
    i_graph = ig.Graph()
    for u in graph.nodes:
        i_graph.add_vertex(id=u, label=graph.nodes[u]['l'])
    for u, v in graph.edges:
        i_graph.add_edge(u,v, key=0,label =graph.edges[u,v]['l'])
    return Graph(i_graph)

class nagydataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return self.data[0]

from config import get_train_config
config = get_train_config()
max_ngv = config["max_ngv"]
max_nge = config["max_nge"]
max_ngvl = config["max_ngvl"]
max_ngel = config["max_ngel"]
if config["share_emb_net"]:
    max_npv = max_ngv
    max_npe = max_nge
    max_npvl = max_ngvl
    max_npel = max_ngel
else:
    max_npv = config["max_npv"]
    max_npe = config["max_npe"]
    max_npvl = config["max_npvl"]
    max_npel = config["max_npel"]





def assign(query, graph, order, matches, selected, 
           candidates, reverse_order, sgldepth, dpth, k, h_vp, h_vg):
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
        next_emb = h_vp[u]
        candidate_emb = []
        if not cur_candidates: return 
        candidate_emb = h_vg[torch.tensor(cur_candidates, dtype = torch.long)]
        dist = torch.sum((candidate_emb - next_emb) ** 2, dim = -1)
        orig_k = k
        k = k if dpth < sgldepth else 1
        k = min(k, int(dist.shape[0]))
        tobeselected = [cur_candidates[int(i)] for i in 
                                        dist.topk(k, largest = False).indices]
        for cur_selected in tobeselected:
            matches.append(cur_selected)
            assert len(matches) == dpth + 1
            selected.add(cur_selected)
            if len(matches) == len(order):
                return matches
            res = assign(query, graph, order, matches, selected, 
                   candidates, reverse_order, sgldepth, dpth + 1, orig_k, h_vp, h_vg)
            if res: return res
            del matches[-1]
            selected.remove(cur_selected)

def calculate_degrees(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        for x in dataset:
            if INDEGREE not in x["pattern"].tdata:
                x["pattern"].tdata[INDEGREE] = x["pattern"].in_degrees()
            if OUTDEGREE not in x["pattern"].tdata:
                x["pattern"].tdata[OUTDEGREE] = x["pattern"].out_degrees()
            if INDEGREE not in x["graph"].tdata:
                x["graph"].tdata[INDEGREE] = x["graph"].in_degrees()
            if OUTDEGREE not in x["graph"].tdata:
                x["graph"].tdata[OUTDEGREE] = x["graph"].out_degrees()
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if INDEGREE not in x["pattern"].ndata:
                x["pattern"].ndata[INDEGREE] = x["pattern"].in_degrees()
            if OUTDEGREE not in x["pattern"].ndata:
                x["pattern"].ndata[OUTDEGREE] = x["pattern"].out_degrees()
            if INDEGREE not in x["graph"].ndata:
                x["graph"].ndata[INDEGREE] = x["graph"].in_degrees()
            if OUTDEGREE not in x["graph"].ndata:
                x["graph"].ndata[OUTDEGREE] = x["graph"].out_degrees()

def compute_largest_eigenvalues(graph):
    if isinstance(graph, dgl.DGLGraph):
        if INDEGREE in graph.ndata:
            in_deg = graph.ndata[INDEGREE].float()
        else:
            in_deg = graph.in_degrees().float()
        if OUTDEGREE in graph.ndata:
            out_deg = graph.ndata[OUTDEGREE].float()
        else:
            out_deg = graph.out_degrees().float()
        u, v = graph.all_edges(form="uv", order="eid")
        max_nd = (out_deg[u] + in_deg[v]).max()
        max_ed = (in_deg[u] + out_deg[v]).max()
    elif isinstance(graph, ig.Graph):
        if INDEGREE in graph.vertex_attributes():
            in_deg = np.asarray(graph.vs[INDEGREE]).astype(np.float32)
        else:
            in_deg = np.asarray(graph.indegree()).astype(np.float32)
        if OUTDEGREE in graph.vertex_attributes():
            out_deg = np.asarray(graph.vs[OUTDEGREE]).astype(np.float32)
        else:
            out_deg = np.asarray(graph.outdegree()).astype(np.float32)
        u, v = np.asarray(graph.get_edgelist()).T
        max_nd = (out_deg[u] + in_deg[v]).max()
        max_ed = (in_deg[u] + out_deg[v]).max()
    else:
        raise ValueError

    node_eigenv = max_nd
    edge_eigenv = max_ed

    return node_eigenv, edge_eigenv

def calculate_eigenvalues(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        pass
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if NODEEIGENV not in x["pattern"].ndata or EDGEEIGENV not in x["pattern"].edata:
                node_eigenv, edge_eigenv = compute_largest_eigenvalues(x["pattern"])
                x["pattern"].ndata[NODEEIGENV] = th.clamp_min(node_eigenv, 1.0).repeat(x["pattern"].number_of_nodes()).unsqueeze(-1)
                x["pattern"].edata[EDGEEIGENV] = th.clamp_min(edge_eigenv, 1.0).repeat(x["pattern"].number_of_edges()).unsqueeze(-1)
            if NODEEIGENV not in x["graph"].ndata or EDGEEIGENV not in x["graph"].edata:
                node_eigenv, edge_eigenv = compute_largest_eigenvalues(x["graph"])
                x["graph"].ndata[NODEEIGENV] = th.clamp_min(node_eigenv, 1.0).repeat(x["graph"].number_of_nodes()).unsqueeze(-1)
                x["graph"].edata[EDGEEIGENV] = th.clamp_min(edge_eigenv, 1.0).repeat(x["graph"].number_of_edges()).unsqueeze(-1)

if __name__ == '__main__':
    print(1)
    if config['graph_path']:
        graph_nx = utilss.load_grf_nx(os.path.abspath(config['graph_path']))
    else:
        graphpath = os.path.abspath(config['prefix'].rstrip('/') + '/' + config['dataname'] + '/data.graph')
        graph_nx = utilss.load_grf_nx(graphpath)
    print(2)

    if config['query_path']:
        querypath = os.path.abspath(config['query_path'])
    else:
        querypath = config['prefix'].rstrip('/') + '/' + config['dataname'] + '/' + 'queries_%d/queries/query_%05d.graph' % (config['nnode'], config['queryno'])
        querypath = os.path.abspath(querypath)
    print(3)
    query_nx = utilss.load_grf_nx(querypath)
    print(4)

    nagygraph = nx2graph(utilss.load_grf_nx(config['prefix'].rstrip('/') + '/%s/data.graph' % config['dataname']))
    nagypattern = nx2graph(utilss.load_grf_nx(querypath))
    print(5)

    for i in range(1000):
        querypath = config['prefix'].rstrip('/') + '/' + config['dataname'] + '/' + 'queries_%d/queries/query_%05d.graph' % (config['nnode'], i)
        querypath = os.path.abspath(querypath)
        try: 
            query_nx = utilss.load_grf_nx(querypath)
        except:
            continue
        nagypattern = nx2graph(utilss.load_grf_nx(querypath))
        dataset = nagydataset([{
                "id": "a-b",
                "pattern": nagypattern,
                "graph": nagygraph,
                "counts": 0,
                "subisomorphisms": []
            }])
        data_loader = DataLoader(
                        dataset,
                        collate_fn=partial(GraphAdjDataset.batchify, return_weights=config["match_weights"]),
                    )
        max_neigenv = 4.0
        max_eeigenv = 4.0
        calculate_degrees(dataset)
        if isinstance(dataset, GraphAdjDataset):
            # calculate_norms(datasets[data_type], self_loop=True) # models handle norms
            calculate_eigenvalues(dataset)
            for x in dataset:
                max_neigenv = max(max_neigenv, x["pattern"].ndata[NODEEIGENV][0].item())
                max_eeigenv = max(max_eeigenv, x["pattern"].edata[EDGEEIGENV][0].item())
        model = load_model('saved_models/%s_%d.pth' % (config['dataname'], config['nnode']))
        import time
        stime = time.time()
        for datapair in data_loader: break
        ids, pattern, graph, counts, (node_weights, edge_weights) = datapair
        output = model(pattern, graph)
        h_vp = output['p_v_rep']
        h_vg = output['g_v_rep']
        order, candidates = utilss.cpp_GQL(config['nnode'], querypath, config['dataname'])
        print(order, len(candidates))
        reverse_order = dict()
        for i in range(len(order)):
            reverse_order[order[i]] = i
        matches = []
        selected = set()
        result = assign(query_nx, graph_nx, order, matches, selected, candidates, reverse_order, len(order) // 2, 0, config['k'], h_vp, h_vg)
        print(time.time() - stime, 's')
        print(result)
