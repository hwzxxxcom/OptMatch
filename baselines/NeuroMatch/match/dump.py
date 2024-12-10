import argparse
import os
import models
from utilss import *
from utils import *


def nx2nx(graph):
    o2n = dict()
    directed = graph.is_directed()
    ngraph = nx.Graph()
    for v in graph.nodes:
        o2n[v] = len(ngraph.nodes)
        ngraph.add_node(o2n[v])
    for u, v in graph.edges:
        ngraph.add_edge(o2n[u], o2n[v])
    return ngraph

class nagydata:
    def __init__(self,x,e):
        self.node_feature = torch.cat([torch.tensor([[1.]]), torch.zeros(x.shape[0] - 1, 1)], dim = 0)
        self.edge_index = e
        self.batch = torch.tensor([0] * x.shape[0])
        self.edge_label_index = e
        self.node_label_index = torch.arange(x.shape[0])
        self.G = [nx.Graph()]
        for i in range(x.shape[0]): self.G[0].add_node(0, node_feature=x[i])
        for u, v in e.T: self.G[0].add_edge(int(u), int(v))

def samp_neighbor(g, center):
    results = [center]
    for i in range(2):
        new_nodes = sum([list(g.neighbors(selected_node)) for selected_node in results], start=[])
        k = min(len(new_nodes), 256)
        new_nodes =random.sample(new_nodes, k)
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
        input = samp_neighbor(query, u)
        next_emb = model.emb_model(nagydata(torch.ones([len(input.nodes), 1]), torch.tensor(list(input.edges)).T))
        candidate_emb = []
        if not cur_candidates: return 
        for cand in cur_candidates:
            input = samp_neighbor(graph, cand)
            candidate_emb.append(model.emb_model(nagydata(torch.ones([len(input.nodes), 1]), torch.tensor(list(input.edges)).T)))
        candidate_emb = torch.cat(candidate_emb, dim = 0)
        dist = torch.sum(candidate_emb - next_emb, -1)
        orig_k = k
        k = k if dpth < sgldepth else 1
        k = min(k, int(dist.shape[0]))
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

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                        help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    enc_parser.set_defaults(conv_type='GIN',
                        method_type='order',
                        dataset='atlas',
                        n_layers=8,
                        batch_size=64,
                        hidden_dim=64,
                        skip="learnable",
                        dropout=0.0,
                        n_batches=1000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=1000,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True)

def build_model(args):
    # build model
    if args.method_type == "order":
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    model.to('cpu')
    if args.test and args.model_path:
        model.load_state_dict(torch.load(args.model_path,
            map_location='cpu'))
    return model

if __name__ == '__main__':
    
    if os.path.exists('/dev/shm/learnsc_filter.ready'):
        os.remove('/dev/shm/learnsc_filter.ready')
    parser = argparse.ArgumentParser()
    parser = (argparse.ArgumentParser(description='Order embedding arguments'))
    parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query-path', default='', type=str)
    parser.add_argument('--graph-path', default='', type=str)
    parser.add_argument('--prefix', default='/home/nagy/TrackCountPredict/data', type=str)
    parser.add_argument('--nnode', default=5, type=int)
    parser.add_argument('--queryno', default=0, type= int)
    parser.add_argument('--k', default=1, type=int)
    args = parser.parse_args()

    if args.graph_path:
        graph = load_grf_nx(os.path.abspath(args.graph_path))
        emb = load_emb(args.prefix.rstrip('/') + '/yeast')
        ng = graph.number_of_nodes()
        emb = torch.concatenate([emb] * (ng // emb.shape[0]) + [emb[:ng % emb.shape[0]]])
    else:
        graphpath = os.path.abspath(args.prefix.rstrip('/') + '/' + args.dataset + '/data.graph')
        graph = load_grf_nx(graphpath)

    model_dir = '../ckpt/'
    
    model = build_model(args)
    model.load_state_dict(torch.load('../ckpt/%s.pth' % args.dataset))
    import time
    stime = time.time()
    if args.query_path:
        querypath = os.path.abspath(args.query_path)
    else:
        querypath = args.prefix.rstrip('/') + '/' + args.dataset + '/' + 'queries_%d/queries/query_%05d.graph' % (args.nnode, args.queryno)
        querypath = os.path.abspath(querypath)

    query = load_grf_nx(querypath)

    embs = []
    for v in tqdm(graph.nodes):
        input = samp_neighbor(graph, v)
        try:
            embs.append(model.emb_model(nagydata(torch.ones([len(input.nodes), 1]), torch.tensor(list(input.edges)).T)).detach())
        except:
            if not input.edges:
                embs.append(model.emb_model(nagydata(torch.ones([len(input.nodes), 1]), torch.zeros([2,0]).type(torch.int32))).detach())
            else: raise
        del input
    embs = torch.cat(embs).tolist()
    embs = np.array(embs, dtype = np.float32)
    np.save('../%s.npy' % args.dataset, embs)

    raise

    order, candidates = cpp_GQL(args.nnode, querypath, args.dataset)

    print(order, len(candidates))
    
    reverse_order = dict()
    for i in range(len(order)):
        reverse_order[order[i]] = i

    matches = []
    selected = set()
    result = assign(query, graph, order, matches, selected, candidates, reverse_order, len(order) // 2, 0, args.k)

    print(time.time() - stime, 's')
    print(result)
