import argparse
import os
from estimator import Estimator, OptMatchEstimator, evaluate_bunch
from utils import *

import random
import numpy as np

def estimate_action(selector: Estimator, matcher: Estimator, counter: Estimator, query_x, query_e, graph_x, 
                    matches, order, candidates, 
                    shallow_depth, deep_depth):
    # 0: extend
    # 1: escape
    # 3: release
    # 4: enumerate
    depth = matches.shape[1]
    if depth >= deep_depth: return 4, -1, -1
    # 0 
    cur_score = evaluate_bunch(counter, matcher, query_x, query_e, graph_x, matches, order, candidates) # 0
    if depth >= 1:
        rev_order_dict = dict((int(order[i]), i) for i in range(len(order)))
        matches_order = [rev_order_dict[int(i)] for i in matches[0]]
        cur_index = matches_order.index(max(matches_order))
        cur_qrynode = matches[0][cur_index]
        sliceindices = list(range(matches.shape[1]))
        del sliceindices[cur_index]
        prt_match = matches[:, sliceindices]
        # 1
        prt_score = evaluate_bunch(counter, matcher, query_x, query_e, graph_x, prt_match, order, candidates) # 1
        h = matcher.estimate_next(queryx, query_e, graph_x, prt_match)[cur_qrynode]
        h_candidates = graph_x[candidates[cur_qrynode]]
        res = selector.evaluate_next(h, h_candidates)
        act1tgt = candidates[cur_qrynode][res.argmax()]
    
        max_rls_score = 0
        max_rls_score_lyr = -1
        for i in range(shallow_depth, matches.shape[1] - 1):
            sliceindices = list(range(matches.shape[1]))
            if i not in matches_order: continue
            try:
                i_index = matches_order.index(i)
            except ValueError as e:
                continue
            del sliceindices[i_index]
            rls_match = matches[:, sliceindices]
            rls_score = evaluate_bunch(counter, matcher, query_x, query_e, graph_x, prt_match, order, candidates)
            if rls_score > max_rls_score:
                max_rls_score = rls_score
                max_rls_score_lyr = i
    else:
        prt_score = -1 
        max_rls_score = -1
    cmd = np.argmax([float(cur_score), float(prt_score), -1, float(max_rls_score)])
    if cmd == 3: 
        lyr = max_rls_score_lyr
        h_res = selector.estimate_good_match_res(query_x, query_e, graph_x, matches)[order[lyr]]
        h = query_x[order[lyr]] + h_res
        h_candidates = graph_x[candidates[order[lyr]]]
        res = selector.evaluate_next(h, h_candidates)
        return 3, lyr, candidates[order[lyr]][res.argmax()]
    if cmd == 1:
        return 1, -1, act1tgt
    return cmd, -1, -1



def start(counter: Estimator, matcher:Estimator, selector:Estimator, graph_x, query_x, query_e, candidates):
    if not os.path.exists('/dev/shm/nagy_match'): os.mkdir('/dev/shm/nagy_match')
    if os.path.exists('/dev/shm/nagy_match/matcher.ready'): 
        os.remove('/dev/shm/nagy_match/matcher.ready')
    origin_candidates = candidates[:]
    print('running')
    while True:
        if not os.path.exists('/dev/shm/nagy_match/searcher.ready'): continue
        with open('/dev/shm/nagy_match/searcher.info') as f:
            order = list(map(int, f.readline().strip().split()))
            matches = list(map(int, f.readline().strip().split()))
            candidates_cur = list(map(int, f.readline().strip().split()))
        dpth = len(matches)
        #candidates[order[dpth]] = candidates_cur
        act, lyr, tgt = estimate_action(selector, matcher, counter, query_x, query_e, graph_x, torch.tensor([order[:dpth], matches], dtype=torch.long), torch.tensor(order), candidates, 1, 3)
        #candidates[order[dpth]] = origin_candidates[order[dpth]]
        with open('/dev/shm/nagy_match/matcher.info', 'w') as f:
            f.write('%d %d %d\n' % (act, lyr, tgt))
            next_emb = matcher.estimate_next(queryx, querye, emb, torch.tensor([order[:dpth], matches]), candidates_cur)[order[dpth]]
            candidate_emb = emb[candidates_cur]
            dist = com_loss(matcher, next_emb, candidate_emb).reshape(-1)
            dist.sort().indices
            for i in dist.sort().indices: f.write('%d ' % i)
        os.remove('/dev/shm/nagy_match/searcher.ready')
        os.remove('/dev/shm/nagy_match/searcher.info')
        with open('/dev/shm/nagy_match/matcher.ready', 'w') as f: pass
'''
search.info:s

1 2 3 4 0
218 237 948
2215 1 3 4
'''
from train import com_loss
# def dist_func(x1, x2):
#     return torch.sqrt(torch.sum((x1 - x2) ** 2, dim = -1))
# def com_loss(model:Estimator, pred, target):
#     return dist_func(model.dis_est2(torch.relu(model.dis_est(pred))), model.dis_can2(torch.relu(model.dis_can(target))))

def assign(query, graph, order, matches, selected, 
           candidates, reverse_order, sgldepth, matcher, counter, selector, dpth, k, rcd_dct, last_act = -1):
    u = order[dpth]
    cur_candidates = []
    for v in candidates[u]:
        if v in selected:
            continue
        valid = True
        for nu in query.neighbors(u):
            if reverse_order[nu] >= dpth: continue
            if (v, matches[reverse_order[nu]]) not in graph.edges: 
                valid = False
                break
        if valid: cur_candidates.append(v)
    if not cur_candidates: return
    if dpth  < 0: 
        raise NotImplementedError
    else:
        if last_act == 4:
            act = 4
            next_emb = matcher.estimate_next(queryx, querye, emb, torch.tensor([order[:dpth], matches]), cur_candidates)[order[dpth]]
            candidate_emb = emb[cur_candidates]
            orig_k = k
            k = k if dpth < sgldepth else 1
            dist = com_loss(matcher, next_emb, candidate_emb).reshape(-1)
            k = min(k, int(dist.shape[0]))
            topk = dist.topk(k, largest = False)
            tobeselected = [cur_candidates[int(i)] for i in 
                                            topk.indices]
            for cur_selected in tobeselected:
                matches.append(cur_selected)
                assert len(matches) == dpth + 1
                selected.add(cur_selected)
                if len(matches) == len(order):
                    return matches
                res = assign(query, graph, order, matches, selected, 
                    candidates, reverse_order, sgldepth, matcher, counter, selector, dpth + 1, orig_k, act)
                if res: return res
                del matches[-1]
                selected.remove(cur_selected)
            return 
        if last_act == -1 or last_act == 1 or last_act == 3:
            act = 0  
        else: act, lyr, tgt = estimate_action(selector, matcher, counter, queryx, querye, emb, torch.tensor([order[:dpth], matches]), torch.tensor(order), candidates, 0, sgldepth)
        if act == 0:
            next_emb = matcher.estimate_next(queryx, querye, emb, torch.tensor([order[:dpth], matches]), cur_candidates)[order[dpth]]
            candidate_emb = emb[cur_candidates]
            orig_k = k
            k = k if dpth < sgldepth else 1
            dist = com_loss(matcher, next_emb, candidate_emb).reshape(-1)
            k = min(k, int(dist.shape[0]))
            topk = dist.topk(k, largest = False)
            tobeselected = [cur_candidates[int(i)] for i in 
                                            topk.indices]
            for cur_selected in tobeselected:
                matches.append(cur_selected)
                if tuple(matches) not in rcd_dct: rcd_dct[tuple(matches)] = 0
                rcd_dct[tuple(matches)] += 1
                assert len(matches) == dpth + 1
                selected.add(cur_selected)
                if len(matches) == len(order):
                    return matches
                res = assign(query, graph, order, matches, selected, 
                    candidates, reverse_order, sgldepth, matcher, counter, selector, dpth + 1, orig_k, rcd_dct, act)
                if res: return res
                del matches[-1]
                selected.remove(cur_selected)
            return 
        if act == 1:
            next_matches = matches[:]
            next_matches[-1] = tgt
            if rcd_dct.get(tuple(next_matches), 0) < k:
                selected.remove(matches[-1])
                matches = next_matches
                if tuple(matches) not in rcd_dct: rcd_dct[tuple(matches)] = 0
                rcd_dct[tuple(matches)] += 1
                selected.add(matches[-1])
                res = assign(query, graph, order, matches, selected, 
                    candidates, reverse_order, sgldepth, matcher, counter, selector, dpth + 1, orig_k, act)
                if res: return res
                selected.remove(matches[-1])
                del matches[-1]
            return 
        if act == 3:
            next_matches = matches[:]
            next_matches[lyr] = tgt
            if rcd_dct.get(tuple(next_matches), 0) < k:
                selected.remove(matches[lyr])
                matches = next_matches
                if tuple(matches) not in rcd_dct: rcd_dct[tuple(matches)] = 0
                rcd_dct[tuple(matches)] += 1
                selected.add(matches[lyr])
                res = assign(query, graph, order, matches, selected, 
                    candidates, reverse_order, sgldepth, matcher, counter, selector, dpth + 1, orig_k, act)
                if res: return res
                selected.remove(matches[-1])
                del matches[-1]
            return 
        

            

if __name__ == '__main__':
    # if os.path.exists('/dev/shm/nagy_match/'): 
    #     for name in os.listdir('/dev/shm/nagy_match/'):
    #         os.remove('/dev/shm/nagy_match/' + name)
    #     os.removedirs('/dev/shm/nagy_match/')
    # os.mkdir('/dev/shm/nagy_match/')
    if os.path.exists('/dev/shm/learnsc_filter.ready'):
        os.remove('/dev/shm/learnsc_filter.ready')
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-path', default='', type=str)
    parser.add_argument('--graph-path', default='', type=str)
    parser.add_argument('--prefix', default='../data', type=str)
    parser.add_argument('--dataname', default='yeast', type=str)
    parser.add_argument('--qsize', default=71, type=int)
    parser.add_argument('--gsize', default=64, type=int)
    parser.add_argument('--hsize', default=64, type=int)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--nlayer', default=1, type=int)
    parser.add_argument('--nnode', default=5, type=int)
    parser.add_argument('--queryno', default=0, type= int)
    parser.add_argument('--shallow-depth', default=2, type=int)
    parser.add_argument('--deep-depth', default=5, type=int)
    parser.add_argument('--neural', action='store_true')
    parser.add_argument('--model', default='nagy', type=str)
    parser.add_argument('--k', default=1, type=int)
    args = parser.parse_args()

    if args.graph_path:
        graph = load_grf_nx(os.path.abspath(args.graph_path))
        emb = load_emb(args.prefix.rstrip('/') + '/yeast')
        ng = graph.number_of_nodes()
        emb = torch.concatenate([emb] * (ng // emb.shape[0]) + [emb[:ng % emb.shape[0]]])
    else:
        emb = load_emb(args.prefix.rstrip('/') + '/' + args.dataname)
        graphpath = os.path.abspath(args.prefix.rstrip('/') + '/' + args.dataname + '/data.graph')
        graph = load_grf_nx(graphpath)

    model_dir = sorted([i for i in os.listdir('../saved_models') if (args.dataname + '_' + str(args.nnode) +'_') in i])[-1]
    Estmtor = OptMatchEstimator if args.model == 'nagy' else Estimator 
    matcher = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer)
    matcher.load_state_dict(torch.load('../saved_models/' + model_dir + '/matcher.pth'))
    counter = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer)
    counter.load_state_dict(torch.load('../saved_models/' + model_dir + '/counter.pth'))
    selector = Estmtor(args.qsize, args.gsize, args.hsize, args.nlayer)
    selector.load_state_dict(torch.load('../saved_models/' + model_dir + '/selector.pth'))

    import time
    if args.query_path:
        querypath = os.path.abspath(args.query_path)
    else:
        querypath = args.prefix.rstrip('/') + '/' + args.dataname + '/' + 'queries_%d/queries/query_%05d.graph' % (args.nnode, args.queryno)
        querypath = os.path.abspath(querypath)

    query = load_grf_nx(querypath)

    queryx, querye = load_grf(querypath)
    queryx = torch.eye(args.qsize)[queryx]
    querye = querye.T
    order, candidates = cpp_GQL(args.nnode, querypath, args.dataname)

    if not args.neural:
        start(counter, matcher, selector, emb, queryx, querye, candidates)

    else:
        print(order, len(candidates))
        
        reverse_order = dict()
        for i in range(len(order)):
            reverse_order[order[i]] = i
        
        matches = []
        selected = set()
        
        stime = time.time()
        result = assign(query, graph, order, matches, selected, candidates, reverse_order, 4 if args.nnode != '24' else 6, matcher, counter, selector, 0, args.k)

        print(time.time() - stime, 's')
        print(result)



# python matcher.py --neural --model nagy --prefix ../data2 --dataname wiki --nnode 8 --queryno 0 --qsize 14