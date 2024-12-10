import numpy as np 
import torch
import os
import networkx as nx
import random

def load_grf(filepath):
    with open(filepath) as f:
        terms = f.readline().split()
        nv, ne = int(terms[1]), int(terms[2])
        vs = []
        es = []
        maxlabel = 0
        for i in range(nv):
            terms = f.readline().strip().split()
            vs.append(int(terms[2]))
            if int(terms[2]) > maxlabel: maxlabel = int(terms[2])
        vs = torch.tensor(vs)
        for i in range(ne):
            terms = f.readline().strip().split()
            es.append([int(terms[1]), int(terms[2])])
        es = torch.tensor(es, dtype=torch.long)
    return (vs, es)

def load_grf_nx(filepath):
    g = nx.Graph()
    with open(filepath) as f:
        terms = f.readline().split()
        nv, ne = int(terms[1]), int(terms[2])
        for i in range(nv):
            terms = f.readline().strip().split()
            g.add_node(int(terms[1]), l=int(terms[2]))
        for i in range(ne):
            terms = f.readline().strip().split()
            g.add_edge(int(terms[1]), int(terms[2]))
    return g

def load_matches(filepath):
    with open(filepath) as f:
        order = list(map(int, f.readline().strip().split()))
        matches = []
        for line in f:
            matches.append(list(map(int, line.strip().split())))
    return matches, order

def load_counts(filepath):
    searches, tracks, counts = [list() for i in range(3)]
    with open(filepath) as f:
        for line in f:
            try:
                search, count = line.strip().split('->')
            except:
                print(line)
                raise
            searches.append(list(map(int, search.strip().split())))
            track, count = map(lambda s: 0 if int(s) < 0 else int(s), count.strip().split(', '))
            tracks.append(track)
            counts.append(count)
    return searches, tracks, counts

def load_emb(path):
    filepath = path.rstrip('/') + '/data_emb.npy'
    return torch.tensor(np.load(filepath))

def load_queries(path, dataname):
    query_names = os.listdir('%s/queries' % path.rstrip('/'))[:40]
    match_names = os.listdir('%s/matches' % path.rstrip('/'))
    count_names = os.listdir('%s/counts' % path.rstrip('/'))
    queries, orders, matches, tracks, counts, searches, candidates = [list() for i in range(7)] 
    count = 0
    for i in range(99999):
        if count >= 40: break
        if 'query_%05d.graph' % i not in query_names: continue
        if 'query_%05d.matches' % i not in match_names: continue
        if 'query_%05d.counts' % i not in count_names: continue
        print('query_%05d.graph' % i)
        queries.append(load_grf('%s/queries' % path.rstrip('/') + '/query_%05d.graph' % i))
        mtc, ord = load_matches('%s/matches' % path.rstrip('/') + '/query_%05d.matches' % i)
        matches.append(mtc)
        orders.append(ord)
        srcs, trks, cnts = load_counts('%s/counts' % path.rstrip('/') + '/query_%05d.counts' % i)
        tracks.append(trks)
        counts.append(cnts)
        searches.append(srcs)
        candidates.append(cpp_GQL(len(queries[-1][0]), '%s/queries' % path.rstrip('/') + '/query_%05d.graph' % i, dataname))
        count += 1
    return queries, matches, orders, searches, tracks, counts, candidates

def load_data(prefix, dataname, n_node):
    path = prefix.rstrip('/') + '/' + dataname
    emb = load_emb(path)
    queries, matches, orders, searches, tracks, counts, candidates = load_queries(f'{path}/queries_{n_node}', dataname)
    return emb, queries, matches, orders, searches, tracks, counts, candidates

def preprocess_queries(queries, emb_mtx):
    return [(emb_mtx[x], e.T) for x, e in queries]

def build_mis_tree(mis_match):
    posis = torch.tensor([int(x) for x, in mis_match[:, 0].nonzero()] + [mis_match.shape[0]])
    counts = posis[1:] - posis[:-1]
    if mis_match.shape[1] == 2:
        return (counts.tolist())
    res = []
    for i in range(len(counts)):
        res.append((int(counts[i]), build_mis_tree(mis_match[posis[i] : posis[i+1], 1:])))
    return res

def get_mis_tree(mtch): 
    mis_match = torch.zeros_like(mtch)
    mis_match[1:] = mtch[:-1]
    mis_match = (mtch != mis_match)
    for i in range(mis_match.shape[1]): mis_match[:, i+1:] += mis_match[:, i: i+1]
    return build_mis_tree(mis_match)

def cpp_GQL(self, query_graph_file, dataname):
    if not os.path.exists('/dev/shm/nagy_matcher'):
        os.mkdir('/dev/shm/nagy_matcher')
    num_query_vertices = self
    alphas = list('abcdefghijklmnopqrstuvwxyz')
    random.shuffle(alphas)
    alphas = ''.join(alphas)
    #print(query_graph_file, ', ', alphas)
    os.system('cp %s /dev/shm/nagy_matcher/%s_%s.graph' % (query_graph_file, dataname, alphas))
    open('/dev/shm/nagy_matcher/%s_%s.ready' % (dataname, alphas), 'w').close()
    while True:
        if os.path.exists('/dev/shm/nagy_matcher/%s_%s.fready' % (dataname, alphas)):
            break
    with open('/dev/shm/nagy_matcher/%s_%s.output' % (dataname, alphas)) as output:
        baseline_visit = output.readlines()
    os.remove('/dev/shm/nagy_matcher/%s_%s.output' % (dataname, alphas))
    os.remove('/dev/shm/nagy_matcher/%s_%s.fready' % (dataname, alphas))
    candidate_count = list()
    order = list(map(int, baseline_visit[0].strip().split()))
    for i in range(len(baseline_visit)):
        if 'Candidate set is:' in baseline_visit[i]:
            candidate_info = baseline_visit[i+1: i+2*num_query_vertices+1]
    candidates = []
    for i in range(len(candidate_info)):
        if i % 2 == 0:
            candidate_count.append(int(candidate_info[i].strip()))
        else:
            candidates.append(list(map(int, candidate_info[i].strip().split())))
    return order, candidates

def TODO():
    raise NotImplementedError