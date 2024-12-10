import sys
import os
import networkx as nx
import torch

dataname = sys.argv[1] #'wordnet'

nagypath = '/home/nagy/TrackCountPredict/data/%s/' % dataname
outpath = '/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s' % dataname


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
            try:
                matches.append(list(map(int, line.strip().split())))
            except:
                print(line)
    return matches, order

def load_counts(filepath):
    searches, tracks, counts = [list() for i in range(3)]
    with open(filepath) as f:
        for line in f:
            search, count = line.strip().split('->')
            searches.append(list(map(int, search.strip().split())))
            track, count = map(lambda s: 0 if int(s) < 0 else int(s), count.strip().split(', '))
            tracks.append(track)
            counts.append(count)
    return searches, tracks, counts

querysetnames = os.listdir(nagypath)
nnodes = [i for i in range(30) if 'queries_%d' % i in querysetnames]
queries = []
matches = []
for nnode in nnodes:
    matchpath = '/home/nagy/TrackCountPredict/data/%s/queries_%d/matches' % (dataname, nnode)
    querypath = '/home/nagy/TrackCountPredict/data/%s/queries_%d/queries' % (dataname, nnode)
    cands = []
    for i in range(1000):
        if ('query_%05d.matches' % i in os.listdir(matchpath) and 
            'query_%05d.graph' % i in os.listdir(querypath)):
            cands.append((nnode, i))
            print(matchpath + '/' + 'query_%05d.matches' % i)
            queries.append(load_grf_nx(querypath + '/' + 'query_%05d.graph' % i))
            matches.append(load_matches(matchpath + '/' + 'query_%05d.matches' % i))
datagraph = load_grf_nx(nagypath + 'data.graph')

totalmatches = sum([len(mtch[0]) * len(mtch[1]) for mtch in matches])
sampling_rate = len(datagraph.edges) / 4 / totalmatches
import random
newnodedict = dict()
for query, (match, order) in zip(queries, matches):
    nodemap = dict()
    for i in range(len(query.nodes)):
        nodemap[i] = len(datagraph.nodes)
        datagraph.add_node(len(datagraph.nodes))
    for u,v in query.edges:
        datagraph.add_edge(nodemap[u], nodemap[v])
    for mch in match:
        if random.random() < sampling_rate:
            for i in range(len(order)):
                datagraph.add_edge(nodemap[order[i]], mch[i])
import numpy as np
def save():
    splittargetpath = '/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/split/target' % dataname
    if not os.path.exists('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s' % dataname):
        os.mkdir('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s' % dataname)
        os.mkdir('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/mapping' % dataname)
        os.mkdir('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/processed' % dataname)
        os.mkdir('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw' % dataname)
        os.mkdir('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/split' % dataname)
        os.mkdir(splittargetpath)
        with open('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/RELEASE_v1.txt' % dataname, 'w') as f: pass
    with open('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/edge.csv' % dataname, 'w') as f:
        for u,v in datagraph.edges:
            u, v = (min(u, v), max(u, v))
            f.write('%d,%d\n'%(u,v))
    with open('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/num-edge-list.csv' % dataname, 'w') as f:
        f.write('%d\n' % len(datagraph.edges))
    with open('/home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/num-node-list.csv' % dataname, 'w') as f:
        f.write('%d\n' % len(datagraph.nodes))
    os.system('gzip /home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/num-node-list.csv' % dataname)
    os.system('gzip /home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/num-edge-list.csv' % dataname)
    os.system('gzip /home/nagy/TrackCountPredict/baselines/Neo-GNNs/dataset/%s/raw/edge.csv' % dataname)
    nodes = list(range(len(datagraph.nodes)))
    torch.save({'edge':np.array([(min(u, v), max(u, v)) for u, v in datagraph.edges])},
               splittargetpath + '/train.pt')
    torch.save({'edge':np.array([(min(u, v), max(u, v)) for u, v in datagraph.edges if random.random() < .1]),
                'edge_neg':np.array([random.sample(nodes, 2)for i in range(int(len(datagraph.edges)*.1))])},
               splittargetpath + '/test.pt')
    torch.save({'edge':np.array([(min(u, v), max(u, v)) for u, v in datagraph.edges if random.random() < .1]),
                'edge_neg':np.array([random.sample(nodes, 2)for i in range(int(len(datagraph.edges)*.1))])},
               splittargetpath + '/valid.pt')
    
save()
        
for mtch, order in matches:
    pass

meta_dict = {'dir_path': 'dataset/%s'%dataname,
             'eval metric': 'hits@20', 
             'download_name': dataname,
             'task type': 'link prediction',
             'version' : 1,
             'url': 'None',
             'add_inverse_edge': True,
             'has_node_attr': False,
             'has_edge_attr': False,
             'split': 'target',
             'additional node files': 'None',
             'additional edge files': 'None',
             'is hetero': False,
             'binary': False}