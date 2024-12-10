class Graph:
    def __init__(self,):
        self.vs = []
        self.es = []
    def save_nsic(self, filepath):
        with open(filepath, 'w') as f:
            f.write('Creator "nagy"\nVersion 1\n')
            f.write('graph\n[\n')
            f.write('  directed 0\n')
            for v, l in enumerate(self.vs):
                f.write('  node\n  [\n    id %d\n    label %d\n  ]\n' % (v, l))
            for u, ues in enumerate(self.es):
                for v, l in ues:
                    f.write('  edge\n  [\n    source %d\n    target %d\n    label %d\n    key 0\n  ]\n' % (u, v, l))
            f.write(']')

def load_graph(path):
    graph = Graph()
    f = open(path)
    nnode, nedge = map(int, f.readline().strip().split()[1:])
    for i in range(nnode):
        line = f.readline()
        if not line.strip(): break
        terms = line.strip().split()
        assert terms[0] == 'v'
        v, l = map(int, terms[1:3])
        assert v == len(graph.vs)
        graph.vs.append(l)
        graph.es.append([])
    for i in range(nedge):
        line = f.readline()
        terms = line.strip().split()
        assert terms[0] == 'e'
        u, v = map(int, terms[1:3])
        assert u < v
        graph.es[u].append((v,l))
    return graph

def save_card(matches, file):
    with open(file, 'w') as f:
        f.write('g_id,counts,subisomorphisms\n')
        f.write('G_N64_E64_NL4_EL4_0,%d,"%s"' % (len(matches), str(matches).replace(' ', '')))
    
import sys
import os
if __name__ == '__main__':
    name = sys.argv[1]
    nnode = int(sys.argv[2])
    nsc = '/home/nagy/TrackCountPredict/data/%s/' % name
    nsic = '/home/nagy/data/SubgraphCounting2/'
    if name + str(nnode) not in os.listdir(nsic): 
        os.system('mkdir %s%s' % (nsic, name + str(nnode)))
    nsic = nsic + name + str(nnode) + '/'
    os.system('mkdir %sdata' % (nsic))
    os.system('mkdir %sgraphs' % (nsic))
    os.system('mkdir %smetadata' % (nsic))
    os.system('mkdir %smodel' % (nsic))
    os.system('mkdir %spatterns' % (nsic))

    g = load_graph(nsc + 'data.graph')

    qrynames = [i for i in os.listdir(nsc + 'queries_%d/queries' % nnode) if '.graph' in i]

    matches = []
    g.save_nsic(nsic+'graphs/' + 'G_N64_E64_NL4_EL4_0.gml')
    for ind, qname in enumerate(qrynames):
        q = load_graph(nsc + 'queries_%d/queries/' % nnode + qname)
        f = open(nsc + 'queries_%d/matches/%s.matches' % (nnode, qname[:11]))
        match = [list(map(int, line.strip().split())) for line in f.read().strip().split('\n')]
        q.save_nsic(nsic+'patterns/' + 'P_N%d_E%d_NL2_%d.gml' % (nnode, nnode, ind))
        save_card(match, nsic+'metadata/' + 'P_N%d_E%d_NL2_%d.csv' % (nnode, nnode, ind))
    
    