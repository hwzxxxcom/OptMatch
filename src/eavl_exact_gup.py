import sys
import subprocess
import time
dataname = sys.argv[1]
nnode = sys.argv[2]



def eval_gql(data_path, query_path):
    print(' '.join(['../gup/target/release/gup', '--probe', '-M', '1',
                                    '--graph', data_path, query_path]))
    stdout = subprocess.run(['../gup/target/release/gup', '--probe', '-M', '1',
                                    '--graph', data_path, query_path],
                                   capture_output = True).stdout.decode('utf8')
    lines = stdout.strip().split('\n')
    for line in lines:
        if 'recursion_count:' in line and 'futile' not in line:
            otrack = int(line.strip().split(':')[-1])
        if 'search_sec:' in line:
            otime = float(line.strip().split(':')[-1])
            
    try:
        print('o', otime, otrack)
    except:
        print('???')
        return -1,-1,-1,-1
    # print(' '.join(['python', 'matcher.py', '--model', 'nagy', '--prefix', '../data', '--graph-path', data_path, 
    #                                 '--dataname', dataname, '--nnode', '8', '--qsize', '14',
    #                                 '--query-path', query_path]))
    # nagymatcher = subprocess.Popen(['python', 'matcher.py', '--model', 'nagy', '--prefix', '../data', '--graph-path', data_path, 
    #                                 '--dataname', dataname, '--nnode', '8', '--qsize', '14',
    #                                 '--query-path', query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # # time.sleep(2)
    # print(' '.join(['../searcher/build/matching/SubgraphSearching.out',
    #                                 '-d', data_path,
    #                                 '-q', query_path]))
    # searcher = subprocess.Popen(['../searcher/build/matching/SubgraphSearching.out',
    #                                 '-d', data_path,
    #                                 '-q', query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stime = time.time()
    # while True:
    #     if searcher.poll() is not None:
    #         lines = searcher.stdout.readlines()
    #         lines = [line.decode('utf8') for line in lines]
    #         break
    #     else:
    #         time.sleep(0.0001)
    #         if time.time() - stime > 60:
    #             print('bad')
    #             searcher.terminate()
    #             return -1,-1,-1,-1
    
    stdout = subprocess.run(['/home/nagy/gup/target/release/gup', '--probe', '-M', '1',
                                    '--graph', data_path, query_path],
                                   capture_output = True).stdout.decode('utf8')
    # try:
    #     stdout = subprocess.run(['../searcher/build/matching/SubgraphSearching.out',
    #                                 '-d', data_path,
    #                                 '-q', query_path],
    #                                capture_output = True, timeout=60).stdout.decode('utf8')
    # except:
    #     print('timeout')
    #     nagymatcher.terminate()
    #     return -1,-1,-1,-1
    lines = stdout.strip().split('\n')
    for line in lines:
        if 'recursion_count:' in line and 'futile' not in line:
            ntrack = int(line.strip().split(':')[-1])
        if 'search_sec:' in line:
            ntime = float(line.strip().split(':')[-1])
    try:
        print('n', ntime, ntrack)
    except:
        print(' '.join(['/home/nagy/gup/target/release/gup', '--probe', '-M', '1',
                                    '--graph', data_path, query_path]))
        return -1, -1, -1, -1
    # nagymatcher.terminate()
    return otime, otrack, ntime, ntrack

import os
names = os.listdir('../data/%s/queries_%d/gup_queries'%(dataname,int(nnode)))
with open('dataname_nnode.txt', 'w') as f:
    for name in names:
        otime, otrack, ntime, ntrack = eval_gql('../data/%s/data_graph' % dataname, '../data/%s/queries_%d/gup_queries/'%(dataname,int(nnode)) + name)
        if otime < 0: continue
        f.write('%f,%f,%d,%d,%s\n' % (otime, ntime, otrack, ntrack, name))