import subprocess
import os
import sys
from tqdm import tqdm
dataname = sys.argv[1]
nnode = sys.argv[2]
qsize = sys.argv[3]
k = sys.argv[4]

count = 0
succ = 0
times = []

files = os.listdir('../data/%s/queries_%d/queries/' % (dataname, int(nnode)))
lst = [i for i in range(1000) if 'query_%05d.graph' % i in files]
for i in tqdm(lst[:40]):
    out = subprocess.run(['python', 'matcher.py', '--neural', '--model', 'nagy', '--prefix', '../data', 
                          '--dataname', dataname, '--nnode', nnode, '--qsize', qsize, '--queryno', str(i), '--k', k], capture_output= True)
    output = out.stdout.decode('utf8').strip()
    #print(output)
    #print(output.split('\n'))
    try:
        _, time, output = output.split('\n')
    except:
        print(i)
        print(' '.join(['python', 'matcher.py', '--neural', '--model', 'nagy', '--prefix', '../data', 
                          '--dataname', dataname, '--nnode', nnode, '--qsize', qsize, '--queryno', str(i), '--k', k]))
        continue

    if output.lower() != 'none':
        succ += 1
    count += 1
    times.append(float(time.strip().split()[0]))

print(succ, count, times)

files = os.listdir('../data/%s/queries_%d/easy_queries/' % (dataname, int(nnode)))
lst = [i for i in range(1000) if 'query_%05d.graph' % i in files]
succ2, count2 = 0, 0
times2 = []
for i in tqdm(lst[:80 - count]):
    out = subprocess.run(['python', 'matcher.py', '--neural', '--model', 'nagy', '--prefix', '../data', 
                          '--dataname', dataname, '--nnode', nnode, '--qsize', qsize, '--query-path', '../data/%s/queries_%s/easy_queries/query_%05d.graph' % (dataname, nnode, i), '--k', k], capture_output= True)
    output = out.stdout.decode('utf8').strip()
    #print(output)
    #print(output.split('\n'))
    try:
        _, time, output = output.split('\n')
    except:
        print(i)
        print(' '.join(['python', 'matcher.py', '--neural', '--model', 'nagy', '--prefix', '../data', 
                          '--dataname', dataname, '--nnode', nnode, '--qsize', qsize, '--query-path', '../data/%s/queries_%s/easy_queries/query_%05d.graph' % (dataname, nnode, i), '--k', k]))
        continue

    if output.lower() != 'none':
        succ2 += 1
    count2 += 1
    times2.append(float(time.strip().split()[0]))
print(succ2, count2, times2)

import numpy as np
print(succ + succ2, count + count2, np.median(times + times2), np.mean(times + times2))