import subprocess
import os
import time

name = 'ciwordnet'
nq = 20

st = 0
count = 0
scnt = 0
for i in range(1000):
    if os.path.exists('/home/nagy/data/LearnSC/%s/queries_%d/query_%05d.graph' % (name, nq, i)):
        count += 1
        cmd = subprocess.run(['python', 'matcher.py', '--graph-path', '/home/nagy/data/LearnSC/%s/data.graph' % name, '--query-path',
                        '/home/nagy/data/LearnSC/%s/queries_%d/query_%05d.graph' % (name, nq, i)], capture_output=True)
        try: 
            t = float(cmd.stdout.decode('utf8').strip().split()[0])
            st += t
            scnt += 1
            print(t)
        except:
            print('-----')
    if count >= 50: break
print('avgt = %8.6f' % (st / scnt) )
print('rate = %.2f' % (scnt / count))