import sys
import os

dataname = sys.argv[1]
os.system('cp /home/nagy/TrackCountPredict/data/%s/data.graph ./data/%s.graph' % (dataname, dataname))
os.system('./run.sh %s' % dataname)



