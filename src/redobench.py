import pickle, string, os, logging, argparse, sys
import datatools.lattice as lat
import core.ops as op
import numpy as np
import numpy.linalg as LA
import itertools as it
from datetime import datetime as dt
import mdtools.timescape as TS
import mdtools.deshaw as DE

import bench.db as db

ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
ascii_latin = ''.join([chr(i) for i in it.chain(range(384,460), range(550, 590))])
k_domain = label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek + ascii_latin
tok   = lambda x: ''.join(sorted([k_domain[i] for i in x]))
toidx = lambda x: [ord(i)-97 for i in x]

home = os.getenv('HOME')
TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]
Kr = [2, 52, 56, 60, 116, 258, 311, 460, 505, 507, 547, 595, 640, 642, 665, 683, 728, 767, 851, 1244, 1485, 1629, 1636]
Km = [2, 3, 4, 23, 26, 52, 53, 54, 55, 56, 60, 79, 108, 109, 110, 111, 112, 116, 151, 164, 170, 171, 204, 217, 218, 224, 240, 241, 258, 272, 291, 292, 293, 294, 310, 311, 342, 343, 344, 360, 361, 380, 391, 392, 393, 394, 408, 409, 410, 411, 412, 430, 441, 451, 453, 457, 458, 459, 460, 461, 462, 479, 480, 499, 500, 501, 504, 505, 506, 507, 508, 509, 510, 527, 528, 529, 546, 547, 548, 549, 550, 551, 552, 553, 554, 574, 593, 594, 595, 596, 597, 598, 599, 600, 620, 621, 638, 639, 640, 641, 642, 643, 665, 682, 683, 684, 685, 687, 709, 725, 726, 728, 729, 752, 767, 768, 770, 771, 811, 851, 889, 1055, 1121, 1122, 1124, 1187, 1215, 1216, 1217, 1240, 1244, 1245, 1246, 1271, 1272, 1273, 1379, 1380, 1381, 1402, 1403, 1424, 1425, 1445, 1484, 1485, 1486, 1544, 1629, 1636, 1637, 1640, 1641, 1642, 1645, 1646, 1649]
cutoff = 8

def LABEL10(L, theta=0.9):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  return 'T%d' % A if A_amt < theta or (L[0] != L[-1]) else 'W%d' % A

# ARG PARSER
# if ps1 not in sys:
parser = argparse.ArgumentParser()
parser.add_argument('support', type=int)
# parser.add_argument('clusters', type=int)
args = parser.parse_args()
support = args.support

# if ps1 not in sys:
LOGFILE = home + '/work/latt_intrinsics/bench_%d_R.log'% support
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(format='%(message)s', filename=LOGFILE, level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


logging.info('SUPPORT,%d', support)

logging.info('Loading sigma.. .')
sigma = np.load(home+'/work/results/DE_basin_sigma_Km.npy')
logging.info('Loading distance space...')
DS = 10*np.load('../data/de_ds_mu.npy')
logging.info("DS loaded")

CM = DS<cutoff
CMr, Dr = CM[:,Kr], DS[:,Kr]
CMm, Dm = CM[:,Km], DS[:,Km]

delabel = np.load(home+'/work/results/DE_label_full.npy')
DW = []
for i in range(42):
  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    DW.append((a+i*100000, b+i*100000))

dL = [delabel[a:b] for a,b in DW]
DE_LABEL = [LABEL10(i,.9) for i in dL]

SPT = [i[0] for i in db.runquery('select distinct support from latt order by support')]
NC  = [i[0] for i in db.runquery('select distinct numclu from latt order by numclu')]

mf, lf = {}, {}
dl, ik  = {}, {}
key, clu, cent, var, Gm = {}, {}, {}, {}, {}

s=support
mf[s], lf[s] = lat.maxminer(CMr, s)
dl[s], ik[s] = lat.derived_lattice(mf[s], Dr, CMr)
pickle.dump(dl[s], open(home + '/work/latt_intrinsics/dlat2_%d.p' % support, 'wb'))
for num_clu in NC:
  key[s], clu[s], cent[s], var[s], Gm[s] = lat.cluster_harch(dl[s], CMr, Dr, theta=.5, num_k=num_clu, dL=None, verbose=False)  
  w, t = lat.score_clusters(clu[s], Dr, cent[s], var[s], Gm[s], sigma, DE_LABEL)
  for k in TBIN10:
    logging.info('SCORE,W,%d,%d,%s,%.5f', support, num_clu, k, w[k])
  for k in TBIN10:
    logging.info('SCORE,T,%d,%d,%s,%.5f', support, num_clu, k, t[k])