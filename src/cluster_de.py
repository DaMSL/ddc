import pickle, string, os, logging, argparse, sys
import datatools.lattice as lat
import core.ops as op
import numpy as np
import numpy.linalg as LA
import itertools as it
from datetime import datetime as dt
import mdtools.timescape as TS
import mdtools.deshaw as DE

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
if not sys.ps1:
  parser = argparse.ArgumentParser()
  parser.add_argument('support', type=int)
  # parser.add_argument('clusters', type=int)
  parser.add_argument('clu', type=int, default=12)
  parser.add_argument('--log', default=None)
  args = parser.parse_args()
  support = args.support
  num_clu = 12
  LOGFILE = args.log
else:
  support, num_clu = 910, 12


if not sys.ps1 and LOGFILE:  
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

label = DE.loadlabels_aslist()
bL = [label[int(i/22.09)] for i in range(91116)]

delabel = np.load(home+'/work/results/DE_label_full.npy')
DW = []
for i in range(42):
  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    DW.append((a+i*100000, b+i*100000))

dL = [delabel[a:b] for a,b in DW]
DE_LABEL = [LABEL10(i,.9) for i in dL]

logging.info('Loading Lattice')

support = 4550
iset    = pickle.load(open(home + '/work/latt_intrinsics/iset_%d.p' % support, 'rb')); len(iset)
dlat    = pickle.load(open(home + '/work/latt_intrinsics/dlat_%d.p' % support, 'rb')); len(dlat)
keylist, clulist, centroid, variance, G = lat.cluster_harch(dlat, CMr, Dm, theta=.5, num_k=num_clu, dL=None, verbose=False)
clu = {k:c for k,c in zip(keylist, clulist)}; lat.printclu(clu, bL)

clu_by_state = [[] for i in range(5)]
for idx, (k,c) in enumerate(zip(keylist, clulist)):
  size = len(c)
  if size < 100:
    continue
  bc  = np.bincount([bL[i] for i in c], minlength=5)
  state = np.argmax(bc)
  stperc = 100*bc[state] / size
  clu_by_state[state].append((idx, stperc, sum(bc)))

# ID best clusters
hist = lambda x: np.histogram(x, bins=48, range=(4,12))[0]
C = [max(cl, key=lambda x: x[1])[0] for cl in clu_by_state]
KrD = k_domain[:len(Kr)]
cluD = [DS[clulist[c]] for c in C]
kdistr = {k: {} for k in KrD}
for m, k in enumerate(KrD):
  idx = Kr[m]
  alld = hist(DS[:,idx])
  kdistr[k]['All'] = alld/np.sum(alld)
  for st, c in enumerate(C):
    d = hist(cluD[st][:,idx])
    kdistr[k]['%d'%st] = d/np.sum(d)

pickle.dump(kdistr, open('kdistr', 'wb'))

for k in KrD:
  P.show_distr(kdistr[k], xscale=(4,10), showlegend=True, states={str(i):i for i in range(5)},\
    xlabel='Distance (in Angstroms)', ylabel='Frequency', fname='distr_'+k, latex=True)

# TO Score Clusters based on total PDF:
well, tran = lat.score_clusters(clulist, Dr, centroid, variance, G, sigma, DE_LABEL)

# TBIN10 = sorted(set(DE_LABEL))
for k in TBIN10:  logging.info('SCORE,%d,W,%d,%d,%s,%.5f', seqnum, support, num_clu, k, well[k])

for k in TBIN10:  logging.info('SCORE,%d,T,%d,%d,%s,%.5f', seqnum, support, num_clu, k, tran[k])



for k,v in kdistr['q']: print(k, v)

  elms = (n, k, len(v), state, stperc, bc) if incldist else (n, k, len(v), state, stperc)
  clusterlist.append(elms)
  # print('%2d.'%n, '%-15s'%k, '%4d '%len(v), 'State: %d  (%4.1f%%)' % (state, stperc))
  n += 1
for i in sorted(clusterlist, key =lambda x : x[2], reverse=True):
  if incldist:
    print('%3d.  %-17s%5d  /  State: %d  (%5.1f%%)   %s' % i)
  else:
    print('%3d.  %-17s%5d  /  State: %d  (%5.1f%%)' % i)


logging.info('\n\n----------------------')
logging.info('SIZE,iset,%d', len(iset))
logging.info('SIZE,dlat,%d', len(dlat))

clunumlist = [1000, 500, 250, 100, 50, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6]
for seqnum in range(1,3):
  for num_clu in clunumlist:
    logging.info('\n\n----------------------')
    logging.info('NUM_CLUSTER,%d', num_clu)
    logging.info('----------------------')
    logging.info('Clustering Lattice:')

    logging.info('Scoring Lattice:')

# keylist, clulist, centroid, variance, G = lat.cluster_harch(dlat, CMr, Dr, theta=.5, num_k=num_clu, dL=dL, verbose=True)
# well, tran = lat.score_clusters(clulist, Dr, centroid, variance, G, sigma, DE_LABEL)
# FOR BASE VALUES
base = defaultdict(int)
for i in L: base[i] += 1



for k,v in sorted(cnt.items()): print(k, v/91116)






  # logging.info('%2d %12s /  size=%3d   var=%5.2f' % (n, k, cSize, np.sum(ew)))
    # if len(iset) < MIN_EV:
    #   sc_size = -2 * (MIN_EV - len(iset)) / MIN_EV
    # clscore = sc_var + sc_size
    # logging.info('   %5d   d=%5.3f  v=%5.3f    %s'%      (i, s, sigma[i], '  ', slab(i)))

  # idx = toidx(k)
  # logging.info('%2d %12s /  size=%3d   var=%5.2f' % (n, k, cSize, np.sum(ew)))
    # if len(iset) < MIN_EV:
    #   sc_size = -2 * (MIN_EV - len(iset)) / MIN_EV
    # clscore = sc_var + sc_size
    # logging.info('   %5d   d=%5.3f  v=%5.3f    %s'%      (i, s, sigma[i], '  ', slab(i)))





sys.exit(0)

### CALCULATE SIGMA
  # logging.info('%s', Km2)
  # logging.info('\nProcessing:')
  # sigma_km2 = []
  # for i in range(42):
  #   ds = 10*np.load(home+'/work/results/DE_ds/DE_ds_full_%02d.npy' % i)
  #   ds_km = []
  #   for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
  #     cov = np.cov(ds[a:b][:,Km2].T)
  #     ew,_= LA.eigh(cov)
  #     sigma_km2.append(np.sum(ew))
  #     ds_km.append(ds[a:b][:,Km2])
  #   np.save(home+'/work/results/DE_km/DE_ds_km_%02d' % i, np.array(ds_km))
  #   logging.info('  Processed', i)

  # sigma = np.array(sigma_km2)
  # np.save(home+'/work/results/DE_basin_sigma_Km', sigma)
  # logging.info('ALL DONE!')
