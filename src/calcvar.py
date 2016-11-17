import pickle, string, os, logging, argparse, sys
import datatools.lattice as lat
import core.ops as op
import numpy as np
import numpy.linalg as LA
import itertools as it
from datetime import datetime as dt
import mdtools.timescape as TS

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

# ARG PARSER
parser = argparse.ArgumentParser()
parser.add_argument('support', type=int)
parser.add_argument('clusters', type=int)
args = parser.parse_args()
support = args.support
num_clu = args.clusters

def LABEL10(L, theta=0.75):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  # if A_amt < theta or (L[0] != L[-1] and A_amt < (.5 + .5 * theta)):
  if A_amt < theta or (L[0] != L[-1]):
    return 'T%d' % A
  else:
    return 'W%d' % A


# LOGGING
LOGFILE = home + '/work/latt_intrinsics/cluster_%d_%d.log'% (support, num_clu)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format='%(message)s', filename=LOGFILE, level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)



logging.info('SUPPORT,%d', support)
logging.info('NUM_CLUSTER,%d', num_clu)

sigma = np.load(home+'/work/results/DE_basin_sigma_Km.npy')
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
slab = lambda x: ''.join([str(i) for i in dL[x]])
iswell = lambda x: (dL[x] == dL[x][0]).all()
getstate  = lambda x: int(np.median(dL[x])) if iswell(x) else 5+np.argmax(np.bincount(dL[x]))


iset    = pickle.load(open(home + '/work/latt_intrinsics/iset_%d.p' % support, 'rb')); len(iset)
dlat    = pickle.load(open(home + '/work/latt_intrinsics/dlat_%d.p' % support, 'rb')); len(dlat)
logging.info('\n\n----------------------')
logging.info('SIZE,iset,%d', len(iset))
logging.info('SIZE,dlat,%d', len(dlat))

logging.info('Clustering Lattice:')

SKIPCLU = False
if SKIPCLU:
  clu = pickle.load(open('clu.p', 'rb'))
  score = pickle.load(open('score.p', 'rb'))
  elmlist = pickle.load(open('elmlist.p', 'rb'))
else:
  clu, score, elmlist = lat.clusterlattice(dlat, CMr, Dm, iset, num_k=num_clu)

cluk = list(clu.keys())
clulist = [clu[k] for k in cluk]

variance = np.zeros(len(cluk))
for n, k in enumerate(cluk):
  if len(clulist[n]) == 1:
    continue
  cov = np.cov(DS[clulist[n]][:,Km].T)
  ew,_ = LA.eigh(cov)
  variance[n] = np.sum(ew)

variance2 = np.zeros(len(cluk))
for n, k in enumerate(cluk):
  if len(clulist[n]) == 1:
    continue
  cov = np.cov(DS[clulist[n]][:,toidx(k)].T)
  ew,_ = LA.eigh(cov)
  variance2[n] = np.sum(ew)



basin_score_well = np.zeros(91116)
basin_score_tran = np.zeros(91116)
multi_basins = []
max_sigma = np.max(sigma)
cluster_score_well, cluster_score_tran = np.zeros(len(cluk)), np.zeros(len(cluk))

C_ws, C_wv = .4, .6
C_ts, C_tv = .1, .90

B_wd, B_wv = .9, .1
B_td, B_tv = .25, .75

max_var1 = np.max(variance[np.nonzero(variance)])
max_var2 = np.max(variance2[np.nonzero(variance2)])
sizeFunc = op.makeLogisticFunc(2, 0.001, 91116/len(clulist))
logging.info("CLUSTER_SUMMARY   %d   %d", support, num_clu)
for n, k in enumerate(cluk):
  cSize = len(clulist[n])
  cVar  = variance2[n]
  sc_var  = 0 if cVar == 0 else cVar / max_var2
  sc_size = sizeFunc(cSize) -1 #      max(-2, 1 - (len(cluk)*cSize / (91116)))
  cluster_score_well[n] = max(0, C_wv *(1-sc_var)    + C_ws * sc_size)  
  cluster_score_tran[n] = max(0, C_tv * sc_var       + C_ts * sc_size)  
  # cnt = np.bincount(list(it.chain(*[[c for c in dL[i]] for i in clulist[n]])))
  cnt = np.bincount([getstate(i) for i in clulist[n]], minlength=6)
  logging.info('%2d %15s |  sz=%6d  |  var=%7.2f | scW=%5.2f  scT%5.2f)  %s' % 
    (n, k, cSize, cVar, cluster_score_well[n], cluster_score_tran[n], cnt/sum(cnt)))
  max_dist  = np.max([s for _,s in elmlist[n]])
  max_sigma = np.max([sigma[i] for i,_ in elmlist[n]])
  for i, s in elmlist[n]:
    sc_dist = s / max_dist
    sc_var  = sigma[i] / max_sigma
    basin_score_well[i] = max(0.0001, basin_score_well[i], B_wd * sc_dist + B_wv * sc_var)
    basin_score_tran[i] = max(0.0001, basin_score_tran[i], B_td * sc_dist + B_tv * sc_var)

cluster_basin_score_well, cluster_basin_score_tran = [],[]
for n in range(len(cluk)):
  blist_well = [(i, basin_score_well[i]) for i,s in elmlist[n]]
  blist_tran = [(i, basin_score_tran[i]) for i,s in elmlist[n]]
  cluster_basin_score_well.append(sorted(blist_well, key=lambda x: x[1], reverse=True))
  cluster_basin_score_tran.append(sorted(blist_tran, key=lambda x: x[1], reverse=True))
  # logging.info('WELL:')
  # for i, sc in cluster_basin_score_well[-1][:3]:
  #   logging.info('   %5.2f  %s', sc, slab(i))
  # logging.info('TRAN:')
  # for i, sc in cluster_basin_score_tran[-1][:3]:
  #   logging.info('   %5.2f  %s', sc, slab(i))
  # logging.info('')

pdf_cluster_well = cluster_score_well / np.sum(cluster_score_well)
pdf_cluster_tran = cluster_score_tran / np.sum(cluster_score_tran)
pdf_basin_well, pdf_basin_tran   = np.zeros(91116), np.zeros(91116)
for n in range(len(cluk)):
  binscore_well = {k: 0 for k in TBIN10}
  binscore_tran = {k: 0 for k in TBIN10}
  totW = np.sum([x for _,x in cluster_basin_score_well[n]])
  totT = np.sum([x for _,x in cluster_basin_score_tran[n]])
  for i, sc in cluster_basin_score_well[n]:
    pdf_basin_well[i] = sc/totW * pdf_cluster_well[n]
    binscore_well[LABEL10(dL[i], .8)] += sc/totW
  for i, sc in cluster_basin_score_tran[n]:
    pdf_basin_tran[i] = sc/totT * pdf_cluster_tran[n]
    binscore_tran[LABEL10(dL[i], .8)] += sc/totT
  # print('CLUSTER: %15s  pW=%4.2f   pT=%4.2f' % (cluk[n], pdf_cluster_well[n], pdf_cluster_tran[n]))
  # pprint(binscore_well)
  # pprint(binscore_tran)
  # print(' ')

binscore_well = {k: 0 for k in TBIN10}
binscore_tran = {k: 0 for k in TBIN10}
for i in range(91116):
  binscore_well[LABEL10(dL[i],.8)] += pdf_basin_well[i]
  binscore_tran[LABEL10(dL[i],.8)] += pdf_basin_tran[i]

totW = np.sum(list(binscore_well.values()))
totT = np.sum(list(binscore_tran.values()))
for k in TBIN10:
  binscore_well[k] /= totW
  binscore_tran[k] /= totT
# pprint(binscore_well)
# pprint(binscore_tran)


for k in TBIN10:
  logging.info('SCORE,W,%d,%d,%s,%.5f', support, num_clu, k, binscore_well[k])

for k in TBIN10:
  logging.info('SCORE,T,%d,%d,%s,%.5f', support, num_clu, k, binscore_tran[k])



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