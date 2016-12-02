import pickle, string, os, logging
import mdtools.deshaw as DE
import itertools as it
from collections import defaultdict, OrderedDict
import datatools.lattice as lat
import datatools.datareduce as DR
import mdtools.timescape as TS


# TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]
TBIN10 = ['W0', 'T0', 'W1', 'T1', 'W2', 'T2', 'W3', 'T3', 'W4', 'T4']
BIN5 = ['State-%d'%i for i in range(5)]
BIN5 = ['%d'%i for i in range(5)]
tidx = {k: i for i, k in enumerate(TBIN10)}

def LABEL10(L, theta=0.9):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  return 'T%d' % A if A_amt < theta or (L[0] != L[-1]) else 'W%d' % A




#  SCRAPE LABEL LOG FILE
for expname in ['biased6', 'uniform6']:
  data = []
  with open('../results/{0}_prov.log'.format(expname)) as provfile:
    for line in provfile.read().strip().split('\n'):
      if line.startswith('BASIN'):
        _,bid,targ,actual,labelseq = line.split(',')
        data.append((bid,targ,actual,labelseq))
  h, w, t = np.identity(10), np.identity(5), np.identity(5)
  for _,a,b,_ in data:
    i, j = tidx[a], tidx[b]
    h[i][j] += 1
    w[int(a[1])][int(b[1])] += 1
    if a[0] == 'T':
      t[int(a[1])][int(b[1])] += 1
  h_norm = (h.T/h.sum()).T
  w_norm = (w.T/w.sum()).T
  t_norm = (t.T/t.sum()).T
  # P.heatmap(np.rot90(h_norm,3).T, TBIN10[::-1], TBIN10, expname+'_accuracy_10bin', ylabel='Start State', xlabel='Output Distribution')
  P.heatmap(np.rot90(w_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_states', ylabel='Start State', xlabel='Output Distribution', latex=True)
  # P.heatmap(np.rot90(t_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_trans', ylabel='Start State', xlabel='Output Distribution')


outbin = {k:defaultdict(list) for k in ['all', 'W', 'T']}
exp = 'lattice2'
h, w, t = np.identity(10), np.identity(5), np.identity(5)
for exp in ['lattice2', 'lattrans']:
  lab_all = np.load(home+'/work/results/label_{0}.npy'.format(exp)).astype(int)
  with open(home+'/work/results/{0}/jclist'.format(exp)) as inf: 
    idlist = inf.read().strip().split('\n')
  for i, tid in enumerate(idlist):
    for a,b in TS.TimeScape.windows(home+'/work/jc/{0}/{1}/{1}_transitions.log'.format(exp,tid)):
      outbin['all'][tid].append(LABEL10(lab_all[i][a:b]))
      if exp=='lattice2': 
        outbin['W'][tid].append(LABEL10(lab_all[i][a:b]))
      else:
        outbin['T'][tid].append(LABEL10(lab_all[i][a:b]))

for k,v in outbin['all'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    h[i][j] += 1

for k,v in outbin['W'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    w[int(a[1])][int(b[1])] += 1

for k,v in outbin['T'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    t[int(a[1])][int(b[1])] += 1

expname='lattice'
h_norm = (h.T/h.sum()).T
w_norm = (w.T/w.sum()).T
t_norm = (t.T/t.sum()).T
# P.heatmap(np.rot90(h_norm,3).T, TBIN10[::-1], TBIN10, expname+'_accuracy_10bin', ylabel='Start State', xlabel='Output Distribution')
P.heatmap(np.rot90(w_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_states', ylabel='Start State', xlabel='Output Distribution', latex=True)
P.heatmap(np.rot90(t_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_accuracy_trans', ylabel='Start State', xlabel='Output Distribution')





def LABEL10(L, theta=0.75):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  # if A_amt < theta or (L[0] != L[-1] and A_amt < (.5 + .5 * theta)):
  if A_amt < theta and (L[0] != L[-1]):
    return 'T%d' % A
  else:
    return 'W%d' % A

# BASIN WINDOWS -- DESHAW
DW = []
for i in range(42):
  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    DW.append((a+i*100000, b+i*100000))

DMin = []
for i in range(42):
  for m in TS.TimeScape.read_log(home+'/work/timescape/desh_%02d_minima.log'%i):
    DMin.append(m + i*100000)

res = {}
delab = np.load(home+'/work/results/DE_label_full.npy').astype(int)
with open(home+'/work/results/trajout_deshaw', 'w') as out:
  for i, (m, (a,b)) in enumerate(zip(DMin, DW)):
    key = m
    cnt = np.bincount(delab[a:b], minlength=5)
    res[key] = cnt
    _=out.write('%d,%07d,%s\n' % (i, key, ','.join([str(i) for i in cnt])))

# BASIN WINDOWS -- ADAPTIVE
expmap = {'Uniform':'uniform5', 'Biased':'biased5', 'Lattice':'lattice2', 'LatTrans':'lattrans'}
idlist, traj = defaultdict(list), defaultdict(list)
for k,exp in expmap.items():
  with open(home+'/work/results/{0}/jclist'.format(exp)) as inf: 
    idlist[k] = inf.read().strip().split('\n')
  # for i in range(len(idlist[k])):
  #   traj[k].append(md.load(home+'/work/results/{0}/{0}_{1:03d}.dcd'.format(exp, i), top=topop))

cent = np.load('../data/bpti-alpha-dist-centroid.npy')
outbin, lab_all = defaultdict(list), {}
for k,exp in expmap.items():
  lab_all[k] = np.load(home+'/work/results/label_{0}.npy'.format(exp)).astype(int)
  for i, tid in enumerate(idlist[k]):
    for a,b in TS.TimeScape.windows(home+'/work/jc/{0}/{1}/{1}_transitions.log'.format(exp,tid)):
      outbin[tid].append(LABEL10(lab_all[k][i][a:b]))

#DEShaw labels
lab_all['DEShaw'] = np.load(home+'/work/results/DE_label_full.npy').astype(int)
for m, (a,b) in zip(DMin, DW):
  outbin[m] = [LABEL10(lab_all['DEShaw'][a:b])]


pmap  = {k: {} for k in expmap.keys()}
stbin = {k: {} for k in expmap.keys()}
for k,v in expmap.items():
  with open(home+'/work/results/trajout_{0}'.format(v)) as infile:
    src = infile.read().strip().split('\n')
    for line in src:
      e = line.split(',')
      if e[2].isdigit():
        key, idx = int(e[2]), 0
      else:
        key, idx = e[2].split('_')[0], int(e[2].split('_')[1])
      pmap[k][key] = e[1]
      stbin[k][e[1]] = outbin[key][idx]
      # res[e[1]] = 
      # res[e[1]] = np.array([int(i) for i in e[3:]])

# for k,v in pmap['Biased'].items(): print('%10s'%str(k), stbin[k], '-->', stbin[v], res[v]/sum(res[v]))

wellmap = {k: np.zeros(shape=(5,5)) for k in expmap.keys()}
tranmap = {k: np.zeros(shape=(5,5)) for k in expmap.keys()}

bincount = {k: {b: np.zeros(10, dtype=int) for b in TBIN10} for k in expmap.keys()}
binindex = {k: i for i,k in enumerate(TBIN10)}
for k, mlist in stbin.items():
  for src, srcbin in mlist.items():
    for b in outbin[src]:
      bincount[k][srcbin][binindex[b]] += 1
    if src not in stbin:
      continue
    t, s = stbin[src]
    if t == 'T':
      tranmap[k][int(s)] += res[v]/sum(res[v])
    else:
      wellmap[k][int(s)] += res[v]/sum(res[v])

for k in wellmap.keys():
  for i in range(5):
    if wellmap[k][i][i] == 0:
      wellmap[k][i][i] = 1
    if tranmap[k][i][i] == 0:
      tranmap[k][i][i] = 1

wellmap_n = {k: np.array([v[i]/sum(v[i]) for i in range(5)]) for k,v in wellmap.items()}
tranmap_n = {k: np.array([v[i]/sum(v[i]) for i in range(5)]) for k,v in tranmap.items()}

for k in expmap.keys():
  P.heatmap(wellmap_n[k][::-1], [0,1,2,3,4], [0,1,2,3,4], title='Accuracy_Well_%s' % k, xlabel='Start State', ylabel='Output Distribution')
  P.heatmap(tranmap_n[k][::-1], [0,1,2,3,4], [0,1,2,3,4], title='Accuracy_Tran_%s' % k, xlabel='Start State', ylabel='Output Distribution')

