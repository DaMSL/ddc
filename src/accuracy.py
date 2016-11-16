import pickle, string, os, logging
import mdtools.deshaw as DE
import itertools as it
from collections import defaultdict, OrderedDict
import datatools.lattice as lat
import datatools.datareduce as DR
import mdtools.timescape as TS

TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]
# ['W0', 'W1', 'W2', 'W3', 'W4', 'T0', 'T1', 'T2', 'T3', 'T4']

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

