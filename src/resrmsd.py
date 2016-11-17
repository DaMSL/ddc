import pickle, string, os, logging
import mdtools.deshaw as DE
import itertools as it
from collections import defaultdict, OrderedDict
import datatools.lattice as lat
import datatools.datareduce as DR
import mdtools.timescape as TS

TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]
# ['W0', 'W1', 'W2', 'W3', 'W4', 'T0', 'T1', 'T2', 'T3', 'T4']

REF = 

afilt,hfilt = DE.FILTER['alpha'], DE.FILTER['heavy']
cent = np.load('../data/bpti-alpha-dist-centroid.npy')
topo = md.load(home+'/work/jc/serial2/de0_0/de0_0.pdb')
topoa = topo.atom_slice(topo.top.select_atom_indices('alpha'))
deL = DE.loadlabels_aslist()
bL = [deL[int(i/22.09)] for i in range(91116)]

traj, rmsd = [], []
for tnum in range(42):
  p, d = DE.getHistoricalTrajectory_prot(tnum)
  traj.append(md.load(d, top=p).atom_slice(afilt))

resrms = []
ccent = np.load('../data/bpti-alpha-cartesian-centroid.npy')
for tnum, tr in enumerate(detraj):
  minlist = TS.TimeScape.read_log(home+'/work/timescape/desh_%02d_minima.log'%tnum)
  minima = tr.slice(minlist)
  minima.superpose(topoa)
  for m in minima.xyz:
    resrms.append(LA.norm(ref - m, axis=1))


  ds = DR.distance_space(minima)
  state = np.array([np.argmin(LA.norm(cent-i, axis=1)) for i in ds])
  resrms = [np.zeros(58)]
  for in range(len(minima[1:])):
    resrms.append(LA.norm(minima[i].xyz - minima[i-1], axis=2))

  for m, s in zip(minima, state):
    resrms.append(LA.norm(m.xyz - ccent[s], axis=0))

for i in range(1, 91116): 
  CM[i] = np.abs(resrms[max(0, i-5):i].mean(0)-resrms[i: min(91116, i+5)].mean(0)) > theta

a, b, Z = 0, 30000, 300
for sc in range(8):
  data = {'State':bL[a:b]}
  for rd in range(7):
    rdu = (sc*8 + rd)
    if rdu >= 58:
      break
    # data['Rd%d' % rdu] = mvavg(10* resrms[a:b][:,rdu], Z)
    data['R%d' % rdu] = 10*resrms[a:b][:,rdu]
  P.lines(data, title='rms_res_%02d'%sc, ylim=(-.25, 4.25))

import time
for i in range(30):
  time.sleep(60)
  print('SLEEPING', i)


  st = dt.now()
  


  print('LOADED #', i, (dt.now()-st).total_seconds())




# BASIN WINDOWS -- DESHAW
DW = []
for i in range(42):



  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    DW.append((a+i*100000, b+i*100000))

for i in range(42):
  for m in :
    DMin.append(m + i*100000)

res = {}
delab = np.load(home+'/work/results/DE_label_full.npy').astype(int)
with open(home+'/work/results/trajout_deshaw', 'w') as out:
  for i, (m, (a,b)) in enumerate(zip(DMin, DW)):
    key = m
    cnt = np.bincount(delab[a:b], minlength=5)
    res[key] = cnt
    # _=out.write('%d,%07d,%s\n' % (i, key, ','.join([str(i) for i in cnt])))

