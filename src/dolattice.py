import mdtools.deshaw as DE
import itertools as it
import pickle, string
from collections import defaultdict, OrderedDict
import datatools.lattice as lat
import numpy as np
import os
import logging

home = os.getenv('HOME')
k_domain = string.ascii_lowercase + string.ascii_uppercase
tok   = lambda x: ''.join(sorted([k_domain[i] for i in x]))

DS = 10*np.load('../data/de_ds_mu.npy')
label = DE.loadlabels_aslist()

scale = 5
blabel = [label[int(i/22.09)] for i in range(91116)]
bL = [blabel[i] for i in range(0, 90000, scale)]
D = DS[range(0, 90000, scale)]

cutoff = 8.
CM = (D<cutoff).astype(int)
K2 = lat.reduced_feature_set(CM, .02); len(K2)
F2 = [i for i in K2 if D[:,i].clip(0, cutoff).std() > .2]; len(F2)
F2std = [x[0] for x in sorted([(i, D[:,i].clip(0, cutoff).std()) for i in F2], key = lambda x: x[1])]
prs = list(it.combinations(range(58), 2))
plist = [prs[i] for i in F2std]
restrack = [[] for i in range(58)]
for a, b in plist:
  restrack[a].append((a,b))
  restrack[b].append((a,b))

remres = set([i for i, r in enumerate(restrack) if len(r) > 0])
rev_pr_idx = {k: i for i, k in enumerate(prs)}
minsize = 1
F3 = set()
while len(remres) > 0:
  selprs = [p for p in restrack if len(p) == minsize]
  for sp in selprs:
    for a, b in sp:
      if a not in remres and b not in remres:
        continue
      if a in remres:
        remres.remove(a)
      if b in remres:
        remres.remove(b)
      F3.add(rev_pr_idx[(a,b)])
  minsize += 1

Kr = sorted(F3)
CMr, Dr = CM[:,Kr], D[:,Kr]

logging.info('FINAL Input Matrix:  %s', Dr.shape)
U = lat.unique_events(CMr)
logging.info('\n MAX-MINER running')
MFIS = lat.maxminer(CMr, 100)
pickle.dump(MFIS, open(home + '/work/mfis.p', 'wb'))
logging.info('\n Max Miner Complete. Constructing derived lattice')
dlat, Ik = lat.dlattice_mm(MFIS, Dr, CMr, 100)
logging.info('\n ALL DONE! Pickling Out')
pickle.dump(Ik, open(home + '/work/iset.p', 'wb'))
pickle.dump(dlat, open(home + '/work/dlat.p', 'wb'))
