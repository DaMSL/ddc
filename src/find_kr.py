import mdtools.deshaw as DE
import itertools as it
import pickle
import string
from collections import defaultdict, OrderedDict
import datatools.lattice as lat
import numpy as np
import os
import logging

home = os.getenv('HOME')

ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
k_domain = label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek

tok   = lambda x: ''.join(sorted([k_domain[i] for i in x]))
tofz  = lambda x: frozenset([k_domain.index(i) for i in x])
toidx = lambda x: [ord(i)-97 for i in x]
fromk = lambda x: np.array([1 if i in x else 0 for i in k_domain])
fromm = lambda x: ''.join(sorted([k_domain[i] for i,m in enumerate(x) if m == 1]))

dist  = lambda k: Dr[Ik[k]][:,toidx(k)] #if len(k) > 1 else Dr[Ik[k]][toidx(k)]
dist1d = lambda k, f: Dr[Ik[k]][:,toidx(f)]
histo = lambda d: np.histogram(d, 20, (4, 8))[0]

nhist1d = lambda k, f: np.histogram(Dr[Ik[k]][:,toidx(f)], 24, (4, 8))[0] / len(Ik[k])

DS = 10*np.load('../data/de_ds_mu.npy')
label = DE.loadlabels_aslist()
blabel = [label[int(i/22.09)] for i in range(91116)]

cutoff = 8.
scale = 1
bL = [blabel[i] for i in range(0, 91116, scale)]
D = DS[range(0, 91116, scale)]
CM = (D<cutoff)
K2 = lat.reduced_feature_set(CM, .02); len(K2)
F2 = [i for i in K2 if D[:,i].clip(0, cutoff).std() > .3]; len(F2)
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
CMr, Dr = CM[:,Kr], D[:,Kr]; CMr.sum()/np.multiply(*CM.shape)
logging.info('FINAL Input Matrix:  %s', Dr.shape)
print('Kr:\n' str(Kr))