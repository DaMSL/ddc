import mdtraj as md
import numpy as np
import redis
import os
import math
import datetime as dt
from numpy import linalg as LA

from datatools.datareduce import *
from datatools.rmsd import *
import  mdtools.deshaw as deshaw

HOST = 'bigmem0026'
RAW_ARCHIVE = os.getenv('HOME') + '/work/bpti'
PDB_PROT   = RAW_ARCHIVE + '/bpti-prot.pdb'
topo = md.load(PDB_PROT)
filt = topo.top.select_atom_indices('alpha')
home = os.environ['HOME']

r = redis.StrictRedis(host=HOST, decode_responses=True)
cent = np.load(home+'/ddc/bpti-alpha-cart-centroid-well.npy')
r.delete('label:hist')
for size in ['sm', 'md', 'lg']:
  r.delete('label:hist:%s'%size)
raw_obs = []
st = dt.datetime.now()

SM = 0.25
MD = .5
LG = 1

theta = {'sm':.25, 'md':.5, 'lg':1.}
raw_obs = {'sm':[], 'md':[], 'lg':[]}

missing = 0
points = deshaw.loadpts(skip=1000, filt=deshaw.FILTER['alpha'])
rmsd = []
for num, pt in enumerate(points):
  # rmsd.append(np.array([np.sum([LA.norm(pt[a] - C[a]) for a in range(58)]) for C in cent]))
  rmsd.append(np.array([LA.norm(pt - C) for C in cent]))
  prox = [np.argsort(i) for i in rmsd]

pipe = r.pipeline()
for i, rm in enumerate(rmsd):
  A = prox[i][0]
  B = prox[i][1]
  for size in ['sm', 'md', 'lg']:
    if (rm[B] - rm[A]) > theta[size]:
      raw_obs[size].append((A, A))
    else:
      raw_obs[size].append((A, B))
  pipe.rpush('label:hist:%s'%size, (A, B))
if num > 0 and num % 100000 == 0:
  print('Saving thru index #', num)
  pipe.execute()
  pipe = r.pipeline()

pipe.execute()

ab = [(A, B) for A in range(5) for B in range(5)]

for size in ['sm', 'md', 'lg']:
  raw_label   = {b: 0 for b in ab}
  for ro in raw_obs[size]:
    raw_label[ro] += 1
  print("DISTRO for SIZE:  ", size)
  for i in sorted(ab):
    print(i, '%7d' % raw_label[i])


