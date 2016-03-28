import mdtraj as md
import numpy as np
import redis
import os
import math
import datetime as dt
from numpy import linalg as LA

from datatools.datareduce import *
from  mdtools.deshaw import *
from datatools.rmsd import *

HOST = 'bigmem0003'
RAW_ARCHIVE = os.getenv('HOME') + '/work/bpti'
PDB_PROT   = RAW_ARCHIVE + '/bpti-prot.pdb'
topo = md.load(PDB_PROT)
filt = topo.top.select_atom_indices('alpha')
home = os.environ['HOME']
lab = loadlabels_aslist()


r = redis.StrictRedis(host=HOST, decode_responses=True)
filelist = r.lrange('xid:filelist', 0, -1)
dcent = np.load(home+'/ddc/data/gen-alpha-cartesian-centroid.npy')
# r.delete('label:raw')

theta = {'sm':.05, 'md':.25, 'lg':.33}
ab = [(A, B) for A in range(5) for B in range(5)]
szlist = ['sm', 'md', 'lg']
topoa = topo.atom_slice(FILTER['alpha'])

nwidth = 10
noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
cw = [.92, .94, .96, .99, .99]

for size in ['sm', 'md', 'lg']:
  r.delete('label:raw:%s'%size)
st = dt.datetime.now()
pipe = r.pipeline()


missing = 0
raw_obs = {'sm':[], 'md':[], 'lg':[]}
srcbin = []
trajlist = []
for num, tr in enumerate(filelist):
  pdb = tr.replace('dcd', 'pdb')
  if (not os.path.exists(tr)) or (not os.path.exists(pdb)):
    missing += 1
    continue
  jc = r.hgetall('jc_' + os.path.splitext(os.path.basename(tr))[0])
  srcbin.append(jc['src_bin'])
  traj = md.load(tr, top=pdb)
  if traj.n_frames < 1000:
    continue
  traj.unitcell_vectors = np.array([np.identity(3) * 5.126 for i in range(traj.n_frames)])
  traj.unitcell_angles = np.array([[90,90,90] for i in range(traj.n_frames)])
  traj.unitcell_lengths = np.array([[5.126,5.126,5.126] for i in range(traj.n_frames)])
  traj.center_coordinates()
  traj = traj.atom_slice(FILTER['alpha'])  
  trajlist.append(traj)
  # ds = distance_space(traj)
  filteredxyz = np.array([noisefilt(traj.xyz, i) for i in range(len(traj.xyz))])
  # ds = [pdist(i) for i in filteredxyz]
  rms = np.array([np.array([cw[i]*LA.norm(dcent[i]-p) for i in range(5)]) for p in filteredxyz])
  prox = [np.argsort(i) for i in rms]
  for i, rm in enumerate(rms):
    A = prox[i][0]
    B = prox[i][1]
    for size in ['sm', 'md', 'lg']:
      if (rm[B] - rm[A]) > theta[size]:
        raw_obs[size].append((A, A))
        pipe.rpush('label:raw:%s'%size, (A, A))
      else:
        raw_obs[size].append((A, B))
        pipe.rpush('label:raw:%s'%size, (A, B))
  if num > 0 and num % 500 == 0:
    print('Saving thru file #', num)
    pipe.execute()
    pipe = r.pipeline()

pipe.execute()

raw_label = {sz: {b: 0 for b in ab} for sz in szlist}
for size in szlist:
  for ro in raw_obs[size]:
    raw_label[size][ro] += 1

for i in sorted(ab):
  print(i, '%7d' % raw_label['sm'][i], '%7d' % raw_label['sm'][i], '%7d' % raw_label['lg'][i])


pipe = r.pipeline()
for i in rms_w:
  prox = np.argsort(i)
  A = prox[0]
  B = prox[1]
  if (i[B] - i[A]) > .33:
    obs_w.append((A, A))
    pipe.rpush('label:raw:lg', (A, A))
  else:
    obs_w.append((A, B))
    pipe.rpush('label:raw:lg', (A, B))

pipe.execute()

cnt_w = {b:0 for b in ab}
for i in obs_w:
  cnt_w[i] += 1


for b in sorted(ab):
  print(b, '%8d'%cnt_raw[b], '%8d'%cnt_w[b])