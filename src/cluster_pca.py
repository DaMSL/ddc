import mdtraj as md
import numpy as np
import redis
import os
import math
import datetime as dt
from numpy import linalg as LA

from datatools.datareduce import *
from datatools.rmsd import *
from mdtools.deshaw import *

HOST = 'compute0038'
PDB_PROT   = RAW_ARCHIVE + '/bpti-prot.pdb'
topo = md.load(PDB_PROT)
filt = topo.top.select_atom_indices('alpha')
topoa = topo.atom_slice(FILTER['alpha'])
home = os.environ['HOME']

r = redis.StrictRedis(host=HOST, decode_responses=True)
filelist = r.lrange('xid:filelist', 0, -1)

takesample = lambda data, x: np.array([data[i] for i in range(0, len(data), x)])

def projgraph(data, stride, title, L=None):
  samp = np.array([data[i] for i in range(0, len(data), stride)])
  X,Y,Z = (samp.T[i] for i in (0,1,2))
  graph.dotgraph3D(X,Y,Z,title)
  if L is not None:
    sampL = np.array([L[i] for i in range(0, len(data), stride)])
    graph.dotgraph3D(X,Y,Z,title, L=sampL)

def projgraph2D(data, stride, title, L=None):
  samp = np.array([data[i] for i in range(0, len(data), stride)])
  X,Y = (samp.T[i] for i in (0,1))
  graph.dotgraph(X,Y,title)
  if L is not None:
    sampL = np.array([L[i] for i in range(0, len(data), stride)])
    graph.dotgraph(X,Y,title, L=sampL)


# POST CALC Convariance and sort by origin bin
covar = {b:[] for b in ab}
for num, tr in enumerate(filelist[:100]):
  if num % 50 == 0:
    print ('NUM:  ', num)
  pdb = tr.replace('dcd', 'pdb')
  if (not os.path.exists(tr)) or (not os.path.exists(pdb)):
    continue
  jc = r.hgetall('jc_' + os.path.splitext(os.path.basename(tr))[0])
  traj = md.load(tr, top=pdb)
  if traj.n_frames < 1000:
    continue
  jc = r.hgetall('jc_' + os.path.splitext(os.path.basename(tr))[0])
  srcbin.append(jc['src_bin'])
  A, B, = eval(jc['src_bin'])
  traj = md.load(tr, top=pdb)
  traj = traj.atom_slice(FILTER['alpha'])  
  covar[(A, B)].extend(calc_covar(traj.xyz, .2, 1, .1))

COV = np.array(covar)
DEcov = np.load('data/covar_1ns.npy')

# DEShaw PCA for each global state

sampleset= [[] for i in range(5)]
for i in range(0, 1031250, 25):
  idx = min(i//250, 4124)
  state = labels[idx]
  sampleset[state].append(DEcov[i])

DEpca = [PCA(n_components = .99) for i in range(5)]
for i in range(5):
  print('Running PCA for state', i)
  DEpca[i].fit(sampleset[i])

for i in range(5):
  print('Projecting State', i)
  proj = DEpca[i].transform(sampleset[i])
  train = np.array([proj[i] for i in range(0, len(proj), 4)])
  for N in [4, 5, 6, 7, 8, 9]:
    genKM = KMeans(n_clusters=N)
    genKM.fit(train)
    L = genKM.predict(proj)
    projgraph(proj, 6, 'DE_PCA_st%d_N%d'%(i, N), L)





gxyz_train = np.array([gxyz[i] for i in range(0, len(gxyz), 10)])

data = COV
for ktype in range(5):
  proj = kpcax[ktype].transform(data)
  train = np.array([proj[i] for i in range(0, len(proj), 4)])
  for N in [5]:
    genKM = KMeans(n_clusters=N)
    genKM.fit(train)
    L = genKM.predict(proj)
    projgraph(proj, 6, 'gen_KPCA_%s_%d'%(kpcax[ktype].kernel, N), L)




kpcax = calc_kpca(XYZ, n_comp=30)

for k in kpcax:
  proj = k.transform(XYZ)
  X, Y, Z = (proj[:,i] for i in [0,1,2])
  graph.dotgraph3D(X, Y, Z, 'KPCA_xyz_%s'%k.kernel)


covar = np.array(covar)
  # ds = distance_space(traj)
  # nwidth = 10
  # noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
  # filteredxyz = np.array([noisefilt(traj.xyz, i) for i in range(len(traj.xyz))])
  # ds = [pdist(i) for i in filteredxyz]
  # rms = np.array([np.array([cw[i] * LA.norm(cent[i]-p) for i in range(5)]) for p in ds])
  rms = np.array([md.rmsd(traj, ctraj, i) for i in range(5)])
  prox = [np.argsort(i) for i in rms]
  for i, rm in enumerate(rms):
    A = prox[i][0]
    B = prox[i][1]
    if i % 500 == 0:
      print(prox[i], A, B)
    for size in ['sm', 'md', 'lg']:
      if (rm[B] - rm[A]) > theta[size]:
        raw_obs[size].append((A, A))
      else:
        raw_obs[size].append((A, B))




#### =============   PCA/KMeans for discovery Learning
takesample = lambda data, x: np.array([data[i] for i in range(0, len(data), x)])

# For testing with DEShaw
DEcov = np.load('../data/covar_1ns.npy')

# Using Gen data
NUMFILES = 150
dcdlist = r.lrange('xid:filelist', 0, 10)
covmat = []
fmap = []
trajlist = []
for i, d in enumerate(dcdlist):
  if i % 100 == 0:
    print('Loading file #', i)
  traj = md.load(d, top=d.replace('dcd', 'pdb'))
  traj.atom_slice(filt, inplace=True)
  trajlist.append(traj)
  cov = calc_covar(traj.xyz, .2, 1, slide=.1)
  for k in range(len(cov)):
    fmap.append(d)
  covmat.extend(cov)


allpts = []
for i, d in enumerate(files1):
  if i % 100 == 0:
    print('Loading file #', i)
  traj = md.load(d, top=d.replace('dcd', 'pdb'))
  traj.atom_slice(filt, inplace=True)
  allpts.extend(traj.xyz)

  cov = calc_covar(traj.xyz, .2, 1, slide=.1)
  for k in range(len(cov)):
    fmap.append(d)
  covmat.extend(cov)



# Incremental PCA:




np.save('biasedcov200ns', np.array(covmat))

samp250 = takesample(DEcov, 250)
samp100 = takesample(DEcov, 250)

samp = {k: takesample(DEcov, k) for k in [250, 100, 25, 10]}
samp = {k: takesample(np.array(allpts), k) for k in [100, 10]}
sample = samp[100]
test = samp[10]

pca1 = calc_pca(sample)
pca1.fit(sample)
ptrain = pca1.transform(sample)

ptest = pca1.transform(test)

kpca = calc_kpca(sample, kerneltype='sigmoid', n_comp=30)
ktrain = pca.transform(sample)
ktest = pca.transform(test)

km1 = KMeans(5)
km1.fit(train)
Lk = km1.predict(test)

km = [KMeans(i) for i in range(5, 20)]

for k in km:
  k.fit(samp)
  k.predict(samp)

var = OrderedDict()
for k in km:
  N_k = np.bincount(k.labels_)
  V_k = np.zeros(k.n_clusters)
  for i, pt in enumerate(samp):
    clust = k.labels_[i]
    V_k[clust] += LA.norm(pt-k.cluster_centers_[clust])
  var[k.n_clusters] = np.sum(V_k)

X = []
Y = []
# for k, v in var.items():
for k in km:
  X.append(k.n_clusters)
  Y.append(k.inertia_)

plt.plot(X, Y)
plt.savefig('elbow.png')
plt.close()



V_k = [np.sum([ for center in ]) / 

gmm = calc_gmm(train, 5, 'full')
Lg = gmm.predict(test)

mu, clust = find_centers(ktrain, 5)
Lk = classify(ktest, mu)

show_compare(Lk)

def show_compare(L):
  AB = [[0 for a in range(5)] for B in range(5)]
  D = len(L) // 4125
  try:
    for i, l in enumerate(L):
      AB[l][min(int(lab[i//D]), len(lab))] += 1
  except IndexError as e:
    pass  #Ignore end of list values
  np.bincount(L)
  for a in range(5):
    for b in range(5):
      print(a, b, AB[a][b])