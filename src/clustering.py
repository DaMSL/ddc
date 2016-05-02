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

HOST = 'localhost'
PORT = 6384
PDB_PROT   = RAW_ARCHIVE + '/bpti-prot.pdb'
topo = md.load(PDB_PROT)
filt = topo.top.select_atom_indices('alpha')
topoa = topo.atom_slice(FILTER['alpha'])
home = os.environ['HOME']

r = redis.StrictRedis(host=HOST, port=PORT, decode_responses=True)
filelist = r.lrange('xid:filelist', 0, -1)

takesample = lambda data, x: np.array([data[i] for i in range(0, len(data), x)])


def loadNP(r, key):
  elm = r.hgetall(key)
  if elm == {}:
    return None
  header = json.loads(elm['header'])
  arr = np.fromstring(elm['data'], dtype=header['dtype'])
  return arr.reshape(header['shape'])


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
cov = {b:[] for b in ab}
covar = []
avgxyz = []
for num, tr in enumerate(filelist[:1000]):
  if num % 50 == 0:
    print ('NUM:  ', num)
  pdb = tr.replace('dcd', 'pdb')
  if (not os.path.exists(tr)) or (not os.path.exists(pdb)):
    continue
  jc = r.hgetall('jc_' + os.path.splitext(os.path.basename(tr))[0])
  traj = md.load(tr, top=pdb)
  if traj.n_frames < 1000:
    continue
  traj = traj.atom_slice(FILTER['alpha'])  
  covar.extend(DR.calc_covar(traj.xyz, .2, 1, .05))
  for i in range(25, len(traj.xyz), 50):
    if i+200 > len(traj.xyz):
      break
    avgxyz.append(np.mean(traj.xyz[i:i+50], axis=0))

covar = np.array(covar)
variance = np.array([np.diag(i) for i in covar])
avgxyz = np.array(avgxyz).reshape(len(avgxyz), 174)


st=dt.datetime.now()
gmm.fit(X3)
print((dt.datetime.now()-st).total_seconds())

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
  for n_components in n_components_range:
    # Fit a mixture of Gaussians with EM
    gmm = GMM(n_components=n_components, covariance_type=cv_type)
    gmm.fit(X3)
    bic.append(gmm.bic(X3))
    if bic[-1] < lowest_bic:
      lowest_bic = bic[-1]
      best_gmm = gmm

  cov[(A, B)].extend(calc_covar(traj.xyz, .2, 1, .1))


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



def label(data, NUM_K):
  centroid, clusters = KM.find_centers(data, NUM_K)
  label, wght = KM.classify_score(data, centroid)
  return label

def getPts(db, N):
  numfiles = 1.25 * (N/1000)
  dcdlist = r.lrange('xid:filelist', 0, round(numfiles))
  ptlist = []
  for i, d in enumerate(dcdlist):
    if i % 100 == 0:
      print('Loading file #', i)
    traj = md.load(d, top=d.replace('dcd', 'pdb'))
    traj.atom_slice(filt, inplace=True)
    ptlist.extend(traj.xyz)
    if len(ptlist) > N:
      break
  return np.array(ptlist)

def getIndexByState(db, S=0, N=None, raw=False):
  key = 'label:raw:lg'
  if raw or not db.exists(key):
    key = 'label:rms'
  lim = -1 if N is None else S+N
  obs = db.lrange(key, S, lim)
  grp = [[] for i in range(5)]
  for i, o in enumerate(obs):
    A, B = eval(o)
    grp[A].append(i)
  return grp

def getIndexByBin(db, S=0, N=None, raw=False):
  key = 'label:raw:lg'
  if raw or not db.exists(key):
    key = 'label:rms'
  lim = -1 if N is None else S+N
  obs = db.lrange(key, S, lim)
  grp = {b: [] for b in [(A,B) for A in range(5) for B in range(5)]}
  for i, o in enumerate(obs):
    grp[eval(o)].append(i)
  return grp

DB={'UNIFORM':u, 'BIASED':b, 'REWEIGHT':r, 'PARALLEL':p, 'SERIAL':s}

keylist = ['SERIAL', 'PARALLEL', 'UNIFORM', 'BIASED', 'REWEIGHT']

state=4
indexlist = {}
indexlist['UNIFORM'] = getIndexByState(DB['UNIFORM'], 150000, 200000)
indexlist['REWEIGHT'] = getIndexByState(DB['REWEIGHT'], 150000, 200000)
indexlist['BIASED'] = getIndexByState(DB['BIASED'], 1300000, 200000, True)
indexlist['PARALLEL'] = getIndexByState(DB['PARALLEL'], 0, 200000)
indexlist['SERIAL'] = getIndexByState(DB['SERIAL'], 0, 200000, True)

# Get 200K Sample Points
labellist = {}
labellist['UNIFORM'] = getIndexByBin(DB['UNIFORM'], 150000, 200000)
labellist['REWEIGHT'] = getIndexByBin(DB['REWEIGHT'], 500000, 200000)
labellist['BIASED'] = getIndexByBin(DB['BIASED'], 1300000, 200000, True)
labellist['PARALLEL'] = getIndexByBin(DB['PARALLEL'], 0, 200000)
labellist['SERIAL'] = getIndexByBin(DB['SERIAL'], 0, 200000, True)

# BackProject HD Data
data = {}
for key in keylist:
  data[key] = {}
  for b in ab:
    print('PROCESSING: ', key, b)
    data[key][b] = bpoff.backProjection(DB[key], labellist[key][b])

# Save it
well = {}
tran = {}
both = {}
for key, bins in data.items():
  well[key] = []
  tran[key] = [[] for i in range(5)]
  both[key] = [[] for i in range(5)]
  for A in range(5):
    both[key][A].extend(bins[(A,B)])
    for B in range(5):
      if A == B:
        well[key].append(bins[(A,A)])
      else:
        tran[key][A].extend(bins[(A,B)])
  for state in range(5):
    well[key][state] = np.array(well[key][state]).reshape(len(well[key][state]), 174)
    tran[key][state] = np.array(tran[key][state]).reshape(len(tran[key][state]), 174)
    both[key][state] = np.array(tran[key][state]).reshape(len(tran[key][state]), 174)

  #   samp[k].append()
  # for k, v in bins.items():
  #   x.extend(i)
  # samp[k] = np.array(x).reshape(len(x), 174)
  # np.save('sample_%s'%k, np.array(x))

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

import dograph as G

pca = {k: [PCA(n_components=20) for i in range(5)] for k in keylist}

PC={}
dbL = {}
ktype = ['linear', 'poly', 'rbf', 'sigmoid']

sw = {}
st = {}
for key in keylist:
  N = len(both[key][4])
  sw[key] = np.array([both[key][4][i] for i in np.random.choice(range(N), min(1000, N//10))])
  st[key] = np.array([both[key][4][i] for i in np.random.choice(range(N), min(10000, N))])

kpca = {kt: {k: [KernelPCA(n_components=20, kernel=kt) for i in range(5)] for k in keylist} for kt in ktype}
km = {kt: {k: [KMeans() for i in range(5)] for k in keylist} for kt in ktype}
for kt in ['linear', 'rbf', 'sigmoid']:
  for key in ['PARALLEL', 'BIASED', 'REWEIGHT']:
      state = 4
      print('KPCAL:', kt, key, state)
      kpca[kt][key][state].fit(sw[key])
      T = kpca[kt][key][state].transform(sw[key])
      PC = kpca[kt][key][state].transform(st[key])
      km[kt][key][state].fit(T)
      L = km[kt][key][state].predict(PC)
      G.dotgraph3D(PC[:,0], PC[:,1], PC[:,2],'KPCA-%s_%s_%d'%(kt,key,state), L=L)

for key in ['PARALLEL', 'BIASED', 'REWEIGHT']:
  for state in range(5):
    print(key, state, km[key][state].inertia_/len(well[key][state]))


from sklearn.cluster import DBSCAN
dbL[key] = DBSCAN().fit_predict(PC[key]) 
    

translist = {}
welllist = {}
for key in keylist:
  translist[key] = [[] for i in range(5)]
  welllist[key] = [[] for i in range(5)]
  for k, v in labellist[key].items():
    A, B = k
    if A == B:
      welllist[key][A] = v
    else:
      translist[key][A].extend(v)
      translist[key][B].extend(v)
  for state in range(5):
    welllist[key][state] = np.array(welllist[key][state]).reshape(len(welllist[key][state]), 174)
    translist[key][state] = np.array(translist[key][state]).reshape(len(translist[key][state]), 174)

for state in range(5):
  print('%d,%s' % (state, ','.join([str(x) for x in [len(indexlist[k][state]) for k in keylist]])))

for b in binlist:
  print('%s,%s' % (b, ','.join([str(x) for x in [len(labellist[k][b]) for k in keylist]])))

for state in range(5):
  print('%d,%s' % (state, ','.join([str(x) for x in [len(translist[k][state]) for k in keylist]])))

for state in range(5):
  print('%d,%s' % (state, ','.join([str(x) for x in [len(welllist[k][state]) for k in keylist]])))



for key in DB.keys():
  print('Processing: ', key)
  indexlist = getIndexByState(DB[key], 50000)

state=4
for state in range(5):
  print('State, ', state)
  idxlist = np.random.choice(welllist['UNIFORM'][state], 1000)
  raw = bpoff.backProjection(DB['PARALLEL'], idxlist)
  kp[state].fit(raw.reshape(1000, 174))

wght = {k: [None for i in range(5)] for k in keylist}
for state in range(5):
state=4 
for key in DB.keys():
    print('Processing: ', key, ' STATE', state)
    N = len(translist[key][state])
    if N < 1000:
      idxlist = translist[key][state]
    else:
      idxlist = np.random.choice(translist[key][state], 1000)
    raw = bpoff.backProjection(DB[key], idxlist)
    # X = KPCA[state].project(raw)
    X = kp[state].transform(raw.reshape(len(raw), 174))
    centroid, clusters = KM.find_centers(X, 6)
    L, wght[key][state] = KM.classify_score(X, centroid)
    G.dotgraph(X[:,0], X[:,1], 'Trans_%d_%s' % (state, key), L)
    G.dotgraph3D(X[:,0], X[:,1], X[:,2], 'Trans_%d_%s' % (state, key), L)




bpoff.backProjection(r, )

dotgraph3D(P4[:,0], P4[:,1], P4[:,2], 'testkpca4', L=rlabel)
dotgraph3D(X[:,0], X[:,1], X[:,2], 'testunif4', L=U4L)
dotgraph3D(Y[:,0], Y[:,1], Y[:,2], 'testbias4', L=B4L)



#### =============   PCA/KMeans for discovery Learning
takesample = lambda data, x: np.array([data[i] for i in range(0, len(data), x)])

# For testing with DEShaw
DEcov = np.load('../data/covar_1ns.npy')

# Using Gen data
NUMFILES = 150
dcdlist = r.lrange('xid:filelist', 0, 1000)
covmat = []
fmap = []
trajlist = []
st = datetime.now()
for i, d in enumerate(dcdlist):
  if i % 50 == 0:
    print('Loading file #', i, (datetime.now()-st).total_seconds())
  p = d.replace('dcd', 'pdb')
  if (not os.path.exists(d)) or (not os.path.exists(p)):
    continue
  traj = md.load(d, top=p)
  traj.atom_slice(filt, inplace=True)
  trajlist.append(traj)
  cov = calc_covar(traj.xyz, .2, 1, slide=.1)
  for k in range(len(cov)):
    fmap.append(d)
  covmat.extend(cov)

print((datetime.now()-st).total_seconds())

X=np.array([np.hstack(x[i][:i] for i in range(174)) for x in covmat])
st = dt.datetime.now()
for i in range(4):
  ipca2.partial_fit(X[5500+(500*i):6000+(500*i)])
  print((dt.datetime.now()-st).total_seconds())  
#ipca.fit(X[:5000])


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

def showtree(kdt):
  enc = kdt.encode()
  klist = [k for k in sorted(enc.keys()) if k.startswith('1') or k.startswith('0')]
  for k in klist: 
    l = 0 if enc[k]['elm'] is None else len(enc[k]['elm'])
    if l > 0:
      vol = kdt.volume(k)
      adj_vol = math.pow(vol, 1/6)
      density = l / adj_vol 
      print('%-12s %4d %10.1f %6.1f %6.1f' % (k, l, vol, adj_vol*adj_vol, density))

kdt1 = kdtree.KDTree(100, maxdepth=9, data=P1, method='middle')
showtree(kdt1)


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