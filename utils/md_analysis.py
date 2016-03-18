
import mdtraj as md
import datetime as dt
import os
import sys
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as IPCA
from sklearn.mixture import GMM
import socket

np.set_printoptions(precision=3, suppress=True)
HOME = os.environ['HOME']
host = socket.gethostname()

if host == 'mddb2':
  pdb = '/mddb2/data/md/bpti/bpti-prot.pdb'
  # dcd = lambda i: 'bpti-prot-100-%02d.dcd' % i
  dcd = lambda i: '/mddb2/data/md/bpti/bpti-prot-%02d.dcd' % i
else:
  pdb = HOME + '/work/bpti/bpti-prot.pdb'
  dcd = lambda i: HOME + '/work/bpti/bpti-prot-%02d.dcd' % i

# To Common / utils
now = lambda: dt.datetime.now()
diff  = lambda st: print((now()-st).total_seconds())
    
topo = md.load(pdb)
f_min = topo.top.select_atom_indices('minimal')
f_alpha = topo.top.select_atom_indices('alpha')
f_heavy = topo.top.select_atom_indices('heavy')





def loadpts(skip=40, filt=None):
  pts = []
  for i in range(42):
    print('loading file: ', i)
    traj = md.load(dcd(i), top=pdb, stride=skip)
    if filt is not None:
      traj.atom_slice(filt, inplace=True)
    for i in traj.xyz:
      pts.append(i)
  return np.array(pts)


#Frame Every 10 ns
xyz = loadpts(skip=1, filt=f_alpha)


def calc_covar(xyz, size_ns, framestep):
  """Calculates the variance-covariance for sets of frames over the
  given trajectory of pts. 
  Note that Window size is calculated in picoseconds, which assumes 
  the framestep size is provide in picoseconds
  This returns a matrix whose rows are the variable variances for each
  windows
  TODO:  do overlapping
  """
  nDIM = len(filterType) * 3
  winsize = int((size_ns * 1000) // framestep)  # conv to ps and divide by frame step
  variance = np.zeros(shape=(len(xyz)//winsize, nDIM))
  st = now()
  for i in range(0, len(xyz), winsize):
    if i % 100000 == 0:
      print ("Calc: ", i)
    cm = np.cov(xyz[i:i+winsize].reshape(winsize, nDIM).T)
    variance[math.floor(i//winsize)] = cm.diagonal()
  print((now()-st).total_seconds())
  lab = '%dns'%size_ns if size_ns > 1 else '%dps' % int(size_ns*1000)
  np.save(HOME+'/ddc/data/covar_%s'%lab, variance)
  return variance

def calc_pca(xyz, title):
  n_dim = np.prod(xyz.shape[1:])
  pca = PCA(n_components = .99)
  pca.fit(xyz.reshape(len(xyz), n_dim))
  np.save('pca_%s_comp' %title, pca.components_)
  np.save('pca_%s_var' %title, pca.explained_variance_ratio_)
  np.save('pca_%s_mean' %title, pca.mean_)
  np.save('pca_%s_applied' %title, pca.transform(xyz.reshape(len(xyz), n_dim)))
  return pca

def calc_gmm(xyz, N, ctype='full'):
  n_dim = np.prod(xyz.shape[1:])
  gmm = GMM(n_components=N, covariance_type=ctype)
  gmm.fit(xyz.reshape(len(xyz), n_dim))
  np.save('gmm_%d_%s_mean' % (N, ctype), gmm.means_)
  np.save('gmm_%d_%s_wgt' % (N, ctype), gmm.weights_)
  np.save('gmm_fit_%d_%s' % (N, ctype), gmm.predict(xyz.reshape(len(xyz), n_dim)))
  return gmm


covar1ns = calc_covar(xyz, 1, 250)
gmm1ns = calc_gmm(xyz, 15, 'gmm_1ns_15')

deshaw(xyz)

st = now()
pca = calc_pca(variance, 'covar')
diff(st)

st = dt.datetime.now()
# Cov Mat for every 20ns trajectory
covmat = []
for i in range(3, len(xyz)):
  if i%1000 == 0:
    print(i)
  covmat.append(covmatrix(xyz[i-3:i]))

# Calc Eigen Vectors
ew = []
ev = []
for i, cm in enumerate(covmat):
  if i%1000 == 0:
    print(i)
  w, v = LA.eigh(cm)
  ew.append(w)
  ev.append(v)

NUM_PC = 3
eigenspace = np.zeros(shape=(len(ev), NUM_PC*174))
for i, e in enumerate(ev):
  eigenspace[i] = e[:,-1] + e[:,-2] + e[:,-3]   #<<----?????

# DO PCA over eigenspace


covmat = covmatrix(xyz)

def loadlabels():
  with open('bpti_labels_ms.txt') as src:
    lines = src.read().strip().split('\n')
  label = [int(l.split()[1]) for l in lines]
  print(len(label), ' labels')
  return label

def loadbpti():
  pdb='/bpti/bpti-prot/bpti-prot.pdb'
  dcd = lambda x: '/bpti/bpti-prot/bpti-prot-%02d.dcd' % x
  tr = []
  for i in range(11):
    print ('loading ', i)
    start = dt.datetime.now()
    tr.append(md.load(dcd(i), top=pdb))
    end = dt.datetime.now()
    print((end-start).total_seconds())
  return tr

def get_pairlist(traj):
  N = traj.n_atoms
  pairs = np.array([(i, j) for i in range(N-1) for j in range(i+1, N)])
  return pairs


#  RMS CALCULATIONS
#  Calculate distance space (for all pairs of atoms) for ea frame

# sums = np.zeros(shape=(5, 58, 3))
def distance_space(traj):
  """Convert a trajectory (or traj list) to distance space
     By default, this will compute ALL pair-wise distances and return
     a vector (or list of vectors if list of trajectories is provided)
  """
  if isinstance(traj, list):
    pairs = get_pairlist(traj[0])
    return [md.compute_distances(k,pairs) for k in traj]
  else:
    pairs = get_pairlist(traj)
    return md.compute_distances(traj,pairs)

def calc_bpti_centroid(traj_list):
  """Given a trajectory list of frames corresponding to the pre-labeled DEShaw
  dataset, calcualte centroids:
    - Groups frames by states and calcuate average (x,y,z) for all atoms
    This wil exclude any near-transition states and does a best fit of 
    using only in-state points (non-transition ones)
  """
  # Assuming distance space (with alpha-filter), hence 1653 dimensions
  sums = np.zeros(shape=(5, 1653))
  cnts = [0 for i in range(5)]
  for n, traj in enumerate(prdist):
    for i in range(0, len(traj), 40):
      try:
        idx = (n*400)  + (i // 1000)
        state = label[idx]
        # Exclude any near transition frames
        if idx < 3 or idx > 4121:
          continue
        if state == label[idx-2] == label[idx-1] == label[idx+1] == label[idx+2]:
          sums[state] += traj[i]
          cnts[state] += 1
      except IndexError as err:
        pass # ignore idx errors due to tail end of DEShaw data
  cent = [sums[i] / cnts[i] for i in range(5)]
  return (np.array(cent))


def check_bpti_rms(traj_list, centroid, skip=40):
  hit = 0
  miss = 0
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      predicted_state = np.argmin(dist)
      if labeled_state == predicted_state:
        hit += 1
      else:
        miss += 1
  print ('Hit rate:  %5.2f  (%d)' % ((hit/(hit+miss)), hit))
  print ('Miss rate: %5.2f  (%d)' % ((miss/(hit+miss)), miss))


def check_bpti_rms_trans(traj_list, centroid, skip=40):
  traj_list = prdist
  centroid = cent_d
  skip=100
  exact_hit = 0
  total_miss = 0
  det_trans = 0
  total = 0
  tdist = 0.
  wdist = 0.
  tlist = []
  theta = 0.15
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      prox = np.argsort(dist)
      A = prox[0]
      B = prox[1]
      delta = dist[B] - dist[A]
      if delta > theta:
        B = A
      if labeled_state == A == B:
        exact_hit += 1
        wdist += delta
      elif labeled_state == A or labeled_state == B:
        det_trans += 1
        tdist += delta
        tlist.append((idx, (A, B), labeled_state))
      else:
        total_miss += 1
      total += 1

print ('Hit rate:  %5.2f  (%d)' % ((exact_hit/(total)), exact_hit))
print ('Miss rate: %5.2f  (%d)' % ((total_miss/(total)), total_miss))
print ('Trans rate:  %5.2f  (%d)' % ((det_trans/(total)), det_trans))


tlist
check_bpti_rms_trans(prdist, cent_d, skip=100)

np.save('bpti-prot-centroid', centroid)


def covmatrix(traj):
  n_frames = traj.shape[0]
  n_atoms = traj.shape[1]*3
  A = traj.reshape(n_frames, n_atoms)
  a = A - np.mean(A, axis=0)
  cov = np.dot(a.T, a)/n_frames
  return cov



#check:

print('Hit=', hit)
print('Miss=', miss)



