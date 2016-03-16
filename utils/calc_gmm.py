
import mdtraj as md
import datetime as dt
import os
import sys
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
import socket
import math

HOME = os.environ['HOME']
host = socket.gethostname()

if host == 'mddb2':
  pdb = '/mddb2/data/md/bpti/bpti-prot.pdb'
  # dcd = lambda i: 'bpti-prot-100-%02d.dcd' % i
  dcd = lambda i: '/mddb2/data/md/bpti/bpti-prot-%02d.dcd' % i
else:
  pdb = HOME + '/work/bpti/bpti-prot.pdb'
  dcd = lambda i: HOME + '/work/bpti/bpti-prot-%02d.dcd' % i


now = lambda: dt.datetime.now()
diff  = lambda st: print((now()-st).total_seconds())
    
topo = md.load(pdb)
f_min = topo.top.select_atom_indices('minimal')
f_alpha = topo.top.select_atom_indices('alpha')
f_heavy = topo.top.select_atom_indices('heavy')
filterType = f_alpha

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
  winsize = int((size_ns * 1000)) // framestep  # conv to ps and divide by frame step
  variance = np.zeros(shape=(len(xyz) // winsize, nDIM))
  st = now()
  for i in range(0, len(xyz), winsize):
    if i % 100000 == 0:
      print ("Calc: ", i)
    cm = np.cov(xyz[i:i+winsize].reshape(len(xyz[i:i+winsize]), nDIM).T)
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
gmm1ns = calc_gmm(xyz, 15)