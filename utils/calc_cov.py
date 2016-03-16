import mdtraj as md
import datetime as dt
import os
import sys
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

HOME = os.environ['HOME']
# For MARCC
pdb = HOME + '/work/bpti/bpti-prot.pdb'
dcd = lambda i: HOME + '/work/bpti/bpti-prot-%02d.dcd' % i

# For MDDB2
# pdb = '/mddb2/data/md/bpti/bpti-prot.pdb'
# dcd = lambda i: '/mddb2/data/md/bpti/bpti-prot-%02d.dcd' % i

topo = md.load(pdb)
f_min = topo.top.select_atom_indices('minimal')
f_alpha = topo.top.select_atom_indices('alpha')
f_heavy = topo.top.select_atom_indices('heavy')

now = lambda: dt.datetime.now()



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


filterType = f_alpha
xyz = loadpts(skip=1, filt=filterType)









def calc_covar(xyz, size_ns, slide_ns, framestep):
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

