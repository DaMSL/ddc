#!/usr/bin/env python
import logging


import mdtraj as md
import datetime as dt
import os
import sys
import pickle

import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as IPCA
from sklearn.mixture import GMM

from scipy.spatial.distance import pdist
import socket


np.set_printoptions(precision=3, suppress=True)



__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

hyperspaces = ['cartesian', 'distance']


def get_pairlist(traj):
  """Returns set of pairs of all atom indices from given trajectory
  Assumes traj is either an MD Trajector or an ND-Array
  """
  N = traj.n_atoms if isinstance(traj, md.Trajectory) else traj.shape[1]
  pairs = np.array([(i, j) for i in range(N-1) for j in range(i+1, N)])
  return pairs


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


def calc_covar(xyz, size_ns, framestep):
  """Calculates the variance-covariance for sets of frames over the
  given trajectory of pts. 
  Input xyz is NDarray
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


def calc_rmsd(traj, centroid, space='cartesian', title=None):
  """Calculated the RMSD from each point in traj to each of the centroids
  Input passed can be either a trajectory or an array-list object
  Title, if defined, will be used in the output filename
  """
  # TODO: Check dimenstionality of source points and centroid for given space
  observations = traj.xyz if isinstance(traj, md.Trajectory) else traj
  if space == 'distance':
    pairs = get_pairlist(observations)


  rmsd = np.zeros(shape=(len(xyz), len(centroid)))
  for n, pt in enumerate(observations):
    # TODO:  Check Axis here
    rmsd[n] = np.array([np.sum(LA.norm(pt - C)) for C in centroid])
  return rmsd



def calc_pca(xyz, title=None):
  n_dim = np.prod(xyz.shape[1:])
  pca = PCA(n_components = .99)
  pca.fit(xyz.reshape(len(xyz), n_dim))
  if title is not None:
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





def PCA(src, pc, numpc=3):
  # TODO: Check size requirements
  projection = np.zeros(shape=(len(src), numpc))
  for i, s in enumerate(src):
    np.copyto(projection[i], np.array([np.dot(s.flatten(),v) for v in pc[:numpc]]))
  return projection

def load_trajectory(dcdfile, pdbfile=None):
  if pdbfile is None:
    pdbfile = dcdfile.replace('.dcd', '.pdb')
  traj = md.load(dcdfile, top=pdbfile)
  return traj

def filter_heavy(source, pdbfile=None):
  """
  Apply heavy atom filter. If no PDB file is provided, assume it is
  saved along side the dcd file, but with .pdb extension
  """
  if isinstance(source, md.Trajectory):
    traj = source
  else:
    # TODO: Check source is a string
    dcdfile = source
    traj = load_trajectory(dcdfile, pdbfile)

  filt = traj.top.select_atom_indices(selection='heavy')
  traj.atom_slice(filt, inplace=True)
  
  # ref = deshaw.deshawReference()
  # traj.superpose(ref, frame=0)
  return traj


