"""
Data Reduction Methods and techqniques
"""

import datetime as dt
import os
import sys
import pickle
import socket
import logging
import math

import mdtraj as md
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import IncrementalPCA as IPCA
from scipy.spatial.distance import pdist

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

hyperspaces = ['cartesian', 'distance']

def get_pairlist(traj):
  """Returns set of pairs of all atom indices from given trajectory
  Assumes traj is either an MD Trajector or an ND-Array
  """
  N = traj.n_atoms if isinstance(traj, md.Trajectory) else traj.shape[1]
  pairs = np.array([(i, j) for i in range(N-1) for j in range(i+1, N)])
  return pairs

def distance_space(traj, top=None, pairs=None):
  """Convert a trajectory (or traj list) to distance space
     By default, this will compute ALL pair-wise distances and return
     a vector (or list of vectors if list of trajectories is provided)
     If a NDarray of conformations is passed, the associated topology 
     must be provided to recreate the trajectory
  """
  if isinstance(traj, np.ndarray):
    if top is None:
      logging.error("Cannot calculate distances from NDArray. Topology is needed.")
      return None
    else:
      traj = md.Trajectory(traj, top)
  if isinstance(traj, list):
    if pairs is None:
      pairs = get_pairlist(traj[0])
    return [md.compute_distances(k,pairs) for k in traj]
  else:
    if pairs is None:
      pairs = get_pairlist(traj)
    return md.compute_distances(traj,pairs)

def calc_var(xyz, size_ns, framestep, slide=None):
  """Calculates the variance-covariance for sets of frames over the
  given trajectory of pts and RETURN Variance Vectors
  Input xyz is NDarray
  Note that Window size is calculated in picoseconds, which assumes 
  the framestep size is provide in picoseconds
  This returns a matrix whose rows are the variable variances for each
  windows
  Slide = shift amount for each window. If not provided, the slide will 
  assume to be the same as the window size (and hence no overlap). Otherwise
  this should be provided in ns
  TODO:  do overlapping
  """
  nDIM = np.prod(xyz.shape[1:])
  winsize = int((size_ns * 1000) // framestep)  # conv to ps and divide by frame step
  shift = winsize if slide is None else int(slide * 1000)
  n_windows = math.floor((len(xyz) * framestep) / shift)
  variance = []
  # st = dt.datetime.now()
  for i in range(0, len(xyz), shift):
    if i+winsize > len(xyz):
      break
    cm = np.cov(xyz[i:i+winsize].reshape(winsize, nDIM).T)
    variance.append(cm.diagonal())
  # print((dt.datetime.now()-st).total_seconds())
  # lab = '%dns'%size_ns if size_ns > 1 else '%dps' % int(size_ns*1000)
  # np.save(HOME+'/ddc/data/covar_%s'%lab, variance)
  return np.array(variance)

def calc_covar(xyz, size_ns, framestep, slide=None):
  """Calculates the variance-covariance for sets of frames over the
  given trajectory of pts and return COVARIANCE Matrices
  Input xyz is NDarray
  Note that Window size is calculated in picoseconds, which assumes 
  the framestep size is provide in picoseconds
  This returns a matrix whose rows are the variable variances for each
  windows
  Slide = shift amount for each window. If not provided, the slide will 
  assume to be the same as the window size (and hence no overlap). Otherwise
  this should be provided in ns
  TODO:  do overlapping
  """
  nDIM = np.prod(xyz.shape[1:])
  winsize = int((size_ns * 1000) // framestep)  # conv to ps and divide by frame step
  shift = winsize if slide is None else int(slide * 1000)
  n_windows = math.floor((len(xyz) * framestep) / shift)
  covariance = []
  # st = dt.datetime.now()
  for i in range(0, len(xyz), shift):
    if i+winsize > len(xyz):
      break
    cv = np.cov(xyz[i:i+winsize].reshape(winsize, nDIM).T)
    covariance.append(cv)
  return np.array(covariance)

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


def filter_alpha(source, pdbfile=None):
  """
  Apply heavy atom filter. If no PDB file is provided, assume it is
  saved along side the dcd file, but with .pdb extension
  """
  if isinstance(source, md.Trajectory):
    traj = source
  else:
    dcdfile = source
    traj = load_trajectory(dcdfile, pdbfile)

  filt = traj.top.select_atom_indices(selection='alpha')
  traj.atom_slice(filt, inplace=True)
  
  return traj

