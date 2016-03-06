#!/usr/bin/env python
import numpy as np
import mdtraj as md

import simmd.deshaw as deshaw

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)

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
  Apply heavy atom filter. If no PDD file is provided, assume it is
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
  
  # Set DEShaw Reference point
  # ref = deshaw.deshawReference()
  # traj.superpose(ref, frame=0)
  return traj


