#!/usr/bin/env python

"""
Offline Analysis of DCD Trajectory
"""

import argparse
import os
 
# For efficient zero-copy file x-fer
import mdtraj as md
import numpy as np
from numpy import linalg as LA

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"
 


if __name__ == '__main__':
  pdb = md.load('coord.pdb')

  elist = ['tran'+str(a) for a in [23,24,30,32,34,40,41,42]]
  elist.extend(['well'+str(a) for a in range(5)])
  print("Elist: ", elist)

  for exp in elist:
    print('Processing: ', exp)
    cfile = exp + '/coord.pdb'
    dfile = exp + '/' + exp + '.dcd'
    pdb = md.load(cfile)
    pfilt = pdb.top.select('protein')
    traj = md.load(dfile, top=cfile, atom_indices=pfilt)
    print('  Loaded N_Frames: ', traj.n_frames)
    traj.save_dcd('output/' + exp + '.dcd')
    print('  File Saved!')


