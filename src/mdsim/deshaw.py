#!/usr/bin/env python

import os
from collections import namedtuple
import logging

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)


# Predefined topology and parameters for DEShaw BPTI
TOPO  = os.getenv('HOME') +  "/bpti/amber/top_all22_prot.inp"
PARM  = os.getenv('HOME') +  "/bpti/amber/par_all22_prot.inp"



# TODO:  Make a Raw Archiver class (& link in with data mover)

# Hard Coded for now
RAW_ARCHIVE = os.getenv('HOME') + '/work/bpti'
PDB_FILE    = os.getenv('HOME') + '/work/bpti/bpti-all.pdb'


label =namedtuple('window', 'time state')

def loadLabels(fn=None):
  if fn is None:
    fn = os.path.join(os.getenv('HOME'), 'ddc', 'bpti_labels_ms.txt')
  label =namedtuple('window', 'time state')
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win



def getLabelList(labels):
  labelset = set()
  for lab in labels:
    labelset.add(lab.state)
  return sorted(list(labelset))


def getDEShawfilename(seqnum, fullpath=False):
    filename = 'bpti-all-%03d.dcd' if int(seqnum) < 1000 else 'bpti-all-%04d.dcd'
    if fullpath:
      filename = os.path.join(RAW_ARCHIVE, filename)
    return filename


def getHistoricalTrajectory(seqnum):
    fname = getDEShawfilename(seqnum)
    dfile = os.path.join(RAW_ARCHIVE, fname % int(seqnum))
    pfile = PDB_FILE
    return pfile, dfile


def loadDEShawTraj(start, end=-1, filt='heavy'):
  if end == -1:
    end = start +1
  trajectory = None
  for seqnum in range(start, end):
    pdbfile, dcdfile = getHistoricalTrajectory(seqnum)
    if not os.path.exists(dcdfile):
      logging.info('%s   File not exists. Continuing with what I got', dcdfile)
      break
    logging.info("LOADING:  %s", os.path.basename(dcdfile))
    traj = md.load(dcdfile, top=pdbfile)
    traj.atom_slice(traj.top.select_atom_indices(selection=filt), inplace=True)
    trajectory = trajectory.join(traj) if trajectory else traj
  return trajectory

def deshawReference(atomfilter='heavy'):
  pdbfile, dcdfile = getHistoricalTrajectory(0)
  traj = md.load(dcdfile, top=pdbfile, frame=0)
  filt = traj.top.select_atom_indices(selection=atomfilter)
  traj.atom_slice(filt, inplace=True)
  return traj



if __name__ == '__main__':

  #  FOR Calculting Centroids and RMSD of ALL conforms in D.E.Shaw Dataset
  settings = systemsettings()
  settings.applyConfig('debug.json')
  win = loadLabels()
  win.append(win[-1])  
  mean = []
  w = []
  traj_ref = deshawReference()
  for dcd in range(4125):
    if dcd % 100 == 0:
      logging.debug("Processing dcd file: %d", dcd)
    traj = loadDEShawTraj(dcd, filt='heavy')
    traj.superpose(traj_ref)
    mean.append(np.mean(traj.xyz, axis=0))
  centroid = np.zeros(shape=(5, 454, 3))
  for i in range(5):
    accum = []
    for k in range(len(mean)):
      if win[k].state == i:
        accum.append(mean[k])
    mn = np.mean(np.array(accum), axis=0)
    np.copyto(centroid[i], mn)
  np.save('centroid.npy', centroid)
  centroid = np.load('centroid.npy')
  rmsd = np.zeros(shape=(4125, 1000, 5))
  with open('rms.out', 'a') as rmsfile:
    for dcd in range(4125):
      traj = loadDEShawTraj(dcd, filt='heavy')
      traj.superpose(traj_ref)
      for f in range(len(traj.xyz)):
        rms = np.array([LA.norm(traj.xyz[f]-centroid[k]) for k in range(5)])
        np.copyto(rmsd[dcd][f], rms)
        rmsfile.write('%d;%d;%d;%d;%s\n' %(dcd, f, win[dcd].state, np.argmin(rms),str(rms.tolist())))
  np.save('rmsd', rmsd)


# rms = np.zeros(shape=(len(tr0), 5))
# for i in range(len(tr0)):
#   np.copyto(rms[i], np.array([LA.norm(tr0.xyz[i]-c1[k]) for k in range(5)]))

# r34 = np.zeros(shape=(1000, 5))
# for i in range(1000):
#   np.copyto(r34[i], np.array([LA.norm(tr34.xyz[i]-c1[k]) for k in range(5)]))




# count = np.zeros(5)
# for i in r33: count[np.argmin(i)] += 1
# rms  = []
# for i, r in enumerate(rm):
#   res = r.replace('[','').replace(']','').split(',')
#   rms.append((i//1000, int(res[1]), int(res[2]), int(res[3]), np.array([float(v) for v in res[4:]])))




# rms_ps = []
# for frame in range(len(rms)):
#   lo = max(0, frame-4)
#   hi = min(frame+5, len(rms))
#   ps = np.mean(np.array([r[4] for r in rms[lo:hi]]), axis=0)
#   rms_ps.append((rms[frame][0], rms[frame][1], rms[frame][2], np.argmin(ps), ps))

# counts = {}
# for i in rms_ps:
#   if i[2] != i[3]:
#     if i[0] not in counts.keys():
#       counts[i[0]] = 0
#     counts[i[0]] += 1