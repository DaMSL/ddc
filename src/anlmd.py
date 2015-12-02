import argparse
import sys
import os

import mdtraj as md
import numpy as np
from numpy import linalg as LA
import nearpy
from nearpy.storage.storage_redis import RedisStorage
from nearpy.hashes import RandomBinaryProjections, PCABinaryProjections

import redisCatalog
from common import *
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def distmatrix(traj):
  n_frames = traj.shape[0]
  n_atoms  = traj.shape[1]
  mean = np.mean(traj, axis=0)
  dist = np.zeros(shape = (n_atoms, n_atoms), dtype=np.float32)
  for A in range(n_atoms):
    for B in range(A, n_atoms):
      delta = LA.norm(mean[A] - mean[B])
      dist[A][B] = delta
      dist[B][A] = delta
  return dist


def covmatrix(traj):
  n_frames = traj.shape[0]
  n_atoms = traj.shape[1]*3
  A = traj.reshape(n_frames, n_atoms)
  a = A - np.mean(A, axis=0)
  cov = np.dot(a.T, a)/n_frames
  return cov


def makeIndex(eg, ev, num_pc=DEFAULT.NUM_PCOMP):
  num_var = len(eg)
  index_size = num_var * num_pc
  index = np.zeros(index_size)
  eigorder = np.argsort(abs(eg))[::-1]
  norm = LA.norm(eigorder, ord=1)
  # np.copyto(index[:num_var], ev[i][:,-1] * eg[i][-1])    # FOr only 1 eigvector
  for n, eig in enumerate(eigorder[:num_pc]):
    # direction = -1 if eg[eig] < 0 else 1
    # np.copyto(index[n*num_var:n*num_var+num_var], direction * ev[:,eig])
    np.copyto(index[n*num_var:n*num_var+num_var], ev[:,eig] * eg[eig] / norm)
  return index


class analysisJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname,  'anl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('dcdFileList', 'LDIndexList')
      self.setState('JCComplete', 'processed', 'anlSplitParam', 'anlDelay', 'indexSize')

      self.modules.add('redis')

      self.buildArchive = False
      self.manual = False

      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = DEFAULT.CPU_PER_NODE

      # TODO: Move to Catalog or define on a per-task basis
      self.winsize = 100
      self.slide   = 50

    def setBuild(self, build=True):
      self.buildArchive = build

    def term(self):
      # For now
      return False

    def split(self):
      split = int(self.data['anlSplitParam'])
      immed = self.data['dcdFileList'][:split]
      return immed, split


    def configElasPolicy(self):
      self.delay = self.data['anlDelay']


    def execute(self, i):

      # TODO: Better Job ID Mgmt, for now hack the filename
      i.replace(':', '_').replace('-', '_')
      jobnum = os.path.basename(i).split('.')[0].split('_')[-1]
      # logging.debug("jobnum = " + jobnum)

      if self.manual:
        dcd, pdb = tuple(map(lambda x: os.path.join(os.path.dirname(i), "%s.%s" % (jobnum, x)), ['dcd', 'pdb']))
      else:
        dcd, pdb = tuple(map(lambda x: os.path.join(DEFAULT.JOB_DIR, jobnum, "%s.%s" % (jobnum, x)), ['dcd', 'pdb']))

      # 1. Load raw data from trajectory file
      traj = md.load(dcd, top=pdb)
      traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)

      logging.debug('Trajectory Loaded: %s - %s', jobnum, str(traj))

      # Ensure trajectory actually contains data to analyze:
      if traj.n_frames < self.winsize:
        logging.warning("Cannot process Trajectory, %s.  Contains %d frames (which is less than Winsize of %d)", jobnum, traj.n_frames, self.winsize)
        return []

      result = {}
      indexSize = self.data['indexSize']
      # 2. Split raw data in WINSIZE chunks and calc eigen vectors
      #   TODO: Retain provenance
      for win in range(0, len(traj.xyz), self.slide):
        if win + self.winsize > len(traj.xyz):
          break
        eg, ev = LA.eigh(distmatrix(traj.xyz[win:win+self.winsize]))
        key = jobnum + ':' + '%04d' % win
        result[key] = makeIndex(eg, ev)

      # To Build Archive online
      if self.buildArchive:
        archive = redisCatalog.dataStore(**archiveConfig)
        engine = getNearpyEngine(archive, indexSize)
        for key, idx in result.items():
          engine.store_vector(idx, key)

      # Index for downstream retrieval
      logging.debug('Saving Index in catalog')

      # Pack & store data
      for k, v in result.items():
        logging.debug(" `%s`:  <vector with dimensions, %s>" % (k, str(v.shape)))
      packed = {k: v.tobytes() for k, v in result.items()}
      
      # Index Key : If/When to update job ID management for downstream data
      index_key = wrapKey('idx', jobnum)
      self.catalog.save({index_key: packed})

      return [jobnum]



    def addArgs(self):
      parser = macrothread.addArgs(self)
      parser.add_argument('-b', '--build')
      parser.add_argument('--winsize', type=int)
      parser.add_argument('--slide', type=int)
      return parser




if __name__ == '__main__':
  mt = analysisJob(schema, __file__)
  #  For archiving
  args = mt.addArgs().parse_args()

  if args.winsize:
    mt.winsize = args.winsize

  if args.slide:
    mt.slide   = args.slide

  if args.debug:
    mt.manual   = True

  if args.build:
    logging.info("Running Single Execution to Build Archive on %s", args.build)
    mt.setBuild()
    mt.execute(args.build)
    sys.exit(0)

  mt.run()

