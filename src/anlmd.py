import argparse
import sys
import os
import sys

import mdtraj as md
import numpy as np
from numpy import linalg as LA
import nearpy
from nearpy.storage.storage_redis import RedisStorage
from nearpy.hashes import RandomBinaryProjections

import redisCatalog
from common import *
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def eigenDecomA(traj):
  '''
  Method A : Pairwise correlation based on each pair of atoms' 
  distance to respective mean
  '''
  n_frames = traj.shape[0]
  N = traj.shape[1]*3
  T = traj.reshape(n_frames, N)
  # t1 = traj.reshape(n_frames, n_atoms_pos)
  mean = np.mean(T, axis=00)
  cov = np.zeros(shape = (N, N))
  for A in range(N):
    # print("Atom # %d" % A, '     ', str(dt.datetime.now()))
    if A % 100 == 0:
      logging.info("Atom # %d" % A)
    for B in range(A, N):
      S = 0
      for i in range(n_frames):
        S += (T[i][A] - mean[A]) * (T[i][B] - mean[B])
      cov[A][B] = cov[B][A] = S / N
  # print ('\n', str(dt.datetime.now()), "  Doing eigenDecomp")
  logging.info("Calculating Eigen")
  # print(str(dt.datetime.now()), "  Calculating Eigen")
  return LA.eigh(cov)


# def eigenDecomA_2(traj):
#   '''
#   Method A : Pairwise correlation based on each pair of atoms' 
#   distance to respective mean
#   '''
#   n_frames = traj.shape[0]
#   N = traj.shape[1]*3
#   T = traj.reshape(n_frames, N)
#   # t1 = traj.reshape(n_frames, n_atoms_pos)
#   mean = np.mean(N, axis=0)
#   cov = np.zeros(shape = (N, N))
#   for A in range(N):
#     # print("Atom # %d" % A, '     ', str(dt.datetime.now()))
#     if A % 10 == 0:
#       logging.info("Atom # %d" % A)
#     for B in range(A, N):
#       s = np.zeros(shape=(3))
#       for i in range(n_frames):
#         s += (traj[i][A] - mean[A]) * (traj[i][B] - mean[B])
#       cov[A][B] = s[0] / n_frames
#       cov[B][A] = s[0] / n_frames
#       cov[A+n_atoms][B+n_atoms] = s[1] / n_frames
#       cov[B+n_atoms][A+n_atoms] = s[1] / n_frames
#       cov[A+2*n_atoms][B+2*n_atoms] = s[2] / n_frames
#       cov[B+2*n_atoms][A+2*n_atoms] = s[2] / n_frames
#   # print ('\n', str(dt.datetime.now()), "  Doing eigenDecomp")
#   logging.info("Calculating Eigen")
#   # print(str(dt.datetime.now()), "  Calculating Eigen")
#   return LA.eigh(cov)


def eigenDecomB(traj):
  '''
  Method B : Feature-like identification based on relative mean distance
  over trajectory for each pair-wise set of atoms
  '''
  logging.debug("CALL: eigenDecomp  ")
  n_frames = traj.shape[0]
  n_atoms  = traj.shape[1]
  # n_atoms_pos = traj.shape[1] * 3
  # t1 = traj.reshape(n_frames, n_atoms_pos)
  mean = np.mean(traj, axis=0)
  dist = np.zeros(shape = (n_atoms, n_atoms), dtype=np.float64)
  for A in range(n_atoms):
    # print("Atom # %d" % A, '     ', str(dt.datetime.now()))
    if A % 100 == 0:
      logging.info("Atom # %d" % A)
    for B in range(A, n_atoms):
      delta = LA.norm(mean[A] - mean[B])
      dist[A][B] = delta
      dist[B][A] = -delta
  # print ('\n', str(dt.datetime.now()), "  Doing eigenDecomp")
  logging.info("Calculating Eigen")
  # print(str(dt.datetime.now()), "  Calculating Eigen")
  return LA.eigh(dist)



class analysisJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'anl')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('dcdFileList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('LDIndexList')
      self.setSplit('anlSplitParam')
      self.modules.add('redis')
      self.buildArchive = False
      self.manual = False

      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = DEFAULT.CPU_PER_NODE



      # TODO: Move to Catalog
      self.winsize = 100
      self.slide   =  50

    def setBuild(self, build=True):
      self.buildArchive = build

    def term(self):
      # For now
      return False

    def split(self):
      split = int(self.data['anlSplitParam'])
      catalog = self.getCatalog()
      immed = catalog.slice('dcdFileList', split)
      return immed



    def execute(self, i):

      # logging.debug('ANL MT. Input = ' + i)

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
      filterMin  = traj.top.select_atom_indices(selection='minimal')
      traj.atom_slice(filterMin, inplace=True)

      logging.debug('Trajectory Loaded: %s - %s', jobnum, str(traj))
      result = {}
      indexSize = 0
      # 2. Split raw data in WINSIZE chunks and calc eigen vectors
      #   TODO: Retain provenance
      for win in range(0, len(traj.xyz) - self.winsize+1, self.slide):
        logging.debug("Processing window %s - %s # " % (jobnum, str(win)))
        eg, ev = eigenDecomA(traj.xyz[win:win+self.winsize])
        eg /= LA.norm(eg)
        ev = np.transpose(ev)   # Transpose eigen vectors
        index = np.zeros(shape=(DEFAULT.NUM_PCOMP, len(ev[0])), dtype=ev.dtype)
        for pc in range(DEFAULT.NUM_PCOMP):
          np.copyto(index[pc], ev[-pc-1] * eg[-pc-1])
        # 3. store index
        if win < 1000:
          key = jobnum + ':' + '%03d' % win
        else:
          key = jobnum + ':' + '%04d' % win
        logging.debug('Cachine Index locally: %s', key)
        result[key] = index.flatten()
        if not indexSize:
          indexSize = len(result[key])
          # logging.debug('Index Size = %d' % indexSize)

      # Create empty lshash and load stored hash
      # OPTION A:  Build Archive online
      if self.buildArchive:
        # logging.debug('Build Archive!  Index stored directly')
        # import redis
        # archive = redis.StrictRedis(port=6380)
        archive = redisCatalog.dataStore(**archiveConfig)

        # logging.debug('Archive Client Created, arch class= %s', str(archive.__class__))
        archive.conn()
        keys = archive.keys()
        # logging.debug('keys loaded')
        # for k in keys:
          # logging.debug("  key: %s", k)
        redis_storage = RedisStorage(archive)
        config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
        if not config:
          logging.error("LSHash not configured")
        #TODO: Gracefully exit
        lshash = RandomBinaryProjections(None, None)
        lshash.apply_config(config)
        engine = nearpy.Engine(indexSize, 
            lshashes=[lshash], 
            storage=redis_storage)
        for key, index in result.items():
          engine.store_vector(index, key)

      # OPTION B:  Index for downstream retrieval
      else:
        logging.debug('Saving Index in catalog')

        # Pack & store data
        for k, v in result.items():
          logging.debug(" `%s`:  %s" % (k, str(v.shape)))
        packed = {k: v.tobytes() for k, v in result.items()}
        
        # Index Key : If/When to update job ID management for downstream data
        index_key = wrapKey('idx', jobnum)
        self.catalog.save({index_key: packed})
        self.data['LDIndexList'].append(index_key)
        logging.debug("DEBUG indexlist follows")
        for l in self.data['LDIndexList']:
          logging.debug(" INDEX: " + l)
        return index_key



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

