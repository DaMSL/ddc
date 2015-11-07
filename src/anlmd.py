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


WINSIZE = 100

HASH_NAME = 'lshash'

def eigenDecomA(traj):
  '''
  Method A : Pairwise correlation based on each pair of atoms' 
  distance to respective mean
  Modified to only correlate along likewise cartesian axes
  '''
  n_frames = traj.shape[0]
  n_atoms = traj.shape[1]
  # t1 = traj.reshape(n_frames, n_atoms_pos)
  mean = np.mean(traj, axis=0)
  cov = np.zeros(shape = (n_atoms*3, n_atoms*3))
  for A in range(n_atoms):
    # print("Atom # %d" % A, '     ', str(dt.datetime.now()))
    if A % 10 == 0:
      logging.info("Atom # %d" % A)
    for B in range(A, n_atoms):
      s = np.zeros(shape=(3))
      for i in range(n_frames):
        s += (traj[i][A] - mean[A]) * (traj[i][B] - mean[B])
      cov[A][B] = s[0] / n_frames
      cov[B][A] = s[0] / n_frames
      cov[A+n_atoms][B+n_atoms] = s[1] / n_frames
      cov[B+n_atoms][A+n_atoms] = s[1] / n_frames
      cov[A+2*n_atoms][B+2*n_atoms] = s[1] / n_frames
      cov[B+2*n_atoms][A+2*n_atoms] = s[1] / n_frames
  # print ('\n', str(dt.datetime.now()), "  Doing eigenDecomp")
  logging.info("Calculating Eigen")
  # print(str(dt.datetime.now()), "  Calculating Eigen")
  return LA.eig(cov)


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
  dist = np.zeros(shape = (n_atoms, n_atoms))
  for A in range(n_atoms):
    # print("Atom # %d" % A, '     ', str(dt.datetime.now()))
    if A % 10 == 0:
      logging.info("Atom # %d" % A)
    for B in range(A, n_atoms):
      delta = LA.norm(mean[A] - mean[B])
      dist[A][B] = delta
      dist[B][A] = -delta
  # print ('\n', str(dt.datetime.now()), "  Doing eigenDecomp")
  logging.info("Calculating Eigen")
  # print(str(dt.datetime.now()), "  Calculating Eigen")
  return LA.eig(dist)



def initialize(archive):

  if os.path.exists('catalog.lock'):
    os.remove('catalog.lock')

  # ref: https://github.com/pixelogik/NearPy
  # Create redis storage adapter
  redis_storage = RedisStorage(archive)

  # Create Hash
  lshash = RandomBinaryProjections(HASH_NAME, 10)

  # Store hash configuration in redis for later use
  logging.debug('Storing Hash in Archive')
  redis_storage.store_hash_configuration(lshash)

  
  indexSize = 284   
  engine = nearpy.Engine(indexSize, lshashes=[lshash])
  numLoadedFile = 0
  numIndices = 0
  indexdir = os.path.join(os.getenv('HOME'), 'scratch')
  for idxnum in range(2000):
    srcFile = os.path.join(indexdir, 'index_%04d.npy' % idxnum)
    if os.path.exists(srcFile):
      numLoadedFile += 1
      source = np.load(srcFile)
      for seqnum, window in enumerate(source):
        eigen = window.reshape(5, 3, 284)
        idx = eigen[0][0]
        engine.store_vector(idx, encodeLabel(idxnum, seqnum))
        numIndices += 1

  logging.debug("Loaded Deshaw data: %d files loaded, total of %d indices", numLoadedFile, numIndices)

  archive.stop()

  # LOAD DEShaw data from saved index files




class analysisJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'simmd')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('dcdFileList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('LDIndexList')
      self.setSplit('anlSplitParam')
      self.buildArchive = False

      self.modules.extend(['redis'])

      self.indexSize = 284


    def term(self):
      # For now
      return False

    def split(self):
      split = int(self.data['anlSplitParam'])
      catalog = self.getCatalog()
      immed = catalog.slice('dcdFileList', split)
      return immed

    def execute(self, i):

      logging.debug('ANL MT. Input = ' + i)

      # TODO: Better Job ID Mgmt, for now hack the filename
      i.replace(':', '_').replace('-', '_')
      jobnum = os.path.basename(i).split('.')[0].split('_')[-1]
      logging.debug("jobnum = " + jobnum)
      

      # 1. Load raw data from trajectory file
      traj = md.load(i, top=DEFAULT.PDB_FILE)
      filterMin  = traj.top.select_atom_indices(selection='minimal')
      traj.atom_slice(filterMin, inplace=True)

      logging.debug('Trajectory Loaded')
      result = {}
      indexSize = 0
      # 2. Split raw data in WINSIZE chunks and calc eigen vectors
      #   TODO: Retain provenance
      for win in range(0, len(traj.xyz) - WINSIZE+1, WINSIZE):
        logging.debug("Running on window # " + str(win))
        eg, ev = eigenDecomB(traj.xyz[win:win+WINSIZE])
        eg /= LA.norm(eg)
        ev = np.transpose(ev)   # Transpose eigen vectors
        index = np.zeros(shape=(DEFAULT.NUM_PCOMP, len(ev[0])), dtype=ev.dtype)
        for pc in range(DEFAULT.NUM_PCOMP):
          np.copyto(index[pc], ev[pc] * eg[pc])
        # 3. store index
        key = jobnum + ':' + '%03d' % win
        logging.debug('Saving Index: %s', key)
        result[key] = index.flatten()
        if not indexSize:
          indexSize = len(result[key])
          logging.debug('Index Size = %d' % indexSize)

      logging.debug("All Indices calculated. Beginning Probing")

      # import redis
      # archive = redis.StrictRedis(port=6380)
      archive = redisCatalog.dataStore(**archiveConfig)

      logging.debug('Archive Client Created, arch class= %s', str(archive.__class__))
      archive.conn()
      keys = archive.keys()
      logging.debug('keys loaded')
      for k in keys:
        logging.debug("  key: %s", k)
      redis_storage = RedisStorage(archive)

      config = redis_storage.load_hash_configuration(HASH_NAME)
      if not config:
        logging.error("LSHash not configured")
        #TODO: Gracefully exit
  
      # Create empty lshash and load stored hash
      lshash = RandomBinaryProjections(None, None)
      lshash.apply_config(config)

      engine = nearpy.Engine(indexSize, 
            lshashes=[lshash], 
            storage=redis_storage)

      # OPTION A:  Build Archive online
      if self.buildArchive:
        logging.debug('Build Archive!  Index stored directly')
        for key, index in result.items():
          engine.store_vector(index, key)

      # OPTION B:  Index for downstream retrieval
      else:
        logging.debug('Saving Index in catalog')

        # ********     PACK / UNPACK

        # TODO:  Pack result for storage!!!!
        self.catalog.save({jobnum: result})
        self.data['LDIndexList'].append(jobnum)








if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--manager')
  parser.add_argument('-w', '--workinput')
  parser.add_argument('-i', '--init', action='store_true')
  parser.add_argument('-d', '--debug')
  args = parser.parse_args()


  archive = redisCatalog.dataStore(**archiveConfig)

  if args.init:
    logging.debug("Initializing the archive.....")
    initialize(archive)
    sys.exit(0)

  registry = redisCatalog.dataStore('catalog')
  mt = analysisJob(schema, __file__)
  mt.setCatalog(registry)


  if args.debug:
    logging.info("Running Single Execution (debugging) Probe on %s", args.debug)
    mt.worker(args.debug)
    sys.exit(0)


  if args.manager:
    mt.manager(fork=False)
  else:
    mt.worker(args.workinput)
