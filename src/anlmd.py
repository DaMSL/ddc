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
from indexing import *

# import logging
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class analysisJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname,  'anl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('dcdFileList', 'LDIndexList')
      self.setState('JCComplete', 'processed', 'anlSplitParam', 'anlDelay', 'indexSize')

      self.modules.add('redis')

      self.buildArchive = False
      self.manual = False

      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = 24

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

      # TODO: Better Job ID Mgmt, for now hack the filename (for analysis)
      i.replace(':', '_').replace('-', '_')
      jobnum = os.path.basename(i).split('.')[0].split('_')[-1]
      # logging.debug("jobnum = " + jobnum)

      if self.manual:
        dcd, pdb = tuple(map(lambda x: os.path.join(os.path.dirname(i), "%s.%s" % (jobnum, x)), ['dcd', 'pdb']))
      else:
        dcd, pdb = tuple(map(lambda x: os.path.join(DEFAULT.JOBDIR, jobnum, "%s.%s" % (jobnum, x)), ['dcd', 'pdb']))

      # 1. Check if source files exist
      if not (os.path.exists(dcd) and os.path.exists(pdb)):
        logging.error('Source Files not found: %s, %s', dcd, pdb)
        return []


      # 2. Load raw data from trajectory file
      traj = md.load(dcd, top=pdb)
      traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)

      # 3. Set DEShaw Reference point
      traj.superpose(deshawReference())

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
        archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
        engine = getNearpyEngine(archive, indexSize)

        #  TODO: Better label
        label = key
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



if __name__ == '__main__':
  mt = analysisJob(__file__)

  mt.parser.add_argument('-b', '--build')
  mt.parser.add_argument('--winsize', type=int)
  mt.parser.add_argument('--slide', type=int)
  args = mt.parser.parse_args()
  #  For archiving
  # args = mt.addArgs().parse_args()

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

