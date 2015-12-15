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
from deshaw import deshawReference

# import logging
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class analysisJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname,  'anl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('rawoutput', 'completesim')
      self.setState('anlSplitParam', 'anlDelay')

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
      logging.debug("Retrieving RAWOUTPUT %s", self.data['rawoutput'])
      split = int(self.data['anlSplitParam'])
      immed = self.data['rawoutput'][:split]
      return immed, split


    def fetch(self, i):

      self.data['centroid'] = self.catalog.loadNPArray('centroid')

      params = {k.decode(): v.decode() for k, v in self.catalog.hgetall(wrapKey('jc', i)).items()}
      for k, v in params.items():
        logging.debug('  %s :  %s', k, str(v))
      return params


    def configElasPolicy(self):
      self.delay = self.data['anlDelay']


    def execute(self, config):

      # 1. Check if source files exist
      if not (os.path.exists(config['dcd']) and os.path.exists(config['pdb'])):
        logging.error('Source Files not found: %s, %s', config['dcd'], config['pdb'])
        return []


      # 2. Load raw data from trajectory file
      traj = md.load(config['dcd'], top=config['pdb'])
      traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)
      
      # Set DEShaw Reference point
      traj.superpose(deshawReference())

      logging.debug('Trajectory Loaded: %s - %s', config['name'], str(traj))

      # 3. Initialize
      numLabels = len(self.data['centroid'])
      statdata = {}
      intransition = False
      numtransitions = 0
      dwell = 0
      stepsize = 500 if 'interval' not in config else int(config['interval'])
      conformlist = []
      uniquebins = set()
      delta_tmat = np.zeros(shape=(numLabels, numLabels))
      theta = .01  
      logging.debug("  THETA  = %0.3f   (static for now)", theta)

      # 4. Calc RMS for each conform to all centroids
      for num, conform in enumerate(traj.xyz):
        dwell += int(stepsize)

        #  Calc RMSD to each centroid
        rms = np.array([LA.norm(conform-cent) for cent in self.data['centroid']])

        #  Sort RMSD by proximity
        rs = np.argsort(rms)
        A = rs[0]

        #  Calc relative proximity for top 2 nearest centroids   (TODO:  Factor in more than top 2)
        relproximity = rms[A] / (rms[A] + rms[rs[1]])
        if relproximity > (.5 - theta):
          B = rs[1] 
          if not intransition:
            logging.info("  Transition:  %d  ->  %d   after,  %d fs  of dwell", A, B, dwell)
            numtransitions += 1
            intransition = True
            dwell = 0
        else:
          B = A
          if intransition:
            intransition = False
            logging.info("  Stable State:  %d    after,  %d fs  of transition", A, dwell)
            dwell = 0

        logging.debug("     Obs:  %d, %d", A, B)
        delta_tmat[A][B] += 1
        uniquebins.add((A,B))
        conformlist.append((num, A, B, relproximity, intransition, dwell))

      # 5. Gather stats for decision history
      statdata['numtransitions'] = numtransitions
      avgdwelltime = {}
      for ub in uniquebins:
        A,B = ub
        avgdwelltime[ub] = np.mean([c[5] for c in conformlist if ub == (c[1],c[2])])
        statdata[ub] = numtransitions



      logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point for classification)", config['name'])
      # TODO:  Feed all conforms into clustering algorithm & update centroid

      logging.debug("  # Observations:      %d", len(conform))
      logging.debug("  # Transisions :      %d", numtransitions)
      logging.debug("  Bins Observed :      %s", str(uniquebins))
      logging.debug("  This Delta:\n%s", str(delta_tmat))

      # TODO: Update/Save Job Candidate History

      delta_key = wrapKey('delta', config['name'])
      self.catalog.storeNPArray(delta_tmat, delta_key)
      return [config['name']  ]





      # result = {}
      # indexSize = self.data['indexSize']
      # # 2. Split raw data in WINSIZE chunks and calc eigen vectors
      # #   TODO: Retain provenance
      # for win in range(0, len(traj.xyz), self.slide):
      #   if win + self.winsize > len(traj.xyz):
      #     break
      #   eg, ev = LA.eigh(distmatrix(traj.xyz[win:win+self.winsize]))
      #   key = jobnum + ':' + '%04d' % win
      #   result[key] = makeIndex(eg, ev)

      # # To Build Archive online
      # if self.buildArchive:
      #   archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
      #   engine = getNearpyEngine(archive, indexSize)

      #   #  TODO: Better label
      #   label = key
      #   for key, idx in result.items():
      #     engine.store_vector(idx, key)

      # # Index for downstream retrieval
      # logging.debug('Saving Index in catalog')

      # # Pack & store data
      # for k, v in result.items():
      #   logging.debug(" `%s`:  <vector with dimensions, %s>" % (k, str(v.shape)))
      # packed = {k: v.tobytes() for k, v in result.items()}


      
      # Index Key : If/When to update job ID management for downstream data
      # index_key = wrapKey('idx', jobnum)
      # self.catalog.save({index_key: packed})



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

