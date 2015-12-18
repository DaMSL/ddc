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
from kvadt import kv2DArray

from collections import deque
# import logging
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class analysisJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname,  'anl')
      # State Data for Simulation MacroThread -- organized by state

      self.setStream('rawoutput', 'completesim')

      self.addImmut('anlSplitParam')
      self.addImmut('anlDelay')
      self.addImmut('centroid')
      self.addImmut('numLabels')
      self.addImmut('terminate')

      self.modules.add('redis')

      self.buildArchive = False
      self.manual = False

      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = 6

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

      key = wrapKey('jc', i)
      params = self.catalog.hgetall(key)
      for k, v in params.items():
        logging.debug('  %s :  %s', k, str(v))
      self.addMut(key, value=params)
      return key


    def configElasPolicy(self):
      self.delay = self.data['anlDelay']


    def execute(self, jobkey):

      config = self.data[jobkey]
      jobname = unwrapKey(jobkey)

      # 1. Check if source files exist
      if not (os.path.exists(config['dcd']) and os.path.exists(config['pdb'])):
        logging.error('Source Files not found: %s, %s', config['dcd'], config['pdb'])
        return []


      # 2. Load raw data from trajectory file
      traj = md.load(config['dcd'], top=config['pdb'])
      traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)
      
      # Set DEShaw Reference point
      ref = deshawReference()
      traj.superpose(ref, frame=0)

      logging.debug('Trajectory Loaded: %s - %s', config['name'], str(traj))

      # 3. Calc RMS for each conform to all centroids
      numLabels = self.data['numLabels']
      rmslist = np.zeros(shape=(len(traj.xyz), numLabels))
      for i, conform in enumerate(traj.xyz):
        np.copyto(rmslist[i], np.array([LA.norm(conform-cent) for cent in self.data['centroid']]))

      # 4. Initialize conformation analysis data
      numtransitions = 0
      dwell = 0
      dwelltime = []
      stepsize = 500 if 'interval' not in config else int(config['interval'])
      conformlist = []
      uniquebins = set()
      delta_tmat = np.zeros(shape=(numLabels, numLabels))
      # logging.debug("  THETA  = %0.3f   (static for now)", theta)

      # 5. Account for noise -- for now use avg over 4 ps 
      noise = DEFAULT.OBS_NOISE
      nwidth = noise//(2*stepsize)
      noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
      filtrms = np.array([noisefilt(rmslist, i) for i in range(len(rmslist))])

      Ap = Bp = None

      # 6.  Iterate over filtered RMS's and mark observations
      translist = []
      sample = {}
      lastbin = None
      for num, rms in enumerate(filtrms):
        dwell += int(stepsize)

        #  Sort RMSD by proximity
        prox = np.argsort(rms)
        A = prox[0]

        #  Calc relative proximity for top 2 nearest centroids   (TODO:  Factor in more than top 2)
        # relproximity = rms[A] / (rms[A] + rms[rs[1]])
        # B = rs[1] if relproximity > (.5 - theta) else A
        # Use of Absolute Calculation


        # THETA Calc derived from static run. it is based from the average std dev of all rms's from a static run
        #   of BPTI without solvent. It could be dynamically calculated, but is hard coded here
        #  The theta is divided by four based on the analysis of DEShaw:
        #   est based on ~3% of DEShaw data in transition (hence )
        avg_stddev = 0.34119404492089034
        theta = avg_stddev / 4.

        # Rel vs Abs:
        # proximity = abs(rms[prox[1]] - rms[A]) / (rms[prox[1]] + rms[A])  #relative
        proximity = abs(rms[prox[1]] - rms[A])    #abs

        B = prox[1] if proximity < theta else A

        status = 'STABLE'
        start = 0
        #  1st observation
        if num == 0:
          curbin = (A, B)

        #  Same bin observed as previous
        elif num < len(filtrms)-1 and (A, B) == (Ap, Bp):
          pass

        #  Noise
        elif dwell <= noise:
          logging.info("        (filtering noise)  ")
          status = 'NOISE'
          start = num

        # Log last observed transition
        elif num == len(filtrms):
          logging.debug("last one")
          status = 'TRANS'
          translist.append((Ap,Bp,dwell))
          curbin = (A, B)

          # Grab a sample for the candidate pool  (Can cache here)
          samp = start + ((num - start) // 2)
          logging.debug("Grabbing sample frame: %d  %s", samp, str((Ap, Bp)))
          sample[(Ap, Bp)] = samp
          start = num

        #  Transition Identified
        else:
          logging.debug("new bin")
          status = 'TRANS'
          translist.append((Ap,Bp,dwell))
          curbin = (A, B)

          # Grab a sample for the candidate pool  (Can cache here)
          samp = start + ((num - start) // 2)
          logging.debug("Grabbing sample frame: %d  %s", samp, str((Ap, Bp)))
          sample[(Ap, Bp)] = samp

          if Ap == Bp:
            logging.info("  MD was stable in state %d for %d fs", Ap, dwell)
          else:
            logging.info("  MD was in transition (%d, %d) for %d fs", Ap, Bp, dwell)


          dwell = 0



        logging.debug("     Obs:  %d, %d", A, B)
        delta_tmat[A][B] += 1

        # Save observations for this conformation
        conformlist.append((num, A, B, proximity, dwell, str(rms.tolist()), status))

        Ap = A
        Bp = B


      tlist = deque()

      for t in translist:
        if len(tlist) == 0:
          tlist.append(t)
        elif (t[0], t[1]) == (tlist[-1][0], tlist[-1][1]):
          last = tlist.pop()
          tlist.append((t[0], t[1], t[2] + last[2]))
        else:
          tlist.append(t)

      logging.debug("TRANSITION LIST:")
      for t in tlist:
        logging.debug('%s', str(t))
      # logging.debug("******  DEBUG  *******  TERMINATING  ******")
      # sys.exit(0)


      # 6. Gather stats for decision history & downstream processing
      self.data[jobkey]['numtransitions'] = numtransitions
      logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point for classification)", config['name'])
      # TODO:  Feed all conforms into clustering algorithm & update centroid

      logging.debug("  # Observations:      %d", len(conformlist))
      logging.debug("  # Transisions :      %d", numtransitions)
      logging.debug("  Bins Observed :      %s", str(uniquebins))
      logging.debug("  This Delta:\n%s", str(delta_tmat))

      # TODO: Move to abstract data ??????
      seqnum = str(self.seqNumFromID())
      self.catalog.set(wrapKey('conform', seqnum), pickle.dumps(conformlist))
      self.catalog.set(wrapKey('translist', config['name']), pickle.dumps(tlist))
      self.catalog.hset('anl_sequence', config['name'], seqnum)
      self.catalog.storeNPArray(rmslist, wrapKey('rmslist', str(self.seqNumFromID())))
      self.catalog.storeNPArray(delta_tmat, wrapKey('delta', config['name']))

      # Add Sample Frames to the Candidate Pool
      for tbin, frame in sample.items():
        A, B = tbin
        key = kv2DArray.key('candidatePool', A, B)
        length = self.catalog.llen(key)
        if length >= DEFAULT.CANDIDATE_POOL_SIZE:
          self.catalog.lpop(key)
        candidate = '%s:%03d' % (jobname,frame)

        logging.info("New Job Candidate for (%d, %d): %s   poolsize=%d",
          A, B, candidate, length)
        self.catalog.rpush(key, candidate)

      sample[(A, B)]

      return [config['name']]





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

