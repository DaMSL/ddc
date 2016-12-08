#!/usr/bin/env python
import argparse
import sys
import os
import fcntl
import logging
from collections import namedtuple, deque

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *
from core.slurm import slurm
from core.kvadt import kv2DArray
from core.kdtree import KDTree
from mdsim.deshaw import deshawReference
import mdsim.datareduce as datareduce
from macro.macrothread import macrothread
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)

transObservation = namedtuple('transitionObservation', 'frame A B dwell')

def reservoirSampling(dataStore, hiDimData, subspaceIndex, subspaceHash, resizeFunc, label, labelNameFunc):

  # hiDimData --> Delta X = indexed from 0...N (actual points in hi-dim space)
  # subspaceIndex -> Global index for Delta S_m  (index for the projected points from Delta X)
  # subspaceHashDelta --> Hash table for variables discovered (or tiled, labeled, etc..)
  #      label/bin/hcube --> list of indecies into DeltaX / DeltaS (0...N)

  for key in subspaceHash.keys():
    storeKey  = 'rsamp:%s:%s' % (label, labelNameFunc(key))
    rsampfile = os.path.join(DEFAULT.DATADIR, 'rSamp_%s' % (labelNameFunc(key)))
  
    while True:
      rsize     = dataStore.llen(storeKey)
          
      # Newly discovered Label
      if rsize == 0:
        logging.debug('New Data Label --> new reservoir Sample (acquiring lock...)')
        try:
          # Check to ensure lock is not already acquired
          lock = os.open(rsampfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as ex:
          logging.debug("Sample File exists (someone else has acquired it). Backing off rand 1..5 seconds and re-checking")
          time.sleep(np.random.randint(4)+1)
          continue

        reservoirSamp = np.zeros(shape=resizeFunc(len(subspaceHash[key])))

        # Assume subspaceHash[l] < MAX_SIZE (otherwise, use random.choice to select MAX_SIZE)
        #  TODO: Pipeline & optimize (here and below)
        for i, si in enumerate(subspaceHash[key]):
          np.copyto(reservoirSamp[i], hiDimData[si])
          dataStore.rpush(storeKey, subspaceIndex[si])

      # Reservoir Sample already exists
      else:
        logging.debug('Old Data Label. Retreiving sample from : %s', rsampfile + '.npy')

        try:
          # Check to ensure lock is not already acquired
          lock = os.open(rsampfile, os.O_RDWR)
          fcntl.lockf(lock, fcntl.LOCK_EX)

        except FileNotFoundError as ex:
          logging.error("Reservoir Sample File not found for `%s`: %s" % (label, labelNameFunc(key)))

        reservoirSamp = np.load(rsampfile + '.npy')
        
        # New points can fit inside reservoir
        if rsize + len(subspaceHash[key]) < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Undersized Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(rsize + len(subspaceHash[key])), refcheck=False)
          for i, si in enumerate(subspaceHash[key]):
            np.copyto(reservoirSamp[rsize+i], hiDimData[si])
            dataStore.rpush(storeKey, subspaceIndex[si])

        # Some new points can fit inside reservoir (sample what can fit)
        elif rsize < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Nearly Full Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(DEFAULT.MAX_RESERVOIR_SIZE), refcheck=False)
          sample = np.random.choice(subspaceHash[key], DEFAULT.MAX_RESERVOIR_SIZE - rsize)
          for i, si in enumerate(sample):
            np.copyto(reservoirSamp[key][rsize+i], hiDimData[sample])
            dataStore.rpush(storeKey, subspaceIndex[sample])

        # Implement Eviction policy & replace with new points
        else:
          logging.debug('Full Reservoir: %d', rsize)
          evictNum = min(len(subspaceHash[l]), DEFAULT.MAX_RESERVOIR_SIZE // 20)         #  5% of reservoir -- for now
          evict = np.random.choice(DEFAULT.MAX_RESERVOIR_SIZE, evictNum)
          sample = np.random.choice(subspaceHash[key], evictNum)
          for i in range(evictNum):
            np.copyto(reservoirSamp[key][evict[i]], hiDimData[sample])
            dataStore.rset(storeKey, evict[i], subspaceIndex[sample])

      logging.debug("Saving Reservoir Sample File: %s", os.path.basename(rsampfile))
      np.save(rsampfile, reservoirSamp)
      os.close(lock)
      break



class analysisJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname,  'anl')

      #  Analysis Thread State Data
      self.setStream('rawoutput', 'completesim')

      self.addImmut('anlSplitParam')
      self.addImmut('anlDelay')
      self.addImmut('centroid')
      self.addImmut('numLabels')
      self.addImmut('terminate')

      self.addImmut('pcaVectors')

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
      bench = microbench()
      config = self.data[jobkey]
      jobname = unwrapKey(jobkey)

      # RETRIEVE from Cache   --> Local Mem:
      allux = AlluxioClient()

      trajfile = config['name'] + '.dcd'
      # >>>>Storing DCD into shared memory on this node
      ramdisk = '/dev/shm/out/'
      if not os.path.exists(ramdisk):
        os.mkdir(ramdisk)
      logging.info("Created ramdisk at: %s", ramdisk)
      bench.mark('mk_ramdisk')
      allux.cp(trajfile, ramdisk)
      bench.mark('cpFromAllux')
      logging.info("Copied to shared mem: %s", os.listdir(ramdisk))
      dcd_ramfile = os.path.join(ramdisk, trajfile)


    # 1. Check if source files exist
      # logging.debug("1. Check for file")
      # if not (os.path.exists(config['dcd']) and os.path.exists(config['pdb'])):
      #   logging.error('Source Files not found: %s, %s', config['dcd'], config['pdb'])
      #   return []

    # 2. Load raw data from trajectory file
      logging.debug("2. Load DCD")
      # bench.start()
      # traj = datareduce.filter_heavy(config['dcd'], config['pdb'])
      # load from Local Mem Now
      traj = datareduce.filter_heavy(dcd_ramfile, config['pdb'])
      bench.mark('File_Load')
      logging.debug('Trajectory Loaded: %s (%s)', config['name'], str(traj))

    # 3. Update Catalog with HD points (TODO: cache this)
      #  TODO: Pipeline all
      # off-by-1: append list returns size (not inserted index)
      #  ADD index to catalog
      # Off by 1 error for index values
      file_idx = self.catalog.append({'xid:filelist': [config['dcd']]})[0]
      delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
      global_idx = self.catalog.append({'xid:reference': delta_xid_index})
      global_xid_index_slice = [x-1 for x in global_idx]
      bench.mark('Indx_Update')


    # 4. Update higher dimensional index
      # Logical Sequence # should be unique seq # derived from manager (provides this
      #  worker's instantiation with a unique ID for indexing)
      mylogical_seqnum = str(self.seqNumFromID())
      self.catalog.hset('anl_sequence', config['name'], mylogical_seqnum)

      # INSERT NEW points here into cache/archive
      logging.debug(" Loading new conformations into cache....TODO: NEED CACHE LOC")
      # for i in range(traj.n_frames):
      #   cache.insert(global_xid_index_slice[i], traj.xyz[i])

    # 5a. Subspace Calcuation: RMS
    #------ A:  RMSD  ------------------
      #     S_A = rmslist

      # 1. Calc RMS for each conform to all centroids
      logging.debug("3. RMS Calculation")
      numLabels = len(self.data['centroid'])
      numConf = len(traj.xyz)
      rmsraw = np.zeros(shape=(numConf, numLabels))
      for i, conform in enumerate(traj.xyz):
        np.copyto(rmsraw[i], np.array([LA.norm(conform-cent) for cent in self.data['centroid']]))
      logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

      # 2. Account for noise
      #    For now: noise is user-configured; TODO: Factor in to Kalman Filter
      noise = DEFAULT.OBS_NOISE
      stepsize = 500 if 'interval' not in config else int(config['interval'])
      nwidth = noise//(2*stepsize)
      noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)

      # Notes: Delta_S == rmslist
      rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

      # 3. Append new points into the data store. 
      #    r_idx is the returned list of indices for each new RMS point
      #  TODO: DECIDE on retaining a Reservoir Sample
      #    for each bin OR to just cache all points (per strata)
      #  Reservoir Sampliing is Stratified by subspaceHash
      # logging.debug('Creating reservoir Sample')
      # reservoirSampling(self.catalog, traj.xyz, rIdx, subspaceHash, 
      #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
      #     'rms', lambda key: '%d_%d' % key)
      r_idx = []
      pipe = self.catalog.pipeline()
      for si in rmslist:
        pipe.rpush('subspace:rms', bytes(si))
      idxlist = pipe.execute()
      for i in idxlist:
        r_idx.append(int(i) - 1)

      logging.debug("R_Index Created (rms).")

      # 4. Apply Heuristics Labeling
      logging.debug('Applying Labeling Heuristic')
      rmslabel = []
      subspaceHash = {}
      for i, rms in enumerate(rmslist):
        #  Sort RMSD by proximity & set state A as nearest state's centroid
        prox = np.argsort(rms)
        A = prox[0]

        #  Calc Absolute proximity between nearest 2 states' centroids
        # THETA Calc derived from static run. it is based from the average std dev of all rms's from a static run
        #   of BPTI without solvent. It could be dynamically calculated, but is hard coded here
        #  The theta is divided by four based on the analysis of DEShaw:
        #   est based on ~3% of DEShaw data in transition (hence )
        avg_stddev = 0.34119404492089034
        theta = avg_stddev / 4.

        # NOTE: Original formulate was relative. Retained here for reference:  
        # Rel vs Abs: Calc relative proximity for top 2 nearest centroids   
        # relproximity = rms[A] / (rms[A] + rms[rs[1]])
        # B = rs[1] if relproximity > (.5 - theta) else A
        # proximity = abs(rms[prox[1]] - rms[A]) / (rms[prox[1]] + rms[A])  #relative
        proximity = abs(rms[prox[1]] - rms[A])    #abs

        #  (TODO:  Factor in more than top 2, better noise)
        #  Label secondary sub-state
        B = prox[1] if proximity < theta else A
        rmslabel.append((A, B))
        if (A, B) not in subspaceHash:
          subspaceHash[(A, B)] = []
          logging.debug("Found Label: %s", str((A, B)))
        subspaceHash[(A, B)].append(i)

      # Update Catalog
      idxcheck = self.catalog.append({'label:rms': rmslabel})
      bench.mark('RMS')


    # 5b. Subspace Calcuation: PCA
    #------ B:  PCA  -----------------
      # 1. Project Pt to PC's for each conform (top 3 PC's)
      logging.debug("Using following PCA Vectors: %s", str(self.data['pcaVectors'].shape))
      pcalist = datareduce.PCA(traj.xyz, self.data['pcaVectors'], numpc=3)

      # 2. Apend subspace in catalog
      p_idx = []
      pipe = self.catalog.pipeline()
      for si in pcalist:
        pipe.rpush('subspace:pca', bytes(si))
      idxlist = pipe.execute()
      for i in idxlist:
        p_idx.append(int(i) - 1)
      logging.debug("P_Index Created (pca) for delta_S_pca")

      # 3. Performing tiling over subspace
      #   For Now: Load entire tree into local memory
      hcube_mapping = json.loads(self.catalog.get('hcube:pca'))
      logging.debug('# Loaded keys = %d', len(hcube_mapping.keys()))

      # 4. Pull entire Subspace (for now)  
      #   Note: this is more efficient than inserting new points
      #   due to underlying Redis Insertions / Index look up
      #   If this become a bottleneck, may need to write custom redis client
      #   connection to persist connection and keep socket open (on both ends)
      #   Or handle some of this I/O server-side via Lua scipt
      packed_subspace = self.catalog.lrange('subspace:pca', 0, -1)
      subspace_pca = np.array([np.fromstring(x) for x in packed_subspace])

      # TODO: accessor function is for 1 point (i) and 1 axis (j). 
      #  Optimize by changing to pipeline  retrieval for all points given 
      #  a list of indices with an axis (if nec'y)
      logging.debug("Reconstructing the tree...")
      hcube_tree = KDTree.reconstruct(hcube_mapping, subspace_pca)

      # logging.debug("Inserting Delta_S_pca into KDtree (hcubes)")
      # for i in range(len(pcalist)):
      #   hcube_tree.insert(pcalist[i], p_idx[i])

      # TODO: Ensure hcube_tree is written to catalog
      # TODO: DECIDE on retaining a Reservoir Sample
      # reservoirSampling(self.catalog, traj.xyz, r_idx, subspaceHash, 
      #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
      #     'pca', 
      #     lambda key: '%d_%d' % key)
      bench.mark('PCA')

      bench.show()
      return [config['name']]


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

