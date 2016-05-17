#!/usr/bin/env python
"""
MULTIVARIATE VERSION
"""
import argparse
import sys
import os
import sys
import math

import json
import datetime as dt
from random import choice, randint
from collections import namedtuple, deque, OrderedDict
from threading import Thread

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *
from macro.macrothread import macrothread
from core.kvadt import kv2DArray
from core.slurm import slurm
from core.kdtree import KDTree
import datatools.datareduce as datareduce
import datatools.datacalc as datacalc
import mdtools.deshaw as deshaw
import datatools.kmeans as KM
# from datatools.pca import calc_kpca, calc_pca, project_pca
from datatools.pca import PCAnalyzer, PCAKernel, PCAIncremental
from datatools.approx import ReservoirSample
from mdtools.simtool import generateNewJC
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient
from bench.timer import microbench
from bench.stats import StatCollector

import plot as G


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

np.set_printoptions(precision=5, suppress=True)

# SIMULATE_RATIO = 50



class controlJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('completesim', None)

      self.addMut('jcqueue')
      self.addMut('converge')
      self.addMut('ctlIndexHead')
      self.addMut('ctlCountHead')
      self.addImmut('ctlSplitParam')
      self.addImmut('ctlDelay')
      self.addImmut('numLabels')
      self.addImmut('terminate')
      self.addImmut('numresources')
      self.addImmut('ctlBatchSize_min')
      self.addImmut('ctlBatchSize_max')
      self.addImmut('dcdfreq')
      self.addImmut('runtime')
      self.addImmut('exploit_factor')
      self.addImmut('max_observations')
      self.addImmut('sim_step_size')
      self.addImmut('observe:count')
          
      self.addAppend('timestep')
      # self.addAppend('observe')
      self.addMut('runtime')


      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = 24

      self.modules.add('namd/2.10')

      self.trajlist_async = deque()

      self.cacheclient = None

      # For stat tracking
      self.cache_hit = 0
      self.cache_miss = 0

      # Optimization on global xid list
      self.xidreference = None




    def term(self):
      numobs = self.catalog.llen('label:rms')
      if numobs > self.data['max_observations']:
        logging.info('Terminating at %d observations', numobs)
        return True
      else:
        return False

    def split(self):

      settings = systemsettings()

      # Batch sizes should be measures in abosolute # of observations
      workloadList = deque(self.data['completesim'])
      minbatchSize = round(self.data['ctlBatchSize_min'])
      maxbatchSize = int(self.data['ctlBatchSize_max'])
      batchAmt = 0
      logging.debug('Controller will launch with batch size between: %d and %d observations', minbatchSize, maxbatchSize)
      num_simbatchs = 0
      compsimlen = self.catalog.llen('completesim')
      projected_numobs = self.data['ctlIndexHead']
      logging.debug("Current Workload List: %s.   Tracking %d items in completesim queue. \
        # Obs Processed thus far:  %d", str(workloadList), compsimlen, projected_numobs)
      while projected_numobs < self.data['max_observations'] and len(workloadList) > 0:
        if int(workloadList[0]) + batchAmt > maxbatchSize:
          if batchAmt == 0:
            logging.warning('CAUTION. May need to set the max batch size higher (cannot run controller)')
            batchAmt += int(workloadList.popleft())
          break
        next_batch = int(workloadList.popleft())
        batchAmt += next_batch
        projected_numobs += next_batch
        num_simbatchs += 1
        logging.debug('  Batch amount up to: %d', batchAmt)

      # Don't do anything if the batch size is less than the min (unless we're at the limit):
      if batchAmt < minbatchSize and projected_numobs < self.data['max_observations']:
        return [], None
      else:
        return [batchAmt], num_simbatchs

    # DEFAULT VAL for i for for debugging
    def fetch(self, batchSize):
      """Fetch determines the next thru index for this control loop
      Note that batchSize is measured in ps. Thru Index should return
      the next index to process
      """
      start_index = max(0, self.data['ctlIndexHead'])
      thru_index = min(start_index + int(batchSize), self.catalog.llen('subspace:feal')) - 1

      return thru_index

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']

    def backProjection(self, index_list):
      """Perform back projection function for a list of indices. Return a list 
      of high dimensional points (one per index). Check cache for each point and
      condolidate file I/O for all cache misses.
      """

      logging.debug('--------  BACK PROJECTION:  %d POINTS ---', len(index_list))
      bench = microbench('bkproj', self.seqNumFromID())

      # reverse_index = {index_list[i]: i for i in range(len(index_list))}

      source_points = []
      cache_miss = []

      self.trajlist_async = deque()
      
      # DEShaw topology is assumed here
      bench.start()

      # Derefernce indices to file, frame tuple:
      historical_framelist = []
      generated_framelist = []

      if self.xidreference is None:
        self.xidreference = self.catalog.lrange('xid:reference', 0, -1)

      # pipe = self.catalog.pipeline()
      logging.debug('Select Index List size = %d', len(index_list))
      for idx in index_list:
        # Negation indicates  historical index:
        index = int(idx)
        if index < 0:
          file_index, frame = deshaw.refFromIndex(-idx)
          historical_framelist.append((file_index, frame))
          # logging.debug('[BP] DEShaw:  file #%d,   frame#%d', file_index, frame)
        else:
          generated_framelist.append(self.xidreference[index])
          # pipe.lindex('xid:reference', index)

      # Load higher dim point indices from catalog
      # logging.debug('Exectuting...')  
      # start = dt.datetime.now()
      # generated_framelist = pipe.execute()
      # logging.debug('...Exectuted in %4.1f sec', ((dt.datetime.now()-start).total_seconds()))  

      # start = dt.datetime.now()
      # all_idx = self.catalog.lrange('xid:reference', 0, -1)
      # logging.debug('Got ALL pts in %4.1f sec', ((dt.datetime.now()-start).total_seconds()))  


      bench.mark('BP:LD:Redis:xidlist')


      ref = deshaw.topo_prot  # Hard coded for now

      # Group all Historical indidces by file number and add to frame Mask 
      logging.debug('Group By file idx (DEshaw)')
      historical_frameMask = {}
      for i, idx in enumerate(historical_framelist):
        file_index, frame = idx
        if file_index not in historical_frameMask:
          historical_frameMask[file_index] = []
        historical_frameMask[file_index].append(frame)

      for k, v in historical_frameMask.items():
        logging.debug('[BP] Deshaw lookups: %d, %s', k, str(v))


      # Group all Generated indidces by file index 
      logging.debug('Group By file idx (Gen data)')
      groupbyFileIdx = {}
      for i, idx in enumerate(generated_framelist):
        file_index, frame = eval(idx)
        if file_index not in groupbyFileIdx:
          groupbyFileIdx[file_index] = []
        groupbyFileIdx[file_index].append(frame)

      # Dereference File index to filenames
      logging.debug('Deref fileidx -> file names')
      generated_frameMask = {}
      generated_filemap = {}
      for file_index in groupbyFileIdx.keys():
        filename = self.catalog.lindex('xid:filelist', file_index)
        if filename is None:
          logging.error('Error file not found in catalog: %s', filename)
        else:
          key = os.path.splitext(os.path.basename(filename))[0]
          generated_frameMask[key] = groupbyFileIdx[file_index]
          generated_filemap[key] = filename
      bench.mark('BP:GroupBy:Files')

      #  Ensure the cache is alive an connected
      logging.debug('Check Cache client')
      self.cacheclient.connect()

      # Check cache for historical data points
      logging.debug('Checking cache for %d DEShaw points to back-project', len(historical_frameMask.keys()))
      for fileno, frames in historical_frameMask.items():
        # handle 1 frame case (to allow follow on multi-frame, mix cache hit/miss)
        if len(frames) == 1:
          datapt = self.cacheclient.get(fileno, frames[0], 'deshaw')
          dataptlist = [datapt] if datapt is not None else None
        else:
          dataptlist = self.cacheclient.get_many(fileno, frames, 'deshaw')
        if dataptlist is None:
          self.cache_miss += 1
          # logging.debug('[BP] Cache MISS on: %d', fileno)
          cache_miss.append(('deshaw', fileno, frames))
        else:
          self.cache_hit += 1
          # logging.debug('[BP] Cache HIT on: %d', fileno)
          source_points.extend(dataptlist)

      # Check cache for generated data points
      logging.debug('Checking cache for %d Generated points to back-project', len(generated_frameMask.keys()))
      for filename, frames in generated_frameMask.items():
        # handle 1 frame case (to allow follow on multi-frame, mix cache hit/miss)
        if len(frames) == 1:
          datapt = self.cacheclient.get(filename, frames[0], 'sim')
          dataptlist = [datapt] if datapt is not None else None
        else:
          dataptlist = self.cacheclient.get_many(filename, frames, 'sim')
        if dataptlist is None:
          self.cache_miss += 1
          # logging.debug('[BP] Cache MISS on: %s', filename)
          cache_miss.append(('sim', generated_filemap[filename], frames))
        else:
          self.cache_hit += 1
          # logging.debug('[BP] Cache HIT on: %s', filename)
          source_points.extend(dataptlist)


      # Package all cached points into one trajectory
      logging.debug('Cache hits: %d points.', len(source_points))
      if len(source_points) > 0:
        source_traj_cached = md.Trajectory(source_points, ref.top)
      else:
        source_traj_cached = None

      # All files were cached. Return back-projected points
      if len(cache_miss) == 0:
        return source_traj_cached
        
      # Add high-dim points to list of source points in a trajectory
      # Optimized for parallel file loading
      source_points_uncached = []
      logging.debug('Sequentially Loading all trajectories')
      for miss in cache_miss:
        ftype, fileno, framelist = miss
        if ftype == 'deshaw':
          pdb, dcd = deshaw.getHistoricalTrajectory_prot(fileno)
          traj = md.load(dcd, top=pdb)
        elif ftype == 'sim':
          traj = datareduce.load_trajectory(fileno)
        selected_frames = traj.slice(framelist)
        source_points_uncached.extend(selected_frames.xyz)
        bench.mark('BP:LD:File')

      logging.debug('All Uncached Data collected Total # points = %d', len(source_points_uncached))
      source_traj_uncached = md.Trajectory(np.array(source_points_uncached), ref.top)
      bench.mark('BP:Build:Traj')
      # bench.show()

      logging.info('--------  Back Projection Complete ---------------')
      if source_traj_cached is None:
        return source_traj_uncached
      else:
        return source_traj_cached.join(source_traj_uncached)

    def execute(self, thru_index):
      """Executing the Controler Algorithm. Load pre-analyzed lower dimensional
      subspaces, process user query and identify the sampling space with 
      corresponding distribution function for each user query. Calculate 
      convergence rates, run sampler, and then execute fairness policy to
      distribute resources among users' sampled values.
      """
      logging.debug('CTL MT')

    # PRE-PROCESSING ---------------------------------------------------------------------------------
      logging.debug("============================  <PRE-PROCESS>  =============================")

      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

      settings = systemsettings()
      bench = microbench('ctl_%s' % settings.name, self.seqNumFromID())
      stat = StatCollector('ctl_%s' % settings.name, self.seqNumFromID())

      # Connect to the cache
      self.cacheclient = CacheClient(settings.APPL_LABEL)

      # create the "binlist":
      numLabels = self.data['numLabels']
      numresources = self.data['numresources']


    # LOAD all new subspaces (?) and values

      ##### BARRIER
      self.wait_catalog()

      # Load new RMS Labels -- load all for now
      bench.start()
      logging.debug('Loading RMS Labels')
      start_index = max(0, self.data['ctlIndexHead'])

      # labeled_pts_rms = self.catalog.lrange('label:rms', self.data['ctlIndexHead'], thru_index)
      logging.debug(" Start_index=%d,  thru_index=%d,   ctlIndexHead=%d", start_index, thru_index, self.data['ctlIndexHead'])

      feallist = [np.fromstring(i) for i in r.lrange('subspace:feal', 0, -1)]
      num_pts = len(feallist)
      self.data['ctlIndexHead'] = thru_index
      thru_count = self.data['observe:count']

      logging.debug('##NUM_RMS_THIS_ROUND: %d', num_pts)
      stat.collect('numpts', len(feallist))


    # Calculate variable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERGENCE>  =========================")

      # Bootstrap current sample for RMS
      logging.info("Feature Landscapes for %d points loaded. Calculating PDF.....", len(feallist))

      #  Static for now
      blocksize = 5000
      mv_convergence = op.bootstrap_block(feallist, blocksize)
      global_landscape = np.mean(feallist, axis=0)
      stat.collect('convergence', mv_convergence)
      stat.collect('globalfeal', global_landscape)
      logging.info('MV Convergence values:\nCONV,%s', ','.join(['%5.3f'%i for i in mv_convergence]))
      logging.info('Global Feature Landscape:\nFEAL,%s', ','.join(['%5.3f'%i for i in global_landscape]))

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")

      ##### BARRIER
      self.wait_catalog()
      selected_index_list = []

      # QUERY PROCESSING & SAMPLING BELOW to select indices. 
      EXPERIMENT_NUMBER = self.experiment_number
      logging.info("RUNNING EXPER CONFIGURATION #%d", EXPERIMENT_NUMBER)


      ###### EXPERIMENT #5:  BIASED (Umbrella) SAMPLER
      if EXPERIMENT_NUMBER == 10:

        # Create the KD Tree from all feature landscapes (ignore first 5 features)
        kd = KDTree(100, 15, np.array(feallist), 'median')

        # Collect hypercubes
        hc = kd.getleaves()

        for key, hcube in hc.items():
          hc_feal = [feallist[i] for i in hc['elm']
          hc['feal'] = np.mean(hc_feal, axis=0)

        #  Det scale and/or sep scales for each feature set
        desired = 10 - global_landscape

        #  Calc euclidean dist to each mean HC's feal
        nn = {k: LA.norm(desired[5:] - v['feal'][5:]) for k,v in hc.items()}

        #  Grab top N Neighbors (10 for now)
        neighbors = sorted(nn.items(), key=lambda x: x[1])[:10]

        ## DATA SAMPLER
        nn_keys = [i for i,w in neighbors]
        nn_wgts = np.array([w for i,w in neighbors])
        nn_wgts /= np.sum(nn_wgts)  # normalize

        # TODO:   Load DEShaw labeled indices -- ASSUME NONE
        # if self.catalog.exists('label:deshaw'):
        #   logging.info("Loading DEShaw historical points.... From Catalog")
        #   rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
        # else:
        #   logging.info("Loading DEShaw historical points.... From File (and recalculating)")
        #   rmslabel = deshaw.labelDEShaw_rmsd()

        # deshaw_samples = {b:[] for b in binlist}
        # for i, b in enumerate(rmslabel):
        #   deshaw_samples[b].append(i)

        # coord_origin = []
        # conv_vals = np.array([v for k, v in sorted(convergence_rms.items())])
        # norm_pdf_conv = conv_vals / sum(conv_vals)
        # logging.info("Umbrella Samping PDF (Bootstrapping):")
        # sampled_distro_perbin = {b: 0 for b in binlist}

        while numresources > 0:
          # First sampling is BIASED
          selected_hc = np.random.choice(nn_keys, p=nn_wgts)

          # Second is UNIFORM (within the HCube)
          index = np.random.choice(hc[selected_hc]['elm'])
          selected_index_list.append(index)
          coord_origin.append(('sim', index, binlist[selected_bin], '%d-D'%A))
          numresources -= 1


    # Back Project to get new starting Coords for each sample  
      logging.debug("=======================  <INPUT PARAM GENERATION>  =================")
      logging.info('All Indices sampled. Back projecting to high dim coords')
      sampled_set = []
      for i in selected_index_list:
        traj = self.backProjection([i])
        sampled_set.append(traj)
      bench.mark('Sampler')

    # Generate new starting positions
      runtime = self.data['runtime']
      jcqueue = OrderedDict()
      for i, start_traj in enumerate(sampled_set):
        jcID, params = generateNewJC(start_traj)

        # TODO:  Update/check adaptive runtime, starting state
        jcConfig = dict(params,
              name    = jcID,
              runtime = runtime,     # In timesteps
              dcdfreq = self.data['dcdfreq'],           # Frame save rate
              interval = self.data['dcdfreq'] * self.data['sim_step_size'],                       
              temp    = 310,
              timestep = self.data['timestep'],
              gc      = 1,
              origin  = coord_origin[i][0],
              src_index = coord_origin[i][1],
              src_bin  = coord_origin[i][2],
              src_hcube = coord_origin[i][3],
              application   = settings.APPL_LABEL)

        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in jcConfig.items():
          logging.debug("   %s:  %s", k, str(v))

        #  Add to the output queue & save config info
        jcqueue[jcID] = jcConfig
        logging.info("New Job Candidate Completed:  %s   #%d on the Queue", jcID, len(jcqueue))

      bench.mark('GenInputParams')

    #  POST-PROCESSING  -------------------------------------
      logging.debug("============================  <POST-PROCESSING & OUTPUT>  =============================")
          
      self.wait_catalog()

      # Clear current queue, mark previously queues jobs for GC, push new queue
      qlen = self.catalog.llen('jcqueue')
      logging.debug('Current queue len;   %s', str(qlen))
      if qlen > 0:
        curqueue = self.catalog.lrange('jcqueue', 0, -1)
        logging.info("Marking %d obsolete jobs for garbage collection", len(curqueue))
        for jc_key in curqueue:
          key = wrapKey('jc', jc_key)
          config = self.catalog.hgetall(key)
          config['gc'] = 0
          # Add gc jobs it to the state to write back to catalog (flags it for gc)
          self.addMut(key, config)
        self.catalog.delete('jcqueue')

      #  CATALOG UPDATES
      self.catalog.rpush('datacount', len(labeled_pts_rms))

      #  EXPR 7 Update:
      if EXPERIMENT_NUMBER > 5:
        # self.catalog.storeNPArray(np.array(centroid), 'subspace:covar:centroid:%d' % cov_iteration)
        self.catalog.rpush('subspace:covar:thruindex', len(covar_pts))

      # Update cache hit/miss
      hit = self.cache_hit
      miss = self.cache_miss
      logging.info('##CACHE_HIT_MISS %d %d  %.3f', hit, miss, (hit)/(hit+miss))
      self.catalog.rpush('cache:hit', self.cache_hit)
      self.catalog.rpush('cache:miss', self.cache_miss)

      self.data['jcqueue'] = list(jcqueue.keys())

      logging.debug("   JCQUEUE:  %s", str(self.data['jcqueue']))
      # Update Each new job with latest convergence score and save to catalog(TODO: save may not be nec'y)
      logging.debug("Updated Job Queue length:  %d", len(self.data['jcqueue']))
      for jcid, config in jcqueue.items():
        # config['converge'] = self.data['converge']
        self.addMut(wrapKey('jc', jcid), config)
 
      bench.mark('PostProcessing')
      print ('## TS=%d' % self.data['timestep'])
      bench.show()
      stat.show()

      return list(jcqueue.keys())




if __name__ == '__main__':

  mt = controlJob(__file__)

  # GENDATA -- to manually generate pdb/psf files for specific DEShaw reference points
  GENDATA = False

  if GENDATA:
    wells = [(0, 2099, 684),
            (1, 630, 602),
            (2, 2364, 737),
            (3, 3322, 188),
            (4, 2108, 258)]
    print('GEN DATA!')

    #  REDO WELLs
    for well in wells:
      state, win, frame = well

      logging.error(' NEED TO REDO Well start point gen')

      pdbfile, archiveFile = getHistoricalTrajectory(win)

      jcID, params = generateNewJC(archiveFile, ppdbfile, frame, jcid='test_%d'%state)

      jcConfig = dict(params,
          name    = jcID,
          runtime = 20000,
          interval = 500,
          temp    = 310,
          state   = state,
          weight  = 0.0,
          timestep = 0,
          gc      = 1,
          application   = DEFAULT.APPL_LABEL)

      print("Data Generated! ")
      print("Job = ", jcID)
      for k, v in jcConfig.items():
        logging.info('%s: %s', k, str(v))


      catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
      catalog.hmset('jc_'+jcID, jcConfig)
    sys.exit(0)

  mt.run()

