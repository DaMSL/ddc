#!/usr/bin/env python

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


def q_select (T, value, limit=None):
  """SELECT operator
  """
  idx_list = []
  for i, elm in enumerate(T):
    if elm == value:
      idx_list.append(i)
  if limit is None or len(idx_list) < limit:
    return idx_list
  else:
    return np.random.choice(idx_list, limit)


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
      thru_index = min(start_index + int(batchSize), self.catalog.llen('label:rms')) - 1

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
      binlist = [(A, B) for A in range(numLabels) for B in range(numLabels)]

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

      # To update Thetas
      rmslist = [np.fromstring(i) for i in self.catalog.lrange('subspace:rms', 0, -1)]
      obslist = self.catalog.lrange('label:rms', 0, -1)
      # translist=[str((a,b)) for a in [StateA, StateB] for b in [StateA, StateB]]

      # reduce to transition distributions
      # if EXPERIMENT_NUMBER > 10:
      #   thetas  = self.catalog.loadNPArray('thetas')
      #   thetas_updated = np.zeros(shape=(numLabels, numLabels))
      # # Map diff's to matrix list
      #   diffM = [[[] for a in range(numLabels)] for b in range(numLabels)]
      #   for rms, obs in zip(rmslist, obslist):
      #     A, B = np.argsort(rms)[:2]
      #     diffM[A][B].append = rms[A] - rms[B]

      #   trans_factor = self.data['transition_sensitivity']
      #   for A in range(0, numLabels-1):
      #     for B in range(A+1, numLabels):
      #       X = sorted(diffM[A][B] + diffM[B][A])
      #       crossover = 0
      #       for i, x in enumerate(X):
      #         if x > 0:
      #           crossover = i
      #           break
      #       print('Crossover at Index #', crossover)

      #       # Find local max gradient  (among 50% of points)
      #       zoneA = int((1-trans_factor) * crossover)
      #       zoneB = crossover + int(trans_factor * (len(X) - crossover))
      #       gradA = zoneA + np.argmax(np.gradient(X[zoneA:crossover]))
      #       gradB = crossover + np.argmax(np.gradient(X[crossover:zoneB]))
      #       thetaA = X[gradA]
      #       thetaB = X[gradB]
      #       thetas_updated[A][B] = thetaA
      #       thetas_updated[B][A] = thetaB

        # Push Updated Thetas -- should this be transactional?
        # self.catalog.storeNPArray('thetas', thetas_updated)

      # labeled_pts_rms = self.catalog.lrange('label:rms', start_index, thru_index)
      labeled_pts_rms = obslist[start_index:thru_index]
      
      num_pts = len(labeled_pts_rms)
      self.data['ctlIndexHead'] = thru_index
      thru_count = self.data['observe:count']

      logging.debug('##NUM_RMS_THIS_ROUND: %d', num_pts)
      stat.collect('numpts', len(labeled_pts_rms))

      # # Load Previous count by bin  -- can make this more efficient
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        A, B = v_i
        pipe.llen('varbin:rms:%d_%d' % (A, B))
      bincounts_raw = pipe.execute()
      bincounts = [int(i) for i in bincounts_raw]

      # pipe = self.catalog.pipeline()
      # for v_i in binlist:
      #   A, B = v_i
      #   pipe.llen('varbin:rms:%d_%d' % (A, B))

      # logging.debug('Global Counts retrieved for %d keys, %s', len(bincounts), str(bincounts))

    # Calculate variable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERGENCE>  =========================")

      # Bootstrap current sample for RMS
      logging.info("RMS Labels for %d points loaded. Calculating PDF.....", len(labeled_pts_rms))

      logging.info("Posterer PDF: This Sample ONLY")
      count_rms_local, pdf_rms_local = datacalc.posterior_prob(labeled_pts_rms, withcount=True)
      for v_i in binlist:
        if v_i not in pdf_rms_local.keys():
          pdf_rms_local[v_i] = 0.

      # Update current Bootstrap, push to catalog, and then get all previous bootstraps
      vals_iter = {}
      counts_local = {}
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        if v_i not in count_rms_local.keys():
          counts_local[v_i] = 0
        else:
          counts_local[v_i] = count_rms_local[v_i]
        pipe.rpush('bootstrap:rms:%d_%d'%(v_i), counts_local[v_i])
      pipe.execute()
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        pipe.lrange('bootstrap:rms:%d_%d'%(v_i), 0, -1)
      bootlists = pipe.execute()

      # Calculate Current Global Convergence based on RMSD metric
      bootstrap = {}
      convergence_rms = {}
      for i, v_i in enumerate(binlist):
        bootstrap[v_i] = [int(k) for k in bootlists[i]]
        # if len(bootlists[i]) > 0:
        #   logging.info('Bootlist for %s: %s', v_i, bootlists[i])
        mean, CI, stddev, err = datacalc.bootstrap_std(bootstrap[v_i])
        if CI == 0 or mean == 0:
          convergence_rms[v_i] = 1.
        else:    
          convergence_rms[v_i] = CI / mean
        # logging.debug('Convergence: `%s`:  %6.3f', str(v_i), convergence_rms[v_i])
      stat.collect('convergence', [v for k, v in sorted(convergence_rms.items())])

      bench.mark('PostPDF:This')

      #  LOAD ALL POINTS TO COMPARE CUMULTIVE vs ITERATIVE:
      #  NOTE: if data is too big, cumulative would need to updated separately

      #  FOR EARLIER EXPIRIMENTS: We pulled ALL pts to calc global PDF
      # all_pts_rms = self.catalog.lrange('label:rms', 0, thru_index)
      # pdf_rms_cuml = datacalc.posterior_prob(all_pts_rms)
      logging.info("Posterer PDF: For ALL DATA Points")
      total = sum(bincounts)
      logging.debug('Total Obs count = %d', total)
      pdf_rms_global = {binlist[v_i]: bincounts[v_i]/total for v_i in range(len(binlist))}

      # # Retrieve Previous Cuml stats
      # vals_cuml = {}
      # counts_global = {}
      # for i, elm in enumerate(bincounts):
      #   val = int(elm) if elm is not None else 0
      #   counts_global[binlist[i]] = val

      # # Calculate incremental PDF
      # pdf_rms_global = datacalc.incremental_posterior_prob(counts_global, labeled_pts_rms)
      bench.mark('PostPDF:All')

      # for v_i in binlist:
      #   A, B = v_i
      #   pipe.rpush('pdf:global:%d_%d' % (A, B), pdf_rms_cuml[v_i])
      #   pipe.rpush('pdf:local:%d_%d' % (A, B), vals_iter[v_i])

      logging.info("PDF Comparison for all RMSD Bins")
      logging.info('##VAL TS   BIN:      Local     Global     Convergence') 
      obs_by_state = [0 for i in range(numLabels)]
      obs_by_state_local = [0 for i in range(numLabels)]
      for v_i, key in enumerate(sorted(binlist)):
        A, B = key
        obs_by_state_local[A] += counts_local[key]
        obs_by_state[A] += bincounts[v_i]
        logging.info('##VAL %03d %s:    %5.2f      %5.2f      %5.2f' % 
          (self.data['timestep'], str(key), 
            pdf_rms_local[key]*100, pdf_rms_global[key]*100, convergence_rms[key]*100))

      logging.info("OBS_LOCAL,%s", ','.join([str(i) for i in obs_by_state_local]))
      logging.info("OBS_GLOBAL,%s", ','.join([str(i) for i in obs_by_state]))

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")

      ##### BARRIER
      self.wait_catalog()
      selected_index_list = []

      # QUERY PROCESSING & SAMPLING BELOW to select indices. 
      EXPERIMENT_NUMBER = self.experiment_number
      logging.info("RUNNING EXPER CONFIGURATION #%d", EXPERIMENT_NUMBER)

      #####  Experment #1: Round-Robin each of 25 bins (do loop to prevent select from empty bin)
      if EXPERIMENT_NUMBER == 1:
        target_bin = self.data['timestep']
        while True:
          A = (target_bin % 25) // 5
          B = target_bin % 5
          label = str((A, B))
          logging.debug('SELECT points in label, %s', str(label))
          rms_indexlist = q_select(labeled_pts_rms, label, limit=500)
          if len(rms_indexlist) > 0:
            break

      #####  EXPERIMENT #2,3:  Select the least converged start state from RMS
      if EXPERIMENT_NUMBER in [2, 3]:
        conv_startstate = [0 for i in range(5)]
        conv_num = [0 for i in range(5)]
        for k, p in sorted(pdf_rms.items()):
          A, B = eval(k)
          conv_startstate[A] += float(p[3])
          conv_num[A] += 1
        for i in range(5):
          if conv_num[i] == 0:
            conv_startstate[i] = 0
          else:    
            conv_startstate[i] /= conv_num[i]
          logging.debug("##STATE_CONV %d %4.2f", i, conv_startstate[i])
        target_state = np.argmax(conv_startstate)
        rms_indexlist = []
        # Using Appoximation (otherwise this would be VERY long)
        selectsizelimit = 1000 #self.data['backproj:approxlimit'] // 4
        while True:
          logging.debug('SELECT points which started in state: %d', target_state)
          for B in range(5):
            if target_state == B:
              continue
            label = str((target_state, B))
            rms_indexlist.extend(q_select(labeled_pts_rms, label, limit=selectsizelimit))
          if len(rms_indexlist) > 0:
            break
          # If no points, then select a random start state until we get one:
          logging.debug("NO POINTS to Sample!! Selecting a random start state")
          target_state = np.random.randint(5)

      ###### EXPERIMENT #4:  UNIFORM (w/updated Convergence)
      if EXPERIMENT_NUMBER == 4:

        ## UNIFORM SAMPLER
        quota = numresources // len(binlist)
        pipe = self.catalog.pipeline()
        logging.debug("Uniform Sampling on %d resources", numresources)

        # Get Sim Generated Data counts
        for A, B in binlist:
          pipe.llen('varbin:rms:%d_%d' % (A,B))
        results = pipe.execute()
        idx_length = [int(i) for i in results]

        # Get DEShaw Labeled Data
        rmslabel = deshaw.labelDEShaw_rmsd()
        deshaw_samples = {b:[] for b in binlist}
        for i, b in enumerate(rmslabel):
          deshaw_samples[b].append(i)

        coord_origin = []
        while numresources > 0:
          # Uniformally sample one bin
          sampled_bin = np.random.randint(len(binlist))

          # Check local generated sim data for index point
          if idx_length[sampled_bin] is not None and idx_length[sampled_bin] > 0:
            sample_num = np.random.randint(idx_length[sampled_bin])
            logging.debug('SAMPLER: selecting sample #%d from bin %s', sample_num, str(binlist[sampled_bin]))
            index = self.catalog.lindex('varbin:rms:%d_%d' % binlist[sampled_bin], sample_num)
            selected_index_list.append(index)
            coord_origin.append(('sim', index, binlist[sampled_bin], '%d-D'%A))
            numresources -= 1

          # If none, get one form DEShaw
          elif len(deshaw_samples[binlist[sampled_bin]]) > 0:
            index = np.random.choice(deshaw_samples[binlist[sampled_bin]])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', index, str(binlist[sampled_bin]))
            # Negation indicates an historical index number
            selected_index_list.append(-index)
            coord_origin.append(('deshaw', index, binlist[sampled_bin], '%d-D'%A))
            numresources -= 1

          # Otherwise, just keep sampling
          else:
            logging.info("NO Candidates for bin: %s", binlist[sampled_bin])

      ###### EXPERIMENT #5:  BIASED (Umbrella) SAMPLER
      if EXPERIMENT_NUMBER == 5:

        ## UMBRELLA SAMPLER

        # Since convergence is measured to zero, we want to sample with BIAS to
        #  the least converged. Thus, the currently calculated convergence gives us
        #  umbrella function over our convergence already applied

        # Load DEShaw labeled indices
        if self.catalog.exists('label:deshaw'):
          logging.info("Loading DEShaw historical points.... From Catalog")
          rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
        else:
          logging.info("Loading DEShaw historical points.... From File (and recalculating)")
          rmslabel = deshaw.labelDEShaw_rmsd()

        deshaw_samples = {b:[] for b in binlist}
        for i, b in enumerate(rmslabel):
          deshaw_samples[b].append(i)

        coord_origin = []
        conv_vals = np.array([v for k, v in sorted(convergence_rms.items())])
        norm_pdf_conv = conv_vals / sum(conv_vals)
        logging.info("Umbrella Samping PDF (Bootstrapping):")
        sampled_distro_perbin = {b: 0 for b in binlist}

        while numresources > 0:
          # First sampling is BIASED
          selected_bin = np.random.choice(len(binlist), p=norm_pdf_conv)
          A, B = binlist[selected_bin]
          sampled_distro_perbin[binlist[selected_bin]] += 1
          if bincounts[selected_bin] is not None and bincounts[selected_bin] > 0:
            # Secondary Sampling is Uniform
            sample_num = np.random.randint(bincounts[selected_bin])
            logging.debug('SAMPLER: selecting sample #%d from bin %s', 
              sample_num, str(binlist[selected_bin]))
            index = self.catalog.lindex('varbin:rms:%d_%d' % binlist[selected_bin], 
              sample_num)
            selected_index_list.append(index)
            coord_origin.append(('sim', index, binlist[selected_bin], '%d-D'%A))
            numresources -= 1
          elif len(deshaw_samples[binlist[selected_bin]]) > 0:
            index = np.random.choice(deshaw_samples[binlist[selected_bin]])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', 
              index, str(binlist[selected_bin]))
            # Negation indicates an historical index number
            selected_index_list.append(-index)
            coord_origin.append(('deshaw', index, binlist[selected_bin], '%d-D'%A))
            numresources -= 1
          else:
            logging.info("NO Candidates for bin: %s", binlist[selected_bin])

      ###### EXPERIMENT #6, 7:  REWEIGHT OPERATOR, et al
      if EXPERIMENT_NUMBER in [6, 7]:
        #  1. RUN PCA on Covariance Matrices and project using PCA
        #  2. Calculate KMeans using varying K-clusters
        #  3. Score each point with distance to centroid
        #  4. B = Select the smallest half of clusters
        #  5. Build state 3 and 4 KD-Tree using top N-PC for each (from sampled PCA)
        #  6. Run KMeans on each (???) for label/weight of H-Cubes in KD Tree (????)
        #       ALT-> use HCUbe size as its weight
        #  7. A = HCubes for states 3 (and 4)
        #  8. Reweight A into both state 3 and state 4 (B) HCubes
        #  9. ID Overlap
        # 10. Apply Gamme Function

        logging.info("=====  Covariance Matrix PCA-KMeans Calculation (STEP-A)")
        logging.info("Retrieving All Covariance Vectors")

        covar_raw = self.catalog.lrange('subspace:covar:pts', 0, -1)
        covar_pts = np.array([np.fromstring(x) for x in covar_raw])
        covar_index = self.catalog.lrange('subspace:covar:xid', 0, -1)
        covar_fidx = self.catalog.lrange('subspace:covar:fidx', 0, -1)
        logging.info("    Pulled %d Covariance Vectors", len(covar_pts))
        logging.info("Calculating Kernel PCA on Covariance (or Pick your PCA Algorithm here)")
        stat.collect('kpcatrainsize', len(covar_pts))


        # LOAD KPCA, CHECK SIZE/STALENESS & RE_CALC ONLY IF NEEDED
        kpca_key = 'subspace:covar:kernel'
        kpca = PCAnalyzer.load(self.catalog, kpca_key)
        update_kcpa = False
        if kpca is None:
          kpca = PCAKernel(10, 'sigmoid')
          kpca.solve(covar_pts)
          update_kcpa = True

        #  Check if the Kernel is old, but cap at 5K Pts (is this enough?)
        elif min(5000, len(covar_pts)) > (kpca.trainsize * 1.5):
          logging.info('PCA Kernel is old (Updating it). Trained on data set of size %d. Current Data set is %d pts.', kpca.trainsize, len(covar_pts))
          if len(covar_pts) > 5000:
            traindata = np.random.choice(covar_pts, 5000)
          else:
            traindata = covar_pts
          kpca.solve(traindata)
          update_kcpa = True

        bench.mark('CaclKPCA_COV')        
        logging.info("Projecting Covariance to PC")
        subspace_covar_pts = kpca.project(covar_pts)
        bench.mark('ProjKPCA')

        # OW/ PROJECT NEW PTS ONLY -- BUT RETAIN grouped index of all points

        # TODO:  FOR NOW use 5  (AND SAME WITH KMEANS RE-CALC)
        # TODO:  Using 8 with EXP 7  (AND SAME WITH KMEANS RE-CALC)
        NUM_K = 8
        logging.info('Running KMeans on covariance data for K =  %d  (TODO: vary K)', NUM_K)
        centroid, clusters = KM.find_centers(subspace_covar_pts, NUM_K)
        bench.mark('CalcKMeans_COV')

        # TODO: Eventually implement per-point weights
        cov_label, cov_wght = KM.classify_score(subspace_covar_pts, centroid)
        cluster_sizes = np.bincount(cov_label)

        # For now, use normalized cluster sizes as the weights inverted (1-X)
        cov_clust_wgts = (1 - cluster_sizes / np.sum(cluster_sizes))
        logging.info('KMeans complete: bincounts is   %s', str(cluster_sizes))
        stat.collect('KMeansCluster', cluster_sizes)

        cov_iteration = self.catalog.get('subspace:covar:count')
        cov_iteration = 0 if cov_iteration is None else cov_iteration
        logging.info("Storing current centroid results (this is iteration #%d", cov_iteration)


        # self.catalog.storeNPArray(np.array(centroid), 'subspace:covar:centroid:%d' % cov_iteration)
        # self.catalog.rpush('subspace:covar:thruindex', len(covar_pts))
        bench.mark('KmeansComplete')

        logging.info("=====  SELECT points from smallest 4 clusters (of STEP-A)")

        # TODO:  Here is the update for newly labeled data
        groupby_label = {k: [] for k in range(NUM_K)}
        for i, L in enumerate(cov_label):
          groupby_label[L].append(i)

        MAX_SAMPLE_SIZE   = 250   # Max # of cov "pts" to back project
        COVAR_SIZE        = 200   # Ea Cov "pt" is 200 HD pts. -- should be static based on user query
        MAX_PT_PER_MATRIX =  20   # 10% of points from Source Covariance matrix

        # FOR EXPERIMENT # 7 -->  We are using approximiation here. Will need to 
        #  Save approx amt.
        #   After analysis ~ 250 max pull pts should be small enough per (500 total)
        # Select smaller clusters from A
        size_order = np.argsort(cluster_sizes)
        Klist = size_order[0:5]  # [, 0:2]
        cov_select = [[] for k in range(NUM_K)]
        cov_weights = [[] for k in range(NUM_K)]
        for L in Klist:
          if len(groupby_label[L]) <= MAX_SAMPLE_SIZE:
            covlist = groupby_label[L]
          else:
            covlist = np.random.choice(groupby_label[L], MAX_SAMPLE_SIZE)
          for cov in covlist:
            selected_hd_idx = np.random.choice(COVAR_SIZE, MAX_PT_PER_MATRIX).tolist()
            cov_select[L].extend([int(covar_index[cov]) + i for i in selected_hd_idx])
            cov_weights[L].extend([cov_wght[cov] for i in range(MAX_SAMPLE_SIZE)])

        logging.info("Selection Operation results:")
        approx_factor = {}
        logging.info('KMeans Cluster Sizes were:   %s', str(cluster_sizes))
        for L in Klist:
          logging.info("  Selected %d points for cluster # %d  (of size %d)  (from KMeans on Covariance matrices)", len(cov_select[L]), L, cluster_sizes[L])
          approx_factor[L] = len(cov_select[L]) / (COVAR_SIZE * len(groupby_label[L]))
          logging.info("##APPOX: %d  %f", L, approx_factor[L])
        bench.mark('SampleKMCluster')


        back_proj_cov_traj = []
        for k in range(len(cov_select)):
          logging.info('Collecting covariance points from KMeans Cluster %d', k)
          if len(cov_select[k]) == 0:
            logging.info("NOT selecting cluster # %d from covariance KMeans clusters (none or filtered data)", k)
          else:
            logging.info("Projecting cluster # %d from covariance KMeans clusters (%d point)", k, len(cov_select[k]))
            source_cov = self.backProjection(cov_select[k])
            # TODO: Weight Preservation when back-projecting
            alpha = datareduce.filter_alpha(source_cov)
            logging.debug('Back Projected %d points to HD space: %s', len(cov_select[k]), str(alpha))
            bench.mark('BackProject:COV_To_HD_%d' % k)
            back_proj_cov_traj.append(alpha)


        ####  BARRIER 
        self.wait_catalog()


        # KD Tree for states from Reservoir Sample of RMSD labeled HighDim
        reservoir = ReservoirSample('rms', self.catalog)

        logging.info("=====  BUILD HCube Tree(s) Using Smallest State(s) (FROM RMSD Obsevations) ")
        hcube_A = {}
        hcube_B = {}
        hcube_B_wgt = {}
        wgt_A = {}
        wgt_B = {}

        hcube_list = {}

        state_order = deque(np.argsort(obs_by_state))
        state_list = deque()
        logging.info("Scanning current set of observed states and finding the smallest with data")

        #  Use a rarty factor to determine what is/is not a rare event
        rarity_factor = sum(obs_by_state) / len(obs_by_state)
        logging.info('Targeting states with fewer than %d observations (these could be considered "rare")', rarity_factor)
        while len(state_order) > 0:
          A = state_order.popleft()
          logging.debug('Checking state %d with %s observations.', A, obs_by_state[A])
          if obs_by_state[A] > 0 and obs_by_state[A] < rarity_factor:
            state_list.append(A)
        logging.info('Selected the following: %s', str(list(state_list)))


        logging.info("=====  PROJECT KMeans clustered data into HCube KD Tree(s)")
        projected_state = []
        for state in state_list:
          hcube_B[state] = {}
          hcube_B_wgt[state] = {}

          # Load Vectors
          logging.info('Loading subspace and kernel for state %d', state)

          # LOAD KPCA Kernel matrix
          kpca_key = 'subspace:pca:kernel:%d' % state
          kpca = PCAnalyzer.load(self.catalog, kpca_key)

          data_raw = self.catalog.lrange('subspace:pca:%d' % state, 0, -1)
          data = np.array([np.fromstring(x) for x in data_raw])
          if len(data) == 0:
            logging.error('Raw PCA data points should exist for state %d.. Try another state', state)
            continue
          projected_state.append(state)
          logging.info('Building KDtree from data of size: %s', str(data.shape))
          kdtree = KDTree(100, maxdepth=8, data=data)
          bench.mark('KDTreeBuild_%d' % state)

          # Back-project all points to higher dimension <- as consolidatd trajectory
          n_total = 0
          for alpha in back_proj_cov_traj:
            # project down into PCA space for this state
            cov_proj_pca = kpca.project(alpha.xyz)
            logging.debug('Project to PCA:  %s', str(cov_proj_pca.shape))

            # Map each projected point into existing hcubes for this State
            #  A -> PCA  and B -> COV
            #   TODO:  Insert here and deal with collapsing geometric hcubes in KDTree
            # Initiaze the set of "projected" hcubes in B
            # Project every selected point into PCA and then group by hcube
            for i in range(len(cov_proj_pca)):
              hcube = kdtree.project(cov_proj_pca[i], probedepth=8)
              if hcube not in hcube_B[state]:
                hcube_B[state][hcube] = []
                hcube_B_wgt[state][hcube] = 0
              # TODO: Preserve per-point weight and use that here
              hcube_B[state][hcube].append(i)
              hcube_B_wgt[state][hcube] += 1
              n_total += 1
            logging.debug('Projected %d points into PCA.', len(cov_proj_pca))
          bench.mark('Project:HD_To_PCA_%d' % state)

          # FOR NOW: calc aggegrate average Wght for newly projected HCubes
          wgt_B[state] = {k: v/n_total for k, v in hcube_B_wgt[state].items()}

          bench.mark('ProjComplete')

          logging.info("=====  REWEIGHT with Overlappig HCubes  (TODO: fine grained geometric overlap (is it necessary)")
          hcube_list[state] = sorted(hcube_B[state].keys())
          # Get all original HCubes from A for "overlapping" hcubes
          hcube_A[state] = {}
          logging.debug('HCUBELIST:   %s', str(hcube_list[state]))
          for k in hcube_list[state]:
            logging.debug('Retrieving:  %s', str(k))
            hcube_A[state][k] = kdtree.retrieve(k)
          total = sum([len(v) for k,v in hcube_A[state].items()])
          wgt_A[state] = {k: len(v)/total for k, v in hcube_A[state].items()}

        #  GAMMA FUNCTION EXPR # 1 & 2
        # gamma = lambda a, b : a * b

        #  GAMMA FUNCTION EXPR # 3
        # gamma = lambda a, b : (a + b) / 2

        #  GAMMA FUNCTION EXPR # 6
        # gamma = lambda a, b : (a + b) / 2

        #  GAMMA FUNCTION EXPR # 7
        gamma = lambda a, b : (a + b)

        # TODO: Factor in RMS weight
        comb_wgt = {}
        for state in projected_state:
          for hc in hcube_list[state]:
            comb_wgt[(state, hc)] =  gamma(wgt_A[state][hc], wgt_B[state][hc])
        total = sum(comb_wgt.values())

        # Normalize
        for k, v in comb_wgt.items():
          comb_wgt[k] = v / total
        bench.mark('GammaFunc')


        ####  User Query Convergence

        ##### BARRIER
        self.wait_catalog()

        logging.info("=====  CALCULATE User Query Convergence")
        logging.info('Resultant HCube Data (for bootstrap)')
        # Get current iteration number
        iteration = int(self.catalog.incr('boot:qry1:count'))

        for state in projected_state:
          # Ensure all currently projected HCubes have same # of observations
          hc_conv_keys = self.catalog.keys('boot:qry1:conv:%d:*' % state)
          offset = len('boot:qry1:conv:%d' % state)
          all_hc = [k[offset:] for k in hc_conv_keys]

          # Get ALL convergence data for all HCubes
          pipe = self.catalog.pipeline()
          for k in hc_conv_keys:
            pipe.lrange(k, 0, -1)
          hc_conv_vals_aslist = pipe.execute()
          hc_conv_vals = {}
          for i, k in enumerate(hc_conv_keys):
            if hc_conv_vals_aslist[i] is None or len(hc_conv_vals_aslist[i]) == 0:
              hc_conv_vals[k[offset:]] = []
            else:
              hc_conv_vals[k[offset:]] = [float(val) for val in hc_conv_vals_aslist[i]]

          pipe = self.catalog.pipeline()
          # Ensure convergence is consistently calculated for all HCubes overlapping
          # in this interation. 
          #  If it's a child HCube, pad with the convergence of the parent
          #  If it's a newly disovered HCube (or one which was not recently discovered) pad with zeros
          for hc in hcube_list[state]:
            if hc in hc_conv_vals and (len(hc_conv_vals[hc]) == iteration - 1):
              logging.debug('HC, `%s` is up to date with %d values', hc, len(hc_conv_vals[hc]))
              continue
            if hc not in hc_conv_vals:
              hc_conv_vals[hc] = []
            logging.debug('HC, `%s` is missing data.  LEN=%d  Iter=%d', hc, len(hc_conv_vals[hc]), (iteration-1))
            # Newly discovered HCube: First Check if parent was discovered and use those stats
            if hc not in all_hc:
              for j in range(1, len(hc)-1):
                # Find the parent HCube
                if hc[:-j] in all_hc:
                  for val in hc_conv_vals[hc[:-j]]:
                    hc_conv_vals[hc].append(val)
                    pipe.rpush('boot:qry1:conv:%s' % k, val)
                  break

            # Pad with zeroes (for previous non-projected points) as needed
            for j in range(len(hc_conv_vals[hc]), iteration-1):
              pipe.rpush('boot:qry1:conv:%s' % k, 0)

        # Then push new data:
        for k, v in comb_wgt.items():
          state, hc = k
          norm_wgt = (v/total) 
          logging.debug('   %20s ->  %4d A-pts (w=%0.3f)  %4d B-pts (w=%0.3f)     (GAMMAwgt=%0.3f)', 
            k, len(hcube_A[state][hc]), wgt_A[state][hc], len(hcube_B[state][hc]), wgt_B[state][hc], comb_wgt[k])
          pipe.rpush('boot:qry1:conv:%d:%s' % k, iteration, norm_wgt)
          # pipe.hset('boot:qry1:conv:%s' % k, iteration, norm_wgt)
        pipe.execute()

        state_conv = {}
        convergence = {}
        for state in projected_state:
          convergence[state] = {}
          logging.info('Calculating Convergence for %d keys ', len(hcube_B[state].keys()))
          for k in hcube_list[state]:
            convdata = self.catalog.lrange('boot:qry1:conv:%d:%s' % (state, k), 0, -1)
            bootstrap = [float(i) for i in convdata]
            if iteration > 1:
              mean, CI, stddev, err = datacalc.bootstrap_std(bootstrap)
              convergence[state][k] = CI / mean  # Total Convergence calculation per key
            else:
              convergence[state][k] = 1.0
            logging.info('##CONV %0d %d %10s %.4f', iteration, state, k, convergence[state][k])

          if len(convergence[state].values()) == 0:
            logging.info('Not enough data collected yet.')
            state_conv[state] = 1.
          else:  
            tot_conv_data = np.array(list(convergence[state].values()))
            conv_max = np.max(tot_conv_data)
            conv_min = np.min(tot_conv_data)
            conv_mean = np.mean(tot_conv_data)
            state_conv[state] = min(1.0, conv_mean)
            logging.info('Other convergence vals: min=%.4f, max=%.4f, mean=%.4f', conv_min, conv_max, conv_mean)

          logging.info('##TOTAL_CONV,%d,%.4f', state,state_conv[state])
          stat.collect('qry1conv:%d'%state, state_conv[state])

        # PUSH User's Total Convergence here
          self.catalog.rpush('boot:qry1:TOTAL:%d'%state, state_conv[state])
          bench.mark('Bootstrap')

      ###### EXPERIMENT #8:  REWEIGHT OPERATOR (Improved)
      if EXPERIMENT_NUMBER == 8:

        #  1. RUN KPCA on <<?????>> (sample set) and project all pts
        #  2. Calculate K-D Tree on above
        #  3. Score each point with distance to centroid
        #  4. B = Select the smallest half of clusters
        #  5. Build state 3 and 4 KD-Tree using top N-PC for each (from sampled PCA)
        #  6. Run KMeans on each (???) for label/weight of H-Cubes in KD Tree (????)
        #       ALT-> use HCUbe size as its weight
        #  7. A = HCubes for states 3 (and 4)
        #  8. Reweight A into both state 3 and state 4 (B) HCubes
        #  9. ID Overlap
        # 10. Apply Gamme Function

        logging.info("=====  Covariance Matrix PCA-KMeans Calculation (B)")
        logging.info("Retrieving All Covariance Vectors")
        home = os.getenv('HOME')
        # cfile = home + '/work/DEBUG_COVAR_PTS'
        # DO_COVAR = True
        # if DO_COVAR: 
        #   if os.path.exists(cfile + '.npy'):
        #     covar_pts = np.load(cfile + '.npy')
        #     logging.debug('Loaded From File')
        #   else: 
        #     covar_raw = self.catalog.lrange('subspace:covar:pts', 0, -1)
        #     covar_pts = np.array([np.fromstring(x) for x in covar_raw])
        #     np.save(cfile, covar_pts)
        #     logging.debug('Loaded From Catalog & Saved')

        covar_raw = self.catalog.lrange('subspace:covar:pts', 0, -1)
        covar_pts = np.array([np.fromstring(x) for x in covar_raw])
        covar_index = self.catalog.lrange('subspace:covar:xid', 0, -1)
        logging.debug('Indiced Loaded. Retrieving File Indices')
        covar_fidx = self.catalog.lrange('subspace:covar:fidx', 0, -1)

        logging.info("    Pulled %d Covariance Vectors", len(covar_pts))
        logging.info("Calculating Incremental PCA on Covariance (or Pick your PCA Algorithm here)")
        stat.collect('kpcatrainsize', len(covar_pts))

        # FOR incrementatl PCA:
        NUM_PC = 6
        ipca_key = 'subspace:covar:ipca'
        ipca = PCAnalyzer.load(self.catalog, ipca_key)
        if ipca is None:
          logging.info('Creating a NEW IPCA')
          ipca = PCAIncremental(NUM_PC)
          lastindex = 0
        else:
          lastindex = ipca.trainsize
          logging.info('IPCA Exists. Trained on %d pts. Will update with incremental batch of %d NEW pts', 
            ipca.trainsize, len(covar_pts)-ipca.trainsize)

        # For incremental, partial solve using only newer pts (from the last "trainsize")
        if len(covar_pts)-lastindex > 0:
          ipca.solve(covar_pts[lastindex:])
          logging.info("Incrementatl PCA Updated. Storing Now...")

          ####  BARRIER 
          self.wait_catalog()
          ipca.store(self.catalog, ipca_key)

        bench.mark('CaclIPCA_COV')        
        logging.info("IPCA Saved. Projecting Covariance to PC")

        # cfile = home + '/work/DEBUG_SUBCOVAR_PTS'
        # if os.path.exists(cfile + '.npy'):
        #   subspace_covar_pts = np.load(cfile + '.npy')
        # else: 
        subspace_covar_pts = ipca.project(covar_pts)
          # np.save(cfile, subspace_covar_pts)

        bench.mark('ProjIPCA')

        # OW/ PROJECT NEW PTS ONLY -- BUT RETAIN grouped index of all points
        logging.info('Building Global KD Tree over Covar Subspace with %d data pts', len(subspace_covar_pts))

        global_kdtree = KDTree(50, maxdepth=4, data=subspace_covar_pts, method='median')
        hcube_global = global_kdtree.getleaves()

        # FOR DEBUGGING -- USE ONLY 3 GLOBAL HCUBES
        # hcube_global_ALL = global_kdtree.getleaves()
        # hcube_global = {}
        # num = 0
        # for k, v in hcube_global_ALL.items():
        #   hcube_global[k] = v
        #   num += 1
        #   if num == 3:
        #     break



        # hcube_global = global_kdtree.getleaves()
        logging.info('Global HCubes: Key  Count  Volume  Density  (NOTE DEBUGGING ONLY 3 USED)')
        for k in sorted(hcube_global.keys()):
          v = hcube_global[k]
          logging.info('%-10s        %6d %8.1f %6.1f', k, v['count'], v['volume'], v['density'])

        if self.filelog:
          keys = hcube_global.keys()
          self.filelog.info('global,keys,%s',','.join(keys))
          self.filelog.info('global,count,%s',','.join([str(hcube_global[k]['count']) for k in keys]))
          self.filelog.info('global,volume,%s',','.join([str(hcube_global[k]['volume']) for k in keys]))
          self.filelog.info('global,density,%s',','.join([str(hcube_global[k]['density']) for k in keys]))

        logging.info("=====  SELECT Sampling of points from each Global HCube  (B)")
        s = sorted(hcube_global.items(), key=lambda x: x[1]['count'])
        hcube_global = {x[0]: x[1] for x in s}

        MAX_SAMPLE_SIZE   = 750   # Max # of cov "pts" to back project
        COVAR_SIZE        = 200   # Ea Cov "pt" is 200 HD pts. -- should be static based on user query
        MAX_PT_PER_MATRIX =   6   # 5% of points from Source Covariance matrix

        counter = 0
        for key in hcube_global.keys():
          counter += 1
          if hcube_global[key]['count']  <= MAX_SAMPLE_SIZE:
            cov_index = hcube_global[key]['elm']
            hcube_global[key]['samplefactor'] = 1
          else:
            cov_index = np.random.choice(hcube_global[key]['elm'], MAX_SAMPLE_SIZE)
            hcube_global[key]['samplefactor'] = len(hcube_global[key]['elm']) / MAX_SAMPLE_SIZE
          hcube_global[key]['idxlist'] = []
          for cov in cov_index:
            selected_hd_idx = np.random.choice(COVAR_SIZE, MAX_PT_PER_MATRIX).tolist()
            hcube_global[key]['idxlist'].extend([int(covar_index[cov]) + i for i in selected_hd_idx])
            # cov_weights[L].extend([cov_wght[cov] for i in range(MAX_SAMPLE_SIZE)])
          logging.info('Back Projecting Global HCube `%s`  (%d out of %d)', key, counter, len(hcube_global.keys()))
          source_cov = self.backProjection(hcube_global[key]['idxlist'])
          hcube_global[key]['alpha'] = datareduce.filter_alpha(source_cov)
          logging.debug('Back Projected %d points to HD space: %s', 
            len(hcube_global[key]['idxlist']), str(hcube_global[key]['alpha']))

        bench.mark('Global_BackProj')
        ####  BARRIER 
        self.wait_catalog()

        # KD Tree for states from Reservoir Sample of RMSD labeled HighDim
        reservoir = ReservoirSample('rms', self.catalog)

        logging.info("=====  BUILD HCube Tree(s) Using Smallest State(s) (FROM RMSD Obsevations) ")
        hcube_list = {}

        logging.info("Scanning current set of observed bins and finding all smallest with data (excluding largest 2)")
        hcube_local = {}

        logging.info("=======================================================")
        logging.info("   PROJECT Global HCubes into Per-Bin HCube KD Tree(s)")
        logging.info("=======================================================\n")
        projected_bin = []
        overlap_hcube = {k: {} for k in hcube_global.keys()}
        TEST_TBIN = [(i,j) for i in range(0,5) for j in range(0,5)]
        # TEST_TBIN = [(2, 4)]
        for tbin in TEST_TBIN:
          logging.info("Project Global HCubes into local subspace for %s", str(tbin))
          # Load Vectors
          logging.info('Loading subspace and kernel for bin %s', str(tbin))

          # LOAD KPCA Kernel matrix
          kpca_key = 'subspace:pca:kernel:%d_%d' % tbin
          kpca = PCAnalyzer.load(self.catalog, kpca_key)

          data_raw = self.catalog.lrange('subspace:pca:%d_%d' % tbin, 0, -1)
          data = np.array([np.fromstring(x) for x in data_raw])
          if len(data) == 0:
            logging.error('No Raw PCA data points for bin %s.... Going to next bin', str(tbin))
            continue
          projected_bin.append(tbin)
          logging.info('Building KDtree over local %s bin from observations matrix of size: %s', str(tbin), str(data.shape))
          kdtree = KDTree(50, maxdepth=4, data=data, method='median')
          hcube_local[tbin] = kdtree.getleaves()
          logging.info('LOCAL KD-Tree Completed for %s:', str(tbin))
          for k in hcube_local[tbin].keys():
            logging.info('    `%-9s`   #pts:%6d   density:%9.1f', 
              k, len(hcube_local[tbin][k]['elm']), hcube_local[tbin][k]['density'])

          if self.filelog:
            keys = hcube_local[tbin].keys()
            A,B = tbin
            self.filelog.info('local,%d_%d,keys,%s',A,B,','.join(keys))
            self.filelog.info('local,%d_%d,count,%s',A,B,','.join([str(hcube_local[tbin][k]['count']) for k in keys]))
            self.filelog.info('local,%d_%d,volume,%s',A,B,','.join([str(hcube_local[tbin][k]['volume']) for k in keys]))
            self.filelog.info('local,%d_%d,density,%s',A,B,','.join([str(hcube_local[tbin][k]['density']) for k in keys]))          
          bench.mark('KDTreeBuild_%d_%d' % tbin)

          # Back-project all points to higher dimension <- as consolidatd trajectory
          # NOTE:  Exploration VS Exploitation HERE
          #   EXPORE:  Project higher dense hcubes
          #   EXPLOIT: Project lower dense hcubes
          #  For now, do both
          n_total = 0
          logging.debug('Global Hcubes to Project (%d):  %s', len(hcube_global.keys()), str(hcube_global.keys()))
          for key, hc in hcube_global.items():
            overlap_hcube[key][tbin] = {}

            cov_proj_pca = kpca.project(hc['alpha'].xyz)

            logging.debug('PROJECT: Global HCube `%-9s` (%d pts) ==> Local KDTree %s  ', 
              key, len(cov_proj_pca), str(tbin))
            for i, pt in enumerate(cov_proj_pca):
              hcube = kdtree.probe(pt, probedepth=9)
              # NOTE: Retaining count of projected pts. Should we track individual pts???
              if hcube not in overlap_hcube[key][tbin]:
                overlap_hcube[key][tbin][hcube] = {
                    'idxlist': hcube_local[tbin][hcube]['elm'],
                    'wgt': hcube_local[tbin][hcube]['density'], 
                    'vol': hcube_local[tbin][hcube]['volume'], 
                    'num_projected': 0}
              overlap_hcube[key][tbin][hcube]['num_projected'] += 1
            for k, v in sorted(overlap_hcube[key][tbin].items()):
              logging.debug('          to ==> Local HCube `%-9s`: %6d points', k, v['num_projected'])
          
            

        #  GAMMA FUNCTION EXPR # 1 & 2
        # gamma = lambda a, b : a * b

        #  GAMMA FUNCTION EXPR # 3
        # gamma = lambda a, b : (a + b) / 2

        #  GAMMA FUNCTION EXPR # 6
        # gamma = lambda a, b : (a + b) / 2

        #  GAMMA FUNCTION EXPR # 7
        # gamma = lambda a, b : (a + b)

        #  GAMMA FUNCTION EXPR # 8
        gamma1 = lambda a, b : (a * b)
        gamma2 = lambda a, b : (a + b) / 2
        gamma3 = lambda a, b : (a + b)

        # Initialize reweight matrix and create index-keys
        
        glb_idx = {k: i for i, k in enumerate(hcube_global.keys())}
        loc_idx = {}  
        explore_matrix = {} 
        exploit_matrix = {} 
        explore_pdf = {} 
        exploit_pdf = {} 

        # TODO: Factor in RMS weight
        for tbin in binlist:

          if tbin not in hcube_local.keys():
            logging.info("Subspace %s has not been projected yet.", str(tbin))
            continue
          explore_matrix[tbin] = {}
          exploit_matrix[tbin] = {}

          loc_idx[tbin] = {k: i for i, k in enumerate(hcube_local[tbin].keys())}        
          explore_matrix[tbin] = np.zeros(shape=(len(glb_idx.keys()), len(loc_idx[tbin].keys())))
          exploit_matrix[tbin] = np.zeros(shape=(len(glb_idx.keys()), len(loc_idx[tbin].keys())))

          # for tbin in sorted(bin_list):
          logging.info('')
          logging.info('BIPARTITE GRAPH for %s', str(tbin))
          bipart = {}
          edgelist = []
          b_density_vals = [hcube_global[hcB]['density'] for hcB in hcube_global.keys()]
          b_total_density = np.sum(b_density_vals)
          b_max_density = np.max(b_density_vals)

          b_volume_vals = [hcube_global[hcB]['volume'] for hcB in hcube_global.keys()]
          b_total_volume = np.sum(b_volume_vals)
          b_max_volume = np.max(b_volume_vals)

          for hcB in hcube_global.keys():
            num_B  = hcube_global[hcB]['count']
            wgt1_B = hcube_global[hcB]['density']
            wgt1_B_norm = wgt1_B / b_total_density
            wgt1_V = hcube_global[hcB]['volume']
            wgt1_V_norm = wgt1_V / b_total_volume
            if tbin not in overlap_hcube[hcB]:
              continue
            a_total_density = np.sum([val['wgt'] for val in overlap_hcube[hcB][tbin].values()])
            a_total_volume  = np.sum([val['vol'] for val in overlap_hcube[hcB][tbin].values()])
            for hcA, hcA_data in overlap_hcube[hcB][tbin].items():
              edge = {}
              if hcA not in bipart:
                bipart[hcA] = []  
              num_proj  = hcA_data['num_projected']
              wgt_A  = hcA_data['wgt']
              wgt_A_norm = wgt_A / a_total_density
              wgt_A_scaled = wgt_A_norm * b_max_density
              wgt_A_vol = (hcA_data['vol'] / a_total_volume) * b_max_volume
              wgt2_B = wgt1_B*num_proj
              edge['combW1'] = gamma1(wgt_A, wgt1_B)
              edge['combW2'] = gamma1(wgt_A, wgt2_B)
              edge['combW3'] = gamma2(wgt_A, wgt1_B)
              edge['combW4'] = gamma2(wgt_A, wgt2_B)
              edge['combW5'] = gamma2(wgt_A*num_proj, wgt2_B)
              edge['combW6'] = gamma1(wgt_A_scaled, wgt1_B)
              edge['combW7'] = gamma1(wgt_A_vol, wgt1_V)
              edge['num_A']  = len(hcA_data['idxlist'])
              edge['num_B']  = num_B
              edge['num_proj']  = num_proj
              edge['wgt_A']  = wgt_A
              edge['wgt_A_scaled']  = wgt_A_scaled
              edge['wgt1_B'] = wgt1_B
              edge['wgt2_B'] = wgt2_B
              edge['hcA'] = hcA
              edge['hcB'] = hcB
              bipart[hcA].append(edge)
              edgelist.append((hcA, hcB, num_proj))

              exploit_matrix[tbin][glb_idx[hcB]][loc_idx[tbin][hcA]] = gamma1(wgt_A_scaled, wgt1_B)
              explore_matrix[tbin][glb_idx[hcB]][loc_idx[tbin][hcA]] = gamma1(wgt_A_vol, wgt1_V)

          if len(bipart) == 0:
            logging.info("NO DATA FOR %s", str(tbin))
            continue
          logging.info('')
          # logging.info('A (# Pts) H-Cube        <--- B H-Cube (# proj/total Pts)      wgt_A  wB1:density wB2:Mass     A*B1     A*B2     AVG(A,B1)     AVG(A,B2)   AVG(A2,B2)   AVG(A,B1)scaled')
          logging.info('A (# Pts) H-Cube        <--- B H-Cube (# proj/total Pts)      wgt_A1(scaled)   wgt_B1     EXPLOIT(A*B)   EXPLORE (volA*volB)')
          for k, v in bipart.items():
            for edge in v:
#               logging.info('A (%(num_A)4d pts) `%(hcA)-8s` <--- `%(hcB)9s`  (%(num_B)4d / %(num_proj)4d pts) B %(wgt_A)9.1f \
# %(wgt1_B)9.1f %(wgt2_B)9.1f %(combW1)9.1f %(combW2)9.1f %(combW3)9.1f %(combW4)9.1f %(combW5)9.1f  %(combW6)9.1f ' % edge)
              logging.info('A (%(num_A)4d pts) `%(hcA)-8s` <--- `%(hcB)9s`  (%(num_B)4d / %(num_proj)4d pts) B %(wgt_A_scaled)9.1f \
%(wgt1_B)9.1f %(combW6)9.1f %(combW7)9.1f' % edge)
              if self.filelog:
                A,B = tbin
                self.filelog.info('edge,%d_%d,%s,%s,%d',A,B,edge['hcA'],edge['hcB'],edge['combW6'])

          #  Normalized by Row & Col, then combine
          exploit_pdf[tbin] = np.sum(exploit_matrix[tbin], axis=0)/np.sum(exploit_matrix[tbin])
          explore_pdf[tbin] = np.sum(explore_matrix[tbin], axis=0)/np.sum(explore_matrix[tbin])

          # Prepare nodes for Bipartite graph
          # nA = set()
          # nB = set()
          # elist = []
          # for e in edgelist:
          #   a, b, z = e
          #   if z <= 5:
          #     continue
          #   nA.add(a)
          #   nB.add(b)
          #   elist.append((a,b,z))
          # nAKeys = sorted(nA)[::-1]
          # nBKeys = sorted(nB)[::-1]
          # sizesA = [hcube_local[tbin][n]['count'] for n in nAKeys]
          # sizesB = [hcube_global[n]['count']*3 for n in nBKeys]
          # idxA = {key: i for i, key in enumerate(nAKeys)}
          # idxB = {key: i for i, key in enumerate(nBKeys)}
          # edges = [(idxA[a], idxB[b], z) for a, b, z in elist]
          # G.bipartite(sizesA,sizesB,edges,sizesA,sizesB,'bipartite_%d_%d' % tbin)

        bench.mark('Reweight_complete')
        bench.show()
        # logging.info('STOPPING HERE!!!!')
        # sys.exit(0)

        ####  User Query Convergence

        ##### BARRIER
        self.wait_catalog()

        # EXECUTE SAMPLER
        logging.debug("=======================  <DATA SAMPLER>  =========================")

        ## UMBRELLA SAMPLER

        # Since convergence is measured to zero, we want to sample with BIAS to
        #  the least converged. Thus, the currently calculated convergence gives us
        #  umbrella function over our convergence already applied

        # Load DEShaw labeled indices
        if self.catalog.exists('label:deshaw'):
          logging.info("Loading DEShaw historical points.... From Catalog")
          rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
        else:
          logging.info("Loading DEShaw historical points.... From File (and recalculating)")
          rmslabel = deshaw.labelDEShaw_rmsd()
          pipe = self.catalog.pipeline()
          for rms in rmslabel:
            pipe.rpush('label:deshaw', rms)
          pipe.execute()

        deshaw_samples = {b:[] for b in binlist}
        for i, b in enumerate(rmslabel):
          deshaw_samples[b].append(i)

        for k, v in deshaw_samples.items():
          logging.debug("DESHAW:  %s  %d", str(k), len(v))

        coord_origin = []
        conv_vals = np.array([v for k, v in sorted(convergence_rms.items())])
        norm_pdf_conv = conv_vals / sum(conv_vals)
        logging.info("Umbrella Samping PDF (Bootstrapping):")
        sampled_distro_perbin = {b: 0 for b in binlist}

        while numresources > 0:
          # First sampling is BIASED
          selected_bin = np.random.choice(len(binlist), p=norm_pdf_conv)
          A, B = tbin = binlist[selected_bin]
          sampled_distro_perbin[tbin] += 1
          if bincounts[selected_bin] is not None and bincounts[selected_bin] > 0:
  
            if tbin not in hcube_local.keys():
              sample_num = np.random.randint(bincounts[selected_bin])
              logging.info("No SubSpace for %s. Reverting to Uniform [Still Incubating] selecting sample #%d", 
                str(tbin), sample_num)
            else:
              # REWEIGHT SAMPLING: Secondary Sampling is either explore/exploit from Reweight Op 
              exploit = np.random.random(1) < self.data['exploit_factor']
              if exploit:
                expl_scheme = 'Exploit'
                hcube_idx = np.random.choice(len(exploit_pdf[tbin]), p=exploit_pdf)
              else:
                expl_scheme = 'Explore'
                hcube_idx = np.random.choice(len(explore_pdf[tbin]), p=explore_pdf)
              selected_hcube = loc_idx[hcube_idx]

              # Uniform sample within the chosen bin
              sample_num = np.random.choice(hcube_local[tbin][selected_hcube]['elm'])
              logging.debug('REWEIGHT SAMPLER [%s]: selecting sample #%d from bin %s', 
                expl_scheme, sample_num, str(tbin))

            index = self.catalog.lindex('varbin:rms:%d_%d' % tbin, sample_num)
            selected_index_list.append(index)
            coord_origin.append(('sim', index, tbin, '%d-D'%A))
            numresources -= 1
          elif len(deshaw_samples[tbin]) > 0:
            index = np.random.choice(deshaw_samples[tbin])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s  [Still Incubating]', 
              index, str(tbin))
            # Negation indicates an historical index number
            selected_index_list.append(-index)
            coord_origin.append(('deshaw', index, tbin, '%d-D'%A))
            numresources -= 1
          else:
            logging.info("No Candidates for bin: %s. Re-Sampling", tbin)

#---------------------------
        # logging.info("=====  CALCULATE User Query Convergence")
        # logging.info('Resultant HCube Data (for bootstrap)')
        # # Get current iteration number
        # iteration = int(self.catalog.incr('boot:qry1:count'))

        # for state in projected_state:
        #   # Ensure all currently projected HCubes have same # of observations
        #   hc_conv_keys = self.catalog.keys('boot:qry1:conv:%d:*' % state)
        #   offset = len('boot:qry1:conv:%d' % state)
        #   all_hc = [k[offset:] for k in hc_conv_keys]

        #   # Get ALL convergence data for all HCubes
        #   pipe = self.catalog.pipeline()
        #   for k in hc_conv_keys:
        #     pipe.lrange(k, 0, -1)
        #   hc_conv_vals_aslist = pipe.execute()
        #   hc_conv_vals = {}
        #   for i, k in enumerate(hc_conv_keys):
        #     if hc_conv_vals_aslist[i] is None or len(hc_conv_vals_aslist[i]) == 0:
        #       hc_conv_vals[k[offset:]] = []
        #     else:
        #       hc_conv_vals[k[offset:]] = [float(val) for val in hc_conv_vals_aslist[i]]

        #   pipe = self.catalog.pipeline()
        #   # Ensure convergence is consistently calculated for all HCubes overlapping
        #   # in this interation. 
        #   #  If it's a child HCube, pad with the convergence of the parent
        #   #  If it's a newly disovered HCube (or one which was not recently discovered) pad with zeros
        #   for hc in hcube_list[state]:
        #     if hc in hc_conv_vals and (len(hc_conv_vals[hc]) == iteration - 1):
        #       logging.debug('HC, `%s` is up to date with %d values', hc, len(hc_conv_vals[hc]))
        #       continue
        #     if hc not in hc_conv_vals:
        #       hc_conv_vals[hc] = []
        #     logging.debug('HC, `%s` is missing data.  LEN=%d  Iter=%d', hc, len(hc_conv_vals[hc]), (iteration-1))
        #     # Newly discovered HCube: First Check if parent was discovered and use those stats
        #     if hc not in all_hc:
        #       for j in range(1, len(hc)-1):
        #         # Find the parent HCube
        #         if hc[:-j] in all_hc:
        #           for val in hc_conv_vals[hc[:-j]]:
        #             hc_conv_vals[hc].append(val)
        #             pipe.rpush('boot:qry1:conv:%s' % k, val)
        #           break

        #     # Pad with zeroes (for previous non-projected points) as needed
        #     for j in range(len(hc_conv_vals[hc]), iteration-1):
        #       pipe.rpush('boot:qry1:conv:%s' % k, 0)

        # # Then push new data:
        # for k, v in comb_wgt.items():
        #   state, hc = k
        #   norm_wgt = (v/total) 
        #   logging.debug('   %20s ->  %4d A-pts (w=%0.3f)  %4d B-pts (w=%0.3f)     (GAMMAwgt=%0.3f)', 
        #     k, len(hcube_A[state][hc]), wgt_A[state][hc], len(hcube_B[state][hc]), wgt_B[state][hc], comb_wgt[k])
        #   pipe.rpush('boot:qry1:conv:%d:%s' % k, iteration, norm_wgt)
        #   # pipe.hset('boot:qry1:conv:%s' % k, iteration, norm_wgt)
        # pipe.execute()

       #  # state_conv = {}
       #  # convergence = {}
       #  # for state in projected_state:
       #  #   convergence[state] = {}
       #  #   logging.info('Calculating Convergence for %d keys ', len(hcube_B[state].keys()))
       #  #   for k in hcube_list[state]:
       #  #     convdata = self.catalog.lrange('boot:qry1:conv:%d:%s' % (state, k), 0, -1)
       #  #     bootstrap = [float(i) for i in convdata]
       #  #     if iteration > 1:
       #  #       mean, CI, stddev, err = datacalc.bootstrap_std(bootstrap)
       #  #       convergence[state][k] = CI / mean  # Total Convergence calculation per key
       #  #     else:
       #  #       convergence[state][k] = 1.0
       #  #     logging.info('##CONV %0d %d %10s %.4f', iteration, state, k, convergence[state][k])

       #  #   if len(convergence[state].values()) == 0:
       #  #     logging.info('Not enough data collected yet.')
       #  #     state_conv[state] = 1.
       #  #   else:  
       #  #     tot_conv_data = np.array(list(convergence[state].values()))
       #  #     conv_max = np.max(tot_conv_data)
       #  #     conv_min = np.min(tot_conv_data)
       #  #     conv_mean = np.mean(tot_conv_data)
       #  #     state_conv[state] = min(1.0, conv_mean)
       #  #     logging.info('Other convergence vals: min=%.4f, max=%.4f, mean=%.4f', conv_min, conv_max, conv_mean)

       #  #   logging.info('##TOTAL_CONV,%d,%.4f', state,state_conv[state])
       #  #   stat.collect('qry1conv:%d'%state, state_conv[state])

       #  # # PUSH User's Total Convergence here
       #  #   self.catalog.rpush('boot:qry1:TOTAL:%d'%state, state_conv[state])
       #  #   bench.mark('Bootstrap')


       #  # EXECUTE SAMPLER
       #  logging.debug("=======================  <DATA SAMPLER>  =========================")

       #  ##### BARRIER
       #  self.wait_catalog()

       #  #  Exploration vs Exploitation SamplingUmbrella Sampling  (For exploration)
       #  #   or use weights as exploitation
       #  #  could also use convegence.....
       #  # Factor in convergence. Umbrella sampling, divided by convergence:
       #  #  Converence (approaches zero for more converged bins) will be more sampled
       #  #  Smaller bins will also be preferred in the sampling
       #  #  TODO: Add linear regression and pick 'trending' convergence bins
       #  hcube_pdf = []
       #  hcube_selection = []
       #  for state in projected_state:
       #    for k in hcube_list[state]:
       #      if convergence[state][k] <= 0:
       #        logging.info('Detected that %s has totally converged (not sure if this is even possible)', k)
       #      else:

       #        # Apply Convergence Factor:  More converged will increase Prob.
       #        select_weight = (1 - comb_wgt[(state,k)]) / convergence[state][k]
       #        hcube_pdf.append(select_weight)
       #        hcube_selection.append((state,k))
       #        logging.info('### TOTAL_WGT %d-%s  %.4f', state, k, select_weight)
       #  hcube_pdf = np.array(hcube_pdf)
       #  hcube_pdf /= np.sum(hcube_pdf)

       # # TODO:  Number & Runtime of each job <--- Resource/Convergence Dependant
       #  coord_origin = []
       #  if len(hcube_list) == 0:
       #    logging.info('Incubating: Lacking sufficient Data to sample')
       #    rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
       #    deshaw_samples = {b:[] for b in binlist}
       #    # Adjust sampling Size for index
       #    sample_factor = 4125000 // len(rmslabel)
       #    for i, b in enumerate(rmslabel):
       #      deshaw_samples[b].append(i*sample_factor)

       #    # Uniform sample while incubating
       #    itr = 0
       #    while numresources > 0:
       #      sample_bin = itr % len(binlist)
       #      A, B = binlist[sample_bin]
       #      index = np.random.choice(deshaw_samples[binlist[sample_bin]])
       #      logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', index, str(binlist[sample_bin]))
       #      # Negation indicates an historical index number
       #      selected_index_list.append(-index)
       #      hcube_label = '%d-D' % (selected_state)
       #      coord_origin.append(('deshaw', index, binlist[sample_bin], hcube_label))
       #      numresources -= 1
       #      itr += 1

       #  while numresources > 0:
       #    # First selection is biased using PDF from above
       #    hcube = np.random.choice(np.arange(len(hcube_selection)), p=hcube_pdf)
       #    selected_state, selected_hcube = hcube_selection[hcube]
       #    # Secondary Sampling is Uniform (OR Could Be Weighted again (see write up))
       #    # selected_index = np.random.choice(list(hcube_A[selected_hcube]) + list(hcube_B[selected_hcube]))

       #    #  ONLY Sample from data points of HCube A
       #    selected_index = np.random.choice(list(hcube_A[selected_state][selected_hcube]))
       #    logging.debug('SAMPLER: selected sample #%d from hcube %s', selected_index, selected_hcube)
       #    selected_index_list.append(selected_index)
       #    hcube_label = '%d_%s' % (selected_state, selected_hcube)
       #    # TODO: Batch this catalog request or delay it
       #    origin_bin = self.catalog.lindex('label:rms', selected_index)
       #    coord_origin.append(('sim', selected_index, origin_bin, hcube_label))
       #    numresources -= 1


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

