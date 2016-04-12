#!/usr/bin/env python

import argparse
import sys
import os
import sys
import math
import json
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
from datatools.pca import PCAnalyzer, PCAKernel
from datatools.approx import ReservoirSample
from mdtools.simtool import generateNewJC
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient
from bench.timer import microbench
from bench.stats import StatCollector


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

np.set_printoptions(precision=5, suppress=True)

SIMULATE_RATIO = 50


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
      self.addImmut('backproj:approxlimit')
      self.addImmut('numresources')
      self.addImmut('ctlBatchSize_min')
      self.addImmut('ctlBatchSize_max')
      self.addImmut('dcdfreq')
      self.addImmut('runtime')
      self.addImmut('sim_step_size')
          
      self.addImmut('launch')
      self.addAppend('timestep')
      # self.addAppend('observe')
      self.addMut('runtime')

      self.addImmut('pcaVectors')


      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = 24

      self.modules.add('namd/2.10')

      self.trajlist_async = deque()

      self.cacheclient = None

      # For stat tracking
      self.cache_hit = 0
      self.cache_miss = 0

    def term(self):
      # For now
      return False

    def split(self):

      settings = systemsettings()

      # Batch sizes should be measures in abosolute # of observations
      workloadList = deque(self.data['completesim'])
      minbatchSize = int(self.data['ctlBatchSize_min'])
      maxbatchSize = int(self.data['ctlBatchSize_max'])
      batchAmt = 0
      logging.debug('Controller will launch with batch size between: %d and %d observations', minbatchSize, maxbatchSize)
      num_simbatchs = 0
      compsimlen = self.catalog.llen('completesim')
      logging.debug("Current Workload List: %s.   Tracking %d items in completesim queue", str(workloadList), compsimlen)
      while batchAmt < maxbatchSize and len(workloadList) > 0:
        if int(workloadList[0]) + batchAmt > maxbatchSize:
          if batchAmt == 0:
            logging.warning('CAUTION. May need to set the max batch size higher (cannot run controller)')
            batchAmt += int(workloadList.popleft())
          break
        batchAmt += int(workloadList.popleft())
        num_simbatchs += 1
        logging.debug('  Batch amount up to: %d', batchAmt)

      # Don't do anything if the batch size is less than the min:
      if batchAmt < minbatchSize:
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

    # FOR THREADING
    # def trajLoader(self, indexlist):
    #   for idx, framelist in indexlist:
    #     logging.debug(" Threaded Traj Loader for idx#%d", idx)
    #     if idx >= 0:
    #       filename = self.catalog.lindex('xid:filelist', idx)
    #       traj = datareduce.load_trajectory(filename)
    #     else:
    #       filename = deshaw.getDEShawfilename(-idx, fullpath=True) % (-idx)
    #       traj = md.load(filename, top=deshaw.PDB_FILE)
    #       # traj = deshaw.loadDEShawTraj(-1 * idx, filt='all')
    #     traj.atom_slice(traj.top.select('protein'), inplace=True)
    #     self.trajlist_async.append(traj.slice(framelist))
    #     logging.debug('FILE ld complete for %s.  # Traj Loaded = %d', filename, len(self.trajlist_async))

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
      pipe = self.catalog.pipeline()
      for idx in index_list:
        # Negation indicates  historical index:
        index = int(idx)
        if index < 0:
          file_index, frame = deshaw.refFromIndex(-idx)
          historical_framelist.append((file_index, frame))
          # logging.debug('[BP] DEShaw:  file #%d,   frame#%d', file_index, frame)
        else:
          pipe.lindex('xid:reference', index)

      # Load higher dim point indices from catalog
      generated_framelist = pipe.execute()

      bench.mark('BP:LD:Redis:xidlist')


      ref = deshaw.topo_prot  # Hard coded for now

      # Group all Historical indidces by file number and add to frame Mask 
      historical_frameMask = {}
      for i, idx in enumerate(historical_framelist):
        file_index, frame = idx
        if file_index not in historical_frameMask:
          historical_frameMask[file_index] = []
        historical_frameMask[file_index].append(frame)

      for k, v in historical_frameMask.items():
        logging.debug('[BP] Deshaw lookups: %d, %s', k, str(v))


      # Group all Generated indidces by file index 
      groupbyFileIdx = {}
      for i, idx in enumerate(generated_framelist):
        file_index, frame = eval(idx)
        if file_index not in groupbyFileIdx:
          groupbyFileIdx[file_index] = []
        groupbyFileIdx[file_index].append(frame)

      # Dereference File index to filenames
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
          logging.debug('[BP] Cache MISS on: %d', fileno)
          cache_miss.append(('deshaw', fileno, frames))
        else:
          self.cache_hit += 1
          logging.debug('[BP] Cache HIT on: %d', fileno)
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
          logging.debug('[BP] Cache MISS on: %s', filename)
          cache_miss.append(('sim', generated_filemap[filename], frames))
        else:
          self.cache_hit += 1
          logging.debug('[BP] Cache HIT on: %s', filename)
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

    # LOAD all new subspaces (?) and values
      # Load new RMS Labels -- load all for now
      bench.start()
      logging.debug('Loading RMS Labels')
      start_index = max(0, self.data['ctlIndexHead'])

      # labeled_pts_rms = self.catalog.lrange('label:rms', self.data['ctlIndexHead'], thru_index)
      logging.debug(" Start_index=%d,  thru_index=%d,   ctlIndexHead=%d", start_index, thru_index, self.data['ctlIndexHead'])
      labeled_pts_rms = self.catalog.lrange('label:rms', start_index, thru_index)
      
      num_pts = len(labeled_pts_rms)
      self.data['ctlIndexHead'] = thru_index

      thru_count = self.catalog.get('observe:count')

      logging.debug('##NUM_RMS_THIS_ROUND: %d', num_pts)
      stat.collect('numpts', len(labeled_pts_rms))
      self.catalog.rpush('datacount', len(labeled_pts_rms))

    # Calculate variable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERGENCE>  =========================")

      # Bootstrap current sample for RMS
      logging.info("RMS Labels for %d points loaded. Calculating PDF.....", len(labeled_pts_rms))

      logging.info("Posterer PDF: This Sample ONLY")
      pdf_rms_iter = datacalc.posterior_prob(labeled_pts_rms)

      vals_iter = {}
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        A, B = v_i
        val = 0 if v_i not in pdf_rms_iter.keys() else pdf_rms_iter[v_i]
        vals_iter[v_i] = val
        pipe.rpush('pdf:local:%d_%d' % (A, B), val)
      pipe.execute()
      bench.mark('PostPDF:This')

      #  LOAD ALL POINTS TO COMPARE CUMULTIVE vs ITERATIVE:
      #  NOTE: if data is too big, cumulative would need to updated separately

      #  FOR EARLIER EXPIRIMENTS: We pulled ALL pts to calc global PDF
      # all_pts_rms = self.catalog.lrange('label:rms', 0, thru_index)
      # pdf_rms_cuml = datacalc.posterior_prob(all_pts_rms)
      logging.info("Posterer PDF: ALL DATA Points Collected")

      # Retrieve Previous Cuml stats
      vals_cuml = {}
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        A, B = v_i
        pipe.llen('varbin:rms:%d_%d' % (A, B))
      results = pipe.execute()
      logging.debug('Global Counts retrieved for %d keys', len(results))

      for i, elm in enumerate(results):
        val = int(elm) if elm is not None else 0
        vals_cuml[binlist[i]] = val

      # Calculate incremental PDF
      pdf_rms_cuml = datacalc.incremental_posterior_prob(vals_cuml, labeled_pts_rms)

      pipe = self.catalog.pipeline()
      for v_i in binlist:
        A, B = v_i
        pipe.rpush('pdf:global:%d_%d' % (A, B), pdf_rms_cuml[v_i])
      pipe.execute()
      bench.mark('PostPDF:All')

      logging.info("PDF Comparison for all RMSD Bins")
      logging.info('##VAL TS   BIN:  Local  Global') 
      obs_by_state = [0 for i in range(numLabels)]
      for key in sorted(binlist):
        A, B = key
        obs_by_state[A] += int(vals_iter[key]*num_pts) + int(vals_cuml[key] * thru_index)
        logging.info('##VAL %03d %s:    %0.4f    %0.4f' % 
          (self.data['timestep'], str(key), vals_iter[key], vals_cuml[key]))

      logging.info("Total Observations by state: %s", str(obs_by_state))

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")
      #   Using RMS and PCA, umbrella sample transitions out of state 3

      EXPERIMENT_NUMBER = 7

      logging.info("RUNNING EXPER CONFIGURATION #%d", EXPERIMENT_NUMBER)
      # 1. get all points in some state
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
        selectsizelimit = self.data['backproj:approxlimit'] // 4
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
        pipe = self.catalog.pipeline()
        for b in binlist:
          pipe.llen('varbin:rms:%d_%d' % b)
        length = pipe.execute()


        logging.info('Variable bins sizes follows')
        idxlist = {}
        for i, b in enumerate(binlist):
          idxlist[b] = length[i]
          logging.info('  %s:  %d', b, length[i])

        ## UNIFORM SAMPLER
        numresources = self.data['numresources']


        quota = numresources // len(binlist)
        sel = 0
        pipe = self.catalog.pipeline()
        logging.debug("Uniform Sampling on %d resources", numresources)

        rmslabel = deshaw.labelDEShaw_rmsd()
        deshaw_samples = {b:[] for b in binlist}
        for i, b in enumerate(rmslabel):
          deshaw_samples[b].append(i)

        selected_indices = []
        coord_origin = []
        while numresources > 0:
          i = sel%25
          if length[i] is not None and length[i] > 0:
            sample_num = np.random.randint(length[i])
            logging.debug('SAMPLER: selecting sample #%d from bin %s', sample_num, str(binlist[i]))
            index = self.catalog.lindex('varbin:rms:%d_%d' % binlist[i], sample_num)
            selected_indices.append(index)
            coord_origin.append(('sim', index, binlist[i]))
            numresources -= 1
          elif len(deshaw_samples[binlist[i]]) > 0:
            index = np.random.choice(deshaw_samples[binlist[i]])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', index, str(binlist[i]))
            # Negation indicates an historical index number
            selected_indices.append(-index)
            coord_origin.append(('deshaw', index, binlist[i]))
            numresources -= 1
          else:
            logging.info("NO Candidates for bin: %s", binlist[i])
          sel += 1

        
        sampled_set = []
        for i in selected_indices:
          traj = self.backProjection([i])
          sampled_set.append(traj)
        bench.mark('Sampler')
     
      ###### EXPERIMENT #5:  BIASED (Umbrella) SAMPLER
      if EXPERIMENT_NUMBER == 5:
        pipe = self.catalog.pipeline()
        for b in binlist:
          pipe.llen('varbin:rms:%d_%d' % b)
        num_gen_samples = pipe.execute()

        logging.info('Variable bins sizes follows')
        idxlist = {}
        for i, b in enumerate(binlist):
          idxlist[b] = num_gen_samples[i]
          logging.info('  %s:  %d', b, num_gen_samples[i])

        numresources = self.data['numresources']

        ## UMBRELLA SAMPLER

        # Since convergence is measured to zero, we want to sample with BIAS to
        #  the least converged. Thus, the currently calculated convergence gives us
        #  umbrella function over our convergence already applied

        # Load DEShaw labeled indices
        if self.catalog.exists('label:deshaw'):
          logging.info("Loading DEShaw historical points.... From Cache")
          rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
        else:
          logging.info("Loading DEShaw historical points.... From File (and recalculating)")
          rmslabel = deshaw.labelDEShaw_rmsd()

        deshaw_samples = {b:[] for b in binlist}
        for i, b in enumerate(rmslabel):
          deshaw_samples[b].append(i)

        sample_distro_iter = []
        coord_origin = []

        norm_pdf_iter = convergence_iter / sum(convergence_iter)
        norm_pdf_cuml = convergence_cuml / sum(convergence_cuml)

        logging.info("Umbrella Samping PDF (Using Iterative Bootstrapping):")

        sampled_distro_perbin = {b: 0 for b in binlist}

        while numresources > 0:
          # First sampling is BIASED
          selected_bin = np.random.choice(len(binlist), p=norm_pdf_iter)
          sampled_distro_perbin[binlist[selected_bin]] += 1
          if num_gen_samples[selected_bin] is not None and num_gen_samples[selected_bin] > 0:
            # Secondary Sampling is Uniform
            sample_num = np.random.randint(num_gen_samples[selected_bin])
            logging.debug('SAMPLER: selecting sample #%d from bin %s', sample_num, str(binlist[selected_bin]))
            index = self.catalog.lindex('varbin:rms:%d_%d' % binlist[selected_bin], sample_num)
            sample_distro_iter.append(index)
            coord_origin.append(('sim', index, binlist[selected_bin]))
            numresources -= 1
          elif len(deshaw_samples[binlist[selected_bin]]) > 0:
            index = np.random.choice(deshaw_samples[binlist[selected_bin]])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', index, str(binlist[selected_bin]))
            # Negation indicates an historical index number
            sample_distro_iter.append(-index)
            coord_origin.append(('deshaw', index, binlist[selected_bin]))
            numresources -= 1
          else:
            logging.info("NO Candidates for bin: %s", binlist[selected_bin])

        sampled_set = []
        for i in sample_distro_iter:
          traj = self.backProjection([i])
          sampled_set.append(traj)
        bench.mark('Sampler')

      ###### EXPERIMENT #6:  REWEIGHT OPERATOR, et al
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
        logging.info("Retrieving Covariance Matrices")
        covar_raw = self.catalog.lrange('subspace:covar:pts', 0, -1)
        covar_pts = np.array([np.fromstring(x) for x in covar_raw])
        covar_fidx = self.catalog.lrange('subspace:covar:fidx', 0, -1)
        covar_index = self.catalog.lrange('subspace:covar:xid', 0, -1)
        logging.info("    Pulled %d Covariance Matrices", len(covar_pts))
        logging.info("Calculating Kernel PCA on Covariance (Pick your PCA Algorithm here)")
        stat.collect('kpcatrainsize', len(covar_pts))


        # LOAD KPCA, CHECK SIZE/STALENESS & RE_CALC ONLY IF NEEDED
        kpca_key = 'subspace:covar:kernel'
        kpca = PCAnalyzer.load(self.catalog, kpca_key)
        if kpca is None:
          # kpca = PCAKernel(None, 'rbf')
          kpca = PCAKernel(10, 'sigmoid')
          kpca.solve(covar_pts)

        #  Check if the Kernel is old, but cap at 5K Pts (is this enough?)
        if min(5000, len(covar_pts)) > (kpca.trainsize * 1.5):
          logging.info('PCA Kernel is old (Updating it). Trained on data set of size %d. Current reservoir is %d pts.', tsize, rsize)
          kpca.solve(traindata)

        bench.mark('CaclKPCA_COV')        
        logging.info("Projecting Covariance to PC")
        pca_cov_pts = kpca.project(covar_pts)
        bench.mark('ProjKPCA')

        # OW/ PROJECT NEW PTS ONLY -- BUT RETAIN grouped index of all points

        # TODO:  FOR NOW use 5  (AND SAME WITH KMEANS RE-CALC)
        # TODO:  Using 8 with EXP 7  (AND SAME WITH KMEANS RE-CALC)
        NUM_K = 8
        logging.info('Running KMeans on covariance data for K =  %d  (TODO: vary K)', NUM_K)
        centroid, clusters = KM.find_centers(pca_cov_pts, NUM_K)
        bench.mark('CalcKMeans_COV')

        # TODO: Eventually implement per-point weights
        cov_label, cov_wght = KM.classify_score(pca_cov_pts, centroid)
        cluster_sizes = np.bincount(cov_label)
        # For now, use normalized cluster sizes as the weights
        cov_clust_wgts = cluster_sizes / np.sum(cluster_sizes)
        logging.info('KMeans complete: bincounts is   %s', str(cluster_sizes))
        stat.collect('KMeansCluster', cluster_sizes)

        cov_iteration = self.catalog.get('subspace:covar:count')
        cov_iteration = 0 if cov_iteration is None else cov_iteration
        logging.info("Storing current centroid results (this is iteration #%d", cov_iteration)
        self.catalog.storeNPArray(np.array(centroid), 'subspace:covar:centroid:%d' % cov_iteration)
        self.catalog.rpush('subspace:covar:thruindex', len(covar_pts))
        bench.mark('KmeansComplete')

        logging.info("=====  SELECT points from smallest 2 clusters (of STEP-A)")

        # TODO:  Here is the update for newly labeled data
        groupby_label = {k: [] for k in range(NUM_K)}
        for i, L in enumerate(cov_label):
          groupby_label[L].append(i)

        MAX_SAMPLE_SIZE   = 250   # Max # of cov "pts" to back project
        COVAR_SIZE        = 200   # Ea Cov "pt" is 200 HD pts. -- should be static based on user query
        MAX_PT_PER_MATRIX =  50   # 10% of points from Source Covariance matrix

        # FOR EXPERIMENT # 7 -->  We are using approximiation here. Will need to 
        #  Save approx amt.
        #   After analysis ~ 250 max pull pts should be small enough per (500 total)
        # Select smaller clusters from A
        size_order = np.argsort(cluster_sizes)
        Klist = size_order[0:2]  # [, 0:2]
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
        for L in Klist:
          logging.info("Selected %d points for cluster # %d  (from KMeans on Covariance matrices)", len(cov_select[L]), L)
          approx_factor[L] = len(cov_select[L]) / (COVAR_SIZE * len(groupby_label[L]))
          logging.info("##APPOX: %d  %f", L, approx_factor[L])
        bench.mark('SampleKMCluster')

        # KD Tree for states from Reservoir Sample of RMSD labeled HighDim
        reservoir = ReservoirSample('rms', self.catalog)

        logging.info("=====  BUILD HCube Tree(s) Using Smallest State(s) (FROM RMSD Obsevations) ")
        hcube_B = {}
        hcube_B_wgt = {}



        state_order = deque(np.argsort(obs_by_state))
        state_list = deque()
        logging.info("Scanning current set of observed states and finding the smallest with data (TODO: multiple states)")
        while len(state_order) > 0:
          A = state_order.popleft()
          if obs_by_state[A] > 0:
            state_list.append(A)

        logging.info("=====  PROJECT KMeans clustered data into HCube KD Tree(s)")
        for state in state_list:
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
          logging.info('Building KDtree from data of size: %s', str(data.shape))
          kdtree = KDTree(100, maxdepth=8, data=data)
          bench.mark('KDTreeBuild_%d' % state)

          # Back-project all points to higher dimension <- as consolidatd trajectory
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

              # 2. project into PCA space for this state
              # cov_proj_pca = project_pca(alpha.xyz, pca_vect, settings.PCA_NUMPC)
              
              cov_proj_pca = kpca.project(alpha.xyz)
              logging.debug('Project to PCA:  %s', str(cov_proj_pca.shape))
              bench.mark('Project:HD_To_PCA_%d' % k)

              # 3. Map into existing hcubes 
              #  A -> PCA  and B -> COV
              #   TODO:  Insert here and deal with collapsing geometric hcubes in KDTree
              # Initiaze the set of "projected" hcubes in B
              # Project every selected point into PCA and then group by hcube
              for i in range(len(cov_proj_pca)):
                hcube = kdtree.project(cov_proj_pca[i], probedepth=8)
                if hcube not in hcube_B:
                  hcube_B[hcube] = []
                  hcube_B_wgt[hcube] = []
                # TODO: Preserve per-point weight and use that here
                hcube_B[hcube].append(i)
                hcube_B_wgt[hcube].append(cov_clust_wgts[k])
              logging.debug('Projected %d points into PCA.', len(cov_proj_pca))

          # FOR NOW: calc aggegrate average Wght for newly projected HCubes
          wgt_B = {k: (1-np.mean(v)) for k, v in hcube_B_wgt.items()}
          break  # hence only 1 state for now

        bench.mark('ProjComplete')

        logging.info("=====  REWEIGHT with Overlappig HCubes  (TODO: fine grained geometric overlap (is it necessary)")
        hcube_list = sorted(hcube_B.keys())
        # Get all original HCubes from A for "overlapping" hcubes
        hcube_A = {}
        for k in hcube_list:
          hcube_A[k] = kdtree.retrieve(k)
        total = sum([len(v) for k,v in hcube_A.items()])
        wgt_A = {k: len(v)/total for k, v in hcube_A.items()}

        #  GAMMA FUNCTION EXPR # 1 & 2
        # gamma = lambda a, b : a * b

        #  GAMMA FUNCTION EXPR # 3
        # gamma = lambda a, b : (a + b) / 2

        #  GAMMA FUNCTION EXPR # 6
        gamma = lambda a, b : (a + b) / 2

        # TODO: Factor in RMS weight
        comb_wgt = {k: gamma(wgt_A[k], wgt_B[k]) for k in hcube_list}
        total = sum(comb_wgt.values())
        bench.mark('GammaFunc')


        ####  User Query Convergence
        logging.info("=====  CALCULATE User Query Convergence")
        logging.info('Resultant HCube Data (for bootstrap)')
        # Get current iteration number
        iteration = int(self.catalog.incr('boot:qry1:count'))

        # Ensure all currently projected HCubes have same # of observations
        hc_conv_keys = self.catalog.keys('boot:qry1:conv:*')
        offset = len('boot:qry1:conv:')
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
        # Ensure convergence is consitently calculated for all HCubes overlapping
        # in this interation. 
        #  If it's a child HCube, pad with the convergence of the parent
        #  If it's a newly disovered HCube (or one which was not recently discovered) pad with zeros
        for hc in hcube_list:
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
          norm_wgt = (v/total) 
          logging.debug('   %20s ->  %4d A-pts (w=%0.3f)  %4d B-pts (w=%0.3f)     (GAMMAwgt=%0.3f)', 
            k, len(hcube_A[k]), wgt_A[k], len(hcube_B[k]), wgt_B[k], comb_wgt[k])
          pipe.rpush('boot:qry1:conv:%s' % k, iteration, norm_wgt)
          # pipe.hset('boot:qry1:conv:%s' % k, iteration, norm_wgt)
        pipe.execute()

        convergence = {}
        logging.info('Calculating Convergence for %d keys ', len(hcube_B.keys()))
        for k in hcube_list:
          convdata = self.catalog.lrange('boot:qry1:conv:%s' % k, 0, -1)
          bootstrap = [float(i) for i in convdata]
          if iteration > 1:
            mean, CI, stddev, err = datacalc.bootstrap_std(bootstrap)
            convergence[k] = CI / mean  # Total Convergence calculation per key
          else:
            convergence[k] = 1.0
          logging.info('##CONV %0d %10s %.4f', iteration, k, convergence[k])

        if len(convergence.values()) == 0:
          logging.info('Not enough data collected yet.')
          total_convergence = 1.
        else:  
          tot_conv_data = np.array(list(convergence.values()))
          conv_max = np.max(tot_conv_data)
          conv_min = np.min(tot_conv_data)
          conv_mean = np.mean(tot_conv_data)
          total_convergence = min(1.0, conv_mean)
          logging.info('Other convergence vals: min=%.4f, max=%.4f, mean=%.4f', conv_min, conv_max, conv_mean)

        logging.info('##TOTAL_CONV QRY1 = %.4f', total_convergence)
        stat.collect('qry1conv', total_convergence)

        # PUSH User's Total Convergence here
        self.catalog.rpush('boot:qry1:TOTAL', total_convergence)
        bench.mark('Bootstrap')

    # EXECUTE SAMPLER
        logging.debug("=======================  <DATA SAMPLER>  =========================")

        #  Exploration vs Exploitation SamplingUmbrella Sampling  (For exploration)
        #   or use weights as exploitation
        #  could also use convegence.....
        # Factor in convergence. Umbrella sampling, divided by convergence:
        #  Converence (approaches zero for more converged bins) will be more sampled
        #  Smaller bins will also be preferred in the sampling
        #  TODO: Add linear regression and pick 'trending' convergence bins
        hcube_pdf = []
        for k in hcube_list:
          if convergence[k] <= 0:
            logging.info('Detected that %s has totally converged (not sure if this is even possible)', k)
          else:
            prob = (1 - comb_wgt[k]) / convergence[k]
            hcube_pdf.append(prob)
            logging.info('### PROB %s  %.4f', k, prob)
        hcube_pdf = np.array(hcube_pdf)
        hcube_pdf /= np.sum(hcube_pdf)

       # TODO:  Number & Runtime of each job <--- Resource/Convergence Dependant
        numresources = self.data['numresources']
        sampled_distro_perbin = {k: 0 for k in hcube_list}
        selected_index_list = []
        coord_origin = []
        if len(hcube_list) == 0:
          logging.info('Incubating: Lacking sufficient Data to sample')
          rmslabel = [eval(x) for x in self.catalog.lrange('label:deshaw', 0, -1)]
          deshaw_samples = {b:[] for b in binlist}
          # Adjust sampling Size for index
          sample_factor = 4125000 // len(rmslabel)
          for i, b in enumerate(rmslabel):
            deshaw_samples[b].append(i*sample_factor)

          # Uniform sample while incubating
          itr = 0
          while numresources > 0:
            sample_bin = itr % len(binlist)
            index = np.random.choice(deshaw_samples[binlist[sample_bin]])
            logging.debug('SAMPLER: selecting DEShaw frame #%d from bin %s', index, str(binlist[sample_bin]))
            # Negation indicates an historical index number
            selected_index_list.append(-index)
            coord_origin.append(('deshaw', index, binlist[sample_bin]))
            numresources -= 1

        while numresources > 0:
          # First selection is biased using PDF from above
          selected_hcube = np.random.choice(hcube_list, p=hcube_pdf)
          sampled_distro_perbin[selected_hcube] += 1
          # Secondary Sampling is Uniform (OR Could Be Weighted again (see write up))
          selected_index = np.random.choice(list(hcube_A[selected_hcube]) + list(hcube_B[selected_hcube]))
          logging.debug('SAMPLER: selected sample #%d from hcube %s', selected_index, selected_hcube)
          selected_index_list.append(selected_index)
          coord_origin.append(('sim', selected_index, selected_hcube))
          numresources -= 1

      logging.info('All Indices sampled. Back projecting to high dim coords')
      sampled_set = []
      for i in selected_index_list:
        traj = self.backProjection([i])
        sampled_set.append(traj)
      bench.mark('Sampler')

    # Generate new starting positions
      runtime = self.catalog.get('runtime')
      if runtime is None or runtime == 0:
        logging.warning("RUNTIME is not set!")
        runtime = 500000
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

