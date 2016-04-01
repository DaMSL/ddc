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
      while batchAmt < maxbatchSize and len(workloadList) > 0:
        if int(workloadList[0]) + batchAmt > maxbatchSize:
          if batchAmt == 0:
            logging.warning('CAUTION. May need to set the max batch size higher (cannot run controller)')
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
        bench.mark('BP:LD:File:%s'%os.path.basename(fileno))

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
      logging.debug("=======================  <SUBSPACE CONVERENCE>  =========================")

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
          kpca = PCAKernel(None, 'rbf')
          kpca.solve(covar_pts)

        #  Check if the Kernel is old, but cap at 5K Pts (is this enough?)
        if min(5000, len(covar_pts)) > (kpca.trainsize * 1.2):
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
              dcdfreq = settings.DCDFREQ,           # Frame save rate
              interval = settings.DCDFREQ * settings.SIM_STEP_SIZE,                       
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



#  SOMEWHAT OLD


# #========================================

#     #  PREPROC
#       numLabels = self.data['numLabels']
#       totalBins = numLabels ** 2

#       logging.debug("Processing output from %d  simulations", len(job_list))

#       obs_delta = np.zeros(shape=(numLabels, numLabels))
      
#       #  Consolidate all observation delta's and transaction lists from recently run simulations
#       translist = {}
#       for job in job_list:
#         logging.debug("  Loading data for simulation: %s", job)

#         jkey = wrapKey('jc', job)

#         # Load Job Candidate params
#         params = self.catalog.load(jkey)

#         # Load the obsevation "delta" 
#         traj_delta  = self.catalog.loadNPArray(wrapKey('delta', job))
#         if traj_delta is not None:
#           obs_delta += traj_delta

#         # Load set of tranations actually visited for the current target bin
#         tbin = eval(params[jkey]['targetBin'])
#         trans = self.catalog.get(wrapKey('translist', job))
#         if trans is not None:
#           trans = pickle.loads(trans)
#           if tbin not in translist.keys():
#             translist[tbin] = []
#           for t in trans:
#             translist[tbin].append(t)

#       #  Update observation matrix
#       observe_before = np.copy(self.data['observe'])
#       self.data['observe'] += obs_delta
#       observe = self.data['observe']
#       launch = self.data['launch']

#     #  RUNTIME   -----------------------------------------
#       logging.info("Adapting Runtimes")
#       for tbin, tlist in translist.items():
#         A, B = tbin
#         time_s = 0
#         time_t = 0
#         num_t  = len(tlist)
#         for t in tlist:        
#           if t[0] == t[1]:
#             time_s += t[2]
#           else:
#             time_t += t[2]

#         run_step = 10000 * np.round(np.log( (num_t * max(1, time_t)) / time_s  ))
#         print("Runtime data for", A, B, ":  ", time_t, time_s, num_t, run_step)

#         currt = self.data['runtime'][A][B]
#         self.data['runtime'][A][B] = min(max(50000, currt+run_step), 500000) 

#         logging.info("Adapting runtime for (%d, %d)  from %7.0f  --->  %7.0f", 
#           A, B, currt, self.data['runtime'][A][B])

#     #  CONVERGENCE   -----------------------------------------
#       logging.debug("============================  <CONVERGENCE>  =============================")

 
#       #  TODO:  Consistency
#       # Weight Calculation: create a logistic function using current observation distribution

#       # Calc convergence on probability distro (before & after)
#       logging.debug("Calculating Probability Distributions for each state...")
#       probDistro_before = np.zeros(shape=(numLabels, numLabels))
#       probDistro        = np.zeros(shape=(numLabels, numLabels))
#       for n in range(numLabels):
#         numTrans_before = np.sum(observe_before[n]) - observe_before[n][n]
#         probDistro_before[n] = observe_before[n] / numTrans_before
#         probDistro_before[n][n] = 0

#         numTrans = np.sum(observe[n]) - observe[n][n]
#         probDistro[n] = observe[n] / numTrans
#         probDistro[n][n] = 0
#       delta        = np.zeros(shape=(numLabels, numLabels))
#       delta = abs(probDistro - probDistro_before)

#       globalconvergence = np.sum(abs(probDistro - probDistro_before))
#       globalconvergence_rate = globalconvergence / len(job_list)

#     #  WEIGHT CALC   -----------------------------------------
#       logging.debug("============================  <WEIGHT CALC>  =============================")

#       bins = [(x, y) for x in range(numLabels) for y in range(numLabels)]
#       logging.debug("Calculating transition rarity...")

#       # 1. Fatigue portion based on # times each bin was "launched"
#       quota = np.sum(launch) / totalBins
#         # Apply a `fatigue` factor; bins are fatigued if run more than their quota
#       fatigue = np.maximum( (quota-launch) / quota, np.zeros(shape=(numLabels, numLabels)))

#       # TODO:   Load New Transition Matrix  if consistency is necessary, otherwise use locally updated tmat

#       # 2. Calculate weight (note: follow 2 are retained for debugging only)
#       #  UPDATED:  Weights CALC as a factor of rare events & instability in delta calc
#       rarity = np.zeros(shape=(numLabels, numLabels))

#         #  Isolate rare events (s.t. rare event seen less than mean)
#       rareObserve = np.choose(np.where(observe.flatten() < np.mean(observe)), observe.flatten())

#         #  Est. mid point of logistic curve by accounting for distribution skew
#       midptObs = np.mean(observe) * skew(observe) + np.median(observe)

#         #  Create the function
#       rarityFunc = makeLogisticFunc(1, 1 / np.std(rareObserve), midptObs)

#       #  3. Apply constants. This can be user influenced
#       alpha = self.data['weight_alpha']
#       beta = self.data['weight_beta']

#       # fatigue = np.zeros(shape=(numLabels, numLabels))
#       logging.debug("Calculating control weights...")
#       weight = {}
#       quotasq = quota**2
#       for i in range(numLabels):
#         for j in range(numLabels):
          
#           # 4. Calculate rarity & weight incorporatin fatigue value
#           rarity[i][j] = rarityFunc(observe[i][j] - midptObs)

#           #  Old function: 
#           #       weight[(i,j)]  = alpha * rarity[i][j] + beta + delta[i][j]
#           #       weight[(i,j)] *= 0 if launch[i][j] > quota else (quota - launch[i][j])**2/quotasq

#           weight[(i,j)] =  alpha * rarity[i][j] + beta * fatigue[i][j]

#       #  5. Sort weights from high to low
#       updatedWeights = sorted(weight.items(), key=lambda x: x[1], reverse=True)

#       #  TODO:  Adjust # of iterations per bin based on weights by
#       #     replicating items in the updatedWeights list

#     #  SCHEDULING   -----------------------------------------
#       logging.debug("============================  <SCHEDULING>  =============================")

#       #  1. Load JC Queue and all items within to get respective weights and projected target bins
#       curqueue = []
#       logging.debug("Loading Current Queue of %d items", len(self.data['jcqueue']))
#       debug = True
#       configlist = self.catalog.load([wrapKey('jc', job) for job in self.data['jcqueue']])
#       for config in configlist.values():

#         # 2. Dampening Factor: proportional to its currency (if no ts, set it to 1)
#         jc_ts = config['timestep'] if 'timestep' in config else 1

#         #  TODO: May want to consider incl convergence of sys at time job was created
#         w_before    = config['weight']
#         config['weight'] = config['weight'] * (jc_ts / self.data['timestep'])
#         # logging.debug("Dampening Factor Applied (jc_ts = %d):   %0.5f  to  %05f", jc_ts, w_before, config['weight'])
#         curqueue.append(config)

#       #  3. (PreProcess current queue) for legacy JC's
#       logging.debug("Loaded %d items", len(curqueue))
#       for jc in range(len(curqueue)):
#         if 'weight' not in curqueue[jc]:
#           curqueue[jc]['weight'] = 1.0

#         if 'gc' not in curqueue[jc]:
#           curqueue[jc]['gc'] = 1

#       #  4. Sort current queue
#       if len(curqueue) > 0:
#         existingQueue = deque(sorted(curqueue, key=lambda x: x['weight'], reverse=True))
#         eqwlow = 0 if np.isnan(existingQueue[0]['weight']) else existingQueue[0]['weight']
#         eqwhi  = 0 if np.isnan(existingQueue[-1]['weight']) else existingQueue[-1]['weight']
#         logging.debug("Existing Queue has %d items between weights: %0.5f - %0.5f", len(existingQueue), eqwlow, eqwhi)
#       else:
#         existingQueue = deque()
#         logging.debug("Existing Queue is empty.")

#       #  5. Det. potential set of  new jobs  (base on launch policy)
#       #     TODO: Set up as multiple jobs per bin, cap on a per-control task basis, or just 1 per bin
#       potentialJobs = deque(updatedWeights)
#       logging.debug("Potential Job Queue has %d items between weights: %0.5f - %0.5f", len(potentialJobs), potentialJobs[0][1], potentialJobs[-1][1])

#       #  6. Prepare a new queue (for output)
#       jcqueue = deque()

#       targetBin = potentialJobs.popleft()
#       oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()
#       selectionTally = np.zeros(shape=(numLabels, numLabels))
#       newJobCandidate = {}

#       #  7. Rank order list of observed transition bins by rare observations for each state (see below)
#       rarityorderstate = np.argsort(observe.sum(axis=1))
#       rarityordertrans = np.zeros(shape=(numLabels,numLabels))
#       for i in range(numLabels):
#         np.copyto(rarityordertrans[i], np.argsort(observe[i]))

#       #  8. Continually pop new/old jobs until max queue size is attained   
#       while len(jcqueue) < DEFAULT.MAX_JOBS_IN_QUEUE:

#         #  8a:  No more jobs
#         if oldjob == None and targetBin == None:
#           logging.info("No more jobs to queue.")
#           break

#         #  8b:  Push an old job
#         if (targetBin == None) or (oldjob and oldjob['weight'] > targetBin[1]) or (oldjob and np.isnan(targetBin[1])):
#           jcqueue.append(oldjob['name'])
#           oldjob['weight'] = 0 if np.isnan(oldjob['weight']) else oldjob['weight']
#           logging.debug("Re-Queuing OLD JOB `%s`   weight= %0.5f", oldjob['name'], oldjob['weight'])
#           oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()

#         #  8c:  Push a new job  (New Job Selection Algorithm)
#         else:

#           #  Job should "start" in is targetBin of (A, B)
#           A, B = targetBin[0]
#           logging.debug("\n\nCONTROL: Target transition bin  %s  (new job #%d,  weight=%0.5f)", str((A, B)), len(newJobCandidate), targetBin[1])

#           # (1)  Start with candidated in the Target Bin's candidate pool
#           cpool = kv2DArray.key('candidatePool', A, B)
#           selectedbin = (A,B)

#           # (2)  Flip direction of transition (if no candidates in that targetbin)
#           if self.catalog.llen(cpool) == 0:
#             logging.info("No DEShaw reference for transition, (%d, %d)  -- checking reverse direction", A, B)
#             cpool = kv2DArray.key('candidatePool', B, A)
#             selectedbin = (B,A)

#           # (3)  Iteratively find another candidate pool from sorted "rare obsevation" list <-- This should find at least 1
#           if self.catalog.llen(cpool) == 0:
#             logging.info("No DEShaw reference for transition, (%d, %d)  -- checking wells in this order: %s", B, A, str(rarityorderstate))
#             for RS in rarityorderstate:
#               cpool = kv2DArray.key('candidatePool', RS, RS)
#               if self.catalog.llen(cpool) == 0:
#                 logging.info("No DEShaw reference for transition, (%d, %d) ", RS, RS)

#               else: 
#                 logging.info("FOUND DEShaw start point from transition, (%d, %d) ", RS, RS)
#                 selectedbin = (RS,RS)
#                 break

#           logging.debug('Final Candidate Popped from Pool %s  of  %d  candidates', cpool, self.catalog.llen(cpool))

#           # (4). Cycle this candidate to back of candidate pool list
#           candidate = self.catalog.lpop(cpool)
#           sourceTraj, srcFrame = candidate.split(':')
#           dstring = "####SELECT_TRAJ@ %s @ %s @ %s @ %s" % (str(targetBin[0]), str(selectedbin), sourceTraj, srcFrame)
#           self.catalog.rpush(cpool, candidate)

#           # (5). Back Projection Function (using newly analzed data to identify start point of next simulation)
#           # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
#           #   or we have a pre-fetch algorithm to get the data
#           # Back-project  <--- Move to separate Function tied to decision history
#           #  Start coordinates are either a DeShaw file (indexed by number) or a generated one
#           if isinstance(sourceTraj, int) or sourceTraj.isdigit():      # It's a DEShaw file
#             pdbfile, archiveFile = getHistoricalTrajectory(sourceTraj)
#           else:
#             archiveFile = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.dcd' % sourceTraj)
#             pdbfile     = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.pdb' % sourceTraj)

#           # (6). Generate new set of params/coords
#           jcID, params = generateNewJC(archiveFile, pdbfile, DEFAULT.TOPO, DEFAULT.PARM, int(srcFrame), debugstring=dstring)

#     #  POST-PROC
#       logging.debug("OBS_MATRIX_DELTA:\n" + str(obs_delta))
#       logging.debug("OBS_MATRIX_UPDATED:\n" + str(observe))
#       logging.debug("LAUNCH_MATRIX:\n" + str(launch))
#       logging.debug("RUNTIMES: \n%s", str(self.data['runtime']))
#       logging.debug("CONVERGENCE_PROB_DISTRO:\n" + str(probDistro))
#       logging.debug("OBS_RARITY:\n" + str(rarity))
#       logging.debug("CONV_DELTA:\n" + str(delta))
#       # logging.debug("CTL_WEIGHT:\n" + str(np.array([[weight[(i,k)] for i in range(numLabels)] or k in range(numLabels)])))

#       logging.info("GLOBAL_CONVERGENCE: %f", globalconvergence)
#       for i in range(5):
#         logging.info("STATE_CONVERGENCE for %d: %f", i, np.sum(delta[i]))

#       logging.info("GLOBAL_RATE_CONV: %f", globalconvergence_rate)
#       for i in range(5):
#         logging.info("STATE_RATE_CONV for %d: %f", i, np.sum(delta[i])/len(job_list))
         




# REAL  OLD COVERGENCE:
      # logging.debug("Observations MAT:\n" + str(tmat))
      # logging.debug("Fatigue:\n" + str(fatigue))

      # # Load Selection Matrix
      # smat = loadNPArray(self.catalog, 'selectionmatrix')
      # if smat is None:
      #   smat = np.full((5,5), 1.)    # SEED Selection matrix (it cannot be 0) TODO: Move to init

      # # Merge Update global selection matrix
      # smat += selectionTally
      # logging.debug("SMAT:\n" + str(smat))
      # # Load Convergence Matrix
      # cmat = loadNPArray(self.catalog, 'convergencematrix')
      # if cmat is None:
      #   cmat = np.full((5,5), 0.04)    # TODO: Move to init
      # logging.debug("CMAT:\n" + str(cmat))

      # # Remove bias from selection and add to observations 
      # #  which gives the unbiased, uniform selection 
      # inverseSelectFunc = lambda x: (np.max(x)) - x

      # #  For now, Assume a 4:1 ratio (input:output) and do not factor in output prob distro
      # unbias = 3 * inverseSelectFunc(smat) + tmat
      # logging.debug("UNBIAS:\n" + str(unbias))

      # # Calculcate "convergence" matrix as unbias(i,j) / sel(i.j)
      # updated_cmat = unbias / np.sum(unbias)
      # logging.debug("CMAT_0:\n" + str(cmat))
      # logging.debug("CMAT_1:\n" + str(updated_cmat))


      # # Calculate global convergence as summed difference:  cmat_t1 - cmat_t0
      # convergence = np.sum(abs(updated_cmat - cmat))


    # def oldcode():
    #   pass
      # TODO:  Treat Archive as an overlay service. For now, wrap inside here and connect to it


      # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
      # redis_storage = RedisStorage(archive)
      # config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
      # if not config:
      #   logging.error("LSHash not configured")
      #   #TODO: Gracefully exit
      #   return []

      # # Create empty lshash and load stored hash
      # lshash = DEFAULT.getEmptyHash()
      # lshash.apply_config(config)
      # indexSize = self.data['num_var'] * self.data['num_pc']
      # logging.debug("INDEX SIZE = %d:  ", indexSize)
      # engine = nearpy.Engine(indexSize, 
      #       lshashes=[lshash], 
      #       storage=redis_storage)

      # Load current set of known states
      #  TODO:  Injection Point for clustering. If clustering were applied
      #    this would be much more dynamic (static fileload for now)
      # labels = loadLabels()
      # labelNames = getLabelList(labels)
      # numLabels = len(labelNames)

 
      # PROBE   ------------------------
      # logging.debug("============================  <PROBE>  =============================")

      # Set initial params for index calculations
      # prevState = -1    # To track each state transition
      # prevTrajectory = None   # To check for unique trajectories
      # decisionHistory = {}    # Holds Decision History data from source JC used to create the data
      # observationDistribution = {}   #  distribution of observed states (for each input trajectory)
      # observationCount = {}
      # statdata = {}
      # newvectors = []
      # obs_delta = np.zeros(shape=(numLabels, numLabels))


      # ### UPDATED USING RMS as probe function


      # for sourceJC in rms_list.keys():
      #   config = {}
      #   self.catalog.load({wrapKey('jc', sourceJC): config})  #TODO: Load these up front
      #   for k, v in config.items():
      #     logging.debug("  %s:  %s", k, str(v))
      #   decisionHistory[sourceJC] = config
      #   if 'state' not in config:
      #     logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
      #     prevState = None  # Do not count transition into first probed index
      #   else:
      #     logging.debug("New Index to analyze, %s: Source JC was previously in state %d", sourceJC, prevState)
      #     prevState = config['state']

      #   sourceJCKey = wrapKey('jc', sourceJC)
      #   self.addToState(sourceJCKey, config)
      #   # statdata[sourceJC] = {}


      #   # Analyze Each conform (Should this move to anlmd execute)
      #   logging.info('Checking RMS for each conform of %d length trajectory', len(rms_list[sourceJC]))
      #   for num, rms in enumerate(rms_list[sourceJC]):

      #     #  Check proximity to nearest 2 centroids
      #     #  TODO: Account for proximity to more than 2 (may be "interesting" data point)



      #   state = A
      #   labeledBin = (A, B)
      #   statdata[sourceJC]['bin'] = str(labeledBin)
      #   statdata[sourceJC]['count'] = count.tolist()
      #   statdata[sourceJC]['clust'] = clust.tolist()


      #     # Increment the transition matrix  
      #     #  (NOTE: SHOULD WE IGNORE 1ST INDEX IN A TRAJECTORY, since it doesn't describe a transition?)
      #     # if prevState == None:
      #     #   prevState = state
      #     logging.debug("    Index classified in bin:  (%d, %d)", A, B)
      #     if prevState is None:
      #       logging.debug("    First Index in Traj")
      #     else:
      #       logging.debug("    Transition (%d, %d)", prevState, state)
      #     logging.debug("      Clustered at: %s", str(clust/np.sum(clust)))
      #     logging.debug("      Counts:       %s", str(count))
      #     logging.debug("      NN Index:     %s", neigh[0][1])

      #     # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
      #     prevState = state






      # NOTE: ld_index is a list of indexed trajectories. They may or may NOT be from
      #   the same simulation (allows for grouping of multiple downstream data into 
      #   one control task). Thus, this algorithm tracks subsequent windows within one
      #   trajectories IOT track state changes. 

      #  Theta is calculated as the probability of staying in 1 state (from observed data)
      # theta = .6  #self.data['observation_counts'][1] / sum(self.data['observation_counts'])
      # logging.debug("  THETA  = %0.3f   (static for now)", theta)

      # for key in sorted(ld_index.keys()):

      #   index = np.array(ld_index[key])   # Get actual Index for this window
      #   sourceJC, frame = key.split(':')  # Assume colon separator
      #   # logging.info(' Index Loaded from %s:   Shape=%s,  Type=%s', sourceJC, str(index.shape), str(index.dtype))

      #   # Get Decision History for the index IF its a new index not previously processed
      #   #  and initialize observation distribution to zeros
      #   if sourceJC != prevTrajectory:
      #     observationDistribution[sourceJC] = np.zeros(5)
      #     observationCount[sourceJC] = np.zeros(5)

      #     #  TODO: Load indices up front with source history
      #     config = {}
      #     self.catalog.load({wrapKey('jc', sourceJC): config})  #TODO: Load these up front
      #     for k, v in config.items():
      #       logging.debug("  %s:  %s", k, str(v))
      #     decisionHistory[sourceJC] = config
      #     if 'state' not in config:
      #       logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
      #     else:
      #       logging.debug("New Index to analyze, %s: Source JC was previously in state %d", sourceJC, prevState)

      #     prevState = None  # Do not count transition into first probed index

      #     sourceJCKey = wrapKey('jc', sourceJC)
      #     self.addToState(sourceJCKey, config)
      #     prevTrajectory = sourceJC
      #     statdata[sourceJC] = {}

      #     # Note:  Other Decision History is loaded here

      #   logging.info("  Probing `%s` window at frame # %s  (state %s)", sourceJC, frame, str(prevState))
      #   statdata[sourceJC][frame] = {}

      #   # Probe historical index
      #   neigh = engine.neighbours(index)
      #   if len(neigh) == 0:
      #     logging.info ("    Found no near neighbors for %s", key)
      #   else:
      #     logging.debug ("    Found %d neighbours:", len(neigh))

      #     #  Track the weighted count (a.k.a. cluster) for this index's nearest neighbors
      #     clust = np.zeros(5)
      #     count = np.zeros(5)
      #     for n in neigh:
      #       nnkey = n[1]
      #       distance = n[2]
      #       trajectory, seqNum = nnkey[2:].split('-')

      #       # CHANGED: Grab state direct from label (assume state is 1 char, for now)
      #       nn_state = int(nnkey[0])  #labels[int(trajectory)].state
      #       # logging.info ("    NN:  %s   dist = %f    state=%d", nnkey, distance, nn_state)
      #       count[nn_state] += 1
      #       clust[nn_state] += abs(1/distance)

      #       # Add total to the original input trajectory (for historical and feedback decisions)
      #       observationCount[sourceJC][nn_state] += 1
      #       observationDistribution[sourceJC][nn_state] += abs(1/distance)

      #     # Classify this index with a label based on the highest weighted observed label among neighbors
      #     clust = clust/np.sum(clust)
      #     order = np.argsort(clust)[::-1]
      #     A = order[0]
      #     B = A if clust[A] > theta else order[1]
      #     obs_delta[A][B] += 1

      #     state = A
      #     labeledBin = (A, B)
      #     statdata[sourceJC][frame]['state'] = str(state)
      #     statdata[sourceJC][frame]['bin'] = str(labeledBin)
      #     statdata[sourceJC][frame]['count'] = count.tolist()
      #     statdata[sourceJC][frame]['clust'] = clust.tolist()


      #     # Increment the transition matrix  
      #     #  (NOTE: SHOULD WE IGNORE 1ST INDEX IN A TRAJECTORY, since it doesn't describe a transition?)
      #     # if prevState == None:
      #     #   prevState = state
      #     logging.debug("    Index classified in bin:  (%d, %d)", A, B)
      #     if prevState is None:
      #       logging.debug("    First Index in Traj")
      #     else:
      #       logging.debug("    Transition (%d, %d)", prevState, state)
      #     logging.debug("      Clustered at: %s", str(clust/np.sum(clust)))
      #     logging.debug("      Counts:       %s", str(count))
      #     logging.debug("      NN Index:     %s", neigh[0][1])

      #     # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
      #     prevState = state


      # Build Decision History Data for the Source JC's from the indices
      # transitionBins = kv2DArray(archive, 'transitionBins', mag=5, dtype=str, init=[])      # Should this load here (from archive) or with other state data from catalog?
                # BUILD ARCHIVE ONLINE:
      #     if DEFAULT.BUILD_ARCHIVE:
      #       label = '%d %s-%s' % (state, sourceJC, frame)
      #       newvectors.append((index, label))

      # if DEFAULT.BUILD_ARCHIVE:
      #   for vect in newvectors:
      #     logging.info("ADDING new Index to the Archive:  %s", vect[1])
      #     engine.store_vector(vect[0], vect[1])
      # else:
      #   logging.debug('BUILD Archive is off (not storing)')

      #  DECISION HISTORY  --------------------------
      # logging.debug("============================  <DECISION HIST>  =============================")


      #  Process output data for each unque input trajectory (as produced by a single simulation)
      # launch_delta = np.zeros(shape=(numLabels, numLabels))
      # for sourceJC, cluster in observationDistribution.items():
      #   logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point here for better classification and/or move to analysis)", sourceJC)
      #   if sum(observationDistribution[sourceJC]) == 0:
      #     logging.debug(" No observed data for, %s", sourceJC)
      #     continue

      #   #  TODO: Another injection point for better classification. Here, the classification is for the input trajectory
      #   #     as a whole for future job candidate selection. 

      #   logging.debug("  Observations:      %s", str(observationCount[sourceJC]))
      #   logging.debug("  Cluster weights:   %s", str(cluster/np.sum(cluster)))
      #   index_distro = cluster / sum(cluster)
      #   # logging.debug("Observations for input index, %s\n  %s", sourceJC, str(distro))
      #   sortedLabels = np.argsort(index_distro)[::-1]    # Should this be normaized to check theta????? or is theta a global calc?
      #   statelist = [int(statdata[sourceJC][f]['state']) for f in sorted(statdata[sourceJC].keys())]

      #   stateA = statelist[0]
      #           # idxcount = np.zeros(len(numLabels))
      #   stateB = None
      #   transcount = 0
      #   for n in statelist[1:]:
      #     transcount += 1
      #     if stateA != n:
      #       stateB = n
      #       logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", sourceJC, stateA, stateB)
      #       break
      #   if stateB is None:
      #     logging.debug(" Trajectory `%s`  classified as staying in state :  %d", sourceJC, stateA)
      #     stateB = stateA
      #     # idxcount[n] += 1
      #   # sortedLabels = np.argsort(idxcount)[::-1]    # Should this be normaized to check theta????? or is theta a global calc?

      #   # stateA = sortedLabels[0]
      #   # stateB = sortedLabels[1] if idxcount[sortedLabels[1]] > 0 else stateA


      #   # Source Trajectory spent most of its time in 1 state
      #   # if max(index_distro) > theta: 
      #   #   logging.debug(" Trajectory `%s`  classified as staying in state :  %d", sourceJC, stateA)
      #   #   stateB = stateA

      #   # # Observation showed some transition 
      #   # else:
      #   #   stateB = sortedLabels[1]
      #   #   logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", sourceJC, stateA, stateB)

      #   #  Add this candidate to list of potential pools (FIFO: popleft if pool at max size):
      #   if len(self.data[candidPoolKey(stateA, stateB)]) >= DEFAULT.CANDIDATE_POOL_SIZE:
      #     del self.data[candidPoolKey(stateA, stateB)][0] 
      #   logging.info("BUILD_CANDID_POOL: Added `%s` to candidate pool (%d, %d)", sourceJC, stateA, stateB)
      #   self.data[candidPoolKey(stateA, stateB)].append(sourceJC)

      #   outputBin = str((stateA, stateB))
      #   if 'targetBin' in self.data[wrapKey('jc', sourceJC)]:
      #     inputBin  = self.data[wrapKey('jc', sourceJC)]['targetBin']
      #     a, b = eval(inputBin)

      #     #  INCREMENT "LAUNCH" Counter
      #     launch_delta[a][b] += 1

      #     logging.info("  Original Target Bin was       :  %s", inputBin) 
      #     logging.info("  Actual Trajectory classified  :  %s", outputBin) 
      #     logging.debug("    TODO: ID difference, update ML weights, provonance, etc..... (if desired)")

      #   # Update other History Data
      #   key = wrapKey('jc', sourceJC)
      #   self.data[key]['actualBin'] = outputBin
      #   self.data[key]['application'] = DEFAULT.APPL_LABEL
      #   logging.debug("%s", str(statdata[sourceJC]))
      #   self.data[key]['indexList'] = json.dumps(statdata[sourceJC])

      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's
      #  Everything above  could actually co-locate with the analysis thread whereby the output of the analysis
      #  is a label or set of labels (vice a low-dim index). 


      #  WEIGHT CALCULATION ---------------------------

      # Increment the timestep




# CACHE CODE
      # Archive File retrieval for all cache-miss points
      # TODO:  May need to deference to local RMS Subspace first, then Back to high-dim
      # bench.mark('Cache:Hit')
      # pipe = self.catalog.pipeline()
      # for pt in cache_miss:
      #   pipe.lindex('xid:reference', pt)
      # framelist = pipe.execute()
      # bench.mark('LD:Redis:xidlist')
      # for i in range(len(framelist)):
      #   if framelist[i] is None:
      #     dcdfile_num = cache_miss[i] // 100
      #     frame_num = (cache_miss[i] - 100*dcdfile_num) * 10
      #     framelist[i] = str((dcdfile_num, frame_num))
          
      # ID unique files and Build index filter for each unique file
      #  Account for DEShaw Files (derived from index if index not in catalog)
      # frameMask = {}
      # for i, idx in enumerate(framelist):
      #   # An index < 0 indicates this was a pre-loaded/pre-labeled dataset
      #   if idx is None:
      #     is_deshaw = True
      #   else:
      #     file_index, frame = eval(idx)
      #     is_deshaw = (file_index < 0)
      #   if is_deshaw:
      #       dcdfile_num = frame // 1000
      #       frame = frame - 1000*dcdfile_num
      #       # Use negation to indicate DEShaw file (not in fileindex in catalog)
      #       file_index = (-1 * dcdfile_num)
      #   if file_index not in frameMask:
      #     frameMask[file_index] = []
      #   frameMask[file_index].append(frame)
      # logging.debug('Back projecting %s points from %d different files.', 
      #   len(framelist), len(frameMask))
      # bench.mark('GroupBy:Files')

      # # Get List of files
      # filelist = {}
      # for file_index in frameMask.keys():
      #   if file_index >= 0:
      #     filelist[file_index] = self.catalog.lindex('xid:filelist', file_index)
      #   else:
      #     filelist[file_index] = deshaw.getDEShawfilename(-1 * file_index)
      # print ('filelist --> ', filelist)