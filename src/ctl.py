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
from mdtools.simtool import generateNewJC
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

np.set_printoptions(precision=5, suppress=True)


def makeLogisticFunc (maxval, steep, midpt):
  return lambda x: maxval / (1 + np.exp(-steep * (midpt - x)))

skew = lambda x: (np.mean(x) - np.median(x)) / np.std(x)


def bootstrap (source, samplesize=.1, N=50, interval=.95):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo

  # Get unique label/category/hcube ID's
  V = set()
  for i in source:
    V.add(i)
  # print ("BS: labels ", str(V))

  #  EXPERIMENT #1, 4+:  SAMPLE SIZE WAS 10%
  L = round(len(source) * samplesize)
  #  EXPERIMENT #2&3:  FIXED SAMPLE SIZE OF 100K (or less)
  # L = min(len(source), 100000)

  # Calculate mu_hat from bootstrap -- REMOVED
  # mu_hat = {}
  # groupby = {v_i: 0 for v_i in V}
  # for s in source:
  #   groupby[s] += 1
  # for v_i in V:
  #   mu_hat[v_i] = groupby[v_i]/L

  # Iterate for each bootstrap and generate statistical distributions
  boot = {i : [] for i in V}
  for i in range(N):
    strap   = [source[np.random.randint(len(source))] for n in range(L)]
    groupby = {v_i: 0 for v_i in V}
    for s in strap:
      groupby[s] += 1
    for v_i in V:
      boot[v_i].append(groupby[v_i]/L)
  probility_est = {}
  for v_i in V:
    P_i = np.mean(boot[v_i])
    delta  = np.array(sorted(boot[v_i]))  #CHECK IF mu or P  HERE
    ciLO = delta[round(N*ci_lo)]
    ciHI = delta[math.floor(N*ci_hi)]
    probility_est[v_i] = (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i)
  return probility_est


def bootstrap_by_state (source, samplesize=.1, N=50, interval=.95):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo

  # Get unique label/category/hcube ID's
  V = set()
  for i in source:
    V.add(i)

  #  EXPERIMENT #1, 4+:  SAMPLE SIZE WAS 10%
  L = round(len(source) * samplesize)
  #  EXPERIMENT #2&3:  FIXED SAMPLE SIZE OF 100K (or less)
  # L = min(len(source), 100000)

  # Iterate for each bootstrap and generate statistical distributions
  boot = {i : [] for i in V}
  for i in range(N):
    strap   = [source[np.random.randint(len(source))] for n in range(L)]
    groupby = {v_i: 0 for v_i in V}
    totals = [0 for i in range(5)]
    for s in strap:
      groupby[s] += 1
    for v_i in V:
      A, B = eval(v_i)
      if A != B:
        totals[A] += groupby[v_i]
    for v_i in V:
      A, B = eval(v_i)
      if A != B:  
        boot[v_i].append(groupby[v_i]/totals[A])
  probility_est = {}
  for v_i in V:
    A, B = eval(v_i)
    if A == B:
      probility_est[v_i] = (0, 0, 0, 0.)
    else:
      P_i = np.mean(boot[v_i])
      delta  = np.array(sorted(boot[v_i]))  #CHECK IF mu or P  HERE
      ciLO = delta[round(N*ci_lo)]
      ciHI = delta[math.floor(N*ci_hi)]
      probility_est[v_i] = (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i)
  return probility_est


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
      minbatchSize = self.data['ctlBatchSize_min']
      maxbatchSize = self.data['ctlBatchSize_max']
      batchAmt = 0
      logging.debug('Controller will launch with batch size between: %d and %d observations', minbatchSize, maxbatchSize)
      num_simbatchs = 0
      while batchAmt < maxbatchSize and len(workloadList) > 0:
        if workloadList[0] + batchAmt > maxbatchSize:
          if batchAmt == 0:
            logging.warning('CAUTION. May need to set the max batch size higher (cannot run controller)')
          break
        batchAmt += int(workloadList.popleft())
        num_simbatchs += 1
        logging.debug('  Batch amount up to: %d', batchAmt)

      # Don't do anything if the batch size is less than the min:
      if batchAmt < minbatchSize:
        return [], self.data['completesim']
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
      bench = microbench()
      source_points = []
      cache_miss = []

      self.trajlist_async = deque()
      
      # DEShaw topology is assumed here
      bench.start()

      # Derefernce indices to file, frame tuple:
      historical_framelist = []
      pipe = self.catalog.pipeline()
      for idx in index_list:
        logging.debug('[BP] Dereferencing Index: %s', str(idx))
        # Negation indicates  historical index:
        index = int(idx)
        if index < 0:
          file_index, frame = deshaw.refFromIndex(-idx)
          historical_framelist.append((file_index, frame))
          logging.debug('[BP] DEShaw:  file #%d,   frame#%d', file_index, frame)
        else:
          pipe.lindex('xid:reference', index)

      # Load higher dim point indices from catalog
      generated_framelist = pipe.execute()
      for i in generated_framelist:
        logging.debug('[BP] De-Referenced Gen Frame: %s', str(i))

      bench.mark('LD:Redis:xidlist')


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
      bench.mark('GroupBy:Files')

      for k, v in generated_frameMask.items():
        logging.debug('[BP] GenData lookups: %s, %s', str(k), str(v))

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
        bench.mark('LD:File:%s'%fileno)

      logging.debug('All Uncached Data collected Total # points = %d', len(source_points_uncached))
      source_traj_uncached = md.Trajectory(np.array(source_points_uncached), ref.top)
      bench.mark('Build:Traj')
      bench.show()

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
      bench = microbench()

    # PRE-PROCESSING ---------------------------------------------------------------------------------
      logging.debug("============================  <PRE-PROCESS>  =============================")

      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

      settings = systemsettings()
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
      
      self.data['ctlIndexHead'] = thru_index

      logging.debug('##NUM_RMS_THIS_ROUND: %d', len(labeled_pts_rms))
      self.catalog.rpush('datacount', len(labeled_pts_rms))
      # varest_counts_rms = {}
      # total = 0
      # for i in labeled_pts_rms:
      #   total += 1
      #   if i not in varest_counts_rms:
      #     varest_counts_rms[i] = 0
      #   varest_counts_rms[i] += 1

      # Load PCA Subspace of hypecubes  (for read only)
      hcube_mapping = json.loads(self.catalog.get('hcube:pca'))
      logging.debug('  # Loaded PCA keys = %d', len(hcube_mapping.keys()))
      bench.mark('LD:Hcubes:%d' % len(hcube_mapping.keys()))
      # TODO: accessor function is for 1 point (i) and 1 axis (j). 
      #  Optimize by changing to pipeline  retrieval for all points given 
      #  a list of indices with an axis
      # LOAD in enture PCA Subspace

      #  TODO: Approximiation HERE <<--------------
      # func = lambda i,j: np.fromstring(self.catalog.lindex('subspace:pca', i))[j]
      packed_subspace = self.catalog.lrange('subspace:pca', 0, -1)
      pca_subspace = np.zeros(shape=(len(packed_subspace), settings.PCA_NUMPC))
      for x in packed_subspace:
        pca_subspace = np.fromstring(x, dtype=np.float32, count=settings.PCA_NUMPC)

      bench.mark('LD:pcasubspace:%d' % len(pca_subspace))
      logging.debug("Reconstructing the tree... (%d pca pts)", len(pca_subspace))
      hcube_tree = KDTree.reconstruct(hcube_mapping, pca_subspace)
      bench.mark('KDTreee_build')

    # Calculate variable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERENCE>  =========================")

      # Bootstrap current sample for RMS
      logging.info("RMS Labels for %d points loaded. Bootstrapping.....", len(labeled_pts_rms))

      pdf_rms = datacalc.posterior_prob(labeled_pts_rms)

      logging.info("Posterer PDF for each variable (for this Set of Data):")
      pipe = self.catalog.pipeline()
      for v_i in binlist:
        key = str(v_i)
        A, B = eval(key)
        val = 0 if key not in pdf_rms.keys() else pdf_rms[key]
        pipe.rpush('boot:%d_%d' % (A, B), val)
        logging.info('##VAL %03d %s:  %0.4f' % (self.data['timestep'], key, val))
      pipe.execute()
      bench.mark('PosteriorPDF:RMS')

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")
      #   Using RMS and PCA, umbrella sample transitions out of state 3

      EXPERIMENT_NUMBER = 4

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

      #  REWEIGHT OPERATION
      reweight = False
      # Back-project all points to higher dimension <- as consolidatd trajectory
      if reweight:
        source_traj = self.backProjection(rms_indexlist)
        traj = datareduce.filter_heavy(source_traj)
        logging.debug('(BACK)PROJECT RMS points in HD space: %s', str(traj))
        bench.mark('BackProject:RMS_To_HD')

        # 2. project into PCA space 
        rms_proj_pca = datareduce.PCA(traj.xyz, self.data['pcaVectors'], 3)
        logging.debug('Projects to PCA:  %s', str(rms_proj_pca.shape))
        bench.mark('Project:RMS_To_PCA')

        # 3. Map into existing hcubes  (project only for now)
        #  A -> PCA  and B -> RMS
        #   TODO:  Insert here and deal with collapsing geometric hcubes in KDTree
        # Initiaze the set of "projected" hcubes in B
        hcube_B = {}
        # Project every selected point into PCA and then group by hcube
        for i in range(len(rms_proj_pca)):
          hcube = hcube_tree.project(rms_proj_pca[i], maxdepth=8)
          # print('POJECTING ', i, '  ', rms_proj_pca[i], '  ---->  ', hcube)
          if hcube not in hcube_B:
            hcube_B[hcube] = []
          hcube_B[hcube].append(i)
        # Gather the hcube stats for wgt calc
        hcube_sizes = [len(k) for k in hcube_B.keys()]
        low_dim = max(hcube_sizes) + 1
        total = sum(hcube_sizes)
        # calc Wght
        wgt_B = {k: len(hcube_B[k])/(total*(low_dim - len(k))) for k in hcube_B.keys()}

        logging.debug('Projected %d points into PCA. Found the following keys:', len(rms_proj_pca))

        # Get all original HCubes from A for "overlapping" hcubes
        hcube_A = {}
        for k in hcube_B.keys():
          hcube_A[k] = hcube_tree.retrieve(k)
        hcube_sizes = [len(k) for k in hcube_A.keys()]
        low_dim = max(hcube_sizes) + 1
        total = sum(hcube_sizes)
        wgt_A = {k: len(hcube_A[k])/(total*(low_dim - len(k))) for k in hcube_B.keys()}

        #  GAMMA FUNCTION EXPR # 1 & 2
        # gamma = lambda a, b : a * b

        #  GAMMA FUNCTION EXPR # 3
        gamma = lambda a, b : (a + b) / 2

        # TODO: Factor in RMS weight
        comb_wgt = {k: gamma(wgt_A[k], wgt_B[k]) for k in hcube_B.keys()}
        total = sum(comb_wgt.values())
        bench.mark('GammaFunc')

        # Umbrella Sampling
        umbrella = OrderedDict()
        for k, v in comb_wgt.items():
          umbrella[k] = (v/total) 
          logging.debug('   %20s ->  %4d A-pts (w=%0.3f)  %4d B-pts (w=%0.3f)     (GAMMAwgt=%0.3f)  (ubrellaW=%0.3f)', 
            k, len(hcube_A[k]), wgt_A[k], len(hcube_B[k]), wgt_B[k], comb_wgt[k], umbrella[k])
        keys = umbrella.keys()
        candidates = np.random.choice(list(umbrella.keys()), 
             size=100, replace=True, p = list(umbrella.values()))
    
    # EXECUTE SAMPLER
      logging.debug("=======================  <DATA SAMPLER>  =========================")

      # 1st Selection level: pick a HCube
      # Select N=20 new candidates
      #  TODO:  Number & Runtime of each job <--- Resource/Convergence Dependant
      # numresources = self.data['numresources']

      # print ('CANDIDATE HCUBES: ', candidates)
      # # 2nd Selection Level: pick an index for each chosen cube (uniform)
      # sampled_set = []
      # for i in candidates:
      #   selected_index = np.random.choice(list(hcube_A[i]) + list(hcube_B[i]))
      #   logging.debug('BACK-PROJECT HighDim Index # %d', selected_index)
      #   traj = self.backProjection([selected_index])
      #   sampled_set.append(traj)
      # bench.mark('Sampler')


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