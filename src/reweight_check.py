#!/usr/bin/env python

import argparse
import sys
import os
import sys
import math
import json
import redis
import itertools
from random import choice, randint
from collections import namedtuple, deque, OrderedDict
from threading import Thread

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *
from core.kdtree import KDTree
import datatools.datareduce as datareduce
import datatools.datacalc as datacalc
import mdtools.deshaw as deshaw
import datatools.kmeans as KM
# from datatools.pca import calc_kpca, calc_pca, project_pca
from datatools.pca import PCAnalyzer, PCAKernel, PCAIncremental
from datatools.approx import ReservoirSample
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

# views:  mapping tuple => list
def collapse_join(origin_keys, views, height=2):
  THETA = 2
  print('\nCOLLAPSE JOIN at LEVEL ', height)
  print('Source Views: ')
  print(views)
  if height == len(origin_keys) or len(views) <= 1:
    return []
  domain_list = [set(c) for c in (itertools.combinations(origin_keys, height))]
  print('Domain List: ',domain_list)
  domain_candidates = {}
  for source  in views.keys():
    print('  Checking source: ', source)
    subset = set(source)
    for domain in domain_list:
      print('  Checking ', subset, ' vs ', domain)
      if subset < domain:
        # map key <--- set(c)
        d_key = tuple(domain)
        if d_key not in domain_candidates.keys():
          domain_candidates[d_key] = set()
        domain_candidates[d_key].add(source)
  print('Domain Candidates: ')
  for k, v in domain_candidates.items():
    print('  ', k, '    ', len(v))
  merge_candidates = {}
  for domain, sources in domain_candidates.items():
    # re-map set(c) <-- key
    if len(sources) <= 1:
      continue
    # Additional Pruning here
    # Query Optimization Here (to ID optimal L & R sources)
    source_list = list(sources)
    merge_candidates[domain] = tuple(source_list[:2])
  print('Merge Candidates: ')
  for k, v in merge_candidates.items():
    print('  ', k, '    ', v)
  target_views = {}
  for domain, pair in merge_candidates.items():
    target_setlist = []
    # Perform 2-way join
    print('2-Way Join on: ', pair)
    L, R = tuple(pair)
    for idxL, valL in views[L]:
      for idxR, valR in views[R]:
        # Final Pruning Here
        idx_set = idxL | idxR
        intersection = valL & valR
        if len(intersection) > THETA:
          target_setlist.append((idx_set, intersection))
    if len(target_setlist) > 0:
      superset = tuple(set(L) | set(R))
      target_views[superset] = target_setlist
  return target_views.update(collapse_join(origin_keys, target_views, height+1))

class reweightJob(object):
    def __init__(self, name, host='localhost', port=6385):

      self.name = name
      self.calc_covar = False

      # For stat tracking
      self.cache_hit = 0
      self.cache_miss = 0
      self.catalog = redis.StrictRedis(host=host, port=port, decode_responses=True)

      FILELOG = False
      if FILELOG:
        self.filelog = logging.getLogger(name)
        self.filelog.setLevel(logging.INFO)
        filename = 'ctl.log'
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(message)s')
        fh.setFormatter(fmt)
        self.filelog.addHandler(fh)
        self.filelog.propagate = False
        logging.info("File Logging is enabled. Logging some output to: %s", filename)
      else:
        self.filelog = None


    def backProjection(self, index_list):
      """Perform back projection function for a list of indices. Return a list 
      of high dimensional points (one per index). Check cache for each point and
      condolidate file I/O for all cache misses.
      """

      logging.debug('--------  BACK PROJECTION:  %d POINTS ---', len(index_list))

      # reverse_index = {index_list[i]: i for i in range(len(index_list))}

      source_points = []
      cache_miss = []

      self.trajlist_async = deque()
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

      logging.debug('All Uncached Data collected Total # points = %d', len(source_points_uncached))
      source_traj_uncached = md.Trajectory(np.array(source_points_uncached), ref.top)

      logging.info('--------  Back Projection Complete ---------------')
      if source_traj_cached is None:
        return source_traj_uncached
      else:
        return source_traj_cached.join(source_traj_uncached)

    def execute(self):
      """Special execute function for the reweight operator -- check/validate.
      """
    # PRE-PROCESSING ---------------------------------------------------------------------------------
      logging.debug("============================  <PRE-PROCESS>  =============================")
      self.cacheclient = CacheClient(self.name)
      numLabels = 5
      binlist = [(A, B) for A in range(numLabels) for B in range(numLabels)]
      labeled_pts_rms = self.catalog.lrange('label:rms', 0, -1)
      num_pts = len(labeled_pts_rms)
      logging.debug('##NUM_OBS: %d', num_pts)

      # TEST_TBIN = [(i,j) for i in range(2,5) for j in range(5)]
      TEST_TBIN = [(2,0), (4,2), (2,2), (4,1)]
      MAX_SAMPLE_SIZE   = 10   # Max # of cov "pts" to back project
      COVAR_SIZE        = 200   # Ea Cov "pt" is 200 HD pts. -- should be static based on user query
      MAX_PT_PER_MATRIX =  3   # 5% of points from Source Covariance matrix



    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")

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
      cfile = home + '/work/DEBUG_COVAR_PTS'
      DO_COVAR = self.calc_covar  # For recalculating covariance matrices (if not pre-calc/stored)
      if DO_COVAR: 
        if os.path.exists(cfile + '.npy'):
          covar_pts = np.load(cfile + '.npy')
          logging.debug('Loaded From File')
        else: 
          covar_raw = self.catalog.lrange('subspace:covar:pts', 0, -1)
          covar_pts = np.array([np.fromstring(x) for x in covar_raw])
          np.save(cfile, covar_pts)
          logging.debug('Loaded From Catalog & Saved')
      covar_index = self.catalog.lrange('subspace:covar:xid', 0, -1)
      logging.debug('Indiced Loaded. Retrieving File Indices')
      covar_fidx = self.catalog.lrange('subspace:covar:fidx', 0, -1)

      if DO_COVAR: 
        logging.info("    Pulled %d Covariance Vectors", len(covar_pts))
        logging.info("Calculating Incremental PCA on Covariance (or Pick your PCA Algorithm here)")

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

        logging.info("IPCA Saved. Projecting Covariance to PC")

      cfile = home + '/work/DEBUG_SUBCOVAR_PTS'
      if os.path.exists(cfile + '.npy'):
        subspace_covar_pts = np.load(cfile + '.npy')
      else: 
        subspace_covar_pts = ipca.project(covar_pts)
        np.save(cfile, subspace_covar_pts)

      # OW/ PROJECT NEW PTS ONLY -- BUT RETAIN grouped index of all points
      logging.info('Building Global KD Tree over Covar Subspace with %d data pts', len(subspace_covar_pts))
      global_kdtree = KDTree(250, maxdepth=8, data=subspace_covar_pts, method='middle')
  

      # hcube_global = global_kdtree.getleaves()

      # FOR DEBUGGING -- USE ONLY 3 GLOBAL HCUBES
      hcube_global_ALL = global_kdtree.getleaves()
      hcube_global = {}
      num = 0
      for k, v in hcube_global_ALL.items():
        hcube_global[k] = v
        num += 1
        if num == 4:
          break

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
        logging.info('Back Projecting Global HCube `%s`  (%d out of %d)', key, counter, len(hcube_global.keys()))
        source_cov = self.backProjection(hcube_global[key]['idxlist'])
        hcube_global[key]['alpha'] = datareduce.filter_alpha(source_cov)
        logging.debug('Back Projected %d points to HD space: %s', 
          len(hcube_global[key]['idxlist']), str(hcube_global[key]['alpha']))

      logging.info('Calculating all HD Distances')
      dist_hd = {}
      dist_ld = {}
      for key in hcube_global.keys():
        T = hcube_global[key]['alpha'].xyz
        N = len(T)
        dist_hd[key] = np.zeros(shape=(N, N))
        dist_ld[key] = {}
        for A in range(0, N):
          dist_hd[key][A][A] = 0
          for B in range(A+1, N):
            dist_hd[key][A][B] = dist_hd[key][B][A] = LA.norm(T[A] - T[B])
        

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

      projlist = []
      for key in sorted(hcube_global.keys()):
        for i in range(len(hcube_global[key]['alpha'].xyz)):
          projlist.append([])
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
        kdtree = KDTree(200, maxdepth=8, data=data, method='middle')
        hcube_local[tbin] = kdtree.getleaves()
        logging.info('LOCAL KD-Tree Completed for %s:', str(tbin))
        for k in sorted(hcube_local[tbin].keys()):
          logging.info('    `%-9s`   #pts:%6d   density:%9.1f', 
            k, len(hcube_local[tbin][k]['elm']), hcube_local[tbin][k]['density'])

        if self.filelog:
          keys = hcube_local[tbin].keys()
          A,B = tbin
          self.filelog.info('local,%d_%d,keys,%s',A,B,','.join(keys))
          self.filelog.info('local,%d_%d,count,%s',A,B,','.join([str(hcube_local[tbin][k]['count']) for k in keys]))
          self.filelog.info('local,%d_%d,volume,%s',A,B,','.join([str(hcube_local[tbin][k]['volume']) for k in keys]))
          self.filelog.info('local,%d_%d,density,%s',A,B,','.join([str(hcube_local[tbin][k]['density']) for k in keys]))          

        n_total = 0
        logging.debug('Global Hcubes to Project (%d):  %s', len(hcube_global.keys()), str(hcube_global.keys()))
        
        pnum = 0
        for key in sorted(hcube_global.keys()):
          overlap_hcube[key][tbin] = {}
          cov_proj_pca = kpca.project(hcube_global[key]['alpha'].xyz)

          logging.debug('PROJECT: Global HCube `%-9s` (%d pts) ==> Local KDTree %s  ', 
            key, len(cov_proj_pca), str(tbin))
          for i, pt in enumerate(cov_proj_pca):
            hcube = kdtree.probe(pt, probedepth=9)
            # NOTE: Retaining count of projected pts. Should we track individual pts -- YES (trying)
            if hcube not in overlap_hcube[key][tbin]:
              overlap_hcube[key][tbin][hcube] = {
                  'idxlist': hcube_local[tbin][hcube]['elm'],
                  'wgt': hcube_local[tbin][hcube]['density'], 
                  'num_projected': 0}
            overlap_hcube[key][tbin][hcube]['num_projected'] += 1

            projlist[pnum].append(hcube)
            pnum += 1

          for k, v in sorted(overlap_hcube[key][tbin].items()):
            logging.debug('   Project ==> Local HCube `%-9s`: %5d points', k, v['num_projected'])
          logging.info('Calculating Lower Dimensional Distances')
          N = len(cov_proj_pca)
          dist_ld[key][tbin] = np.zeros(shape=(N, N))
          for A in range(0, N):
            for B in range(A+1, N):
              dist_ld[key][tbin][A][B] = dist_ld[key][tbin][B][A] = LA.norm(cov_proj_pca[A] - cov_proj_pca[B])



      sets = {}
      proj_bin_list = []
      for tbin in TEST_TBIN:
        if tbin not in hcube_local:
          continue
        proj_bin_list.append(tbin)
        sets[tbin] = {k: set() for k in hcube_local[tbin].keys()}
      for n, proj in enumerate(projlist):
        for i, tbin in enumerate(proj_bin_list):
          sets[tbin][proj[i]].add(n)
        if self.filelog:
          self.filelog.info('%d,%s', n, ','.join(proj))
        logging.info('%d,%s', n, ','.join(proj))

      set_list = {}
      for tbin, view in sets.items():
        set_list[(tbin,)] = []
        for hcube, idxlist in view.items():
          print(tbin, hcube, idxlist)
          set_list[(tbin,)].append((set((hcube,)), idxlist))

      print("Trying Collapse Join")
      out = collapse_join(set_list.keys(), set_list)
      for view in out:
        print(view)

      # def collapse(C):
      #   a = 0
      #   b = 0
      #   N = []
      #   while a < len(C) and b < len(C):
      #     A = sorted(C[a])
      #     B = sorted(C[b])
      #     if A == B:
      #       b += 1
      #     elif A[0] == B[0]:
      #       N.append(set(A)|set(B))
      #       b += 1
      #     else:
      #       a += 1
      #   if len(N) <= 1:
      #     return []
      #   else:
      #     return N + collapse(N)

      # q=collapse(t1)
      # for i in q: print(sorted(i))


      # print('Checking all 2-Way Joins')
      # join2 = {}
      # for a in range(0, len(proj_bin_list)-1):
      #   tA = proj_bin_list[a]
      #   for b in range(a+1, len(proj_bin_list)):
      #     tB = proj_bin_list[b]
      #     join_ss = tuple(set((tA, tB)))
      #     set_list = []
      #     for kA, vA in sets[tA].items():
      #       for kB, vB in sets[tB].items():
      #         join_hc = set((kA, kB))
      #         inter = vA & vB
      #         if len(inter) > 0:
      #           set_list.append((join_hc, inter))
      #     if len(set_list) > 0:
      #       join2[join_ss] = set_list
      # print('2-Way Join Results:')
      # for ss, set_list in join2.items():
      #   for hc, idxlist in set_list:
      #     print(ss, hc, idxlist)


      # print('Checking all 3-Way Joins')
      # join3 = []
      # checked = []
      # for a in range(0, len(join2)-1):
      #   sA, hA, vA = join2[a]
      #   for b in range(a+1, len(join2)):
      #     sB, hB, vB = join2[b]
      #     if sA == sB:
      #       continue
      #     ss, hc = sA | sB, hA | hB
      #     if (ss, hc) in checked[-10:]:
      #       continue
      #     checked.append((ss, hc))
      #     inter = vA & vB
      #     if len(inter) > 0:
      #       join3.append((ss, hc, inter))


      # print('Checking all 4-Way Joins')
      # join4 = []
      # checked = []
      # for a in range(0, len(join3)-1):
      #   sA, hA, vA = join3[a]
      #   for b in range(a+1, len(join3)):
      #     sB, hB, vB = join3[b]
      #     if sA == sB:
      #       continue
      #     ss, hc = sA | sB, hA | hB
      #     if (ss, hc) in checked[-10:]:
      #       continue
      #     checked.append((ss, hc))
      #     inter = vA & vB
      #     if len(inter) > 0:
      #       join4.append((ss, hc, inter))

      # if self.filelog:
      #   for i in join2:
      #     self.filelog.info('%s', str(i))
      #   for i in join3:
      #     self.filelog.info('%s', str(i))
      #   for i in join4:
      #     self.filelog.info('%s', str(i))

      DO_MIN_CHECK = False
      if DO_MIN_CHECK:
        def maxcount(x):
          y={}
          for i in x:
            y[i] = 1 if i not in y else y[i]+1
          return max(y.values())

        print('%% of Points Per HCube with same NN subspaces (e.g. 20%% of points have same NN in 5 sub-spaces')
        argmin_nonzero = lambda x: np.argmin([(i if i>0 else np.inf) for i in x])
        for key in hcube_global.keys():
          # logging.info('Showing MIN / MAX for points from HCube %s:', key)
          minA = {}; maxA={}
          for n in range(len(dist_hd[key])) :
            minA[n]=[] ; maxA[n]=[]
            for tbin in TEST_TBIN:
              if tbin not in dist_ld[key].keys():
                continue
                minA[n].append(0)
                maxA[n].append(0)          
              else:
                minA[n].append(argmin_nonzero(dist_ld[key][tbin][n]))
                maxA[n].append(np.argmax(dist_ld[key][tbin][n]))          
          numsame = np.zeros(len(dist_ld[key].keys())+1)
          for n in range(len(dist_hd[key][n])):
            minH = argmin_nonzero(dist_hd[key][n])
            maxH = np.argmax(dist_hd[key][n])
            minmax = ['%2d/%-2d'%i for i in zip(minA[n], maxA[n])]
            numsamepair = maxcount(minA[n])
            numsame[numsamepair] += 1
            # print('%3d'%n, '%2d/%-2d  '%(minH, maxH), '%s' % ' '.join(minmax), '   [%d]'%numsamepair)
          print(' '.join(['%4.1f%%'%i for i in (100* (numsame/np.sum(numsame)))]))

      print('Stopping HERE!')
      sys.exit(0)
      #  GAMMA FUNCTION EXPR # 8
      gamma1 = lambda a, b : (a * b)
      gamma2 = lambda a, b : (a + b) / 2

      # TODO: Factor in RMS weight
      for tbin in TEST_TBIN:
      # for tbin in sorted(bin_list):
        logging.info('')
        logging.info('BIPARTITE GRAPH for %s', str(tbin))
        bipart = {}
        edgelist = []
        for hcB in hcube_global.keys():
          num_B  = hcube_global[hcB]['count']
          wgt1_B = hcube_global[hcB]['density']
          if tbin not in overlap_hcube[hcB]:
            continue
          for hcA, hcA_data in overlap_hcube[hcB][tbin].items():
            edge = {}
            if hcA not in bipart:
              bipart[hcA] = []  
            num_proj  = hcA_data['num_projected']
            wgt_A  = hcA_data['wgt']
            wgt2_B = wgt1_B*num_proj
            edge['combW1'] = gamma1(wgt_A, wgt1_B)
            edge['combW2'] = gamma1(wgt_A, wgt2_B)
            edge['combW3'] = gamma2(wgt_A, wgt1_B)
            edge['combW4'] = gamma2(wgt_A, wgt2_B)
            edge['num_A']  = len(hcA_data['idxlist'])
            edge['num_B']  = num_B
            edge['num_proj']  = num_proj
            edge['wgt_A']  = wgt_A
            edge['wgt1_B'] = wgt1_B
            edge['wgt2_B'] = wgt2_B
            edge['hcA'] = hcA
            edge['hcB'] = hcB
            bipart[hcA].append(edge)
            edgelist.append((hcA, hcB, num_proj))
        if len(bipart) == 0:
          logging.info("NO DATA FOR %s", str(tbin))
          continue
        logging.info('')
        logging.info('A (# Pts) H-Cube        <--- B H-Cube (# proj/total Pts)      wgt_A  wB1:density wB2:Mass     A*B1     A*B2     AVG(A,B1)     AVG(A,B2)')
        for k, v in bipart.items():
          for edge in v:
            logging.info('A (%(num_A)4d pts) `%(hcA)-8s` <--- `%(hcB)9s`  (%(num_B)4d / %(num_proj)4d pts) B %(wgt_A)9.1f %(wgt1_B)9.1f %(wgt2_B)9.1f %(combW1)9.1f %(combW2)9.1f %(combW3)9.1f %(combW3)9.1f' % edge)
            if self.filelog:
              A,B = tbin
              self.filelog.info('edge,%d_%d,%s,%s,%d',A,B,edge['hcA'],edge['hcB'],edge['num_proj'])

        # Prepare nodes for graph
        nA = set()
        nB = set()
        elist = []
        for e in edgelist:
          a, b, z = e
          if z <= 5:
            continue
          nA.add(a)
          nB.add(b)
          elist.append((a,b,z))
        nAKeys = sorted(nA)[::-1]
        nBKeys = sorted(nB)[::-1]
        sizesA = [hcube_local[tbin][n]['count'] for n in nAKeys]
        sizesB = [hcube_global[n]['count']*3 for n in nBKeys]
        idxA = {key: i for i, key in enumerate(nAKeys)}
        idxB = {key: i for i, key in enumerate(nBKeys)}
        edges = [(idxA[a], idxB[b], z) for a, b, z in elist]
        G.bipartite(sizesA,sizesB,edges,sizesA,sizesB,'bipartite_%d_%d' % tbin)

      logging.info('STOPPING HERE!!!!')
      sys.exit(0)
      return []



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', default='reweight4')
  args = parser.parse_args()

  mt = reweightJob(args.name)
  mt.execute()




