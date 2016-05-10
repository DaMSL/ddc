#!/usr/bin/env python

import argparse
import sys
import os
import sys
import math
import json
import redis
import itertools
from pprint import pprint
from random import choice, randint
from collections import namedtuple, deque, OrderedDict
from threading import Thread

import mdtraj as md
import numpy as np
from numpy import linalg as LA

import redis
import dhist as dh
from datatools.pca import PCAnalyzer, PCAKernel


from core.common import *
from core.kdtree import KDTree
import datatools.datareduce as datareduce
import datatools.datacalc as datacalc
import mdtools.deshaw as deshaw
from mdtools.simtool import generateNewJC
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



class reweightJob(object):
    def __init__(self, name, host='localhost', port=6379):

      self.name = name
      self.calc_covar = False

      # For stat tracking
      self.cache_hit = 0
      self.cache_miss = 0
      self.catalog = redis.StrictRedis(host=host, port=port, decode_responses=True)

      self.dest = redis.StrictRedis(host='compute0330', port=6399, decode_responses=True)

      FILELOG = False
      if FILELOG:
        self.filelog = logging.getLogger(name)
        self.filelog.setLevel(logging.INFO)
        filename = 'kd04.log'
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

        # Check cache for historical data points
        logging.debug('Checking cache for %d DEShaw points to back-project', len(historical_frameMask.keys()))
        for fileno, frames in historical_frameMask.items():
          cache_miss.append(('deshaw', fileno, frames))


        # Check cache for generated data points
        logging.debug('Checking cache for %d Generated points to back-project', len(generated_frameMask.keys()))
        for filename, frames in generated_frameMask.items():
          cache_miss.append(('sim', generated_filemap[filename], frames))


        # Package all cached points into one trajectory
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
        return source_traj_uncached


    def execute(self):
      """Special execute function for the reweight operator -- check/validate.
      """
    # PRE-PROCESSING ---------------------------------------------------------------------------------
      logging.debug("============================  <PRE-PROCESS>  =============================")

      config = systemsettings()
      config.manualConfig('mb_kd04')
      config.envSetup()

      # TEST_TBIN = [(i,j) for i in range(2,5) for j in range(5)]
      TEST_TBIN = [(0,4)]
      TEST_HC = ['0000', '1111', '1100', '0101']
      hcube_local = {}
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
        logging.info('Building KDtree over local %s bin from observations matrix of size: %s', str(tbin), str(data.shape))
        kdtree = KDTree(50, maxdepth=4, data=data, method='median')
        hcube_local[tbin] = kdtree.getleaves()
        logging.info('LOCAL KD-Tree Completed for %s:', str(tbin))
        for k in hcube_local[tbin].keys():
          logging.info('    `%-9s`   #pts:%6d   density:%9.1f', 
            k, len(hcube_local[tbin][k]['elm']), hcube_local[tbin][k]['density'])

        logging.info('STOPPING HERE!!!!')
        sys.exit(0)
        return []

        logging.debug("=======================  <INPUT PARAM GENERATION>  =================")

        jcqueue = OrderedDict()
        for hc in TEST_HC:
          samples = np.random.choice(hcube_local[tbin][hc]['elm'], 25)
          print(hc, '\n', samples)
          for i in samples:
            traj = self.backProjection([i])
            jcID, params = generateNewJC(traj)
            jcConfig = dict(params,
                  name    = jcID,
                  runtime = 50000,     # In timesteps
                  dcdfreq = 500,           # Frame save rate
                  interval = 1000,                       
                  temp    = 310,
                  timestep = 0,
                  gc      = 1,
                  origin  = 'sim',
                  src_index = i,
                  src_bin  = '(0, 4)',
                  src_hcube = hc,
                  application  = 'mb_kd04')
            jcqueue[jcID] = jcConfig

            key = 'jc_'+jcID
            logging.info("Push the following, %s", key)
            pprint(jcConfig)
            self.dest.hmset(key, jcConfig)
            self.dest.rpush('jcqueue', key)

        if self.filelog:
          keys = hcube_local[tbin].keys()
          A,B = tbin
          self.filelog.info('local,%d_%d,keys,%s',A,B,','.join(keys))
          self.filelog.info('local,%d_%d,count,%s',A,B,','.join([str(hcube_local[tbin][k]['count']) for k in keys]))
          self.filelog.info('local,%d_%d,volume,%s',A,B,','.join([str(hcube_local[tbin][k]['volume']) for k in keys]))
          self.filelog.info('local,%d_%d,density,%s',A,B,','.join([str(hcube_local[tbin][k]['density']) for k in keys]))          

      logging.info('STOPPING HERE!!!!')
      sys.exit(0)
      return []




P.heatmap(dd, hc1k, hc2k, title='High_Low_unif_sig_kpca', xlabel='High Dim (x,y,z) KD Tree Leaves', ylabel='Low Dim (KPCA) KD Tree Leaves')

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('name', default='reweight5_xlt')
  # args = parser.parse_args()

  mt = reweightJob('reweight5_xlt')
  mt.execute()




