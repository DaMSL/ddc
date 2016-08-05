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
import core.ops as op
from macro.macrothread import macrothread
from core.kvadt import kv2DArray
from core.slurm import slurm

from core.kdtree import KDTree
import datatools.datareduce as datareduce
import datatools.datacalc as datacalc
import datatools.mvkp as mvkp
import mdtools.deshaw as deshaw
import datatools.kmeans as KM
# from datatools.pca import calc_kpca, calc_pca, project_pca
from datatools.pca import PCAnalyzer, PCAKernel, PCAIncremental
from datatools.approx import ReservoirSample
from mdtools.simtool import generateExplJC, getSimParameters, Peptide 
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient
from bench.timer import microbench
from bench.stats import StatCollector

from sampler.basesample import UniformSampler

from datatools.feature import feal
import plot as G


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.2"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=5, suppress=True)



class controlJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('basin:stream', None)

      self.addMut('jcqueue')
      self.addMut('converge')
      self.addMut('ctl_index_head')
      # self.addImmut('ctlSplitParam')
      # self.addImmut('ctlDelay')
      # self.addImmut('terminate')
      # self.addImmut('numresources')
      # self.addImmut('ctl_min_basin')
      # self.addImmut('dcdfreq')
      # self.addImmut('runtime')
      # self.addImmut('max_observations')
      # self.addImmut('sim_step_size')
      self.addImmut('basin:list')
      self.addImmut('basin:rms')
      self.addAppend('timestep')
      self.addMut('runtime')


      self.addMut('ctlCountHead')
      self.addImmut('exploit_factor')
      self.addImmut('observe:count')
          


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
      numobs = self.catalog.llen('xid:reference')
      if numobs > self.data['max_observations']:
        logging.info('Terminating at %d observations', numobs)
        return True
      else:
        return False

    def split(self):

      settings = systemsettings()
      basinlist = self.data['basin:stream']
      minproc  = int(self.data['ctl_min_basin'])
      logging.info('# New Basins:  %d', len(basinlist))
      logging.info('Min Processing Threshold:  %d', minproc)
      if len(basinlist) < minproc:
        return [], None
      else:
        # Process all new basins
        self.data['basin:stream'] = []
        return [len(basinlist)], len(basinlist)
        
    def fetch(self, batchSize):
      """Fetch determines the next thru index for this control loop
      Note that batchSize is measured in ps. Thru Index should return
      the next index to process
      """
      start_index = max(0, self.data['ctl_index_head'])
      thru_index = min(start_index + int(batchSize), self.catalog.llen('basin:list')) - 1
      return thru_index

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']

    
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

      np.set_printoptions(precision=4, linewidth=150)

      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

      settings = systemsettings()
      bench = microbench('ctl_%s' % settings.name, self.seqNumFromID())
      stat = StatCollector('ctl_%s' % settings.name, self.seqNumFromID())

      # Connect to the cache
      self.cacheclient = CacheClient(settings.APPL_LABEL)

      # create the "binlist":
      numresources = self.data['numresources']


    # LOAD all new subspaces (?) and values

      ##### BARRIER
      self.wait_catalog()

      # Load new RMS Labels -- load all for now
      bench.start()
      logging.debug('Loading RMS Labels')
      start_index = max(0, self.data['ctl_index_head'])

      # labeled_pts_rms = self.catalog.lrange('label:rms', self.data['ctlIndexHead'], thru_index)
      logging.debug(" Start_index=%d,  thru_index=%d", start_index, thru_index)

      # Simplicity: For now, read in ALL RMSD values.
      all_rms = [float(v) for v in self.data['basin:rms'].values()]

    # Calculate variable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERGENCE> (skip)  ===================")

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")

      ##### BARRIER
      self.wait_catalog()
      selected_index_list = []

      # QUERY PROCESSING & SAMPLING BELOW to select indices. 
      EXPERIMENT_NUMBER = self.experiment_number
      logging.info("RUNNING EXPER CONFIGURATION #%d", EXPERIMENT_NUMBER)

      # Basin List will be the list of basin representing the new Job Candidates
      basin_list = []
      all_basins = self.data['basin:list']

      # UNIFORM SAMPLER (BASIC)
      if EXPERIMENT_NUMBER == 12:
        sampler = UniformSampler(all_basins)
        basin_id_list = sampler.execute(numresources)

        # For now retrieve immediately from catalog
        for bid in basin_id_list:
          basin_list.append(self.catalog.hgetall('basin:%s'%bid))

    # Generate new starting positions
      global_params = getSimParameters(self.data, 'gen')
      jcqueue = OrderedDict()
      for basin in basin_list:

        jcID, jcConfig = generateExplJC(basin)
        jcConfig.update(global_params)
        jcConfig['name'] = jcID

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


      # Update cache hit/miss
      # hit = self.cache_hit
      # miss = self.cache_miss
      # logging.info('##CACHE_HIT_MISS %d %d  %.3f', hit, miss, (hit)/(hit+miss))
      # self.catalog.rpush('cache:hit', self.cache_hit)
      # self.catalog.rpush('cache:miss', self.cache_miss)

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
  mt.run()

