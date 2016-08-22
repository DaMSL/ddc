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
from mdtools.simtool import generateExplJC, getSimParameters, generateFromBasin 
from mdtools.structure import Protein 
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient
from bench.timer import microbench
from bench.stats import StatCollector

from sampler.basesample import UniformSampler, CorrelationSampler

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

      # Set the simluation thread as the down stream macrothread node
      self.register_downnode('sim')

      # State Data for Simulation MacroThread -- organized by state
      self.setStream('basin:stream', None)

      self.addMut('jcqueue')
      self.addMut('converge')
      self.addMut('ctl_index_head')
      self.addImmut('basin:list')
      self.addImmut('basin:rms')
      self.addAppend('timestep')
      self.addMut('runtime')
      self.addMut('corr_vector')
      self.addMut('dspace_mu')
      self.addMut('dspace_sigma')
      self.addMut('raritytheta')

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
      if self.catalog.get('ctl:force') is not None:
        logging.info("Flagged to force a control decision.")
        minproc = 0
        logging.info("Resetting control flag.")
        self.catalog.delete('ctl:force')
      logging.info('# New Basins:  %d', len(basinlist))
      logging.info('Min Processing Threshold:  %d', minproc)
      if len(basinlist) < minproc:
        return [], None
      else:
        # Process all new basins: remove from downstream and create key for 
        #  next control job
        ctl_job_id = 'ctl:' + self.seqNumFromID()
        self.addMut(ctl_job_id, self.data['basin:stream'])
        consumed = len(self.data['basin:stream'])
        self.data['basin:stream'] = []
        return [ctl_job_id], consumed
        
    def fetch(self, item):
      """Retrieve this control job's list of basin ID's to process"""
      # First, Define Protein Object
      self.protein = Protein('bpti', self.catalog)

      # Load corr_matrices for all new basins
      self.cmat = {}

      new_basin_list = self.catalog.lrange(item, 0, -1)
      for b in new_basin_list:
        key = 'basin:' + b
        self.data[key] = self.catalog.hgetall(key)
        self.cmat[b] = self.catalog.loadNPArray('basin:cmat:' + b)
        # self.dmu[b] = self.catalog.loadNPArray('basin:dmu:' + b)
        # self.dsigma[b] = self.catalog.loadNPArray('basin:dsigma:' + b)
      return new_basin_list
      # start_index = max(0, self.data['ctl_index_head'])
      # thru_index = min(start_index + int(batchSize), self.catalog.llen('basin:list')) - 1
      # return thru_index

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']

    
    def execute(self, new_basin_list):
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
      topo = self.protein.top

      ##### BARRIER
      self.wait_catalog()

      # Load new RMS Labels -- load all for now
      bench.start()
      logging.debug('Loading RMS Labels')
      start_index = max(0, self.data['ctl_index_head'])

      # labeled_pts_rms = self.catalog.lrange('label:rms', self.data['ctlIndexHead'], thru_index)
      logging.debug(" Start_index=%d,  batch_size=%d", start_index, len(new_basin_list))

      # Simplicity: For now, read in ALL RMSD values.
      # all_rms = [float(v) for v in self.data['basin:rms'].values()]

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

      if EXPERIMENT_NUMBER == 13:

        # PREPROCESS
        N_features_src = topo.n_residues
        N_features_corr = (N_features_src**2 - N_features_src) // 2 
        upt = np.triu_indices(N_features_src, 1)
    
        # FOR NOW: Load from disk
        logging.info('Loading Historical data')
        de_corr_matrix = np.load('data/de_corr_matrix.npy')
        de_dmu = np.load('data/de_ds_mu.npy')
        de_dsig = np.load('data/de_ds_mu.npy')

        basin_corr_matrix_prev =  self.data['corr_vector']
        dspace_mu_prev = self.data['dspace_mu']
        dspace_sigma_prev = self.data['dspace_sigma']

        # MERGE: new basins with basin_corr_matrix, d_mu, d_sigma
        # Get list of new basin IDs
        stat.collect('new_basin', len(new_basin_list))
        cmat, ds_mean, ds_std = [], [], []
        logging.info('Collecting new data')
        for bid in new_basin_list:
          key = 'basin:' + bid
          basin = self.data[key]

          # TODO:  HOW TO STORE THIS???? 
          cmat.append(pickle.loads(basin['corr_vector']))
          ds_mean.append(pickle.loads(basin['d_mu']))
          ds_std.append(pickle.loads(basin['d_sigma']))

          # TODO: FINISH NOISE MODEL

        # Merge new values with old values:
        logging.info('Mering new data')
        if basin_corr_matrix_prev is None or basin_corr_matrix_prev == []:
          C_T = basin_corr_matrix = np.array(cmat)
        else:
          C_T = basin_corr_matrix = np.vstack((basin_corr_matrix_prev, cmat))
  
        if dspace_mu_prev is None or dspace_mu_prev == []:
          D_mu = np.array(ds_mean)
        else:
          D_mu = np.vstack((dspace_mu_prev, ds_mean))
        
        if dspace_sigma_prev is None or dspace_sigma_prev == []:
          D_sigma = np.array(ds_std)
        else:
          D_sigma = np.vstack((dspace_sigma_prev, ds_std)) 
  
        # D_noise = np.zeros(shape=(N_obs, M_reduced))
        cm_all = np.vstack((de_corr_matrix, C_T))
        dmu_all = np.vstack((de_dmu, D_mu))
        dsig_all = np.vstack((de_dsig, D_sigma))
        sampler = CorrelationSampler(cm_all, mu=dmu_all, sigma=dsig_all)
        basin_id_list = sampler.execute(numresources)

        # For now retrieve immediately from catalog
        for index in basin_id_list:
          bid = all_basins[index]
          basin_list.append(self.catalog.hgetall('basin:%s'%bid))

    # Generate new starting positions
      jcqueue = OrderedDict()
      src_traj_list = []
      for basin in basin_list:
        #  TODO:  DET HOW TO RUN FOLLOW ON FROM GEN SIMS
        src_traj_list.append(basin['traj'])
        if basin['traj'].startswith('desh'):
          global_params = getSimParameters(self.data, 'deshaw')
          fileno = int(basin['traj'][-4:])
          frame = int(basin['mindex'])
          jcID, config = generateDEShawJC(fileno, frame, jcid)
        else:
          global_params = getSimParameters(self.data, 'gen')
          src_psf = self.catalog.hget('jc_' + basin['traj'], 'psf')
          global_params.update({'psf': src_psf})
          jcID, config = generateExplJC(basin, jcid=None)

        # jcID, config = generateFromBasin(basin)

        config.update(global_params)
        config['name'] = jcID

        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in config.items():
          logging.debug("   %s:  %s", k, str(v))

        #  Add to the output queue & save config info
        jcqueue[jcID] = config
        logging.info("New Job Candidate Completed:  %s   #%d on the Queue", jcID, len(jcqueue))

      stat.collect('src_traj_list', src_traj_list)
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
 
      self.notify('sim')

      bench.mark('PostProcessing')
      print ('## TS=%d' % self.data['timestep'])
      bench.show()
      stat.show()

      return list(jcqueue.keys())

if __name__ == '__main__':
  mt = controlJob(__file__)
  mt.run()

