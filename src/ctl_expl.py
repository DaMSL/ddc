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
import datatools.lattice as lat

# from datatools.pca import calc_kpca, calc_pca, project_pca
from datatools.pca import PCAnalyzer, PCAKernel, PCAIncremental
from datatools.approx import ReservoirSample
from mdtools.simtool import *
from mdtools.trajectory import bin_label_10, bin_label_25
from mdtools.structure import Protein 
from overlay.redisOverlay import RedisClient
from overlay.cacheOverlay import CacheClient
from bench.timer import microbench
from bench.stats import StatCollector

from sampler.basesample import *

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
      self.addMut('sampler:explore')

      # self.addMut('lattice:max_fis')
      # self.addMut('lattice:low_fis')
      # self.addMut('lattice:dlat')
      # self.addMut('lattice:iset_delta')

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

      self.force_decision = False

      self.parser.add_argument('--force', action='store_true')
      args = self.parser.parse_args()
      if args.force:
        logging.info('FORCING A MANUAL CONTROL DECISION')
        self.force_decision = True


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

      if self.force_decision:
        return None

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

      if self.force_decision:
        new_basin_list = []

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
      selected_basin_list = []
      all_basins = self.data['basin:list']

      # UNIFORM SAMPLER (BASIC)
      if EXPERIMENT_NUMBER == 12:
        sampler = UniformSampler(all_basins)
        basin_id_list = sampler.execute(numresources)

        # For now retrieve immediately from catalog
        for bid in basin_id_list:
          selected_basin_list.append(self.catalog.hgetall('basin:%s'%bid))

      if EXPERIMENT_NUMBER == 17:
        n_features = 1653
        explore_factor = float(self.data['sampler:explore'])
        

        # Load previous distance space
        de_ds_raw = self.catalog.lrange('dspace', 0, -1)
        prev_size = len(de_ds_raw)
        local_basins = {}
        if  prev_size > 0:
          ds_prev = np.zeros(shape=(prev_size, n_features))
          logging.info("Unpickling distance space to array: %s", ds_prev.shape)
          for i, elm in enumerate(de_ds_raw):
            ds_prev[i] = pickle.loads(elm)
          logging.info('Prev DS loaded. Size = %d', len(ds_prev))
        else:
          logging.info('NO Prev DS')
          ds_prev = []

        # Merge new data
        delta_ds = []
        for bid in new_basin_list:
          key = 'basin:' + bid
          logging.debug("  Loading Basin: %s", key)
          basin = self.data[key]
          dmu_ = pickle.loads(self.catalog.get('basin:dmu:'+bid))
          if dmu_ is None:
            logging.debug("No Data for Basin:  %s", bid)
          else:
            delta_ds.append(dmu_)

            label_seq = [int(i) for i in self.catalog.lrange('basin:labelseq:'+bid, 0, -1)]

            label10 = bin_label_10(label_seq)
            label25 = bin_label_25(label_seq)

            basin_index = self.catalog.rpush('dspace', pickle.dumps(dmu_)) - 1
            local_basins[basin_index] = basin
            with self.catalog.pipeline() as pipe:
              pipe.hset(key, 'label:10', label10)
              pipe.hset(key, 'label:25', label25)
              pipe.rpush('bin:10:%s' % label10, basin_index)
              pipe.rpush('bin:25:%d_%d' % label25, basin_index)
              pipe.execute()


        # if not self.force_decision and len(delta_ds) > 0:
        #   logging.debug('Pushing DELTA_DS to catalog: %d', len(delta_ds))
        #   with self.catalog.pipeline() as pipe:
        #     for elm in delta_ds:
        #       pipe.rpush('dspace', pickle.dumps(elm))
        #     pipe.execute()

        if len(delta_ds) > 0 and len(ds_prev) > 0:
          dist_space = np.vstack((ds_prev, np.array(delta_ds)))
        elif len(delta_ds) == 0:
          dist_space = np.array(ds_prev)
        elif len(ds_prev) == 0:
          logging.info('FIRST Set of Distance Coord laoded')
          dist_space = np.array(delta_ds)
        else:
          logging.error("ERROR! NO DISTANCE SPACE IN THE CATALOG")

        logging.debug("Dist_space: %s", dist_space.shape)

        logging.info('Loading Basin Lists')
        bin_labels_10 = ['T0', 'T1', 'T2', 'T3', 'T4', 'W0', 'W1', 'W2', 'W3', 'W4']
        bin_labels_25 = [(a,b) for a in range(5) for b in range(5)]
        bin_list_10 = {k: [int(i) for i in self.catalog.lrange('bin:10:%s' % k, 0, -1)] for k in bin_labels_10}
        bin_list_25 = {k: [int(i) for i in self.catalog.lrange('bin:25:%d_%d' % k, 0, -1)] for k in bin_labels_25}

        # USING 10-BIN LABELS
        distro = [len(bin_list_10[i]) for i in bin_labels_10]

        # Create and invoke the sampler
        logging.info('Running the biased (umbrella) samplers')
        sampler = BiasSampler(distro)
        samplecount = np.zeros(len(bin_labels_10), dtype=np.int16)
        # Find the first index for each bin:
        explore_direction = 1 if explore_factor < .5 else -1
        for i, b in enumerate(bin_list_10):
          if len(b) == 0:
            idx = 0
          else:
            idx = np.floor(explore_factor * (len(b) - 1))
          samplecount[i] = idx

        sel_bins = sampler.execute(numresources)

        logging.info('Processing selected bins to find starting candidates')
        candidate_list = {}
        basin_idx_list = []
        for b in sel_bins:
          target_bin = bin_labels_10[b]
          if target_bin not in candidate_list:
            candidate_list[target_bin] = bin_list_10[target_bin]

          #  TODO:  FULLY IMPLEMENT EXPLORE/EXPLOIT BUT INCL HISTORY/PROVONANCE

            # Lazy Update to centroid -- push to catalog immediately
            # vals = dist_space[bin_list_10[target_bin]]
            # logging.info('Updating Centroid for bin %s,  bindata: %s', target_bin, vals.shape)
            # centroid = np.mean(vals, axis=0)
            # self.catalog.set('bin:10:centroid:%s' % target_bin, pickle.dumps(centroid))
            # dist_center = [LA.norm(centroid - dist_space[i]) for i in bin_list_10[target_bin]]
            # candidate_list[target_bin] = sorted(zip(bin_list_10[target_bin], dist_center), key=lambda x: x[1])

          # basin_idx, basin_diff = candidate_list[target_bin][samplecount[b]]
          # samplecount[b] += explore_direction
          # # Wrap
          # if samplecount[b] == 0:
          #   samplecount = len(candidate_list[target_bin]) - 1
          # if samplecount[b] == len(candidate_list[target_bin]):
          #   samplecount = 0

          # FOR NOW PICK A RANDOM CANDIDATE 
          rand_index = np.random.choice(len(candidate_list[target_bin]))
          basin_idx = candidate_list[target_bin][rand_index]
          logging.info('BIAS SAMPLER:\n   Bin: %s\n   basin: %d     Delta from Center: %6.3f  (note: dist not factored in)', \
            target_bin, basin_idx, 0.)
          basin_idx_list.append(basin_idx)

        for i in basin_idx_list:
          if i < prev_size:
            logging.info("Select index: %s   (Retrieve from Catalog)", i)
            bid = self.data['basin:list'][i]
            basin = self.catalog.hgetall('basin:%s'%bid)
          else:
            logging.info("Select index: %s   (New locally built basin in mem)", i)  
            basin = local_basins[i]
          logging.debug('   BASIN:  %s', basin['id'])
          selected_basin_list.append(basin)


      # LATTICE SAMPLER (WITH HISTORICAL DATA)
      if EXPERIMENT_NUMBER in [13, 14, 16]:

        # PREPROCESS
        if EXPERIMENT_NUMBER == 16:
          n_features = 58  
        else:
          n_features = 1653
    
        logging.info("NUmber of Features: %d", n_features)
        #####   BASIN LIST HERE
        # Get ALL basins metadata:
        old_basin_ids = self.data['basin:list'][:start_index-1]

        # FOR NOW: Load from Catalog
        logging.info('Loading Historical DEShaw data')

        # Explicitly manage distance space load here
        logging.info('Loading raw distance space from catalog')
        de_ds_raw = self.catalog.lrange('dspace', 0, -1)
        ds_prev = np.zeros(shape=(len(de_ds_raw), n_features))
        logging.info("Unpickling distance space to array: %s", ds_prev.shape)
        for i, elm in enumerate(de_ds_raw):
          ds_prev[i] = pickle.loads(elm)

        # MERGE: new basins with basin_corr_matrix, d_mu, d_sigma
        # Get list of new basin IDs
        stat.collect('new_basin', len(new_basin_list))
        delta_cm, delta_ds, = [], []
        logging.info('Collecting new data from basins: %s', new_basin_list)
        for bid in new_basin_list:
          key = 'basin:' + bid
          logging.debug("  Loading Basin: %s", key)
          basin = self.data[key]

          # cm_ = self.catalog.get('basin:cm:'+bid)
          # delta_cm.append(pickle.loads(cm_))

          dmu_ = self.catalog.get('basin:dmu:'+bid)
          if dmu_ is None:
            logging.debug("No Data for Basin:  %s", bid)
          else:
            delta_ds.append(pickle.loads(dmu_))
          # ds_std.append(pickle.loads(dsig_))

        if EXPERIMENT_NUMBER == 13:
          # Merge new values with old values:
          logging.info('Mering new data')
          basin_corr_matrix_prev =  self.data['corr_vector']
          dspace_mu_prev = self.data['dspace_mu']
          # dspace_sigma_prev = self.data['dspace_sigma']

          if basin_corr_matrix_prev is None or basin_corr_matrix_prev == []:
            C_T = basin_corr_matrix = np.array(cmat)
          else:
            C_T = basin_corr_matrix = np.vstack((basin_corr_matrix_prev, cmat))
    
          if dspace_mu_prev is None or dspace_mu_prev == []:
            D_mu = np.array(ds_mean)
          else:
            D_mu = np.vstack((dspace_mu_prev, ds_mean))
          
          # if dspace_sigma_prev is None or dspace_sigma_prev == []:
          #   D_sigma = np.array(ds_std)
          # else:
          #   D_sigma = np.vstack((dspace_sigma_prev, ds_std)) 
    
          # D_noise = np.zeros(shape=(N_obs, M_reduced))
          cm_all = np.vstack((de_cm, C_T))
          dmu_all = np.vstack((de_ds, D_mu))
          # dsig_all = np.vstack((de_dsig, D_sigma))


          # sampler = CorrelationSampler(cm_all, mu=dmu_all, sigma=dsig_all)
          sampler = CorrelationSampler(cm_all, mu=dmu_all)
        
        else:
        # Merge Existing delta with DEShaw Pre-Processed data:
          logging.info('Merging DEShaw with existing generated data')

          # Set parameters for lattice
          Kr = [int(i) for i in self.catalog.lrange('lattice:features', 0, -1)]
          support = int(self.data['lattice:support'])
          dspt = self.catalog.get('lattice:delta_support')
          delta_support = 5 if dspt is None else int(dspt)
          cutoff  = float(self.data['lattice:cutoff'])

          logging.info('PARAMS  Kr:%s\n support:%d  dspt:%d  cutoff:%f', Kr, support, delta_support, cutoff)

          # Load existing (base) lattice data
          logging.info("Unpickling max/low FIS and derived lattice EMD values")
          max_fis    = pickle.loads(self.catalog.get('lattice:max_fis'))
          print('MFIS: ', len(max_fis))
          low_fis    = pickle.loads(self.catalog.get('lattice:low_fis'))
          dlat       = pickle.loads(self.catalog.get('lattice:dlat'))

          # Full DEShaw Index is saved on disk
          logging.info("Loading full Itemset from disk (TODO: Det optimization on mem/time)")
          Ik        = pickle.load(open(settings.datadir + '/iset.p', 'rb'))

          # Item_set Keys (Ik) are only saved as a delta for space conservation
          if os.path.exists(settings.datadir + '/iset_delta.p'):
            Ik_delta  = pickle.load(open(settings.datadir + '/iset_delta.p', 'rb'))
          else:
            Ik_delta = {}

          # Merge previous item set delta with DEShaw index
          logging.info("Merging DEShaw Ik with Delta IK")
          for k,v in Ik_delta.items():
            Ik[k] = np.concatenate((Ik[k], v)) if k in Ik else v

          # Build Lattice Object
          invert_vals = (EXPERIMENT_NUMBER == 16)
          logging.info('Building Existing lattice object (do invert? %s', invert_vals)
          base_lattice=lat.Lattice(ds_prev, Kr, cutoff, support, invert=invert_vals)
          base_lattice.set_fis(max_fis, low_fis)
          base_lattice.set_dlat(dlat, Ik)

          # Build Delta Lattice Object
          logging.info('Building Delta lattice. Num new items: %d', len(delta_ds))

          if not self.force_decision and len(delta_ds) > 0:
            delta_ds = np.array(delta_ds)
            delta_lattice = lat.Lattice(delta_ds, Kr, cutoff, delta_support, invert=invert_vals)
            delta_lattice.maxminer()
            delta_lattice.derive_lattice()

            # Update non-DEShaw delta itemset key index
            logging.info("Updating Itemsets and Distance Space Matrix")
            for k,v in delta_lattice.Ik.items():
              Ik_delta[k] = np.concatenate((Ik_delta[k], v)) if k in Ik_delta else v

            # Save Ik delta to disk
            logging.info("Saving Delta Itemset (to disk)")
            pickle.dump(Ik_delta, open(settings.datadir + '/iset_delta.p', 'wb'))
            

            # Push distance space delta values to catalog
            with self.catalog.pipeline() as pipe:
              for elm in delta_ds:
                pipe.rpush('dspace', pickle.dumps(elm))
              pipe.execute()


            #  Perform incremental maintenance
            logging.info('Merging Delta lattice with Base Lattice')
            base_lattice.merge(delta_lattice)

          # Create the Sampler object (also make clusters)
          #  TODO:  CLuster maintenance
          logging.info('Invoking the Lattice Sampler')

        if settings.EXPERIMENT_NUMBER == 16:
          sampler = LatticeExplorerSampler(base_lattice)
        else:
          sampler = LatticeSampler(base_lattice)

        basin_id_list = sampler.execute(numresources)

        # For now retrieve immediately from catalog
        self.wait_catalog()
        for index in basin_id_list:
          bid = all_basins[index]
          selected_basin_list.append(self.catalog.hgetall('basin:%s'%bid))

      # LATTICE SAMPLER (DE NOVO)
      if EXPERIMENT_NUMBER == 15:

        # PREPROCESS
        N_features_src = topo.n_residues
        N_features_corr = (N_features_src**2 - N_features_src) // 2 
        upt = np.triu_indices(N_features_src, 1)
    
        #####   BASIN LIST HERE
        # Get ALL basins metadata:
        old_basin_ids = self.data['basin:list'][:start_index-1]

        # Explicitly manage distance space load here
        logging.info('Loading raw distance space from catalog')
        ds_raw  = self.catalog.lrange('dspace', 0, -1)
        ds_prev = np.array([pickle.loads(i) for i in ds_raw])

        # MERGE: new basins with basin_corr_matrix, d_mu, d_sigma
        # Get list of new basin IDs
        stat.collect('new_basin', len(new_basin_list))
        delta_cm, delta_ds, = [], []
        logging.info('Collecting new data from basins: %s', new_basin_list)
        for bid in new_basin_list:
          key = 'basin:' + bid
          logging.debug("  Loading Basin: %s", key)
          basin = self.data[key]

          # cm_ = self.catalog.get('basin:cm:'+bid)
          # delta_cm.append(pickle.loads(cm_))

          dmu_ = self.catalog.get('basin:dmu:'+bid)
          if dmu_ is None:
            logging.debug("No Data for Basin:  %s", bid)
          else:
            delta_ds.append(pickle.loads(dmu_))


        # Push distance space delta values to catalog
        if not self.force_decision:
          delta_ds = np.array(delta_ds)
          with self.catalog.pipeline() as pipe:
            for elm in delta_ds:
              pipe.rpush('dspace', pickle.dumps(elm))
            pipe.execute()


        # DENOVO Exploratory Bootstrapping (RMSD)
        explore_factor = float(self.data['sampler:explore'])

        # TODO:  Better transtion plan from explore to exploit
        self.data['sampler:explore'] *= .75   
        executed_basins = self.catalog.lrange('executed', 0, -1)

        if explore_factor > 0:
          logging.info("EXPLORING Most active basins....")
          basindata = [self.catalog.hgetall(bid) for bid in old_basin_ids]
          for bid in new_basin_list:
            basindata.append(self.data['basin:'+bid])          

          basins_with_rms = [b for b in basindata if 'resrms_delta' in b]

          basin_by_rmsd = sorted(basins_with_rms, key=lambda x: float(x['resrms_delta']), reverse=True)
          explore_samples = int(np.floor(numresources * explore_factor))
          logging.info('Num to explore: %d  out of %d', explore_samples, len(basin_by_rmsd))
          idx, num_sampled = 0, 0

          while idx < len(basin_by_rmsd) and num_sampled < explore_samples:
            selb = basin_by_rmsd[idx]
            if selb['id'] in executed_basins:
              logging.info('Selected %s, but it has been executed. Selecting next option', selb['id'])
            else:
              selected_basin_list.append(selb)
              logging.info('  (%d) EXPLORE BASIN:  %s  %f', selb['id'], selb['id'], float(selb['resrms_delta']))
              numresources -= 1
              num_sampled += 1
            idx += 1


        # TODO:  Reduced Feature Sets
        #  Using Reduced Feature Set Alg #2 HERE
        ds_total = np.vstack((ds_prev, delta_ds)) if len(delta_ds) > 0 else ds_prev
        support = int(.01 * len(ds_total))
        cutoff  = 8

        # RE-Calc the whole lattice:
        logging.info("Building the new lattice")
        BUILD_NEW = not self.catalog.exists('lattice:bootstrapped')
        # TODO: Decision to go from build new to incr maint
        if BUILD_NEW:
          tval = .05
          Kr = lat.reduced_feature_set2(ds_total, cutoff, theta=tval, maxk=25)
          retry = 5
          while len(Kr) < 12 and retry > 0:
            tval /= 2
            retry -= 1
            Kr = lat.reduced_feature_set2(ds_total, cutoff, theta=tval, maxk=25)


          base_lattice = lat.Lattice(ds_total, Kr, cutoff, support)
          base_lattice.maxminer()
          base_lattice.derive_lattice()
          with self.catalog.pipeline() as pipe:
            pipe.delete('lattice:kr')
            for i in sorted(Kr):
              pipe.rpush('lattice:kr', i)
            pipe.execute()

        else:
          # Load existing (base) lattice data
          max_fis    = pickle.loads(self.catalog.get('lattice:max_fis'))
          low_fis    = pickle.loads(self.catalog.get('lattice:low_fis'))
          dlat       = pickle.loads(self.catalog.get('lattice:dlat'))
          Ik         = pickle.loads(self.catalog.get('lattice:iset'))
          num_k      = self.catalog.get('lattice:num_k')
          Kr         = [int(i) for i in self.catalog.lrange('lattice:kr', 0, -1)]
          if num_k is None:
            num_k = max(8, min(15, numresources*2))

          # Build Lattice Object
          logging.info('Building Existing lattice object')
          base_lattice=lat.Lattice(ds_prev, Kr, cutoff, support)
          base_lattice.set_fis(max_fis, low_fis)
          base_lattice.set_dlat(dlat, Ik)

          # Build Delta Lattice Object
          logging.info('Building Delta lattice. Num new items: %d', len(delta_ds))
          delta_lattice = lat.Lattice(delta_ds, Kr, cutoff, 1)
          delta_lattice.maxminer()
          delta_lattice.derive_lattice()

          #  Perform incremental maintenance
          logging.info('Merging Delta lattice with Base Lattice')
          base_lattice.merge(delta_lattice)

        # Create the Sampler object (also make clusters)
        #  TODO:  CLuster maintenance

        if numresources > 0:
          logging.info('Invoking the Lattice Sampler')
          sampler = LatticeSampler(base_lattice)
          basin_id_list = sampler.execute(numresources)

          # For now retrieve immediately from catalog
          self.wait_catalog()
          for index in basin_id_list:
            bid = all_basins[index]
            key = 'basin:%s'%bid
            # Check to ensure its not a new basin and that it exists in the DB
            if self.catalog.exists(key):
              logging.debug('KEY EXISTS: %s', key)
              selbasin = self.catalog.hgetall(key)
            else:
              logging.debug('NO KEY: %s\n%s', key, self.data[key])
              selbasin = self.data[key]
            selected_basin_list.append(selbasin)

      bench.mark('GlobalAnalysis')

    # Generate new starting positions
      jcqueue = OrderedDict()
      src_traj_list = []
      for basin in selected_basin_list:


        #  TODO:  DET HOW TO RUN FOLLOW ON FROM GEN SIMS
        src_traj_list.append(basin['traj'])
        if basin['traj'].startswith('desh'):
          global_params = getSimParameters(self.data, 'deshaw')
          fileno = int(basin['traj'][-4:])
          frame = int(basin['mindex'])
          jcID, config = generateDEShawJC(fileno, frame)
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

      with self.catalog.pipeline() as pipe:
        for basin in selected_basin_list:
          pipe.rpush('executed', basin['id'])
        pipe.execute()

      # Append new distance values
      if EXPERIMENT_NUMBER in [14, 16]:
        # Save Ik delta to disk
        logging.info("Saving Delta Itemset (to disk)")
        pickle.dump(Ik_delta, open(settings.datadir + '/iset_delta.p', 'wb'))

        with self.catalog.pipeline() as pipe:
          pipe.set('lattice:max_fis', pickle.dumps(base_lattice.max_fis))
          pipe.set('lattice:low_fis', pickle.dumps(base_lattice.low_fis))
          pipe.set('lattice:dlat', pickle.dumps(base_lattice.dlat))
          pipe.execute()

      if EXPERIMENT_NUMBER == 15:
        with self.catalog.pipeline() as pipe:
          pipe.set('lattice:max_fis', pickle.dumps(base_lattice.max_fis))
          pipe.set('lattice:low_fis', pickle.dumps(base_lattice.low_fis))
          pipe.set('lattice:dlat', pickle.dumps(base_lattice.dlat))
          pipe.execute()
        self.catalog.set('lattice:iset', pickle.dumps(base_lattice.Ik))


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

