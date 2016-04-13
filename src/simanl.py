#!/usr/bin/env python

import argparse
import os
import sys
import shutil
import time
import fcntl
import logging
import math
from collections import namedtuple, deque


# For efficient zero-copy file x-fer
from sendfile import sendfile
import mdtraj as md
import numpy as np
from numpy import linalg as LA
import redis

from core.common import *
from core.slurm import slurm
from core.kvadt import kv2DArray
from core.kdtree import KDTree
from macro.macrothread import macrothread
import mdtools.deshaw as deshaw
import datatools.datareduce as dr
from datatools.rmsd import calc_rmsd
# from datatools.pca import project_pca, calc_pca, calc_kpca
from datatools.pca import PCAnalyzer, PCAKernel
from datatools.approx import ReservoirSample
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient
from overlay.overlayException import OverlayNotAvailable
from bench.timer import microbench
from bench.stats import StatCollector



__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

PARALLELISM = 24
SIM_STEP_SIZE = 2

# Factor used to "simulate" long running jobs using shorter sims
SIMULATE_RATIO = 1

class simulationJob(macrothread):
  """Macrothread to run MD simuation. Each worker runs one simulation
  using the provided input parameters.
    Input -> job candidate key in the data store 
    Execute -> creates the config file and calls namd to run the simulation. 
  """
  def __init__(self, fname, jobnum = None):
    macrothread.__init__(self, fname, 'sim')

    # State Data for Simulation MacroThread -- organized by state
    self.setStream('jcqueue', 'completesim')
    self.addImmut('simSplitParam')
    self.addImmut('simDelay')
    self.addImmut('centroid')
    self.addImmut('pcaVectors')
    self.addImmut('numLabels')
    self.addImmut('terminate')
    self.addImmut('sim_conf_template')
    self.addImmut('dcdfreq')
    self.addImmut('runtime')
    self.addImmut('obs_noise')
    self.addImmut('sim_step_size')
    self.addAppend('xid:filelist')

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('namd/2.10')
    # self.modules.add('namd/2.10-mpi')
    self.modules.add('redis')
    # self.slurmParams['share'] = None

    self.slurmParams['cpus-per-task'] = PARALLELISM
    # self.slurmParams['time'] = '2:00:0'

    self.skip_simulation = False

    self.parser.add_argument('-a', '--analysis', action='store_true')
    self.parser.add_argument('--useid')
    args = self.parser.parse_args()
    if args.analysis:
      logging.info('SKIPPING SIMULATION')
      self.skip_simulation = True

    if args.useid:
      self.job_id = args.useid

  def term(self):
    return False

  def split(self):

    if len(self.data['jcqueue']) == 0:
      return [], None
    split = int(self.data['simSplitParam'])
    immed = self.data['jcqueue'][:split]
    return immed, split

  def configElasPolicy(self):
    self.delay = self.data['simDelay']

  def fetch(self, i):
    # Load parameters from catalog
    key = wrapKey('jc', i)
    params = self.catalog.hgetall(key)
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))
    self.addMut(key, params)
    return params

  def execute(self, job):

  # PRE-PREOCESS ---------------------------------------------------------
    settings = systemsettings()
    bench = microbench('sim_%s' % settings.name, self.seqNumFromID())
    bench.start()
    stat  = StatCollector('sim_%s' % settings.name, self.seqNumFromID())
    mylogical_seqnum = str(self.seqNumFromID())

    # Prepare working directory, input/output files
    conFile = os.path.join(job['workdir'], job['name'] + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file
    USE_SHM = True

    frame_size = (SIMULATE_RATIO * int(job['interval'])) / (1000)
    logging.info('Frame Size is %f  Using Sim Ratio of 1:%d', frame_size, SIMULATE_RATIO)

  # EXECUTE SIMULATION ---------------------------------------------------------
    if self.skip_simulation:

      logging.info('1. SKIPPING SIMULATION.....')
      USE_SHM = False

      job['dcd'] = dcdFile
      key = wrapKey('jc', job['name'])
      self.data[key]['dcd'] = dcdFile

    else:
      logging.info('1. Run Simulation')
      # Prepare & source to config file
      with open(self.data['sim_conf_template'], 'r') as template:
        source = template.read()

      # >>>>Storing DCD into shared memory on this node

      if USE_SHM:
        # ramdisk = '/dev/shm/out/'
        ramdisk = '/tmp/ddc/'
        if not os.path.exists(ramdisk):
          os.mkdir(ramdisk)
        job['outputloc'] = ramdisk
        dcd_ramfile = os.path.join(ramdisk, job['name'] + '.dcd')
      else:
        job['outputloc'] = ''

      with open(conFile, 'w') as sysconfig:
        sysconfig.write(source % job)
        logging.info("Config written to: " + conFile)

      # # Run simulation in parallel
      # if 'parallel' in job:
      #   numnodes = job['parallel']
      #   total_tasks = numnodes * 24
      #   cmd = 'mpiexec -n %d namd2 %s > %s'  % (total_tasks, conFile, logFile)

      # # Run simulation single threaded
      # else:
      #   cmd = 'namd2 %s > %s' % (conFile, logFile)

      # cmd = 'mpirun -n %d namd2 %s > %s' % (PARALLELISM, conFile, logFile)
      check = executecmd('module list')
      logging.debug('%s', check)

      cmd = 'namd2 +p%d %s > %s' % (PARALLELISM, conFile, logFile)

      #  MICROBENCH #1 (file to Lustre)
      # logging.debug("Executing Simulation:\n   %s\n", cmd)
      # bench = microbench()
      # bench.start()
      # stdout = executecmd(cmd)
      # logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
      # bench.mark('SimExec:%s' % job['name'])
      # shm_contents = os.listdir('/dev/shm/out')
      # logging.debug('Ramdisk contents (should have files) : %s', str(shm_contents))
      # shutil.copy(ramdisk + job['name'] + '.dcd', job['workdir'])
      # logging.info("Copy Complete to Lustre.")
      # bench.mark('CopyLustre:%s' % job['name'])
      # shutil.rmtree(ramdisk)
      # shm_contents = os.listdir('/dev/shm')
      # logging.debug('Ramdisk contents (should be empty) : %s', str(shm_contents))
      # bench.show()


      logging.debug("Executing Simulation:\n   %s\n", cmd)

      stdout = executecmd(cmd)

      logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
      bench.mark('SimExec:%s' % job['name'])

      # Internal stats
      sim_length = self.data['sim_step_size'] * int(job['runtime'])
      sim_realtime = bench.delta_last()
      sim_run_ratio =  (sim_realtime/60) / (sim_length/1000000)
      logging.info('##SIM_RATIO %6.3f  min-per-ns-sim', sim_run_ratio)
      stat.collect('sim_ratio', sim_run_ratio)

      if USE_SHM:
        shm_contents = os.listdir(ramdisk)
        logging.debug('Ramdisk contents (should have files) : %s', str(shm_contents))

        if not os.path.exists(dcd_ramfile):
          logging.warning("DCD FILE NOT FOUND!!!! Wait 10 seconds for sim to close it (???)")
          time.sleep(10)

        if not os.path.exists(dcd_ramfile):
          logging.warning("DCD STIILL FILE NOT FOUND!!!!")
        else:
          logging.info("DCD File was found")

      # #  MICROBENCH #2 (file to Alluxio)
      # allux = AlluxioClient()
      # # copy to Aluxio FS
      # allux.put(ramdisk + job['name'] + '.dcd', '/')
      # logging.info("Copy Complete to Alluxio.")
      # bench.mark('CopyAllux:%s' % job['name'])

      # And copy to Lustre
      # shutil.copy(ramdisk + job['name'] + '.dcd', job['workdir'])
      # And copy to Lustre (usng zero-copy):
      if USE_SHM:
        src  = open(dcd_ramfile, 'rb')
        dest = open(dcdFile, 'w+b')
        offset = 0
        dcdfilesize = os.path.getsize(dcd_ramfile)
        while True:
          sent = sendfile(dest.fileno(), src.fileno(), offset, dcdfilesize)
          if sent == 0:
            break
          offset += sent
        logging.info("Copy Complete to Lustre.")
        bench.mark('CopyLustre:%s' % job['name'])
      
      # TODO: Update job's metadata
      key = wrapKey('jc', job['name'])
      self.data[key]['dcd'] = dcdFile

  # ANALYSIS   ------- ---------------------------------------------------------
    #  ANALYSIS ALGORITHM
  # 1. With combined Sim-analysis: file is loaded locally from shared mem
    logging.debug("2. Load DCD")

    # Load full higher dim trajectory
    # traj = datareduce.filter_heavy(dcd_ramfile, job['pdb'])
    if USE_SHM:
      traj = md.load(dcd_ramfile, top=job['pdb'])
    else:
      traj = md.load(dcdFile, top=job['pdb'])
    bench.mark('File_Load')
    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))


  #  DIMENSIONALITY REDUCTION --------------------------------------------------
  # 4-A. Subspace Calcuation: RMS using Alpha-Filter
    #------ A:  RMSD-ALPHA  ------------------
      #     S_A = rmslist
    logging.info('---- RMSD Calculation against pre-defined centroids ----')
      #  RMSD is calculated on the Ca ('alpha') atoms in distance space
      #   whereby all pairwise distances are calculated for each frame.
      #   Pairwise distances are plotted in euclidean space
      #   Distance to each of the 5 pre-calculated centroids is calculated

    # 1. Filter to Alpha atoms
    alpha = traj.atom_slice(deshaw.FILTER['alpha'])

    # 2. (IF USED) Convert to distance space: pairwise dist for all atom combinations
    # alpha_dist = dr.distance_space(alpha)

    # 3. Calc RMS for each conform to all centroids
    # Heuristic centroid weight (TODO: make this trained)
    cw = [.92, .94, .96, .99, .99]

    numLabels = len(self.data['centroid'])
    numConf = len(traj.xyz)
    stat.collect('numpts',numConf)
    rmsraw = calc_rmsd(alpha, self.data['centroid'], weights=cw)
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 4. Account for noise
    #    For now: noise is user-defined; TODO: Factor in to Kalman Filter
    noise = self.data['obs_noise']
    stepsize = 500 if 'interval' not in job else int(job['interval'])
    nwidth = noise//(2*stepsize)
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)

    # Notes: Delta_S == rmslist
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

    # logging.debug("RMS CHECK......")
    # for i in rmslist:
    #   logging.debug("  %s", str(np.argsort(i)))


    # 4. Apply Heuristics Labeling
    logging.debug('Applying Labeling Heuristic')
    rmslabel = []
    binlist = [(a, b) for a in range(numLabels) for b in range(numLabels)]
    label_count = {b: 0 for b in binlist}
    groupbystate = [[] for i in range(numLabels)]
    for i, rms in enumerate(rmslist):
      #  Sort RMSD by proximity & set state A as nearest state's centroid
      prox = np.argsort(rms)
      A = prox[0]

      #  Calc Absolute proximity between nearest 2 states' centroids
      # THETA Calc derived from static run. it is based from the average std dev of all rms's from a static run
      #   of BPTI without solvent. It could be dynamically calculated, but is hard coded here
      #  The theta is divided by four based on the analysis of DEShaw:
      #   est based on ~3% of DEShaw data in transition (hence )
      # avg_stddev = 0.34119404492089034
      theta = settings.RMSD_THETA

      # NOTE: Original formulate was relative. Retained here for reference:  
      # Rel vs Abs: Calc relative proximity for top 2 nearest centroids   
      # relproximity = rms[A] / (rms[A] + rms[rs[1]])
      # B = rs[1] if relproximity > (.5 - theta) else A
      # proximity = abs(rms[prox[1]] - rms[A]) / (rms[prox[1]] + rms[A])  #relative
      proximity = abs(rms[prox[1]] - rms[A])    #abs

      #  (TODO:  Factor in more than top 2, better noise)
      #  Label secondary sub-state
      B = prox[1] if proximity < theta else A
      rmslabel.append((A, B))

      # Add this index to the set of indices for this respective label
      #  TODO: Should we evict if binsize is too big???
      logging.debug('Label for observation #%3d: %s', i, str((A, B)))
      label_count[(A, B)] += 1

      # Group high-dim point by state
      # TODO: Consider grouping by stateonly or well/transitions (5 vs 10 bins)
      groupbystate[A].append(i)

    stat.collect('observe', label_count)
    bench.mark('RMS')
    logging.info('Labeled the following by State:')
    for A in range(numLabels):
      logging.info('## %d  %d', A, len(groupbystate[A]))

  # 4-B. Subspace Calcuation: COVARIANCE Matrix, 200ns windows, Full Protein
  #------ B:  Covariance Matrix  -----------------
    # 1. Project Pt to PC's for each conform (top 3 PC's)
    logging.info('---- Covariance Calculation on 200ns windows (Full Protein, cartesian Space) ----')

    # Calculate Covariance over 200 ps Windows sliding every 100ps
    #  These could be user influenced...
    WIN_SIZE_NS = .2
    SLIDE_AMT_NS = .1
    logging.debug("Calculating Covariance over trajectory. frame_size = %.1f, WINSIZE = %dps, Slide = %dps", 
      frame_size, WIN_SIZE_NS*1000, SLIDE_AMT_NS*1000)
    covar = dr.calc_covar(alpha.xyz, WIN_SIZE_NS, frame_size, slide=SLIDE_AMT_NS)
    bench.mark('CalcCovar')
    logging.debug("Calcualted %d covariance matrices. Storing variances", len(covar)) 


  #  BARRIER: WRITE TO CATALOG HERE -- Ensure Catalog is available
    # try:
    self.wait_catalog()
    # except OverlayNotAvailable as e:
    #   logging.warning("Catalog Overlay Service is not available. Scheduling ASYNC Analysis")


  # Update Catalog with 1 Long Atomic Transaction  
    global_index = []
    with self.catalog.pipeline() as pipe:
      while True:
        try:
          logging.debug('Update Filelist')
          pipe.watch(wrapKey('jc', job['name']))
          file_idx = pipe.rpush('xid:filelist', job['dcd']) - 1
          # HD Points
          logging.debug('Update HD Points')
          for x in range(traj.n_frames):
            # Note: Pipelined insertions "should" return contiguous set of index points
            index = pipe.rpush('xid:reference', (file_idx, x)) - 1
            global_index.append(index - 1) 

          pipe.multi()
          logging.debug('Update RMS Subspace')
          for x in range(traj.n_frames):
            A, B = rmslabel[i]
            # Labeled Observation (from RMSD)
            pipe.rpush('label:rms', rmslabel[x])
            pipe.rpush('varbin:rms:%d_%d' % (A, B), index)
            pipe.rpush('subspace:rms', bytes(rmslist[x]))

          logging.debug('Update OBS Counts')
          for b in binlist:
            pipe.rpush('observe:rms:%d_%d' % b, label_count[b])
          pipe.incr('observe:count')
          pipe.hset('anl_sequence', job['name'], mylogical_seqnum)

          logging.debug('Update Covar Subspace')
          for i, si in enumerate(covar):
            local_index = int(i * frame_size * SLIDE_AMT_NS)
            pipe.rpush('subspace:covar:pts', bytes(si))
            pipe.rpush('subspace:covar:xid', global_index[local_index])
            pipe.rpush('subspace:covar:fidx', (file_idx, local_index))

          logging.debug('Executing')
          pipe.execute()
          break
        except redis.WatchError as e:
          logging.debug('WATCH ERROR')
          continue

    self.data[key]['xid:start'] = global_index[0]
    self.data[key]['xid:end'] = global_index[-1]
    bench.mark('Indx_Update')
    stat.collect('numcovar', len(covar))

  # (Should we Checkpoint here?)

  # 4-C. Subspace Calcuation: PCA BY Strata (PER STATE) using Alpha Filter
  #------ C:  GLOBAL PCA by state  -----------------
  #  Update PCA Vectors for each state with new data
    logging.info('---- PCA Calculation per state over Alpha Filter in cartesian Space ----')
    # TODO:  This will eventually get moved into a User process macrothread 
    #   which will set in between analysis and controller. 
    # For now, we're recalculating using a lock

    # Check if vectors need to be recalculated
    # Connect to reservoir samples
    # TODO: Catalog or Cache?
    reservoir = ReservoirSample('rms', self.catalog)
    # STALENESS_FACTOR = .25   # Recent updates account for 25% of the sample (Just a guess)

    num_inserted = {A: 0 for A in range(numLabels)}

    for A in range(numLabels):
      if len(groupbystate[A]) == 0:
        logging.info('No data received for state %d.  Not processing this state here.', A)
        continue

      updateVectors = False
      kpca_key = 'subspace:pca:kernel:%d' % A
      kpca = PCAnalyzer.load(self.catalog, kpca_key)
      newkpca = False
      if kpca is None:
        # kpca = PCAKernel(None, 'sigmoid')
        kpca = PCAKernel(10, 'sigmoid')
        newkpca = True


      logging.info('PCA:  Checking if current vectors for state %d are out of date', A)
      rsize = reservoir.getsize(A)
      tsize = kpca.trainsize

      #  KPCA is out of date is the sample size is 20% larger than previously used  set
      #  Heuristics --- this could be a different "staleness" factor or we can check it some other way
      if newkpca or rsize > (tsize * 1.5):
        logging.info('PCA Kernel is old (Updating it). Trained on data set of size %d. Current reservoir is %d pts.', tsize, rsize)

        #  Should we only use a sample here??? (not now -- perhaps with larger rervoirs or if KPCA is slow
        traindata = reservoir.get(A)
        if newkpca:
          num_hd_pts = len(groupbystate[A])
          logging.info('Projecting %d points on Kernel PCA for state %d', num_hd_pts, A)
          traindata = np.zeros(shape=((num_hd_pts,)+alpha.xyz.shape[1:]), dtype=np.float32)
          for i, index in enumerate(groupbystate[A]):
            np.copyto(traindata[i], alpha.xyz[index])

        if len(traindata) < 2:
          logging.info("Not enough data to update PC's. Skipping-PCA-%d", A)
          continue
        logging.info('   Performing Kernel PCA (Sigmoid) for state %d using traindata of size %d', A, len(traindata))

        kpca.solve(traindata)

        # NOTE: Pick PCA Algorithm HERE
        # pca = calc_kpca(np.array(traindata), kerneltype='sigmoid')
        # pca = calc_pca(np.array(traindata))
        bench.mark('CalcKPCA_%d'%A)

        # new_vect = pca.alphas_.T
        lock = self.catalog.lock_acquire(kpca_key)
        if lock is None:
          logging.info('Could not lock the PC Kernel for State %d. Not updating', A)
        else:
          kpca.store(self.catalog, kpca_key)
          lock = self.catalog.lock_release(kpca_key, lock)
        bench.mark('ConcurrPCAWrite_%d'%A)

        #  NOTE::::  SET MAX # OF PC's STORED AND/OR KPCA N_COMPONENT SIZE (ILO ALL)

        # Project Reservoir Sample to the Kernel and overwrite current set of points
        #  This should only happen up until the reservior is filled
        # If we are approx above to train, be sure to project all reservor points
        if not newkpca:
          logging.info('Clearing and Re-Projecting the entire reservoir of %d points for State %d.', rsize, A)
          rsamp_lowdim = kpca.project(traindata)
          pipe = self.catalog.pipeline()
          pipe.delete('subspace:pca:%d'%A)
          for si in rsamp_lowdim:
            pipe.rpush('subspace:pca:%d'%A, bytes(si))
          pipe.execute()


      else:
        logging.info('PCA Kernel is good -- no need to change them')

      bench.mark('start_ProjPCA')
      num_hd_pts = len(groupbystate[A])
      logging.info('Projecting %d points on Kernel PCA for state %d', num_hd_pts, A)
      hd_pts = np.zeros(shape=((num_hd_pts,)+alpha.xyz.shape[1:]), dtype=np.float32)
      for i, index in enumerate(groupbystate[A]):
        np.copyto(hd_pts[i], alpha.xyz[index])
      pc_proj = kpca.project(hd_pts)
      bench.mark('ProjPCA_%d'%A)

      # 2. Append subspace in catalog
      pipe = self.catalog.pipeline()
      for si in pc_proj:
        pipe.rpush('subspace:pca:%d' % A, bytes(si))
      pipe.execute()

      logging.debug('Updating reservoir Sample')
      num_inserted[A] = reservoir.insert(A, hd_pts)

    bench.mark('PCA')
    pipe = self.catalog.pipeline()
    for A, num in enumerate(num_inserted):
      if num > 0:
        pipe.rpush('subspace:pca:updates:%d' % A, num)
    pipe.execute()

  # ---- POST PROCESSING
    if USE_SHM:
      shutil.rmtree(ramdisk)
      # shm_contents = os.listdir('/dev/shm')
      shm_contents = os.listdir('/tmp')
      logging.debug('Ramdisk contents (should be empty of DDC) : %s', str(shm_contents))
    
    # For benchmarching:
    # print('##', job['name'], dcdfilesize/(1024*1024*1024), traj.n_frames)
    bench.show()
    stat.show()

    # Return # of observations (frames) processed
    return [numConf]

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
