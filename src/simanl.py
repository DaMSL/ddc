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

from core.common import *
from core.slurm import slurm
from core.kvadt import kv2DArray
from core.kdtree import KDTree
from macro.macrothread import macrothread
import mdtools.deshaw as deshaw
import datatools.datareduce as dr
from datatools.rmsd import calc_rmsd
from datatools.pca import project_pca, calc_pca, calc_kpca
from datatools.approx import ReservoirSample
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient
from bench.timer import microbench


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

PARALLELISM = 24
SIM_STEP_SIZE = 2

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

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('namd/2.10')
    # self.modules.add('namd/2.10-mpi')
    self.modules.add('redis')
    # self.slurmParams['share'] = None

    self.slurmParams['cpus-per-task'] = PARALLELISM
    self.slurmParams['time'] = '2:00:0'

    self.skip_simulation = False

    self.parser.add_argument('-a', '--analysis', action='store_true')
    args = self.parser.parse_args()
    if args.analysis:
      logging.info('SKIPPING SIMULATION')
      self.skip_simulation = True


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

  # def preparejob(self, job):
  #   logging.debug('Simlation is preparing job %s', job)
  #   key = wrapKey('jc', i)
  #   params = self.catalog.hgetall(key)
  #   logging.debug(" Job Candidate Params:")
  #   for k, v in params.items():
  #     logging.debug("    %s: %s" % (k, v))
    # if 'parallel' in job:
    #   numnodes = job['parallel']
    #   total_tasks = numnodes * 24       # Total # cpu per node should be detectable
    #   self.modules.add('namd/2.10-mpi')
    #   self.slurmParams['partition'] = 'parallel'
    #   self.slurmParams['ntasks-per-node'] = 24
    #   self.slurmParams['nodes'] = numnodes
    #   del self.slurmParams['cpus-per-task']


  def fetch(self, i):
    # Load parameters from catalog

    key = wrapKey('jc', i)
    params = self.catalog.hgetall(key)
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    self.addMut(key, params)

    # Increment launch count
    # A, B = eval(params['targetBin'])
    # logging.debug("Increment Launch count for %s", params['targetBin'])
    # self.data['launch'][A][B] += 1

    return params

  def execute(self, job):

  # PRE-PREOCESS ---------------------------------------------------------
    settings = systemsettings()
    bench = microbench('sim', self.seqNumFromID())
    bench.start()

    # Prepare working directory, input/output files
    conFile = os.path.join(job['workdir'], job['name'] + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file
    USE_SHM = True

    # Load common settings:
    # Framesize in ps
    frame_size = (settings.DCDFREQ * settings.SIM_STEP_SIZE) // 1000


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
      sim_length = settings.SIM_STEP_SIZE * int(job['runtime'])
      sim_realtime = bench.delta_last()
      sim_run_ratio =  (sim_realtime/60) / (sim_length/1000000)
      logging.info('##SIM_RATIO %6.3f  min-per-ns-sim', sim_run_ratio)

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

  # 2. Update Catalog with HD points (TODO: cache this)
    file_idx = self.catalog.append({'xid:filelist': [job['dcd']]})[0]
    delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
    global_idx_recv = self.catalog.append({'xid:reference': delta_xid_index})
    global_index = [x-1 for x in global_idx_recv]
    # Note: Pipelined insertions should return contiguous set of index points
    self.data[key]['xid:start'] = global_index[0]
    self.data[key]['xid:end'] = global_index[-1]
    bench.mark('Indx_Update')


  # 3. Update higher dimensional index
    # Logical Sequence # should be unique seq # derived from manager (provides this
    #  worker's instantiation with a unique ID for indexing)
    mylogical_seqnum = str(self.seqNumFromID())
    self.catalog.hset('anl_sequence', job['name'], mylogical_seqnum)

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
    rmsraw = calc_rmsd(alpha, self.data['centroid'], weights=cw)
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 4. Account for noise
    #    For now: noise is user-defined; TODO: Factor in to Kalman Filter
    noise = DEFAULT.OBS_NOISE
    stepsize = 500 if 'interval' not in job else int(job['interval'])
    nwidth = noise//(2*stepsize)
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)

    # Notes: Delta_S == rmslist
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

    # logging.debug("RMS CHECK......")
    # for i in rmslist:
    #   logging.debug("  %s", str(np.argsort(i)))

    # 3. Append new points into the data store. 
    pipe = self.catalog.pipeline()
    for si in rmslist:
      pipe.rpush('subspace:rms', bytes(si))
    idxlist = pipe.execute()

    # 4. Apply Heuristics Labeling
    logging.debug('Applying Labeling Heuristic')
    rmslabel = []
    binlist = [(a, b) for a in range(numLabels) for b in range(numLabels)]
    labeled_points = {b: [] for b in binlist}
    groupbystate = [[] for i in range(numLabels)]
    pipe = self.catalog.pipeline()
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
      pipe.rpush('varbin:rms:%d_%d' % (A, B), global_index[i])
      labeled_points[(A, B)].append(i)

      # Group high-dim point by state
      # TODO: Consider grouping by stateonly or well/transitions (5 vs 10 bins)
      groupbystate[A].append(alpha.xyz[i])

    pipe.execute()
    # Update Catalog
    idxcheck = self.catalog.append({'label:rms': rmslabel})

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
    pipe = self.catalog.pipeline()
    for i, si in enumerate(covar):
      local_index = int(i * frame_size * SLIDE_AMT_NS)
      pipe.rpush('subspace:covar:pts', bytes(si))
      pipe.rpush('subspace:covar:xid', global_index[local_index])
      pipe.rpush('subspace:covar:fidx', (file_idx, local_index))
    idxlist = pipe.execute()


  # 4-C. Subspace Calcuation: Kernel PCA using Covariance Matrix, 200ns windows, Alpha Filter
  #------ C:  Kernel PCA by state  -----------------
  #  Update PCA Vectors for each state with new data
    logging.info('---- PCA Calculation per state over Alpha Filter in cartesian Space ----')
    # TODO:  This will eventually get moved into a User process macrothread 
    #   which will set in between analysis and controller. 
    # For now, we're recalculating using a lock

    # Check if vectors need to be recalculated
    # Connect to reservoir samples
    # TODO: Catalog or Cache?
    reservoir = ReservoirSample('rms', self.catalog)
    STALENESS_FACTOR = .25   # Recent updates account for 25% of the sample (Just a guess)
    for A in range(numLabels):
      if len(groupbystate[A]) == 0:
        logging.info('No data received for state %d.  Not processing this state here.')
        continue
      updateVectors = False
      pc_vectors = self.catalog.loadNPArray('subspace:pca:vectors:%d' % A)
      logging.info('PCA:  Checking if current vectors for state %d are out of date', A)
      rsize = reservoir.getsize(A)
      changelist = self.catalog.lrange('subspace:pca:updates:%d' % A, 0, -1)
      changeamt = np.sum([int(x) for x in changelist])
      staleness = changeamt/rsize if rsize > 0 else 0
      logging.info('##STALE PCA_%d  %.3f', A, staleness)
      logging.info('Vectors are %f%% stale (rsize = %s   changeamt = %d)', staleness, rsize, changeamt)

      # CHECK if PCA needs to be updated (??? Should we delete all old PC data???)
      #  Thought is that this will change early on, but should not change after some time
      #    and this stabilize the PC Vectors for each state (considering the RMSD centroids
      #    are not going to change)
      traindata = None
      if pc_vectors is None or staleness > STALENESS_FACTOR:
        logging.info('PCA Vectors are old, checking if out of tolerance')
        # PCA Vectors are out of date for this bin. Update the vectors....
        sampledata = reservoir.get(A)
        logging.debug('SampleData:  %s', len(sampledata))
        newdata = groupbystate[A]
        logging.debug('NewData:  %s', len(newdata))
        if len(sampledata) + len(newdata) < 2:
          logging.info("Not enough data to update PC's. Skipping-PCA-%d", A)
          continue
        # Include recent data
        if len(sampledata) == 0:
          traindata = np.array(groupbystate[A])
        elif len(groupbystate[A]) == 0:
          traindata = np.array(sampledata)
        else:
          traindata = np.concatenate((np.array(sampledata), np.array(groupbystate[A])))
        logging.info('   Performing Kernel PCA (Sigmoid) for state %d using traindata of size %d', A, len(traindata))

        # NOTE: Pick PCA Algorithm HERE
        pca = calc_kpca(np.array(traindata), kerneltype='sigmoid')
        bench.mark('CalcKPCA_%d'%A)
        new_vect = pca.components_
        updateVectors = True
        if pc_vectors is not None:
          logging.info('# Comp (99%% of PC Coverage):  Before = %d    After = %d', len(pc_vectors), len(new_vect))
          logging.info('Delta of top 90%%:')
          coverage = 0.
          delta = []
          for i, variance in enumerate(pca.explained_variance_ratio_):
            delta.append(LA.norm(pc_vectors[i] - new_vect[i]))
            coverage += variance
            if coverage > .9:
              break
          logging.info('Variance by PC:')
          for i, d in enumerate(delta):
            logging.info(' PC #  %d   var= %6.3f', i, d)
          logging.info('Total delta is %f', np.sum(delta))
          logging.info('NOTE: This is a check for PCA Change. TODO: Det threshold to compare delta with')

        lock = self.catalog.lock_acquire('subspace:pca:vectors:%d' % A)
        if lock is None:
          logging.info('Could not lock the PC Vectors for State %d. Not updating', A)
        else:
          self.catalog.delete('subspace:pca:vectors:%d' % A)
          self.catalog.storeNPArray(new_vect, 'subspace:pca:vectors:%d' % A)
          pc_vectors = new_vect

          # Reset change log
          self.catalog.ltrim('subspace:pca:updates:%d' % A, len(changelist), -1)
        bench.mark('ConcurrPCAWrite_%d'%A)

      else:
        logging.info('  PCA Vectors are fresh -- no need to change them')


      logging.debug('alpha %s     pc_v %s    numpc %s', 
        str(alpha.xyz.shape), str(pc_vectors), str(settings.PCA_NUMPC))
      #  TODO:  Should we delete old points and re-project reservoir as well??
      #   Or keep it and apply decaying factor to the date
      #  DOING BOTH FOR COMPARISON
      bench.mark('start_proj')
      logging.info('Projecting %d points onto %d PCs for state %d', len(groupbystate[A]), settings.PCA_NUMPC, A)
      pc_proj = project_pca(groupbystate[A], pc_vectors, settings.PCA_NUMPC)
      bench.mark('ProjPCA_%d'%A)
      if updateVectors:     
        logging.info('Comparson: Projecting %d points onto %d PCs for state %d', len(traindata), settings.PCA_NUMPC, A)
        pc_proj_rsamp = project_pca(traindata, pc_vectors, settings.PCA_NUMPC)
        bench.mark('ProjPCA_RSAMP_%d'%A)

      # 2. Apend subspace in catalog
      p_idx = []
      pipe = self.catalog.pipeline()
      key = 'subspace:pca:%d' % A
      keyrsamp = 'subspace:pcarsamp:%d' % A
      for si in pc_proj:
        pipe.rpush(key, bytes(si))
      if updateVectors:
        for si in pc_proj_rsamp:
          pipe.rpush(keyrsamp, bytes(si))
      idxlist = pipe.execute()
      logging.info('Stored %d NEWlower dim points %s', len(idxlist), key)
      for i in idxlist:
        p_idx.append(int(i) - 1)
      logging.debug("P_Index Created (pca) for delta_S_pca")

    bench.mark('PCA')

    # 4. Update reservoir sample
    logging.debug('Updating reservoir Sample')
    num_inserted = {A: 0 for A in range(numLabels)}
    for A, ptlist in enumerate(groupbystate):
      if len(ptlist) == 0:
        continue
      num_inserted[A] = reservoir.insert(A, ptlist)

    pipe = self.catalog.pipeline()
    for A, num in enumerate(num_inserted):
      if num > 0:
        pipe.rpush('subspace:pca:updates:%d' % A, num)
    pipe.execute()
    bench.mark('Reservoir')


  # ---- POST PROCESSING
    if USE_SHM:
      shutil.rmtree(ramdisk)
      # shm_contents = os.listdir('/dev/shm')
      shm_contents = os.listdir('/tmp')
      logging.debug('Ramdisk contents (should be empty of DDC) : %s', str(shm_contents))
    
    # For benchmarching:
    # print('##', job['name'], dcdfilesize/(1024*1024*1024), traj.n_frames)
    bench.show()

    # return [job['name']]
    # Return # of observations (frames) processed
    return [numConf]

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
