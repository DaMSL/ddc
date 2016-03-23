#!/usr/bin/env python

import argparse
import os
import sys
import shutil
import time
import fcntl
import logging
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
from datatools.pca import project_pca
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient

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

  # EXECUTE SIMULATION ---------------------------------------------------------
    settings = systemsettings()
    # Prepare & source to config file
    with open(self.data['sim_conf_template'], 'r') as template:
      source = template.read()

    # Prepare working directory, input/output files
    conFile = os.path.join(job['workdir'], job['name'] + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file

    # >>>>Storing DCD into shared memory on this node

    USE_SHM = True
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
    bench = microbench()
    bench.start()

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
      #  TODO: Pipeline all
      # off-by-1: append list returns size (not inserted index)
      #  ADD index to catalog
    file_idx = self.catalog.append({'xid:filelist': [job['dcd']]})[0]
    delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
    global_idx = self.catalog.append({'xid:reference': delta_xid_index})
    global_xid_index_slice = [x-1 for x in global_idx]
    self.data[key]['xid:start'] = global_xid_index_slice[0]
    self.data[key]['xid:end'] = global_xid_index_slice[-1]
    bench.mark('Indx_Update')


  # 3. Update higher dimensional index
    # Logical Sequence # should be unique seq # derived from manager (provides this
    #  worker's instantiation with a unique ID for indexing)
    mylogical_seqnum = str(self.seqNumFromID())
    self.catalog.hset('anl_sequence', job['name'], mylogical_seqnum)

    # INSERT NEW points here into cache/archive .. OR JUS LET CACHE DEAL WITH IT
    logging.debug(" Loading new conformations into cache....TODO: NEED CACHE LOC")
    # for i in range(traj.n_frames):
    #   cache.insert(global_xid_index_slice[i], traj.xyz[i])

  # 4a. Subspace Calcuation: RMS
    #------ A:  RMSD-ALPHA  ------------------
      #     S_A = rmslist
    logging.debug("3. RMS Calculation")
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
    rmsraw = calc_rmsd(alpha_dist, self.data['centroid'])
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 4. Account for noise
    #    For now: noise is user-defined; TODO: Factor in to Kalman Filter
    noise = DEFAULT.OBS_NOISE
    stepsize = 500 if 'interval' not in job else int(job['interval'])
    nwidth = noise//(2*stepsize)
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)

    # Notes: Delta_S == rmslist
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

    logging.debug("RMS CHECK......")
    for i in rmslist:
      logging.debug("  %s", str(np.argsort(i)))

    # 3. Append new points into the data store. 
    #    r_idx is the returned list of indices for each new RMS point
    #  TODO: DECIDE on retaining a Reservoir Sample
    #    for each bin OR to just cache all points (per strata)
    #  Reservoir Sampliing is Stratified by subspaceHash
    # logging.debug('Creating reservoir Sample')
    # reservoirSampling(self.catalog, traj.xyz, rIdx, subspaceHash, 
    #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
    #     'rms', lambda key: '%d_%d' % key)
    # r_idx = []
    pipe = self.catalog.pipeline()
    for si in rmslist:
      pipe.rpush('subspace:rms', bytes(si))
    idxlist = pipe.execute()
    # for i in idxlist:
    #   r_idx.append(int(i) - 1)

    logging.debug("R_Index Created (rms).")

    # 4. Apply Heuristics Labeling
    logging.debug('Applying Labeling Heuristic')
    rmslabel = []

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
      pipe.rpush('varbin:rms:%d_%d' % (A, B), global_xid_index_slice[i])

    pipe.execute()
    # Update Catalog
    idxcheck = self.catalog.append({'label:rms': rmslabel})
    bench.mark('RMS')


  # 5b. Subspace Calcuation: PCA
  #------ B:  PCA  -----------------
    # 1. Project Pt to PC's for each conform (top 3 PC's)
    logging.debug("Using following PCA Vectors: %s", str(self.data['pcaVectors'].shape))

    pc_proj = project_pca(alpha.xyz, self.data['pcaVectors'], settings.PCA_NUMPC)

    # pcalist = datareduce.PCA(traj.xyz, self.data['pcaVectors'], numpc=3)

    # 2. Apend subspace in catalog
    p_idx = []
    pipe = self.catalog.pipeline()
    for si in pc_proj:
      pipe.rpush('subspace:pca', bytes(si))
    idxlist = pipe.execute()
    for i in idxlist:
      p_idx.append(int(i) - 1)
    logging.debug("P_Index Created (pca) for delta_S_pca")

    # 3. Performing tiling over subspace
    #   For Now: Load entire tree into local memory

    #  TODO:  DECONFLICT CONSISTENCY IN UPDATE TO KDTREE -- FOR NOW USE LOCK

    # while True:
    #   lock = self.catalog.get('hcube:pca:LOCK')
    #   if lock is None or int(lock) == 0:
    #     logging.info('HCube KDTree is available for updating')
    #     break
    #   logging.info('Waiting to acquire the HCube Lock.....')
    #   time.sleep(3)

    # logging.info('Acquired the HCube Lock!')
    # self.catalog.set('hcube:pca:LOCK', 1)
    # self.catalog.expire('hcube:pca:LOCK', 30)

    # hcube_mapping = json.loads(self.catalog.get('hcube:pca'))
    # logging.debug('# Loaded keys = %d', len(hcube_mapping.keys()))

    # # 4. Pull entire Subspace (for now)  
    # #   Note: this is more efficient than inserting new points
    # #   due to underlying Redis Insertions / Index look up
    # #   If this become a bottleneck, may need to write custom redis client
    # #   connection to persist connection and keep socket open (on both ends)
    # #   Or handle some of this I/O server-side via Lua scipt
    # packed_subspace = self.catalog.lrange('subspace:pca', 0, -1)
    # subspace_pca = np.zeros(shape=(len(packed_subspace), settings.PCA_NUMPC))
    # for x in packed_subspace:
    #   subspace_pca = np.fromstring(x, dtype=np.float32, count=settings.PCA_NUMPC)

    # # TODO: accessor function is for 1 point (i) and 1 axis (j). 
    # #  Optimize by changing to pipeline  retrieval for all points given 
    # #  a list of indices with an axis (if nec'y)
    # logging.debug("Reconstructing the tree...")
    # hcube_tree = KDTree.reconstruct(hcube_mapping, subspace_pca)

    # logging.debug("Inserting Delta_S_pca into KDtree (hcubes)")
    # for i in range(len(pc_proj)):
    #   hcube_tree.insert(pc_proj[i], p_idx[i])

    # # Ensure hcube_tree is written to catalog
    # #  TODO:  DECONFLICT CONSISTENCY IN UPDATE TO KDTREE -- FOR NOW USE LOCK
    # encoded = json.dumps(hcube_tree.encode())
    # logging.info('Storing in catalog')
    # pipe = self.catalog.pipeline()
    # pipe.delete('hcube:pca')
    # pipe.set('hcube:pca', encoded)
    # pipe.delete('hcube:pca:LOCK')
    # pipe.execute()
    # logging.info('PCA HyperCube KD Tree update (lock released)')

    # TODO: DECIDE on retaining a Reservoir Sample
    # reservoirSampling(self.catalog, traj.xyz, r_idx, subspaceHash, 
    #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
    #     'pca', 
    #     lambda key: '%d_%d' % key)
    bench.mark('PCA')


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
    return [traj.n_frames]

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
