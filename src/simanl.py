#!/usr/bin/env python

import argparse
import os
import sys
import shutil
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
from mdsim.deshaw import deshawReference
import mdsim.datareduce as datareduce
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)

PARALLELISM = 24

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
    self.modules.add('redis')
    # self.slurmParams['share'] = None

    self.slurmParams['cpus-per-task'] = PARALLELISM
    self.slurmParams['time'] = '4:00:0'

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

    # Prepare & source to config file
    with open(self.data['sim_conf_template'], 'r') as template:
      source = template.read()

    # Prepare working directory, input/output files
    conFile = os.path.join(job['workdir'], job['name'] + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file

    # >>>>Storing DCD into shared memory on this node
    ramdisk = '/dev/shm/out/'
    if not os.path.exists(ramdisk):
      os.mkdir(ramdisk)
    job['outputloc'] = ramdisk
    dcd_ramfile = os.path.join(ramdisk, job['name'] + '.dcd')


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
    shm_contents = os.listdir('/dev/shm/out')
    logging.debug('Ramdisk contents (should have files) : %s', str(shm_contents))

    # #  MICROBENCH #2 (file to Alluxio)
    # allux = AlluxioClient()
    # # copy to Aluxio FS
    # allux.put(ramdisk + job['name'] + '.dcd', '/')
    # logging.info("Copy Complete to Alluxio.")
    # bench.mark('CopyAllux:%s' % job['name'])

    # And copy to Lustre
    # shutil.copy(ramdisk + job['name'] + '.dcd', job['workdir'])
    # And copy to Lustre (usng zero-copy):
    src  = open(dcd_ramfile, 'r')
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

    #  ANALYSIS ALGORITHM
    # 1. With combined Sim-analysis: file is loaded locally from shared mem
    logging.debug("2. Load DCD")
    traj = datareduce.filter_heavy(dcd_ramfile, job['pdb'])
    bench.mark('File_Load')
    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))

  # 2. Update Catalog with HD points (TODO: cache this)
      #  TODO: Pipeline all
      # off-by-1: append list returns size (not inserted index)
      #  ADD index to catalog
      # Off by 1 error for index values
    file_idx = self.catalog.append({'xid:filelist': [job['dcd']]})[0]
    delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
    global_idx = self.catalog.append({'xid:reference': delta_xid_index})
    global_xid_index_slice = [x-1 for x in global_idx]
    bench.mark('Indx_Update')

  # 4. Update higher dimensional index
    # Logical Sequence # should be unique seq # derived from manager (provides this
    #  worker's instantiation with a unique ID for indexing)
    mylogical_seqnum = str(self.seqNumFromID())
    self.catalog.hset('anl_sequence', job['name'], mylogical_seqnum)

    # INSERT NEW points here into cache/archive
    logging.debug(" Loading new conformations into cache....TODO: NEED CACHE LOC")
    # for i in range(traj.n_frames):
    #   cache.insert(global_xid_index_slice[i], traj.xyz[i])

    # 5a. Subspace Calcuation: RMS
    #------ A:  RMSD  ------------------
      #     S_A = rmslist

      # 1. Calc RMS for each conform to all centroids
    logging.debug("3. RMS Calculation")
    numLabels = len(self.data['centroid'])
    numConf = len(traj.xyz)
    rmsraw = np.zeros(shape=(numConf, numLabels))
    for i, conform in enumerate(traj.xyz):
      np.copyto(rmsraw[i], np.array([LA.norm(conform-cent) for cent in self.data['centroid']]))
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 2. Account for noise
    #    For now: noise is user-jobured; TODO: Factor in to Kalman Filter
    noise = DEFAULT.OBS_NOISE
    stepsize = 500 if 'interval' not in job else int(job['interval'])
    nwidth = noise//(2*stepsize)
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)

    # Notes: Delta_S == rmslist
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

    # 3. Append new points into the data store. 
    #    r_idx is the returned list of indices for each new RMS point
    #  TODO: DECIDE on retaining a Reservoir Sample
    #    for each bin OR to just cache all points (per strata)
    #  Reservoir Sampliing is Stratified by subspaceHash
    # logging.debug('Creating reservoir Sample')
    # reservoirSampling(self.catalog, traj.xyz, rIdx, subspaceHash, 
    #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
    #     'rms', lambda key: '%d_%d' % key)
    r_idx = []
    pipe = self.catalog.pipeline()
    for si in rmslist:
      pipe.rpush('subspace:rms', bytes(si))
    idxlist = pipe.execute()
    for i in idxlist:
      r_idx.append(int(i) - 1)

    logging.debug("R_Index Created (rms).")

    # 4. Apply Heuristics Labeling
    logging.debug('Applying Labeling Heuristic')
    rmslabel = []
    subspaceHash = {}
    for i, rms in enumerate(rmslist):
      #  Sort RMSD by proximity & set state A as nearest state's centroid
      prox = np.argsort(rms)
      A = prox[0]

      #  Calc Absolute proximity between nearest 2 states' centroids
      # THETA Calc derived from static run. it is based from the average std dev of all rms's from a static run
      #   of BPTI without solvent. It could be dynamically calculated, but is hard coded here
      #  The theta is divided by four based on the analysis of DEShaw:
      #   est based on ~3% of DEShaw data in transition (hence )
      avg_stddev = 0.34119404492089034
      theta = avg_stddev / 4.

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
      if (A, B) not in subspaceHash:
        subspaceHash[(A, B)] = []
        logging.debug("Found Label: %s", str((A, B)))
      subspaceHash[(A, B)].append(i)

    # Update Catalog
    idxcheck = self.catalog.append({'label:rms': rmslabel})
    bench.mark('RMS')


  # 5b. Subspace Calcuation: PCA
  #------ B:  PCA  -----------------
    # 1. Project Pt to PC's for each conform (top 3 PC's)
    logging.debug("Using following PCA Vectors: %s", str(self.data['pcaVectors'].shape))
    pcalist = datareduce.PCA(traj.xyz, self.data['pcaVectors'], numpc=3)

    # 2. Apend subspace in catalog
    p_idx = []
    pipe = self.catalog.pipeline()
    for si in pcalist:
      pipe.rpush('subspace:pca', bytes(si))
    idxlist = pipe.execute()
    for i in idxlist:
      p_idx.append(int(i) - 1)
    logging.debug("P_Index Created (pca) for delta_S_pca")

    # 3. Performing tiling over subspace
    #   For Now: Load entire tree into local memory
    hcube_mapping = json.loads(self.catalog.get('hcube:pca'))
    logging.debug('# Loaded keys = %d', len(hcube_mapping.keys()))

    # 4. Pull entire Subspace (for now)  
    #   Note: this is more efficient than inserting new points
    #   due to underlying Redis Insertions / Index look up
    #   If this become a bottleneck, may need to write custom redis client
    #   connection to persist connection and keep socket open (on both ends)
    #   Or handle some of this I/O server-side via Lua scipt
    packed_subspace = self.catalog.lrange('subspace:pca', 0, -1)
    subspace_pca = np.array([np.fromstring(x) for x in packed_subspace])

    # TODO: accessor function is for 1 point (i) and 1 axis (j). 
    #  Optimize by changing to pipeline  retrieval for all points given 
    #  a list of indices with an axis (if nec'y)
    logging.debug("Reconstructing the tree...")
    hcube_tree = KDTree.reconstruct(hcube_mapping, subspace_pca)

    # logging.debug("Inserting Delta_S_pca into KDtree (hcubes)")
    # for i in range(len(pcalist)):
    #   hcube_tree.insert(pcalist[i], p_idx[i])

    # TODO: Ensure hcube_tree is written to catalog
    # TODO: DECIDE on retaining a Reservoir Sample
    # reservoirSampling(self.catalog, traj.xyz, r_idx, subspaceHash, 
    #     lambda x: tuple([x]+list(traj.xyz.shape[1:])), 
    #     'pca', 
    #     lambda key: '%d_%d' % key)
    bench.mark('PCA')


  # ---- POST PROCESSING
    shutil.rmtree(ramdisk)
    shm_contents = os.listdir('/dev/shm')
    logging.debug('Ramdisk contents (should be empty) : %s', str(shm_contents))
    
    # For benchmarching:
    print('##', job['name'], dcdfilesize/(1024*1024*1024), traj.n_frames)
    bench.show()




    return [job['name']]

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
