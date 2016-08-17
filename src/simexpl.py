#!/usr/bin/env python

import argparse
import os
import sys
import shutil
import time
import fcntl
import logging
import pickle
import tempfile
from datetime import datetime as dt
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
from datatools.feature import feal

from macro.macrothread import macrothread
import mdtools.deshaw as DE
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

from mdtools.timescape import TimeScapeParser
from mdtools.structure import Protein

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

PARALLELISM = 24
SIM_STEP_SIZE = 2

# Factor used to "simulate" long running jobs using shorter sims
SIMULATE_RATIO = 50

class simulationJob(macrothread):
  """Macrothread to run MD simuation. Each worker runs one simulation
  using the provided input parameters.
    Input -> job candidate key in the data store 
    Execute -> creates the config file and calls namd to run the simulation. 
  """
  def __init__(self, fname, jobnum = None):
    macrothread.__init__(self, fname, 'sim')

    # State Data for Simulation MacroThread -- organized by state
    self.setStream('jcqueue', 'basin:stream')
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
    numobs = self.catalog.llen('xid:reference')
    if numobs >= self.data['max_observations']:
      logging.info('Terminating at %d observations', numobs)
      return True
    else:
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

    # First, Define Protein Object
    self.protein = Protein('bpti', self.catalog)

    # Load parameters from catalog
    key = wrapKey('jc', i)
    params = self.catalog.hgetall(key)
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))
    self.addMut(key, params)
    return params

  def execute(self, job):

  # PRE-PREOCESS ----------------------------------------------------------
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

    SIMULATE_RATIO = settings.SIMULATE_RATIO
    if SIMULATE_RATIO > 1:
      logging.warning(" USING SIMULATION RATIO OF %d -- THis is ONLY for debugging", SIMULATE_RATIO)
    frame_size = (SIMULATE_RATIO * int(job['interval'])) / (1000)
    logging.info('Frame Size is %f  Using Sim Ratio of 1:%d', \
      frame_size, SIMULATE_RATIO)

    EXPERIMENT_NUMBER = settings.EXPERIMENT_NUMBER
    logging.info('Running Experiment Configuration #%d', EXPERIMENT_NUMBER)

    # TODO: FOR LINEAGE
    # srcA, srcB = eval(job['src_bin'])
    # stat.collect('src_bin', [str(srcA), str(srcB)])

    traj = None

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

      max_expected_obs = int(job['runtime']) // int(job['dcdfreq'])
      # Retry upto 3 attempts if the sim fails
      MAX_TRY = 3
      for i in range(MAX_TRY, 0, -1):
        min_required_obs = int(max_expected_obs * ((i-1)/(MAX_TRY)))
        logging.debug("Executing Simulation:\n   %s\n", cmd)
        logging.debug('# Obs Expected to see: %d', max_expected_obs)
        stdout = executecmd(cmd)
        logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
        # Check file for expected data
        if USE_SHM:
          traj = md.load(dcd_ramfile, top=job['pdb'])
        else:
          traj = md.load(dcdFile, top=job['pdb'])
        logging.info("Obs Threshold  = %4d", min_required_obs)
        logging.info("#Obs This Traj = %4d", traj.n_frames)
        if traj.n_frames >= min_required_obs:
          logging.info('Full (enough) Sim Completed')
          break
        logging.info('Detected a failed Simulation. Retrying the same sim.')
      
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

        # ALT:  X-Mit to Cache Service and let cache write to disk lazily
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

    # topofile = os.path.join(settings.workdir, self.data['pdb:topo'])
    topo = self.protein.top

    # Load full higher dim trajectory
    # traj = datareduce.filter_heavy(dcd_ramfile, job['pdb'])
    if traj is None:
      if USE_SHM:
        traj = md.load(dcd_ramfile, top=topo)
      else:
        traj = md.load(dcdFile, top=topo)

    # Superpose Coordinates to Common Reference
    traj.superpose(topo)

    bench.mark('File_Load')
    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))


    # Slice / filter traj
    # Calc RMSD / CA-DiH for each obs
    # Run Timescapes agility
    # Parse Timescapes results (from seg.dat)
    # Spatio-Temp cluster
    # For each basin:
      #  Create basin Object  (?)
      #  ID/Store local minima as JC
      #  Calc median RMSD, CA-DiH (phi/psi??), atom-variance
      #  Label via atom variance
      #  Store Data

    # TODO: VERTIFY this filter
    hfilt = self.protein.hfilt()
    pfilt = self.protein.pfilt()
    traj_prot = traj.atom_slice(pfilt)
    traj_heavy = traj.atom_slice(hfilt)


    # Calculate output Distance Space
    # Use of the second side chain atom is discussed in the ref paper on Timescape
    # The atom pairs and selected atoms are in the timescape module
    sc_pairs = TS.side_chain_pairs(traj_prot)
    dist_space = DR.distance_space(traj_prot, pairs=sc_pairs)

    # lm_file = os.path.join(settings.workdir, self.data['pdb:ref:0'])
    # landmark  = md.load(lm_file)

    # Calculate RMSD for ea (note conversion nm -> angstrom)
    # logging.info('Running Metrics on local output:  RMSD')
    # rmsd = 10 * md.rmsd(traj_prot, landmark, 0, hfilt, hfilt, precentered=True)

    # Calc Phi/Psi angles
    # phi_angles = md.compute_phi(traj)[1][0]
    # psi_angles = md.compute_psi(traj)[1][0]
    # phi_lm = md.compute_phi(landmark)[1][0]
    # psi_lm = md.compute_psi(landmark)[1][0]
    # phi = np.array([LA.norm(a - phi_lm) for a in phi_angles])
    # psi = np.array([LA.norm(a - psi_lm) for a in psi_angles])

    # Execute Timescapes agility program to detect spatial-temporal basins
    # Get the frame rate to save from catalog:
    logging.debug('Preprocessing output for TimeScapes: terrain')
    traj_frame_per_ps = int(job['interval']) / 1000.   # jc interval is in fs
    ts_frame_per_ps = int(self.data['timescape:rate'])  # this value is in ps
    frame_rate = int(ts_frame_per_ps / traj_frame_per_ps)

    # FOR DEBUGGING
    logging.warning("DEBUGGING IS ON..... FRAME RATE MANUAL SET TO 1")
    frame_rate = 1

    # Prep file and save locally
    tmp_out = '/tmp/ddc/traj_ts'
    tmp_dcd = tmp_out + '.dcd'
    tmp_pdb = tmp_out + '.pdb'
    output_prefix = os.path.join(job['workdir'], job['name'])
    heavy = traj_heavy.slice(range(0, traj.n_frames, frame_rate))
    heavy.slice(0).save_pdb(tmp_pdb)
    heavy.save_dcd(tmp_dcd)

    # Gaussuan Full Width at Half-Max value affects sliding window size
    # ref:  http://timescapes.biomachina.org/guide.pdf
    gmd_cut1 = int(self.data['timescape:gmd:low'])
    gmd_cut2 = int(self.data['timescape:gmd:hi'])
    gauss_wght_delta = int(self.data['timescape:delta'])

    # Execute timescapes' terrain.py on the pre-processed trajectory
    cmd = 'terrain.py %s %s %d %d %d GMD %s' %
      (tmp_pdb, tmp_dcd, gmd_cut1, gmc_cut2, gauss_wght_delta, output_prefix)
    logging.info('Running Timescapes:\n  %s', cmd)
    stdout = executecmd(cmd)
    logging.info('TimeScapes COMPLETE:\n%s', stdout)

    logging.debug('Parsing Timescapes output')
    ts_parse = TimeScapeParser(tmp_pdb, output_prefix, job['name'], 
      dcd=dcdFile, traj=traj, uniqueid=False)
    basin_list = ts_parse.load_basins(frame_rate=frame_rate)
    corr_matrix = ts_parse.correlation_matrix()

    minima_coords = {}
    basin_rms = {}
    basins = {}

    downstream_list = []
    for i, basin in enumerate(basin_list):
      logging.info('  Processing basin #%2d', i)
      bid = basin.id
      downstream_list.append(bid)

      # Slice out minima coord  & save to disk (for now)
      #  TODO:  Store in memory in cache
      minima_coords[bid] = traj.slice(basin.mindex)
      jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
      minima_coords[bid].save_pdb(jc_filename)

      # METRIC CALCULATION
      a, b = basin.start, basin.end
      corr_vector = np.mean(corr_matrix[a:b], axis=0)
      dspace_mean = np.mean(dist_space[a:b], axis=0)
      dspace_std  = np.std(dist_space[a:b], axis=0)

      # basin_rms[bid] = np.median(rmsd[a:b])

      # Collect Basin metadata
      basin_hash = basin.kv()
      basin_hash['pdbfile'] = jc_filename
      basin_hash['corr_vector'] = pickle.dumps(corr_vector)
      basin_hash['d_mu']        = pickle.dumps(dspace_mean)
      basin_hash['d_sigma']     = pickle.dumps(dspace_std)
      basins[bid] = basin_hash

      logging.info('  Basin data dump: %s', basin_hash)


  #  BARRIER: WRITE TO CATALOG HERE -- Ensure Catalog is available
    # try:
    self.wait_catalog()
    # except OverlayNotAvailable as e:
    #   logging.warning("Catalog Overlay Service is not available. Scheduling ASYNC Analysis")


  # Update Catalog with 1 Long Atomic Transaction  
    with self.catalog.pipeline() as pipe:
      while True:
        try:
          logging.debug('Update Filelist')
          pipe.watch(wrapKey('jc', job['name']))
          file_idx = pipe.rpush('xid:filelist', job['dcd']) - 1

          # HD Points
          logging.debug('Update HD Points')
          start_index = pipe.llen('xid:reference')
          pipe.multi()

          pipe.rpush('xid:reference', *[(file_idx, x) for x in range(traj.n_frames)])
          pipe.rpush('metric:rms', *rmsd)

          logging.debug('Updating %s basins', len(basins))
          for bid in basins.keys():
            pipe.rpush('basin:list', bid)
            # pipe.hset('basin:rms', bid, basin_rms[bid])
            pipe.hmset('basin:%s'%bid, basins[bid])
            pipe.set('minima:%s'%bid, pickle.dumps(minima_coords[bid]))

          pipe.hset('anl_sequence', job['name'], mylogical_seqnum)
          logging.debug('Executing')
          pipe.execute()
          break

        except redis.WatchError as e:
          logging.debug('WATCH ERROR')
          continue

    self.data[key]['xid:start'] = start_index
    self.data[key]['xid:end'] = start_index + traj.n_frames
    bench.mark('Indx_Update')

    # # COPY OUT TimeScapes Logs (to help debug):
    # ts_logfiles = [f for f in os.listdir(tmp_out) if f.endswith('log')]
    # for f in ts_logfiles:
    #   shutil.copy(f, os.path.join(settings.jobdir, job['name']))

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
    return downstream_list

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
