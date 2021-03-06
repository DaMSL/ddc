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
import datatools.datareduce as DR
from datatools.rmsd import calc_rmsd
# from datatools.pca import project_pca, calc_pca, calc_kpca
from datatools.pca import PCAnalyzer, PCAKernel
from datatools.approx import ReservoirSample
from overlay.redisOverlay import RedisClient
from overlay.alluxioOverlay import AlluxioClient
from overlay.overlayException import OverlayNotAvailable
from bench.timer import microbench
from bench.stats import StatCollector

from mdtools.timescape import TimeScapeParser, side_chain_pairs
from mdtools.structure import Protein
from mdtools.trajectory import rms_delta

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

# EXPERIMENT NUMBER REFERENCE:
    # 12 - UNIFORM
    # 13 - LATTICE (DS)
    # 14 - LATTICE (orig)
    # 15 - DENOVO (w/lattice)
    # 16 - LATTICE_T (TRANSITION w/resrmds)
    # 17 - BIASED

# FOR METRICS:::
def LABEL10(L, theta=0.9):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  return 'T%d' % A if A_amt < theta or (L[0] != L[-1]) else 'W%d' % A




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

    # Set the controller as the down stream macrothread node
    self.register_downnode('ctl')

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
    # self.slurmParams['time'] = '4-0:00:0'

    self.skip_simulation = False
    self.skip_timescape = False
    self.skip_notify = False

    self.parser.add_argument('-a', '--analysis', action='store_true')
    self.parser.add_argument('--useid')
    self.parser.add_argument('--skiptimescape', action='store_true')
    self.parser.add_argument('--skipnotify', action='store_true')

    args = self.parser.parse_args()
    if args.analysis:
      logging.info('SKIPPING SIMULATION')
      self.skip_simulation = True

    if args.skiptimescape:
      logging.info('SKIPPING TIMESCAPE')
      self.skip_timescape = True

    if args.skipnotify:
      logging.info('SKIPPING DOWNSTREAM NOTIFY')
      self.skip_notify = True

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

    # # Grab historical basin data's relative time to start (for lineage)
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
        ramdisk = gettempdir()
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
        break
      
      bench.mark('simulation')
      # bench.mark('SimExec:%s' % job['name'])

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

    # Load topology and define filters
    topo = self.protein.pdb
    pdb = md.load(job['pdb'])
    hfilt = pdb.top.select_atom_indices('heavy')
    pfilt = pdb.top.select('protein')
    afilt = pdb.top.select_atom_indices('alpha')

    # Load full higher dim trajectory
    if traj is None:
      if USE_SHM:
        traj = md.load(dcd_ramfile, top=pdb)
      else:
        traj = md.load(dcdFile, top=pdb)

    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))
    traj_prot = traj.atom_slice(pfilt)
    traj_heavy = traj.atom_slice(hfilt)
    traj_alpha = traj.atom_slice(afilt)

    # Superpose Coordinates to Common Reference
    traj_prot.superpose(topo)

    # Calculate output Distance Space
    # Use of the second side chain atom is discussed in the ref paper on Timescape
    # The atom pairs and selected atoms are in the timescape module
    sc_pairs = side_chain_pairs(traj_prot)
    n_features = len(sc_pairs)

    # Use the CA atoms to calculate distance space
    # NOTE THE CONVERSION FROM NM TO ANG!!!!!!
    dist_space = 10 * DR.distance_space(traj_alpha)

    # Get the frame rate to for conversion between source and timescape
    #  NOTE: Timescae is run at a more coarse level
    logging.debug('Preprocessing output for TimeScapes: terrain')
    traj_frame_per_ps = SIMULATE_RATIO * int(job['interval']) / 1000.   # jc interval is in fs
    ts_frame_per_ps = int(self.data['timescape:rate'])  # this value is in ps
    frame_rate = int(ts_frame_per_ps / traj_frame_per_ps)
    logging.debug('%5.2f fr/ps (Traj)     %5.2f fr/ps (TS)    FrameRate= %4.1f', traj_frame_per_ps, ts_frame_per_ps, frame_rate)

    # Execute Timescapes agility program to detect spatial-temporal basins
    output_prefix = os.path.join(job['workdir'], job['name'])
    if not self.skip_timescape:
      # Prep file and save locally in shm
      tmploc = gettempdir()
      ts_out = tmploc + 'traj_ts'
      ts_dcd = ts_out + '.dcd'
      ts_pdb = ts_out + '.pdb'
      heavy = traj_heavy.slice(range(0, traj.n_frames, frame_rate))
      heavy.slice(0).save_pdb(ts_pdb)
      heavy.save_dcd(ts_dcd)

      # Gaussuan Full Width at Half-Max value affects sliding window size
      # ref:  http://timescapes.biomachina.org/guide.pdf
      gmd_cut1 = int(self.data['timescape:gmd:low'])
      gmd_cut2 = int(self.data['timescape:gmd:hi'])
      gauss_wght_delta = int(self.data['timescape:delta'])

      # Execute timescapes' terrain.py on the pre-processed trajectory
      cmd = 'terrain.py %s %s %d %d %d GMD %s' %\
        (ts_pdb, ts_dcd, gmd_cut1, gmd_cut2, gauss_wght_delta, output_prefix)
      logging.info('Running Timescapes:\n  %s', cmd)
      stdout = executecmd(cmd)
      logging.info('TimeScapes COMPLETE:\n%s', stdout)


    # Collect and parse Timescape output
    logging.debug('Parsing Timescapes output')
    ts_parse = TimeScapeParser(job['pdb'], output_prefix, job['name'], 
      dcd=dcdFile, traj=traj, uniqueid=False)
    basin_list = ts_parse.load_basins(frame_ratio=frame_rate)
    n_basins = len(basin_list)

    minima_coords = {}
    basin_rms = {}
    basins = {}

    new_dmu={}
    new_dsig={}
    resid_rms_delta = {}

    stat.collect('num_basin', n_basins)
    downstream_list = []

    # FOR BOOTSTRAPPING USING RMSD
    ref_file = os.path.join(settings.workdir, self.data['pdb:ref:0'])
    logging.info('Loading RMSD reference frame from %s', self.data['pdb:ref:0'])
    refframe = md.load(ref_file)
    ref_alpha = refframe.atom_slice(refframe.top.select_atom_indices('alpha'))
    rmsd = 10*md.rmsd(traj_alpha, ref_alpha)

    # FOR RESIDUE RMSD
    res_rms_Kr = FEATURE_SET_RESID
    resrmsd = 10*np.array([LA.norm(i-ref_alpha.xyz[0], axis=1) for i in traj_alpha.xyz])
    basin_res_rms = np.zeros(shape=(len(basin_list), traj_alpha.n_atoms))

    # LOAD CENROIDS -- todo move to immut
    centroid = pickle.loads(self.catalog.get('centroid:ds'))
    basin_label_list = {}

    # Process each basin
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
      new_dmu[bid]    = np.mean(dist_space[a:b], axis=0)
      new_dsig[bid]   = np.std(dist_space[a:b], axis=0)

      # Collect Basin metadata
      basin_hash = basin.kv()
      basin_hash['pdbfile'] = jc_filename
      basin_hash['rmsd'] = np.median(rmsd[a:b])
      basin_res_rms[i] = np.median(resrmsd[a:b], axis=0)

      # FOR EXP #16 -- Residue RMSD

      # Set relative time (in ns)
      # basin_hash['time'] = reltime_start + (a * frame_rate) / 1000
      basins[bid] = basin_hash

      # LABEL ATEMPORAL DATA
      # Take every 4th frame
      stride = [dist_space[i] for i in range(a,b,4)]
      rms = [LA.norm(centroid - i, axis=1) for i in stride]
      label_seq = [np.argmin(i) for i in rms]
      basin_label_list[bid] = label_seq
      logging.info('  Basin Processed: #%s, %d - %d', basin_hash['traj'], 
        basin_hash['start'], basin_hash['end'])

    # RMSD DELTA (RESIDUE)
    basin_res_rms_delta = np.array([rms_delta(i) for i in basin_res_rms.T]).T
    basin_rms_delta_bykey = {basin.id: basin_res_rms_delta[i] for i, basin in enumerate(basin_list)}
    for k in basins.keys():
      basins[k]['resrms_delta'] = np.sum(basin_rms_delta_bykey[k][res_rms_Kr])


    # TODO:  Use Min Index as snapshot, median (or mean) DistSpace vals for each basin?????

    bench.mark('analysis')
  #  BARRIER: WRITE TO CATALOG HERE -- Ensure Catalog is available
    # try:
    self.wait_catalog()
    # except OverlayNotAvailable as e:
    #   logging.warning("Catalog Overlay Service is not available. Scheduling ASYNC Analysis")

  # FOR EXPERIMENT METRICS
    src_basin = self.catalog.hgetall("basin:" + job['src_basin'])
    with open('/home-1/bring4@jhu.edu/ddc/results/{0}_prov.log'.format(settings.name), 'a') as metric_out:
      for i, basin in enumerate(basin_list):
        bid = basin.id
        label_seq = basin_label_list[bid]
        basin_metric_label = LABEL10(label_seq)
        metric_out.write('BASIN,%s,%s,%s,%s\n'% \
          (bid, src_basin['label:10'], basin_metric_label, ''.join([str(i) for i in label_seq])))

  # Update Index Synchronized data lists
    basin_list_sorted =sorted(basins.keys(), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
    for bid in basin_list_sorted:
      with self.catalog.pipeline() as pipe:
        while True:
          try:
            logging.debug('Updating %s basin indeces and distance space', len(basins))
            pipe.rpush('basin:list', bid)
            pipe.rpush('dspace', pickle.dumps(new_dmu[bid]))
            basin_index,_ = pipe.execute()
            break
          except redis.WatchError as e:
            logging.debug('WATCH ERROR. Someone else is writing to the catalog. Retrying...')
            continue
      basins[bid]['dsidx'] = basin_index - 1

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
          pipe.set('resrms:' + job['name'], resrmsd)

          # Store all basin data
          logging.debug('Updating %s basins', len(basins))
          for bid in sorted(basins.keys(), key=lambda x: (x.split('_')[0], int(x.split('_')[1]))):
            pipe.hmset('basin:'+bid, basins[bid])
            pipe.set('minima:%s'%bid, pickle.dumps(minima_coords[bid]))
            for i in basin_label_list[bid]:
              pipe.rpush('basin:labelseq:'+bid, i)

          pipe.hset('anl_sequence', job['name'], mylogical_seqnum)

          logging.debug('Executing')
          pipe.execute()
          break

        except redis.WatchError as e:
          logging.debug('WATCH ERROR. Someone else is writing to the catalog. Retrying...')
          continue

    self.data[key]['xid:start'] = start_index
    self.data[key]['xid:end'] = start_index + traj.n_frames
    bench.mark('catalog')

  # ---- POST PROCESSING
    # Check for total data in downstream queue  & Notify control manager to run
    # 1. Det. what's still running
    job_queue = slurm.currentqueue()
    n_simjobs = -1    # -1 to exclude self
    for j in job_queue:
      if j['name'].startswith('sw'):
        n_simjobs += 1

    logging.info('SIM WORKER has detected %d other simulations running/pending.', len(job_queue))
    remain_percent = n_simjobs / self.data['numresources']

    # Fault Tolerance: ensure the pipeline persists
    if n_simjobs == 0:
      logging.info('I am the last simulation. Ensuring the controler executes.')
      self.catalog.set('ctl:force', 1)

    # Send downstream notification when less than 10% of jobs are still running
    if not self.skip_notify and remain_percent < .1:
      # Notify the control manager 'CM'
      self.notify('ctl')

    if USE_SHM:
      shutil.rmtree(ramdisk)
      shm_contents = os.listdir('/dev/shm')
      logging.debug('Ramdisk contents (should be empty of DDC) : %s', str(shm_contents))
    
    # For benchmarching:
    bench.show()
    stat.show()

    # Return # of observations (frames) processed
    return downstream_list

if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
