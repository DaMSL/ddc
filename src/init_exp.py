#!/usr/bin/env python

import argparse
import math
import json
import bisect
import datetime as dt

import mdtraj as md
import numpy as np
from numpy import linalg as LA


from core.common import *
import mdtools.deshaw as deshaw
from overlay.redisOverlay import RedisClient
from core.kvadt import kv2DArray
from core.slurm import slurm
from core.kdtree import KDTree
import core.ops as ops
import datatools.datareduce as datareduce
from datatools.rmsd import *
from mdtools.simtool import generateExplJC, Peptide
import mdtools.deshaw as deshaw
import bench.db as db
import plot as P

from mdtools.timescape import *


# For changes to schema
def updateschema(catalog):
  """
    # TODO: Do away with pre-defined schema and infer datatypes
    This is a work around for Redis calls; however, data init
    and uses can be inferred and extracted from redis
  """
  settings = systemsettings()
  if not settings.configured():
    settings.applyConfig()

  dtype_map = {}
  for k, v in settings.schema.items():
    logging.debug('items %s, %s', str(k), str(v))
    dtype_map[k] = str(v)

  for k, v in settings.init.items():
    if k not in dtype_map:
      dtype_map[k] = type(v).__name__
      settings.schema[k] = type(v).__name__

  for k, v in dtype_map.items():
    logging.info('Setting schema:    %s  %s', k, str(v))

  catalog.hmset("META_schema", dtype_map)

def initializecatalog(catalog):
  """Initialize the Catalog for an experiment
  """
  settings = systemsettings()

  logging.debug("Clearing Catalog.")
  if not catalog.isconnected:
    logging.warning('Catalog is not started. Please start it first.')
    sys.exit(0)
  catalog.clear()

  logging.debug("Loading schema into catalog.")
  updateschema(catalog)
  catalog.loadSchema()

  catalog.set('name', settings.name)

  # Set defaults vals for schema
  initvals = {i:settings.init[i] for i in settings.init.keys() if settings.schema[i] in ['int', 'float', 'list', 'dict', 'str']}
  catalog.save(initvals)
  for k, v in initvals.items():
    logging.debug("Initializing data elm %s  =  %s", k, str(v))

def load_seeds(catalog):
    settings = systemsettings()
    idx_list  = [0,1,2,3,4,20,23,24,30,32,34,40,41,42]
    slist = {k: 'seed%d'%k for k in idx_list}
    tlist = {k: 'tr%d.dcd'%k for k in idx_list}
    logging.info('Loading %d seeds for experiment %s', len(slist.keys()), catalog.get('name'))

    seed_dir = os.path.join(settings.WORKDIR, 'seed')
    pdb_file = os.path.join(settings.workdir, catalog.get('pdb:topo'))
    print("PDB FILE: ", settings.workdir, pdb_file)
    topo = md.load(pdb_file)
    bpti = Peptide('bpti', topo)

    logging.info('Topology file: %s  <%s>', pdb_file, str(topo))

    ref_file = os.path.join(settings.workdir, catalog.get('pdb:ref:0'))
    ref_traj = md.load(ref_file)
    logging.info('RMS Reference file: %s  <%s>', ref_file, str(ref_traj))

    hfilt = bpti.get_filter('heavy')

    for idx in idx_list:
      s, t = slist[idx], tlist[idx]
      logging.info('Trajectory SEED:  %s', s)
      tfile = os.path.join(seed_dir, t)
      traj = md.load(tfile, top=topo)
      traj.superpose(topo)

      protein = traj.atom_slice(bpti.get_filter('protein'))

      ####  METRIC GOES HERE
      rms = 10*md.rmsd(protein, ref_traj, 0, hfilt, hfilt, precentered=True)

      # Push to Catalog
      file_idx = catalog.rpush('xid:filelist', tfile) - 1
      start_index = catalog.llen('xid:reference')
      catalog.rpush('xid:reference', *[(file_idx, x) for x in range(traj.n_frames)])
      catalog.rpush('metric:rms', *rms)

      # Process Trajectory as basins
      logging.info("  Seed Loaded, RMS Complete. Loading TimeScape Data...")
      seed_name = 'seed%d'%idx
      ts_traj = TimeScapeTrajectory(pdb_file, os.path.join(seed_dir, 'out', seed_name),\
          seed_name, dcd=tfile, traj=traj)
      ts_traj.load_basins()

      for i, basin in enumerate(ts_traj.basins):
        logging.info('  Processing basin #%2d', i)

        pipe = catalog.pipeline()
        bid = basin.id
        # Store on Disk and in redis
        jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
        minima_frame = traj.slice(basin.mindex)
        minima_frame.save_pdb(jc_filename)

        a, b = basin.start, basin.end

        ###### TODO:   FACTOR IN SAMPLE RATE 

        basin_rms = np.median(rms[a:b])
        
        basin_hash = basin.kv()
        basin_hash['pdbfile'] = jc_filename
        logging.info('  Basin data dump: %s', basin_hash)

        pipe.rpush('basin:list', bid)
        pipe.hset('basin:rms', bid, basin_rms)
        pipe.hmset('basin:%s'%bid, basin_hash)
        pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))
        pipe.execute()

def sample_uniform(basin_list, num):
  need_replace = (len(basin_list) < num)
  candidates = np.random.choice(basin_list, size=num, replace=need_replace)
  return candidates  

def seed_jobs(catalog, num=1):
  """
  Seeds jobs into the JCQueue -- pulled from DEShaw
  Selects equal `num` of randomly start frames from each bin
  to seed as job candidates
  """
  logging.info('Seeding %d jobs per transtion bin', num)
  settings = systemsettings()

  # Load pre-seeded basin list
  basinlist = catalog.lrange('basin:list', 0, -1)
  if len(basinlist) == 0:
    logging.error('No basins loaded. Please seed')
    return

  dcdfreq = int(catalog.get('dcdfreq'))
  runtime = int(catalog.get('runtime'))
  sim_step_size = int(catalog.get('sim_step_size'))
  force_field_dir = catalog.get('ffield_dir')

  # TODO:  Apply sampling Algorith HERE
  seedlist = sample_uniform(basinlist, num)

  # Create new jobs from selected basins
  for seed in seedlist:
    logging.debug('\nSeeding Job: %s ', seed)
    basin = catalog.hgetall('basin:%s'%seed)

    # Generate new set of params/coords
    jcID, params = generateExplJC(traj)

    # Update Additional JC Params and Decision History, as needed
    config = dict(params,
        name    = jcID,
        runtime = runtime,
        dcdfreq = dcdfreq,
        interval = dcdfreq * sim_step_size,                       
        ffield_dir = force_field_dir,
        temp    = 310,
        timestep = 0,
        gc      = 1,
        origin  = 'seed',
        src_traj = seed.traj,
        src_basin = seed.id,
        application   = settings.name)

    # Push to catalog
    logging.info("New Simulation Job Created: %s", jcID)
    for k, v in config.items():
      logging.debug("   %s:  %s", k, str(v))
    catalog.rpush('jcqueue', jcID)
    catalog.hmset(wrapKey('jc', jcID), config)

def resetAnalysis(catalog):
  """Removes all analysis data from the database
  """
  settings = systemsettings()

  keylist0 =['completesim',
            'label:rms',
            'observe:count',
            'rsamp:rms:_dtype',
            'rsamp:rms:_shape',
            'subspace:rms',
            'xid:filelist',
            'xid:reference  ',
            'subspace:covar:fidx',
            'subspace:covar:pts',
            'subspace:covar:xid']
  keylist1 =['subspace:pca:%d',
            'subspace:pca:kernel:%d',
            'subspace:pca:updates:%d',
            'rsamp:rms:%d',
            'rsamp:rms:%d:full',
            'rsamp:rms:%d:spill']
  keylist2 =['observe:rms:%d_%d']

  logging.info("Clearing the database of recent data")
  count = 0
  for key in keylist0:
    count += catalog.delete(key)

  for key in keylist1:
    for A in range(5):
      count += catalog.delete(key % A)

  for key in keylist2:
    for A in range(5):
      for B in range(5):
        count += catalog.delete(key % (A, B))

  logging.info('Removed %d keys', count)

  jobs = catalog.hgetall('anl_sequence')
  logging.info('RE-RUN THESE JOBS')
  orderedjobs = sorted(jobs.items(), key=lambda x: x[1])
  seqnum = 1
  fileseq = 0
  jobfile = open('joblist.txt', 'w')
  for k, v in orderedjobs:
    if seqnum == 100:
      fileseq += 1
      seqnum = 1
    outline = 'src/simanl.py -a --useid="sw-%04d.%02d" -c %s -w %s' % \
      (fileseq,seqnum,settings.name, k)
    print(outline)
    jobfile.write(outline + '\n')
    seqnum += 1
  jobfile.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', default='default')
  parser.add_argument('--seed', action='store_true')
  parser.add_argument('--initjobs', action='store_true')
  parser.add_argument('--onejob', action='store_true')
  parser.add_argument('--initcatalog', action='store_true')
  parser.add_argument('--updateschema', action='store_true')
  # parser.add_argument('--reset', action='store_true')
  parser.add_argument('--all', action='store_true')

  args = parser.parse_args()

  confile = args.name + '.json'

  settings = systemsettings()
  settings.applyConfig(confile)
  catalog = RedisClient(args.name)

  if args.initcatalog or args.all:
    settings.envSetup()
    initializecatalog(catalog)

  if args.seed or args.all:
    load_seeds(catalog)

  if args.initjobs or args.all:
    numresources = int(catalog.get('numresources'))
    seed_jobs(catalog, numresources)

  elif args.onejob:
    seed_jobs(catalog, 1)

  if args.updateschema:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    updateschema(catalog)

  # if args.reset:
  #   resetAnalysis(catalog)