#!/usr/bin/env python

import argparse
import math
import json
import bisect
import datetime as dt
import shutil

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
from mdtools.simtool import *
from mdtools.structure import Protein
import mdtools.deshaw as deshaw
import bench.db as db
import plot as P

from mdtools.timescape import *
from sampler.basesample import *

import datatools.lattice as lat

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

  for k, v in settings.state.items():
    if k not in dtype_map:
      dtype_map[k] = type(v).__name__
      settings.schema[k] = type(v).__name__

  for k, v in settings.sim_params.items():
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
  initvals = {}
  for k, val in settings.state.items():
    initvals[k] = val

  for k, val in settings.sim_params.items():
    initvals[k] = val

  # Manually Load Protein PDF into catalog (for now)
  # TODO:  Make this config and/or in setting depending on # of proteins to run
  # bpti_pdb = os.path.join(settings.workdir, 'bpti', '5PTI.pdb')
  # catalog.set('protein:bpti', bpti_pdb)

  catalog.save(initvals)
  for k, v in initvals.items():
    logging.debug("Initializing data elm %s  =  %s", k, str(v))

def load_historical_DEShaw(catalog):
  """ Load all DEShaw data into basins for processing """
  settings = systemsettings()
  home = os.getenv('HOME')

  # Read in and parse TimeScape output
  file_pref = home+'/work/timescape/desh_' #'/root/heavy/out/expl'
  basin_list = []
  logging.info('Loading all D.E.Shaw Time Scape data and processing basins')
  for i in range(42):  
    nframes = 100000 if i < 41 else 25000
    minima_list = TimeScape.read_log(file_pref + '%02d_minima.log'%i)
    window_list = TimeScape.windows(file_pref + '%02d_transitions.log'%i)
    basin_index = 0
    last = None
    offset = 100000*i
    pipe = catalog.pipeline()
    while basin_index < len(minima_list):
      ### MINIMA IS LOCAL TO FULL 2.5us FILE
      ### WINDOW IS GLOBAL INDEX OVER ALL 4.125Mil Frames
      a, b = window_list[basin_index]
      minima = minima_list[basin_index]
      basin_id = '%07d' % (offset + a)
      local_file_num = offset // 1000 + minima // 1000
      local_file_id  = 'desh_%04d' %  (local_file_num)
      local_minima   = (minima + offset) - local_file_num * 1000
      basin = Basin(local_file_id, (a+offset, b+offset), local_minima, uid='%07d' % (offset + a))
      if last is not None:
        basin.prev = last.id
        basin_list[-1].next = basin.id
      basin_list.append(basin)
      last = basin
      basin_index += 1

      basin_hash = basin.kv()
      # fname = ('bpti-all-%03d.dcd' if local_file_num < 1000 else 'bpti-all-%04d.dcd')%local_file_num
      # basin_hash['pdbfile'] = os.path.join(settings.workir, 'bpti', fname)
      # logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)
      pipe.rpush('basin:list', basin_id)
      pipe.hmset('basin:%s'%basin_id, basin_hash)
    pipe.execute()

  # logging.info('Loading Pre-Calculated Correlation Matrix and mean/stddev vals')
  # corr_matrix = np.load('data/de_corr_matrix.npy')
  # dmu = np.load('data/de_ds_mu.npy')
  # dsig = np.load('data/de_ds_mu.npy')

  # logging.info("Loading Historical data into catalog:  corr_matrix: %s", corr_matrix.shape)
  # catalog.storeNPArray(corr_matrix, 'desh:coor_vector')
  # catalog.storeNPArray(dmu, 'desh:ds_mu')
  # catalog.storeNPArray(dsigma, 'desh:ds_sigma')

  if settings.EXPERIMENT_NUMBER == 14:

    if not os.path.exists(settings.datadir + '/iset.p'):
      os.symlink(settings.workdir + '/iset.p', settings.datadir + '/iset.p')
    if not os.path.exists(settings.datadir + '/iset.p'):
      os.symlink(settings.workdir + '/de_ds_mu.npy', settings.datadir + '/de_ds_mu.npy')
    if not os.path.exists(settings.datadir + '/de_ds_mu.npy'):
      os.symlink(settings.workdir + '/data/de_ds_mu.npy', settings.datadir + '/de_ds_mu.npy')

    dlat    = open(os.path.join(settings.workdir, 'dlat.p'), 'rb').read()
    max_fis = open(os.path.join(settings.workdir, 'mfis.p'), 'rb').read()
    low_fis = open(os.path.join(settings.workdir, 'lowfis.p'), 'rb').read()

    logging.info('Loading max, low FIS and derived lattice')
    with catalog.pipeline() as pipe:
      pipe.set('lattice:max_fis', max_fis)
      pipe.set('lattice:low_fis', low_fis)
      pipe.set('lattice:dlat', dlat)
      pipe.execute()

    logging.info('Loading raw distance from file')
    de_ds = 10*np.load(settings.datadir + '/de_ds_mu.npy')
    logging.info('Loading raw distance space into catalog')
    with catalog.pipeline() as pipe:
      for elm in de_ds:
        pipe.rpush('dspace', pickle.dumps(elm))
      pipe.execute()

  logging.info('DEShaw data loaded. ALL Done!')
  # FOR CREATING CM/MU/SIGMA vals for first time
  # C_T = np.zeros
  # deds = np.load(home + '/work/timescape/deds.npy')
  # for i, basin in enumerate(basin_list):
  #   C_T.append(np.mean(corr_mat[a:b], axis=0))
  #   mu_T.append(np.mean(deds[offset+a:offset+b], axis=0))
  #   sigma_T.append(np.std(deds[offset+a:offset+b], axis=0))
   
  # TODO:  IS THIS NEEDED FOR DEShaw Data? Store on Disk and in redis
  # jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
  # minima_frame = md.load_frame(tfile, basin.mindex, top=topo) if traj is None else traj.slice(basin.mindex)
  # minima_frame.save_pdb(jc_filename)

def load_historical_Expl(catalog):
  """ Load all DEShaw data into basins for processing """
  settings = systemsettings()
  
  # idx_list      = [0,1,2,3,4,20,23,24,30,32,34,40,41,42]
  idx_list      = [0,34]
  tlist         = {k: 'tr%d.dcd'%k for k in idx_list}
  seed_dir      = os.path.join(settings.WORKDIR, 'seed')
  seed_ts_ratio = 16     # TimeScape ran on 4ps frame rate (16x source)

  # Load topology and anscillary data
  # bpti = Protein(bpti, catalog, load=True)
  pdb_file = os.path.join(seed_dir, 'coord.pdb')
  topo = md.load(pdb_file)
  pfilt = topo.top.select('protein')

  logging.info('Topology loaded %s', topo)

  # ID Side Chain pair atoms for each distance space calc 
  sc_pairs = side_chain_pairs(topo.atom_slice(pfilt))
  logging.info('Identified side chains: %d', len(sc_pairs))


  # Process all sorce trajectories
  basin_list = []
  C_T, mu_T, sigma_T = [], [], []
  for idx in idx_list:
    logging.info('Procesing Seed index: %d', idx)
    # Load SRC seed trajetory & calc distance space  -- TODO: make this optional
    tfile = os.path.join(seed_dir, tlist[idx])
    traj = md.load(tfile, top=topo)
    traj.superpose(topo)
    ds = datareduce.distance_space(traj, pairs=sc_pairs)

      # Push to Catalog
    file_idx = catalog.rpush('xid:filelist', tfile) - 1
    start_index = catalog.llen('xid:reference')
    # TODO: Do I still need to index every frame??????
    catalog.rpush('xid:reference', *[(file_idx, x) for x in range(traj.n_frames)])

    # Process Trajectory as basins
    logging.info("  Seed Loaded. Loading TimeScape Data...")
    seed_name = 'tr%d'%idx
    ts_data_path  = os.path.join(seed_dir, 'TEST', seed_name)
    ts_traj = TimeScapeParser(pdb_file, 
        ts_data_path, seed_name, dcd=tfile, traj=traj)
    basin_list = ts_traj.load_basins(frame_ratio=seed_ts_ratio)
    corr_mat   = ts_traj.correlation_matrix()

    for i, basin in enumerate(ts_traj.basins):
      a, b = basin.start, basin.end
      bid = basin.id
      if a > traj.n_frames:
        logging.info('Finished processing all basins for this Trajectory!')
        break

      # Store on Disk and in redis
      jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
      minima_frame = md.load_frame(tfile, basin.mindex, top=topo) if traj is None else traj.slice(basin.mindex)
      minima_frame.save_pdb(jc_filename)

      C_T.append(np.mean(corr_mat[a:b], axis=0))
      mu_T.append(np.mean(ds[a:b], axis=0))
      sigma_T.append(np.std(ds[a:b], axis=0))
     
      basin_hash = basin.kv()
      basin_hash['pdbfile'] = jc_filename
      logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)

      pipe = catalog.pipeline()
      pipe.rpush('basin:list', bid)
      pipe.hmset('basin:%s'%bid, basin_hash)
      pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))
      pipe.execute()

  catalog.storeNPArray(np.array(C_T), 'corr_vector')
  catalog.storeNPArray(np.array(mu_T), 'dspace_mu')
  # catalog.storeNPArray(np.array(sigma_T), 'dspace_sigma')

def load_seeds(catalog, calc_seed_rms=False):
    settings = systemsettings()
    idx_list  = [0,1,2,3,4,20,23,24,30,32,34,40,41,42]
    # seed_frame_length = 8000


    # slist = {k: 'seed%d'%k for k in idx_list}
    tlist = {k: 'tr%d.dcd'%k for k in idx_list}
    logging.info('Loading %d seeds for experiment %s', len(slist.keys()), catalog.get('name'))

    seed_dir = os.path.join(settings.WORKDIR, 'seed')
    seed_frame_rate = 0.25  # in ps
    seed_ts_factor = 16     # TimeScape ran on 4ps frame rate (16x source)

    bpti = Protein(bpti, catalog, load=True)
    pdb_file = bpti.pdbfile

    print("PDB FILE: ", settings.workdir, pdb_file)
    topo = bpti.pdb
    logging.info('Topology file: %s  <%s>', pdb_file, str(topo))

    # ref_file = os.path.join(settings.workdir, catalog.get('pdb:ref:0'))
    # ref_traj = md.load(ref_file)
    # logging.info('RMS Reference file: %s  <%s>', ref_file, str(ref_traj))

    hfilt = bpti.get_filter('heavy')

    ts_rate = catalog.get('timescape:rate')

    for idx in idx_list:
      s, t = slist[idx], tlist[idx]
      logging.info('Trajectory SEED:  %s', s)
      tfile = os.path.join(seed_dir, t)


      # FOR RMSD CALCULATIONS  EXP #12
      # rms_file = os.path.join(seed_dir, 'rms', 'rms%d'%idx)
      # if calc_seed_rms:
      #   traj = md.load(tfile, top=topo)
      #   traj.superpose(topo)
      #   protein = traj.atom_slice(bpti.get_filter('protein'))
      #   ####  METRIC GOES HERE
      #   rms = 10*md.rmsd(protein, ref_traj, 0, hfilt, hfilt, precentered=True)
      #   np.save(rms_file, rms)
      # else:
      #   traj = None
      #   rms = np.load(rms_file + '.npy')

      # Push to Catalog
      file_idx = catalog.rpush('xid:filelist', tfile) - 1
      start_index = catalog.llen('xid:reference')

      # TODO: Do I still need to index every frame??????
      catalog.rpush('xid:reference', *[(file_idx, x) for x in range(traj.n_frames)])
      # catalog.rpush('metric:rms', *rms)

      # Process Trajectory as basins
      logging.info("  Seed Loaded. Loading TimeScape Data...")
      seed_name = 'seed%d'%idx
      ts_traj = TimeScapeParser(pdb_file, 
          os.path.join(seed_dir, 'out', seed_name),
          seed_name, dcd=tfile, traj=traj)
      ts_traj.load_basins(frame_ratio=seed_ts_factor)

      for i, basin in enumerate(ts_traj.basins):
        pipe = catalog.pipeline()
        bid = basin.id
        # Store on Disk and in redis
        jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
        minima_frame = md.load_frame(tfile, basin.mindex, top=topo) if traj is None else traj.slice(basin.mindex)
        minima_frame.save_pdb(jc_filename)


        a, b = basin.start, basin.end
        basin_rms = np.median(rms[a:b])
       
        basin_hash = basin.kv()
        basin_hash['pdbfile'] = jc_filename
        logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)

        pipe.rpush('basin:list', bid)
        pipe.hset('basin:rms', bid, basin_rms)
        pipe.hmset('basin:%s'%bid, basin_hash)
        pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))
        pipe.execute()

def make_jobs(catalog, num=1):
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
  sim_step_size = float(catalog.get('sim_step_size'))
  force_field_dir = os.path.join(settings.workdir, catalog.get('ffield_dir'))
  sim_init = {key: catalog.get(key) for key in settings.sim_params.keys()}

  # Apply sampling Algorithm HERE
  if settings.EXPERIMENT_NUMBER == 12:
    sampler = UniformSampler(basinlist)
    global_params = getSimParameters(sim_init, 'seed')
    seedlist = sampler.execute(num)

  elif settings.EXPERIMENT_NUMBER == 13:
    global_params = getSimParameters(sim_init, 'deshaw')

    logging.info('Loading Pre-Calculated Correlation Matrix and mean/stddev vals')
    corr_matrix = np.load('data/de_corr_matrix.npy')
    dmu = np.load('data/de_ds_mu.npy')
    dsig = np.load('data/de_ds_mu.npy')
    sampler = CorrelationSampler(corr_matrix, dmu, dsig)
    seedlist = [catalog.lindex('basin:list', i) for i in sampler.execute(num)]
    for i in seedlist:
      logging.info("Select index: %s", i)

  elif settings.EXPERIMENT_NUMBER == 14:
    global_params = getSimParameters(sim_init, 'deshaw')
    logging.info('Loading Pre-Calculated Correlation Matrix and mean/stddev vals')

    # de_ds = 10*np.load(settings.datadir + '/de_ds_mu.npy')
    de_ds = []

    logging.info('Loading raw distance space from catalog')
    de_ds_raw = catalog.lrange('dspace', 0, -1)
    logging.info("Unpickling distance space")
    de_ds = np.zeros(shape=(len(de_ds_raw), 1653))
    for i, elm in enumerate(de_ds_raw):
      de_ds[i] = pickle.loads(elm)
    print('DS  : ', de_ds.shape, de_ds[0])

    # de_cm = np.load(settings.datadir + '/de_corr_matrix.npy')

    Kr = FEATURE_SET
    support = 900
    cutoff  = 8

    logging.info('Loading DEShw Pre-Constructed Lattice Data')
    dlat    = pickle.loads(catalog.get('lattice:dlat'))
    print('DLAT: ', len(dlat))
    max_fis = pickle.loads(catalog.get('lattice:max_fis'))
    print('MFIS: ', len(max_fis))
    low_fis = pickle.loads(catalog.get('lattice:low_fis'))
    print('LFIS: ', len(low_fis))
    Ik      = pickle.load(open(settings.datadir + '/iset.p', 'rb'))
    print('Ik  : ', len(Ik))

    logging.info('Building Existing lattice object')
    lattice=lat.Lattice(de_ds, Kr, cutoff, support)
    lattice.set_fis(max_fis, low_fis)
    lattice.set_dlat(dlat, Ik)

    sampler = LatticeSampler(lattice)

    start_indices = sampler.execute(num)
    seedlist = [catalog.lindex('basin:list', i) for i in start_indices]
    for i in seedlist:
      logging.info("Selected index: %s", i)


  else:
    logging.error('No Experiment Defined.')
    return


  # Create new jobs from selected basins
  # psf = os.path.join(settings.workdir, catalog.get('psffile'))

  for seed in seedlist:
    logging.debug('\nSeeding Job: %s ', seed)
    basin = catalog.hgetall('basin:%s'%seed)

    # Generate new set of params/coords
    jcID, config = generateFromBasin(basin)

    # Update Additional JC Params and Decision History, as needed
    config.update(global_params)

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
    load_historical_DEShaw(catalog)
    # if settings.EXPERIMENT_NUMBER == 12:
    #   load_historical_Expl(catalog)
    # elif settings.EXPERIMENT_NUMBER == 13:
    # else:
    #   logging.error("NO experiment defined")
    # # load_seeds(catalog, calc_seed_rms=False)

  if args.initjobs or args.all:
    numresources = int(catalog.get('numresources'))
    make_jobs(catalog, numresources)

  elif args.onejob:
    seed_jobs(catalog, 1)

  if args.updateschema:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    updateschema(catalog)

  # if args.reset:
  #   resetAnalysis(catalog)





### FOR Simple Uniform sampling of expl basins
# def load_seeds(catalog, calc_seed_rms=False):
#     settings = systemsettings()
#     idx_list  = [0,1,2,3,4,20,23,24,30,32,34,40,41,42]
#     seed_frame_length = 8000
#     slist = {k: 'seed%d'%k for k in idx_list}
#     tlist = {k: 'tr%d.dcd'%k for k in idx_list}
#     logging.info('Loading %d seeds for experiment %s', len(slist.keys()), catalog.get('name'))
#     seed_dir = os.path.join(settings.WORKDIR, 'seed')
#     seed_frame_rate = 0.25  # in ps
#     seed_ts_factor = 16     # TimeScape ran on 4ps frame rate (16x source)
#     bpti = Protein(bpti, catalog, load=True)
#     pdb_file = bpti.pdbfile
#     print("PDB FILE: ", settings.workdir, pdb_file)
#     topo = bpti.pdb
#     logging.info('Topology file: %s  <%s>', pdb_file, str(topo))
#     ref_file = os.path.join(settings.workdir, catalog.get('pdb:ref:0'))
#     ref_traj = md.load(ref_file)
#     logging.info('RMS Reference file: %s  <%s>', ref_file, str(ref_traj))
#     hfilt = bpti.get_filter('heavy')
#     ts_rate = catalog.get('timescape:rate')
#     for idx in idx_list:
#       s, t = slist[idx], tlist[idx]
#       logging.info('Trajectory SEED:  %s', s)
#       tfile = os.path.join(seed_dir, t)
#       rms_file = os.path.join(seed_dir, 'rms', 'rms%d'%idx)
#       if calc_seed_rms:
#         traj = md.load(tfile, top=topo)
#         traj.superpose(topo)
#         protein = traj.atom_slice(bpti.get_filter('protein'))
#         rms = 10*md.rmsd(protein, ref_traj, 0, hfilt, hfilt, precentered=True)
#         np.save(rms_file, rms)
#       else:
#         traj = None
#         rms = np.load(rms_file + '.npy')
#       file_idx = catalog.rpush('xid:filelist', tfile) - 1
#       start_index = catalog.llen('xid:reference')
#       catalog.rpush('xid:reference', *[(file_idx, x) for x in range(traj.n_frames)])
#       logging.info("  Seed Loaded. Loading TimeScape Data...")
#       seed_name = 'seed%d'%idx
#       ts_traj = TimeScapeParser(pdb_file, 
#           os.path.join(seed_dir, 'out', seed_name),
#           seed_name, dcd=tfile, traj=traj)
#       ts_traj.load_basins(frame_rate=seed_ts_factor)
#       for i, basin in enumerate(ts_traj.basins):
#         pipe = catalog.pipeline()
#         bid = basin.id
#         jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
#         minima_frame = md.load_frame(tfile, basin.mindex, top=topo) if traj is None else traj.slice(basin.mindex)
#         minima_frame.save_pdb(jc_filename)
#         a, b = basin.start, basin.end
#         basin_rms = np.median(rms[a:b])
#         basin_hash = basin.kv()
#         basin_hash['pdbfile'] = jc_filename
#         logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)
#         pipe.rpush('basin:list', bid)
#         pipe.hset('basin:rms', bid, basin_rms)
#         pipe.hmset('basin:%s'%bid, basin_hash)
#         pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))
#         pipe.execute()