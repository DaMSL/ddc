#!/usr/bin/env python

import argparse
import math
import json
import os
import bisect
import datetime as dt
import shutil

import mdtraj as md
import numpy as np
from numpy import linalg as LA


from core.common import *
from overlay.redisOverlay import RedisClient
from core.slurm import slurm
import core.ops as ops


import mdtools.timescape as TS
import datatools.datareduce as DR


from mdtools.simtool import *
from mdtools.structure import Protein
import mdtools.deshaw as deshaw
import bench.db as db

from mdtools.timescape import *
from sampler.basesample import *

import datatools.lattice as lat

import init_exp as initialize


def bootstrap(catalog, num=4, build_new=False):
  ''' Bootstrap After TimeScape has run on source trajectory '''
  home = os.getenv("HOME")
  support = 1
  cutoff  = 8
  wkdir = home+'/work/jc/serial2/'
  pdb_file = wkdir+'serinit.pdb'

  topo = md.load(pdb_file)
  traj = md.load(wkdir+'serinit.dcd', top=topo)
  afilt = topo.top.select_atom_indices('alpha')
  alpha = traj.atom_slice(afilt)

  W = TS.TimeScape.windows(home+'/work/jc/serial2/serial_transitions.log')
  ts_traj=TS.TimeScapeParser(pdb_file, wkdir+'serial', 'serial', dcd=wkdir+'serinit.dcd', traj=traj)  
  basins = ts_traj.load_basins()

  src_traj = wkdir + 'de0_0/serial.dcd'
  src_topo = md.load(wkdir + 'de0_0/de0_0.pdb')

  ds = []
  dsa = DR.distance_space(alpha)
  for a,b in W: 
    ds.append(dsa[a:b].mean(0))
  ds = 10*np.array(ds[1:])
  cm = ds<cutoff
  fs = lat.reduced_feature_set(cm,.075); len(fs)
  dr, cr = ds[:,fs], cm[:,fs]

  # Note: skip the first basin
  for i, basin in enumerate(ts_traj.basins[1:]):
    pipe = catalog.pipeline()
    bid = basin.id
    # Store on Disk and in redis
    jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
    minima_frame = md.load_frame(src_traj, basin.mindex, top=src_topo)
    minima_frame.save_pdb(jc_filename)

    a, b = basin.start, basin.end
   
    basin_hash = basin.kv()
    basin_hash['pdbfile'] = jc_filename
    logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)

    pipe.rpush('basin:list', bid)
    pipe.hmset('basin:%s'%bid, basin_hash)
    pipe.set('basin:cm:'+bid, pickle.dumps(cm[i]))
    pipe.set('basin:dmu:'+bid, pickle.dumps(ds[i]))
    pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))
    pipe.execute()



  # Re-Construct the Lattice from 
  if build_new:
    mfis,lfis = lat.maxminer(cr, 1)
    dlat, ik = lat.derived_lattice(mfis, dr, cr)
    pickle.dump(mfis, open(wkdir+'mfis.p', 'wb'))
    pickle.dump(lfis, open(wkdir+'lfis.p', 'wb'))
    pickle.dump(iset, open(wkdir+'iset.p', 'wb'))
    pickle.dump(ik, open(wkdir+'iset.p', 'wb'))

  else:

    logging.info('Loading Pre-Constructed Lattice Data')
    dlat = pickle.load(open(wkdir+'dlat.p', 'rb'))
    mfis = pickle.load(open(wkdir+'mfis.p', 'rb'))
    lfis = pickle.load(open(wkdir+'lfis.p', 'rb'))
    ik = pickle.load(open(wkdir+'iset.p', 'rb'))

  with catalog.pipeline() as pipe:
    pipe.set('lattice:max_fis', pickle.dumps(mfis))
    pipe.set('lattice:low_fis', pickle.dumps(lfis))
    pipe.set('lattice:dlat', pickle.dumps(dlat))
    pipe.set('lattice:iset', pickle.dumps(ik))
    pipe.execute()

  # logging.info('Building Existing lattice object')
  # lattice=lat.Lattice(ds, fs, cutoff, support)
  # lattice.set_fis(max_fis, low_fis)
  # lattice.set_dlat(dlat, Ik)
  # sampler = LatticeSampler(lattice)

  # Sample
  cl,sc,el = lat.clusterlattice(dlat, cr, dr, ik, num_k=8)
  cl_list = sorted(el, key=lambda x: len(x))

  # TODO: Check if fan out > single item clusters
  start_indices = [clu[0][0] for clu in cl_list[:num]]
  seedlist = [catalog.lindex('basin:list', i) for i in start_indices]
  sim_init = {key: catalog.get(key) for key in settings.sim_params.keys()}
  global_params = getSimParameters(sim_init, 'seed')
  global_params['psf'] = wkdir+'de0_0/de0_0.psf'

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



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', default='default')
  parser.add_argument('--initcatalog', action='store_true')
  parser.add_argument('--initjobs', action='store_true')
  parser.add_argument('--all', action='store_true')

  args = parser.parse_args()
  confile = args.name + '.json'

  settings = systemsettings()
  settings.applyConfig(confile)
  catalog = RedisClient(args.name)

  if args.initcatalog or args.all:
    settings.envSetup()
    initialize.initializecatalog(catalog)

  if args.initjobs or args.all:
    numresources = int(catalog.get('numresources'))
    bootstrap(catalog, numresources)
