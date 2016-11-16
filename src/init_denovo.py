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
from mdtools.trajectory import rms_delta
import mdtools.deshaw as deshaw
import bench.db as db

from mdtools.timescape import *
from sampler.basesample import *

import datatools.lattice as lat

import init_exp as initialize


def bootstrap_lattice(catalog, num=10, build_new=False):
  ''' Bootstrap After TimeScape has run on source trajectory '''
  home = os.getenv("HOME")
  support = 1
  cutoff  = 8

  start_coord = ['de2586_315', 'de531_20', 'de3765_63', 'de3305_668', 'de1732_139']
  dcdfile = lambda x: home + '/work/data/{0}.dcd'.format(x)
  outloc  = lambda x: home+'/work/jc/denovouniform1/{0}/{0}'.format(x)


  traj_list = {}

  basin_list = catalog.lrange('basin:list', 0, -1)
  if len(basin_list) == 134:
    logging.info('Basin Data already loaded!')
    rms_delta_list = [(i, np.sum(pickle.loads(catalog.get('basin:rmsdelta:'+b)))) for i, b in enumerate(basin_list)]
  else:
    logging.info('Loading all bootstrap data to initialize...')
    basin_list = []
    rms_delta_list = []
    pdb_file = home+'/work/data/alpha.pdb'
    topo = md.load(pdb_file)
    ref_alpha = md.load(home+'/work/' + catalog.get('pdb:ref:0'))
    ref_alpha.atom_slice(ref_alpha.top.select_atom_indices('alpha'), inplace=True)
    res_rms_Kr = FEATURE_SET

    for sc in start_coord:
      dist_space = []
      srcfile = outloc(sc) + '.dcd'
      pdbfile = srcfile.replace('dcd', 'pdb')
      logging.debug('LOADING TRAJ:  %s', srcfile)
      traj = md.load(srcfile, top = pdbfile)
      traj_list[sc] = traj
      alpha = traj.atom_slice(traj.top.select_atom_indices('alpha'))

      logging.info('Grabbing TS data...')
      W = TS.TimeScape.windows(outloc(sc) + '_transitions.log')
      ts_traj = TS.TimeScapeParser(pdbfile, outloc(sc), sc, dcd=srcfile, traj=traj)
      basins = ts_traj.load_basins()

      logging.info("Processing distance space and residue RMS")
      dsa = DR.distance_space(alpha)
      resrmsd = 10*np.array([LA.norm(i-ref_alpha.xyz[0], axis=1) for i in alpha.xyz])
      basin_res_rms = np.zeros(shape=(len(ts_traj.basins), alpha.n_atoms))
      for i, (a,b) in enumerate(W):
        dist_space.append(dsa[a:b].mean(0))
        basin_res_rms[i] = np.median(resrmsd[a:b], axis=0)

      basin_res_rms_delta = np.array([rms_delta(i) for i in basin_res_rms.T]).T
      logging.debug('RMS LEN CHECK:  %d =?= %d    -- Updating RMS Delta',len(basins), len(basin_res_rms_delta))


      for i, basin in enumerate(basins):
        pipe = catalog.pipeline()
        bid = basin.id

        # Store on Disk and in redis
        jc_filename = os.path.join(settings.datadir, 'basin_%s.pdb' % bid)
        logging.info('MIN for %s:   Idx# %d  to %s', bid, basin.mindex, jc_filename)
        minima_frame = traj.slice(basin.mindex)  #md.load_frame(src_traj, basin.mindex, top=src_traj.replace('dcd', 'pdb'))
        minima_frame.save_pdb(jc_filename)

        basin_hash = basin.kv()
        basin_hash['pdbfile'] = jc_filename
        logging.info('  Basin: %(id)s  %(start)d - %(end)d   Minima: %(mindex)d    size=%(len)d' % basin_hash)

        pipe.rpush('basin:list', bid)
        pipe.hmset('basin:%s'%bid, basin_hash)
        pipe.set('basin:dmu:'+bid, pickle.dumps(dist_space[i]))
        pipe.set('minima:%s'%bid, pickle.dumps(minima_frame))

        # FOR RESIDUE RMSD
        resrms_d = np.sum(basin_res_rms_delta[i][res_rms_Kr])
        basin_hash['resrms_delta'] = resrms_d
        rms_delta_list.append((len(basin_list), resrms_d))
        basin_list.append(basin_hash)
        pipe.set('basin:rmsdelta:'+bid, pickle.dumps(basin_res_rms_delta[i]))

        pipe.execute()




  # FOR RESIDUE RMSD

  # FOR SEED SAMPLING USING RMS_DELTA

  # Note: skip the first basin



  # Re-Construct the Lattice from 
  if build_new:
    dist_space = 10*np.array(dist_space)
    cm = ds<cutoff
    fs = lat.reduced_feature_set(cm,.115); len(fs)
    dr, cr = ds[:,fs], cm[:,fs]


    mfis,lfis = lat.maxminer(cr, 1)
    dlat, ik = lat.derived_lattice(mfis, dr, cr)
    pickle.dump(mfis, open(home + '/work/data/denovo_mfis.p', 'wb'))
    pickle.dump(lfis, open(home + '/work/data/denovo_lfis.p', 'wb'))
    pickle.dump(ik, open(home + '/work/data/denovo_iset.p', 'wb'))
    pickle.dump(dlat, open(home + '/work/data/denovo_dlat.p', 'wb'))

  else:

    logging.info('Loading Pre-Constructed Lattice Data')
    dlat = pickle.load(open(home + '/work/data/denovo_dlat.p', 'rb'))
    mfis = pickle.load(open(home + '/work/data/denovo_mfis.p', 'rb'))
    lfis = pickle.load(open(home + '/work/data/denovo_lfis.p', 'rb'))
    ik = pickle.load(open(home + '/work/data/denovo_iset.p', 'rb'))

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

  # Sample -- FOR USING LATTICE TO BOOTSTRAP
  # cl,sc,el = lat.clusterlattice(dlat, cr, dr, ik, num_k=8, invert=True)
  # cl_list = sorted(el, key=lambda x: len(x))

  # TODO: Check if fan out > single item clusters
  # start_indices = [clu[0][0] for clu in cl_list[:num]]

  rms_delta_ranked = [x[0] for x in sorted(rms_delta_list, key=lambda i: i[1], reverse=True)]
  start_indices = rms_delta_ranked[:num]

  seedlist = [catalog.lindex('basin:list', i) for i in start_indices]
  sim_init = {key: catalog.get(key) for key in settings.sim_params.keys()}
  global_params = getSimParameters(sim_init, 'seed')
  global_params['psf'] = home+'/work/jc/serial2/de0_0/de0_0.psf'

  for seed in seedlist:
    logging.debug('\nSeeding Job: %s ', seed)
    basin = catalog.hgetall('basin:%s'%seed)
    catalog.rpush('executed', seed)

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
  parser.add_argument('--lattice', action='store_true')
  parser.add_argument('--centroid', action='store_true')

  args = parser.parse_args()
  confile = args.name + '.json'

  settings = systemsettings()
  settings.applyConfig(confile)
  catalog = RedisClient(args.name)

  if args.initcatalog:
    settings.envSetup()
    initialize.initializecatalog(catalog)

  if args.lattice:
    numresources = int(catalog.get('numresources'))
    bootstrap_lattice(catalog, numresources)

  if args.centroid:
    start_coords = [(2586,315), (531,20), (3765,63), (3305,668), (1732,139)]
    for fileno, frame in start_coords:
      initialize.manual_de_job(catalog, fileno, frame)