#!/usr/bin/env python

import argparse
import os
import sys
import shutil
import time
import logging
from collections import namedtuple, deque

 
# For efficient zero-copy file x-fer
import mdtraj as md
import numpy as np
import redis

from numpy import linalg as LA

from core.common import *
import mdtools.deshaw as deshaw
import datatools.datareduce as dr
from datatools.rmsd import calc_rmsd
from datatools.pca import project_pca
from bench.stats import StatCollector

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"


class offlineAnalysis(object):
  def __init__(self, catalog):
    self.catalog = catalog
    self.centroid = None

  def append(self, data):
    for key in data.keys():
      pipe = self.catalog.pipeline()
      for elm in data[key]:
        pipe.rpush(key, elm)
      results = pipe.execute()
    return results


  def storeNPArray(self, arr, key):
    #  Force numpy version 1.0 formatting
    header = {'shape': arr.shape,
              'fortran_order': arr.flags['F_CONTIGUOUS'],
              'dtype': np.lib.format.dtype_to_descr(np.dtype(arr.dtype))}
    self.catalog.hmset(key, {'header': json.dumps(header), 'data': bytes(arr)})

  def loadNPArray(self, key):
    elm = self.catalog.hgetall(key)
    if elm == {}:
      return None
    header = json.loads(elm['header'])
    arr = np.fromstring(elm['data'], dtype=header['dtype'])
    return arr.reshape(header['shape'])

  def loadcentroids(self):
    centfile = os.environ['HOME'] + '/ddc/data/gen-alpha-cartesian-centroid.npy'
    logging.info("Loading centroids from %s", centfile)
    self.centroid = np.load(centfile)
    self.storeNPArray(self.centroid, 'centroid')


  def analyze_job(self, key, force=False):
    logging.info("OFFLINE ANALYSIS FOR:  %s", key)
    # stat  = StatCollector('sim_naive', '%06d'%num)
    filelist = self.catalog.lrange('xid:filelist', 0, -1)
    job = self.catalog.hgetall('jc_' + key)

    if not force and job['dcd'] in filelist:
      logging.info('This file has already been analyzed and FORCE is not on. Skipping')
      return

  # 1. Get the Traj
    traj = md.load(job['dcd'], top=job['pdb'])
    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))

  # 2. Update Catalog with HD points (TODO: cache this)
    dcdfile = job['dcd']
    self.analyze_dcd(traj)


  def analyze_dcd(self, traj, dcdfile):
    file_idx = self.append({'xid:filelist': dcdfile})[0]
    delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
    global_idx_recv = self.append({'xid:reference': delta_xid_index})
    global_index = [x-1 for x in global_idx_recv]
    # catalog.hset('jc_' + key, 'xid:start', global_index[0])
    # catalog.hset('jc_' + key, 'xid:end', global_index[-1])

  # 3. Update higher dimensional index
    # 1. Filter to Alpha atoms
    alpha = traj.atom_slice(deshaw.FILTER['alpha'])

    if self.centroid is None:
      self.loadcentroids()

    #  Set Weights
    cw = [.92, .94, .96, .99, .99]
    numLabels = len(self.centroid)
    numConf = len(traj.xyz)
    rmsraw = calc_rmsd(alpha, self.centroid, weights=cw)
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 2. Account for noise
    nwidth = 10
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])

    # 3. Append new points into the data store. 
    pipe = self.catalog.pipeline()
    for si in rmslist:
      pipe.rpush('subspace:rms', bytes(si))
    idxlist = pipe.execute()

    # 4. Apply Heuristics Labeling
    rmslabel = []
    binlist = [(a, b) for a in range(numLabels) for b in range(numLabels)]
    label_count = {b: 0 for b in binlist}
    groupbystate = [[] for i in range(numLabels)]
    pipe = self.catalog.pipeline()
    for i, rms in enumerate(rmslist):
      prox = np.argsort(rms)
      A = prox[0]
      theta = .33
      proximity = abs(rms[prox[1]] - rms[A])    #abs
      B = prox[1] if proximity < theta else A
      rmslabel.append((A, B))
      # logging.debug('Label for observation #%3d: %s', i, str((A, B)))
      pipe.rpush('varbin:rms:%d_%d' % (A, B), global_index[i])
      label_count[(A, B)] += 1
      groupbystate[A].append(alpha.xyz[i])

    pipe.execute()
    # Update Catalog
    idxcheck = self.append({'label:rms': rmslabel})

    for b in binlist:
      pipe.rpush('observe:rms:%d_%d' % b, label_count[b])
    pipe.incr('observe:count')
    pipe.execute()
   


if __name__ == '__main__':
  logging.basicConfig(format='%(message)s', level=logging.DEBUG)
  parser = argparse.ArgumentParser()
  parser.add_argument('-j', '--job')
  parser.add_argument('--jobfile')
  parser.add_argument('-d', '--dcd')
  args = parser.parse_args()

  catalog = redis.StrictRedis(port=6380, decode_responses=True)

  task = offlineAnalysis(catalog)

  if args.job:
    task.analyze_job(args.job)
  elif args.jobfile:
    logging.info('Reading all jobs from: %s', args.jobfile)
    with open(args.jobfile) as src:
      for i, job in enumerate(src.read().strip().split('\n')):
        logging.info('Procesing job #%d  (%s)', i, job)
        task.analyze_job(job)
  elif args.dcd:
    traj = md.load(args.dcd, top=os.getenv('HOME')+'/work/jc/serial/k98564b78.pdb')
    task.analyze_dcd(traj, args.dcd)
  else:
    logging.info("Need to provide a job or jobfile")

