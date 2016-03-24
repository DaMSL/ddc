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

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)



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


  def analyze_dcd(self, key, force=False):
    logging.info("OFFLINE ANALYSIS FOR:  %s", key)
    filelist = self.catalog.lrange('xid:filelist', 0, -1)
    job = self.catalog.hgetall('jc_' + key)

    if not force and job['dcd'] in filelist:
      logging.info('This file has already been analyzed and FORCE is not on. Skipping')
      return

    if self.centroid is None:
      self.loadcentroids()

  # 1. Get the Traj
    traj = md.load(job['dcd'], top=job['pdb'])
    logging.debug('Trajectory Loaded: %s (%s)', job['name'], str(traj))

  # 2. Update Catalog with HD points (TODO: cache this)
    file_idx = self.append({'xid:filelist': [job['dcd']]})[0]
    delta_xid_index = [(file_idx-1, x) for x in range(traj.n_frames)]
    global_idx = self.append({'xid:reference': delta_xid_index})
    global_xid_index_slice = [x-1 for x in global_idx]
    job['xid:start'] = global_xid_index_slice[0]
    job['xid:end'] = global_xid_index_slice[-1]

  # 3. Update higher dimensional index
    # Logical Sequence # should be unique seq # derived from manager (provides this
    #  worker's instantiation with a unique ID for indexing)
    logging.debug("3. RMS Calculation")
    # 1. Filter to Alpha atoms
    alpha = traj.atom_slice(deshaw.FILTER['alpha'])

    numLabels = len(self.centroid)
    numConf = len(traj.xyz)
    rmsraw = calc_rmsd(alpha, self.centroid)
    logging.debug('  RMS:  %d points projected to %d centroid-distances', numConf, numLabels)

    # 2. Account for noise
    noise = 10000
    stepsize = 500 if 'interval' not in job else int(job['interval'])
    nwidth = noise//(2*stepsize)
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
    rmslist = np.array([noisefilt(rmsraw, i) for i in range(numConf)])
    logging.debug("RMS CHECK......")
    for i in rmslist:
      logging.debug("  %s", str(np.argsort(i)))

    # 3. Append new points into the data store. 
    pipe = self.catalog.pipeline()
    for si in rmslist:
      pipe.rpush('subspace:rms', bytes(si))
    idxlist = pipe.execute()

    # 4. Apply Heuristics Labeling
    logging.debug('Applying Labeling Heuristic')
    rmslabel = []
    pipe = self.catalog.pipeline()
    for i, rms in enumerate(rmslist):
      #  Sort RMSD by proximity & set state A as nearest state's centroid
      prox = np.argsort(rms)
      A = prox[0]
      theta = .33
      proximity = abs(rms[prox[1]] - rms[A])    #abs
      B = prox[1] if proximity < theta else A
      rmslabel.append((A, B))
      logging.debug('Label for observation #%3d: %s', i, str((A, B)))
      pipe.rpush('varbin:rms:%d_%d' % (A, B), global_xid_index_slice[i])

    pipe.execute()
    # Update Catalog
    idxcheck = self.append({'label:rms': rmslabel})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-j', '--job')
  args = parser.parse_args()

  catalog = redis.StrictRedis(port=6381, decode_responses=True)

  task = offlineAnalysis(catalog)
  task.analyze_dcd(args.job)

