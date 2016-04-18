#!/usr/bin/env python
"""Caching Overlay Service (built on top of Redis)
    Implementation for both the Cache service and the client
"""
import os
import time
import numpy as np
from collections import deque
from threading import Thread, Event
import logging
import argparse
import sys
import subprocess as proc
import shlex
import json
import pickle 

import redis
import mdtraj as md

from core.common import systemsettings, executecmd, getUID
from core.kvadt import kv2DArray, decodevalue
from overlay.overlayService import OverlayService
from overlay.redisOverlay import RedisService
import mdtools.deshaw as deshaw

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

class CacheService(RedisService):
  """
  """

  #  TODO:  Ensure catalog and cache run on separate servers
  def __init__(self, name, port=6380):
    """
    """
    RedisService.__init__(self, name, port)

    self._name_app = name + '_cache'

    self.fetchthread = Thread(target=self.fetcher)
    self.fetchthread.start()
    self.lru_queue = 'cache:lru'
    logging.info('[Cache] Service is initialized.')

  def tear_down(self):
    logging.info("[Cache] Ensuring the fetcher is complete.")
    self.fetchthread.join()
    logging.info("[Cache] Fetcher thread terminated. Tear down complete.")

  def fetcher(self):
    """The fetcher is resposible for loading files and inserting data into
    the cache. It opens a local client connection and blocks until file
    data requests are inserted (by the corresponding remote client).
    When triggered, it will open the file and load data into the cache.
    This is designed to run in a separate thread.
    TODO:  Evictor (in here or separate)
    """
    config = systemsettings()

    # Capacity should be set in GB (in settings)
    capacity  = config.CACHE_CAPACITY * (2**30)
    logging.info('[Fetcher] Fetcher is starting. Preparing capacity of %d GB', config.CACHE_CAPACITY)

    # Create client
    conn = redis.StrictRedis(host='localhost', port=self._port, decode_responses=True)

    # Wait until service is up (and not terminating on launch)
    timeout = 60
    while True:
      try:
        logging.info('[Fetcher] Fetcher is waiting for the service to start... Conn=%s', (self.connection is not None))
        info = conn.info()
        if info is None:
          logging.info('[Fetcher] No commo with the local service. Wait and retry')
          time.sleep(1)
          timeout -= 1
          if timeout == 0:
            logging.error('[Fetcher] Fetch never connected to the cache. Shutting down...')
            self.terminationFlag.set()
            break
        elif info['loading']:
          logging.info('[Fetcher] Service is loading')
          time.sleep(10)
        else:
          conn.client_setname("fetcher")
          break
      except redis.BusyLoadingError as e:
        logging.info('[Fetcher] Service is loading')
        time.sleep(10)
        continue
      except redis.RedisError as e:
        if timeout == 0:
          logging.error('[Fetcher] Fetch never connected to the cache. Shutting down...')
          self.terminationFlag.set()
          break        
        logging.info('[Fetcher] Cannot connect to serice. Will keep trying for %s seconds', timeout)
        time.sleep(1)
        timeout -= 1
        continue

    # then exit if it cannot connect
    if not self.ping() or self.terminationFlag.is_set():
      logging.error('[Fetcher] Cannot connect to the server')
      return

    logging.error('[Fetcher] Fetch found the service. Connecting as a client.')


    block_timeout = 60   #needed to check for service termination (MV-> setting)
    while True:
      if self.terminationFlag.is_set():
        break

      # TODO: make this more dynamic
      try:
        request = conn.blpop(['request:deshaw', 'request:sim'], block_timeout)
      except redis.ConnectionError:
        logging.warning('[Fetcher] LOST local connection. Returning')
        return
      if request is None:
        logging.info('[Fetcher] Fetcher heartbeat.... timeout waiting on requests')
        continue

      logging.info('[Fetcher] Fetcher Got a request...')
      if request[0] == 'request:deshaw':
        fileno = int(request[1])
        key = 'deshaw:%02d' % fileno 
        dcd = deshaw.getDEShawfilename_prot(fileno, fullpath=True)
        pdb = deshaw.PDB_PROT   

      else:
        key = 'sim:%s' % request[1]
        dcd  = os.path.join(config.JOBDIR, request[1], request[1] + '.dcd')
        pdb  = dcd.replace('dcd', 'pdb')

      if conn.exists(key):
        logging.info('File Load request for existing key: %s', key)
        continue

      logging.info('File Request!\n%-20s\n%-20s\n%-20s\n%-20s\n%-20s', 
        request[0], request[1], dcd, pdb, key)

      logging.info('Loading Trajectory')
      if os.path.exists(dcd) and os.path.exists(pdb):
        traj = md.load(dcd, top=pdb)

      # TODO: Check Memory reporting:
        #  'used_memory' -> actual memory used by REDIS
        #  'used_memroy_rss' -> allocated memory (as reported by OS)

      used_mem = int(conn.info('memory')['used_memory'])
      available = capacity - used_mem
      logging.debug('Cache Capacity check:  Used= %6dMB    Capacity= %6dMB   Avail=%d bytes  Data=%d', \
        (used_mem/(2**20)), (capacity/(2**20)), available, traj.xyz.nbytes)

      if available >= traj.xyz.nbytes:
        logging.info('Cache has available capacity')

      while available < traj.xyz.nbytes:
        logging.info('Cache is full. Evicting')
        lru_queue = conn.lrange('cache:lru', 0, -1)

        index = 1
        while True:
          #Since cache hits perform a LRU update -- the evictor needs to check 
          # the queue for any subsequent hits
          if index > len(lru_queue):
            logging.error("Eviction Error. The evictor can't find anything. We may need more capacity (overflow)")
            break
          
          candidate = conn.lpop('cache:lru')
          found = False
          # Linear scan to check remainder of the queue for any newer requests
          for i in range(index, len(lru_cache)):
            if candidate == lru_cache[i]:
              # Found a more recent cache request. Moving to next file in LRU:
              index +=1
              found = True
          if not found:
            break

        # Evict this candidate (TODO:  Separate the evictor to separate module(s))
        conn.delete(candidate)
        used_mem = int(conn.info('memory')['used_memory'])
        available = capacity - used_mem
        logging.debug('Cache Capacity check:  Used= %6dMB    Capacity= %6dMB   Avail=%d bytes  Data=', \
          (used_mem*(2**20)), (capacity*(2**20)), available, traj.xyz.nbytes)

      # Insert all xyz coords into cache
      logging.info('[Cache] Insering %d points to %s', len(traj.xyz), key)
      pipe = conn.pipeline()
      for i in traj.xyz:
        z = pipe.rpush(key, pickle.dumps(i))
      pipe.execute()
      check = conn.llen(key)
      logging.info('[Cache] CHECK: LLEN= %d   for KEY=%s', check, key)


  # TODO: Eventually make ea file load a separate thrad (if performance is needed)
  def loader(self, dcd, pdb):
    # TODO: Loader as a thread
    # Load File from disk
    traj = md.load(dcd, top=pdb)
    # Insert all xyz coords into cache
    pipe = self.conn.pipeline()
    for i in traj.xyz:
      z = pipe.rpush(key, pickle.dumps(i))
    pipe.execute()




class CacheClient(object):

  def __init__(self, name):
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, 'CacheService')
    self.isconnected = False
    self.conn = None
    if self.connect():
      logging.info("[CacheClient] is ready to serve.")
    else:
      logging.info("[CacheClient] has no master (cannot serve).")
  
  def connect(self):
    if self.isconnected and self.conn.ping():
      return True
    try:
      with open(self.lockfile) as master:
        config = master.read().split('\n')[0].split(',')
      if len(config) < 2:
        logging.error('[CacheClient] ERROR. Lock file is corrupt.')
        return False
      self.host = config[0]
      self.port = config[1]
      pool = redis.ConnectionPool(host=self.host, port=self.port, db=0, decode_responses=True)
      if self.isconnected:
        self.conn.connection_pool = pool
        logging.info('[CacheClient] Switched over to new master at %s on port %s', self.host, self.port)
      else:
        self.conn = redis.StrictRedis(host=self.host, port=self.port, connection_pool=pool, decode_responses=True)
        self.conn.client_setname(getUID())
        logging.info('[CacheClient] Connected as client to master at %s on port %s', self.host, self.port)
        self.isconnected = True
      return True
    except FileNotFoundError as ex:
      logging.error('[CacheClient] ERROR. Service is not running. No lockfile found: %s  (Returning False)', self.lockfile)
      return False
    except redis.BusyLoadingError as ex:
      logging.error('[CacheClient] Service is loading.')
      self.isconnected = True
      return True

  def get(self, filename, frame, file_type):
    if not self.connect():
      logging.warning("Cache Service is unavailable")
      return None

    # For now, accepting 2 types ('deshaw' and 'sim')
    if file_type not in ['deshaw', 'sim']:
      logging.warning('[CacheClient] Only accepting "deshaw" and "sim"')
      return None

    logging.info('[CacheClient] Request for %s   [%d]', str(filename), frame)

    if file_type == 'deshaw':
      fileno = int(filename)
      key = 'deshaw:%02d' % fileno
    else:
      key = 'sim:%s' % filename


    data = self.conn.lindex(key, frame)

    # Cache HIT: unpickle & return
    if data is not None:
      logging.info('[CacheClient] CACHE HIT. Updating LRU')
      self.conn.rpush('cache:lru', key)
      return pickle.loads(data)

    # Cache MISS:  Request to have the file cached
    logging.info('[CacheClient] CACHE MISS.  ftype=%s  [%s]', file_type, str(filename))
    if file_type == 'deshaw':
      self.conn.rpush('request:deshaw', str(filename))
    else:
      self.conn.rpush('request:sim', filename)
    return None


  def get_many(self, filename, framelist, file_type):
    if not self.connect():
      logging.warning("Cache Service is unavailable")
      return None

    # For now, accepting 2 types ('deshaw' and 'sim')
    if file_type not in ['deshaw', 'sim']:
      logging.warning('[CacheClient] Only accepting "deshaw" and "sim"')
      return None

    if file_type == 'deshaw':
      fileno = int(filename)
      key = 'deshaw:%02d' % fileno
    else:
      key = 'sim:%s' % filename

    iscached = (self.conn.exists(key) == 1)
    logging.debug('[CacheClient] Request for key %s. Presence: %s', key, str(iscached))
    if iscached:
      pipe = self.conn.pipeline()
      for frame in framelist:
        pipe.lindex(key, frame)
      data = pipe.execute()
    else:
      data = None

    # Cache HIT: unpickle & return
    if data is not None:
      self.conn.rpush('cache:lru', key)
      # TODO: Handle file pages
      points = [pickle.loads(pt) for pt in data]
      return points

    # Cache MISS:  Request to have the file cached
    if file_type == 'deshaw':
      self.conn.rpush('request:deshaw', str(filename))
    else:
      self.conn.rpush('request:sim', filename)
    return None




  def put(self, name, traj):
    if not self.connect():
      logging.warning("Cache Service is unavailable")
      return None
    # Assume we're only putting in new sim files data here
    pipe = self.conn.pipeline()
    for i in traj.xyz:
      pipe.rpush('sim:' + name, pickle.dumps(i))
    pipe.execute()

  def putDEShaw(self, name, traj):
    if not self.connect():
      logging.warning("Cache Service is unavailable")
      return None
    # Assume we're only putting in new sim files data here
    pipe = self.conn.pipeline()
    for i in traj.xyz:
      pipe.rpush('sim:' + name, pickle.dumps(i))
    pipe.execute()






############
### OVERLAY CACHE
# import os
# import mdtraj as md
# import redis
# import datetime as dt
# import pickle

# # intial load of mdfile
# HOME = os.environ['HOME']
# pdb = HOME+"/work/bpti/bpti-prot.pdb"
# st = dt.datetime.now()
# traj= md.load(HOME+"/work/bpti/bpti-prot-%02d.dcd"%0, top=pdb)
# ed = dt.datetime.now()
# print('Loadtime= ', (ed-st).total_seconds())

# # Single File Load = 10.113577


# r = redis.StrictRedis(host='compute0022', decode_responses=True)

# st = dt.datetime.now()
# pipe = r.pipeline()
# for i in traj.xyz:
#   z = pipe.rpush('deshaw:xyz', pickle.dumps(i))

# z = pipe.execute()
# ed = dt.datetime.now()
# print('Loadtime= ', (ed-st).total_seconds())

# # Load time = 53.64

# pickle.loads(r.lindex('deshaw:xyz', 333)

#retrieve