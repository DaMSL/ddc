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

from core.common import systemsettings, executecmd, getUID
from core.kvadt import kv2DArray, decodevalue
from overlay.overlayService import OverlayService
from overlay.redisService import RedisService

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
  def __init__(self, name, port=6379):
    """
    """
    RedisService.__init__(self, name, port)

  def fetcher(self):
    """The fetcher is resposible for loading files and inserting data into
    the cache. It opens a local client connection and blocks until file
    data requests are inserted (by the corresponding remote client).
    When triggered, it will open the file and load data into the cache
    TODO:  Evictor (in here or separate)
    """
    config = systemsettings()

    # Wait until service is up (and not terminating on launch)
    # then exit is this happens
    while self.connection is None and not self.terminationFlag.is_set():
      time.sleep(1)

    if self.connection is None or self.terminationFlag.is_set():
      return

    conn = redis.StrictRedis(
        host='localhost', port=self._port, decode_responses=True)
    conn.client_setname("fetcher")

    block_timeout = 10   #needed to check for service termination
    while true:
      if self.terminationFlag.is_set():
        break

      # TODO: make this more dynamic
      request = conn.blpop('request:bpti', 'request:sim', block_timeout)
      if request is None:
        continue

      if request[0] == 'request:bpti':
        fileno = int(request[1])
        dcd = deshaw.getDEShawfilename_prot(fileno, fullpath=True)
        pdb = deshaw.PDB_PROT_FILE   
        key = 'deshaw:%02d' % fileno 

      else:        
        dcd  = os.path.join(config.JOBDIR, request[1], request[1] + '.dcd')
        pdb  = dcd.replace('dcd', 'pdb')
        key = 'sim:%s' % request[1]

      traj = md.load(dcd, top=pdb)

      # Insert all xyz coords into cache
      pipe = self.conn.pipeline()
      for i in traj.xyz:
        z = pipe.rpush(key, pickle.dumps(i))
      pipe.execute()


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
        logging.info('[CacheClient] Switched over to new master at %s on port %s', host, port)
      else:
        self.conn = redis.StrictRedis.__init__(self, connection_pool=pool, decode_responses=True)
        self.client_setname(getUID())
        logging.info('[CacheClient] Connected as client to master at %s on port %s', host, port)
        self.isconnected = True
      return True
    except FileNotFoundError as ex:
      logging.error('[CacheClient] ERROR. Service is not running. No lockfile found: %s', self.lockfile)
      return True

  def get(self, filename, frame, file_type):
    if not self.connect():
      logging.warning("Cache Service is unavailable")
      return None

    # For now, accepting 2 types ('deshaw' and 'sim')
    if file_type not in ['deshaw', 'sim']:
      logging.warning('[CacheClient] Only accepting "deshaw" and "sim"')
      return None

    if file_type == 'deshaw':
      key = 'deshaw:' + str(filename)
    else:
      key = 'sim:' + filename

    data = self.conn.lindex(key, frame)

    # Cache HIT: unpickle & return
    if data is not None:
      return pickle.loads(data)

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