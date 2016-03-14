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

  def fetcher():
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

    block_timeout = 5   #needed to check for service termination
    while true:
      if self.terminationFlag.is_set():
        break

      # TODO: make this more dynamic
      request = conn.blpop('request:bpti', 'request:sim', block_timeout)
      if request is None:
        continue

      if request[0] == 'request:bpti':
        fileno = int(request[1])
        dcd = deshaw.getDEShawfilename(fileno, fullpath=True) % fileno
        pdb = deshaw.PDB_FILE   # FULL OR PROT HERE?????
        key = 'deshaw:%04d' % fileno    #OR fewer digits?

      else:        
        dcd  = os.path.join(config.JOBDIR, request[1], request[1] + '.dcd')
        pdb  = dcd.replace('dcd', 'pdb')
        key = 'sim:%s' % request[1]

      # Load File from disk
      traj = md.load(dcd, top=pdb)

      # Insert all xyz coords into cache
      pipe = r.pipeline()
      for i in traj.xyz:
        z = pipe.rpush(key, pickle.dumps(i))
      pipe.execute()




class CacheClient(object):

  def __init__(self, name):
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, 'CacheService')
    self.isconnected = False
    self.pool = self.connect()
    if self.pool is not None:
      self.conn = redis.StrictRedis.__init__(self, connection_pool=self.pool, decode_responses=True)
      self.client_setname(getUID())
      logging.info('[CacheClient] Connected as client to master at %s on port %s', host, port)
      self.isconnected = True

  
  def connect(self):
    try:
      with open(self.lockfile) as master:
        config = master.read().split('\n')[0].split(',')
      if len(config) < 2:
        logging.error('[CacheClient] ERROR. Lock file is corrupt.')
        return
      self.host = config[0]
      self.port = config[1]
      pool = redis.ConnectionPool(host=self.host, port=self.port, db=0, decode_responses=True)
      return pool
    except FileNotFoundError as ex:
      logging.error('[Redis Client] ERROR. Service is not running. No lockfile found: %s', self.lockfile)
      return None


  def get(self, filename, frame):
    # Check connection:
    if not self.conn.ping():
      self.pool = connect()
      if self.pool is None:
        logging.warning("Cache Service is currently not available")
        return None
      self.conn.connection_pool = self.pool
      self.isconnected = True
    
    if 'bpti' in filename:
      key = 'deshaw:' + filename
    else:
      key = 'sim:' + filename
    data = self.conn.lindex(key, frame)

    if data is not None:
      return pickle.loads(data)

    # Request to have the file cached
    if 'bpti' in filename:
      self.conn.rpush('request:deshaw', filename)
    else:
      self.conn.rpush('request:sim', filename)
    return None




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