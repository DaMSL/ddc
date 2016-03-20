#!/usr/bin/env python
"""Redis Over Service
    Implementation for both the Redis Server persistent overlay service
    and the client interface
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

import redis

from core.common import systemsettings, executecmd, getUID
from core.kvadt import kv2DArray, decodevalue
from overlay.overlayService import OverlayService

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

class RedisService(OverlayService):
  """
  """

  # REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'

  def __init__(self, name, port=6379):
    """
    """
    OverlayService.__init__(self, name, port)
    self.connection = None

    config = systemsettings()
    if not config.configured():
      # For now assume JSON file
      config.applyConfig(name + '.json')

    self.workdir   = config.WORKDIR  #ini.get('workdir', '.')
    self.redis_conf_template =  config.REDIS_CONF_TEMPLATE #init.get('redis_conf_template', 'templates/redis.conf.temp')
    self.MONITOR_WAIT_DELAY    = config.MONITOR_WAIT_DELAY #ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = config.CATALOG_IDLE_THETA #ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = config.CATALOG_STARTUP_DELAY #ini.get('catalog_startup_delay', 10)

    # Check if a connection exists to do an immediate shutdown request
    if os.path.exists(self.lockfile):
      host, port = self.getconnection()
      self.shutdowncmd = 'redis-cli -h %s -p %s shutdown' % (host, port)

  def ping(self, host='localhost', port=None):
    """Heartbeat to the local Redis Server
    """
    # TODO: Should wrap around try/catch and propagate an IO exception
    if port is None:
      port = self._port
    ping_cmd = 'redis-cli -h %s -p %s ping' % (host, port)
    pong = executecmd(ping_cmd).strip()
    return (pong == 'PONG')

  def prepare_service(self):
    # Prepare 
    with open(self.redis_conf_template, 'r') as template:
      source = template.read()
      logging.info("Redis Source Template loaded")

    # params = dict(localdir=DEFAULT.WORKDIR, port=self._port, name=self._name)
    params = dict(localdir=self.workdir, port=self._port, name=self._name_app)

    # TODO: This should be local
    self.config = self._name_app + "_db.conf"
    with open(self.config, 'w') as config:
      config.write(source % params)
      logging.info("Data Store Config written to  %s", self.config)

    self.launchcmd = 'redis-server %s' % self.config
    self.shutdowncmd = 'redis-cli shutdown'

  def idle(self):
    return False
    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")

    for client in self.connection.client_list():
      if client['name'] == 'monitor':
        continue
      if int(client['idle']) < self.CATALOG_IDLE_THETA:
        logging.debug('[Monitor - %s]  Service was idle for more than %d seconds. Stopping.', self._name_svc, CATALOG_IDLE_THETA)
        return True
    return False

  def handover_to(self):
    """Invoked to handover services from this instance to a new master
    Called when a new slave is ready to receive as a new master
    """
    logging.info("[Overlay - %s] RECEVIED Flag to Handover.", self._name_svc)
    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")

    info = self.connection.info()
    if info['connected_slaves'] == 0:
      logging.error("[Overlay - %s] ERROR. Flagged to Handover. But no slaves are connected.", self._name_svc)
      return

    # TODO: For multiple slave: iterate for each and find first one online.....
    next_master = info['slave0']
    if next_master['state'] != 'online':
      logging.error("[Overlay - %s] ERROR. Slave tabbed for next master `%s` is OFFLINE", self._name_svc, next_master['ip'])
      return
    logging.info("[Overlay - %s] Detected next master at %s.", self._name_svc, next_master)    

    # Become slave of next master (effectively goes READONLY on both ends)

    # TODO: Stop Receiving new connections (e.g. Client pause)
    logging.info("[Overlay - %s] Checking for active clients before I shut down.", self._name_svc)    
    while True:
      active_clients = False
      for client in self.connection.client_list():
        # Ignore other slave(s) 
        if client['name'] == 'monitor' or client['flags'] == 'S':
          continue
        if 'x' in client['flags'] or int(client['multi']) > 0:
          logging.debug('[Monitor - %s]  Found a client processing a pipeline. Waiting.', self._name_svc)
          active_clients = True
        if int(client['idle']) < 3:
          logging.debug('[Monitor - %s]  Found a client idle for less than 3 second. Waiting to stop serving.', self._name_svc)
          active_clients = True
      if active_clients:
        time.sleep(1)
        continue
      break

    logging.info("[Overlay - %s] No active clients releasing Master authority.", self._name_svc)    
    self.connection.slaveof(next_master['ip'], self._port)
    self.master = next_master['ip']

    while True:
      if self.connection.info()['master_link_status'] == 'up':
        logging.debug("[Overlay - %s] New MASTER has assumed responsibility on %s.", self._name_svc, self.master)
        break
      time.sleep(1)

    return self.master

  def handover_from(self):
    """Invoked as a callback when another master service is handing over
    service control (i.e. master duties) to this instance
    """
    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")
    
    if self.master is None:
      logging.error("[Overlay - %s] Handover TO called but no master is defined.", self._name_svc)
      return

    # Connect to the master as a slave
    self.connection.slaveof(self.master, self._port)

    # Wait until the connection to master is established
    while True:
      connected_to_master = (self.connection.info()['master_link_status'] == 'up')
      if connected_to_master:
        break
      logging.debug("[Overlay - %s] New REPLICA is waiting to connect to the Redis Master.", self._name_svc)
      time.sleep(1)

    # Synch DB with the master  (Should we call sync explicitly here???)
    while True:
      info = self.connection.info()
      if info['master_sync_in_progress']:
        logging.debug("[Overlay - %s] SYNC with Redis Master in progress\n  \
          %d Bytes remaining\n \
          %d Seconds since last I/O", self._name_svc, 
          info['master_sync_left_bytes'], info['master_sync_last_io_seconds_ago'])
        time.sleep(1)
      else:
        logging.debug("[Overlay - %s] REPLICA SYNC complete.", self._name_svc)
        break

    # Flag master to go read only and then stop

    # FLAG MASTER
    with open(self.lockfile, 'a') as conn:
      conn.write('\nslave,%s,SYNC' % self._host)

    # Replica prepared to begin as master. It will wait until notified by master to 
    #   Assume responsibility
    while True:
      if self.connection.info()['master_link_status'] != 'up':
        logging.debug("[Overlay - %s] Detected MASTER has gone READONLY.", self._name_svc)
        break
      time.sleep(1)

    # Become the master
    logging.debug("[Overlay - %s] Assumed control as MASTER on %s.", self._name_svc, self._host)
    self.connection.slaveof()

  def tear_down(self):
    logging.info("[Redis] Tear down complete.")

class RedisClient(redis.StrictRedis):

  def __init__(self, name):
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, 'RedisService')
    self.isconnected = False
    try:
      with open(self.lockfile) as master:
        config = master.read().split('\n')[0].split(',')
      if len(config) < 2:
        logging.error('[Client - Redis] ERROR. Lock file is corrupt.')
        return
      host = config[0]
      port = config[1]
      self.pool = redis.ConnectionPool(host=host, port=port, db=0, decode_responses=True)
      redis.StrictRedis.__init__(self, connection_pool=self.pool, decode_responses=True)
      self.client_setname(getUID())
      self.host = host
      self.port = port
      logging.info('[Redis Client] Connected as client to master at %s on port %s', host, port)
      self.isconnected = True
    except FileNotFoundError as ex:
      logging.error('[Redis Client] ERROR. Service is not running. No lockfile found: %s', self.lockfile)

  
  def execute_command(self, *args, **options):
    """Execute a command and return a parsed response
    Catches connection errors and attempts to connect with correct master
    """
    initial_connect = True
    while True:
      pool = self.connection_pool
      command_name = args[0]
      connection = pool.get_connection(command_name, **options)
      try:
          connection.send_command(*args)
          cursor = self.parse_response(connection, command_name, **options)
          connection.disconnect()
          return cursor
      except (ConnectionResetError, ConnectionAbortedError, redis.ReadOnlyError) as e:
        logging.warning('[Redis Client] Current Master is busy. It may be trying to shutdown. Wait and try again')
        time.sleep(3)
        continue
      except (redis.ConnectionError, redis.TimeoutError) as e:
        logging.warning('[Redis Client] Error connecting to %s', str(self.host))
      logging.info('[Redis Client] Rechecking lock')
      try:
        connection.disconnect()
        with open(self.lockfile) as master:
          config = master.read().split('\n')[0].split(',')
          if config[0] == self.host and config[1] == str(self.port):
            if initial_connect:
              time.sleep(5)
              initial_connect = False
              continue
            else:
              logging.warning('[Redis Client] Cannot connect to the master on %s', str(self.host))
              raise redis.ConnectionError
          self.host = config[0]
          self.port = config[1]
          self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, decode_responses=True)
          logging.info('[Redis Client] Changing over to new master on %s, port=%s', self.host, self.port)
      except FileNotFoundError as ex:
        logging.error('[Client - Redis] ERROR. Service is not running.')
        break

  def loadSchema(self):
    logging.debug("Loading system schema")

    self.schema = self.hgetall('META_schema')
    # for k, v in self.schema.items(): print("  ",k, v)

  def clear(self):
    self.flushdb()

  def save(self, data):
    deferredsave = []
    pipe = self.pipeline()
    for key, value in data.items():
      if key not in self.schema.keys():
        deferredsave.append(key)
        continue

      if self.schema[key] == 'list':
        pipe.delete(key)
        for val in value:
          pipe.rpush(key, val)
      elif self.schema[key] == 'dict':
        pipe.hmset(key, value)
      elif self.schema[key] in ['matrix', 'ndarray']:
        deferredsave.append(key)
      else:
        pipe.set(key, value)
      logging.debug("  Saving data elm  `%s` as %s ", key, type(data[key]))

    result = pipe.execute()

    for key in deferredsave:
      if key not in self.schema.keys():
        if isinstance(data[key], list):
          self.delete(key)
          for elm in data[key]:
            self.rpush(key, elm)
        elif isinstance(data[key], dict):
          self.hmset(key, data[key])
        else:
          self.set(key, data[key])
        logging.debug("Dynamically saving %s. Updating schema", key)

      elif self.schema[key] == 'ndarray':
        self.storeNPArray(data[key])
      elif self.schema[key] == 'matrix':
        if isinstance(data[key], np.ndarray):
          matrix = kv2DArray(self, key)
          matrix.set(data[key])
        else:
          matrix = kv2DArray(self, key, )

    return result

  # Retrieve data stored for each key in data & store into data 
  def load(self, keylist):

    if isinstance(keylist, dict):
      logging.error("NOT ACCEPTING DICT HERE")
      sys.exit(0)

    keys = keylist if isinstance(keylist, list) else [keylist]
    data = {}
    deferredload = []

    # Pipeline native data type
    pipe = self.pipeline()
    for key in keys:
      if key not in self.schema.keys():
        deferredload.append(key)
        continue
      if self.schema[key] == 'list':
        pipe.lrange(key, 0, -1)
      elif self.schema[key] == 'dict':
        pipe.hgetall(key)
      elif self.schema[key] in ['matrix', 'ndarray']:
        deferredload.append(key)
      else:
        pipe.get(key)

    vals = pipe.execute()

    #  Deferred load for retrieval of non-native data types
    for key in deferredload:
      if key not in self.schema.keys():
        try:
          value = self.lrange(key, 0, -1)
        except redis.ResponseError as ex:
          try:
            value = self.hgetall(key)
          except redis.ResponseError as ex:
            value = self.get(key)
        data[key] = decodevalue(value)
        self.schema[key] = type(value).__name__

      elif self.schema[key] == 'ndarray':
        data[key] = self.loadNPArray(key)
      
      elif self.schema[key] == 'matrix':
        matrix = kv2DArray(self, key)
        data[key] = matrix.get()

    #  Data Conversion
    # TODO:  MOVE TO JSON or pickel BASED RETRIEVAL (???)

    i = 0

    for key in keys:
      try:
        if key in deferredload:
          continue
        data[key] = decodevalue(vals[i])
        if data[key] is None:
          if self.schema[key] == 'list':
            data[key] = []
          elif self.schema[key] == 'dict':
            data[key] = {}
        i += 1
      except (AttributeError, KeyError) as ex:
        logging.error("ERROR! Loading a BAD KEY:  %s", key)
        logging.error("Trace:  %s", str(ex))
        sys.exit(0)

    return data

  def append(self, data):
    print('CATALOG APPEND')
    deferredappend = []
    pipe = self.pipeline()
    for key, value in data.items():
      logging.debug("Appending data elm  `%s` of type, %s", key, type(data[key]))
      if key not in self.schema.keys():
        logging.warning("  KEY `%s` not found in local schema! Will try Dynamic Append")
        deferredappend.append(key)
      elif self.schema[key] == 'int':
        pipe.incr(key, value)
      elif self.schema[key] == 'float':
        pipe.incrbyfloat(key, value)
      elif self.schema[key] == 'list':
        for val in value:
          pipe.rpush(key, val)
      elif self.schema[key] == 'dict':
        for k, v in value:
          pipe.hset(key, k, v)
      elif self.schema[key] in ['matrix', 'ndarray']:
        deferredappend.append(key)
      else:
        pipe.set(key, value)

    result = pipe.execute()

    for key in deferredappend:
      if self.schema[key] == 'ndarray':
        logging.warning(" NUMPY ARRAY MERGING NOT SUPPORTED")
        # self.storeNPArray(data[key])
      elif self.schema[key] == 'matrix':
        logging.warning("  Merging matrix `%s`:  %s", key, str(data[key]))
        matrix = kv2DArray(self, key)
        matrix.merge(data[key])

    return result

  # Slice off data in-place. Asssume key stores a list
  def slice(self, key, num):
    try:
      data = self.lrange(key, 0, num-1)
      self.ltrim(key, num, -1)
      return data
    except TypeError as e:
      logging.warning('Cannot slice %s  with num=%s  (type=%s)', key, str(num), str(type(num)))
      return None

  # Remove specific items from a list
  #  Given: a list and a set of indices into that list
  def removeItems(self, key, itemlist):
    nullvalue = getUID()

    pipe = self.pipeline()
    for index in itemlist:
      pipe.lset(key, index, nullvalue)

    pipe.lrem(key, 0, nullvalue)
    pipe.execute()

  def storeNPArray(self, arr, key):
    #  Force numpy version 1.0 formatting
    header = {'shape': arr.shape,
              'fortran_order': arr.flags['F_CONTIGUOUS'],
              'dtype': np.lib.format.dtype_to_descr(np.dtype(arr.dtype))}
    self.hmset(key, {'header': json.dumps(header), 'data': bytes(arr)})

  def loadNPArray(self, key):
    elm = self.hgetall(key)
    if elm == {}:
      return None
    header = json.loads(elm['header'])
    arr = np.fromstring(elm['data'], dtype=header['dtype'])
    return arr.reshape(header['shape'])



def test_redismaster():
  server = RedisService('testoverlay')
  server.start()

def test_redisclient(name):
  client = RedisClient(name)
  if not client.isconnected:
    print('I am not connected. Service is not running')
    return
  client.ping()
  print(client.get('foo'))
  print('Running a long pipeline...')
  pipe = client.pipeline()
  for i in range(40000):
    allkeys = pipe.keys('*')
    pipe.set('allkeys', str(allkeys))
  result = pipe.execute()
  print('Pipeline complete.')
  # print('Promote. local slave')
  # time.sleep(5)
  # print('demote master')
  # time.sleep(5)
  # print('Waiting...')
  print(client.incr('foo'))
  print(client.get('foo'))


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.add_argument('name')
#   parser.add_argument('--client', action='store_true')
#   parser.add_argument('--stop', action='store_true')
#   args = parser.parse_args()

#   if args.client:
#     testClient(args.name)
#     sys.exit(0)

#   settings = systemsettings()
#   settings.applyConfig('%s.json' % args.name)

#   server = RedisService(args.name)
#   if args.stop:
#     server.stop()
#   else:
#     server.start()




