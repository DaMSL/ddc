import redis
import socket
import sys
import os
import subprocess as proc
import abc

import common
from catalog import catalog

import logging
logger = common.setLogger()


class dataStore(catalog, redis.StrictRedis):
  def __init__(self, lockfile, host='localhost', port=6379, db=0):

    redis.StrictRedis.__init__(self)

    self.lockfile = lockfile
    self.host = host
    self.port = port
    self.database = db

    self.conn()


  def conn (self, host='localhost'):

    # # Check if it's already started and connected
    
    if self.exists():
      return

    # If already started by another node, get connection info
    if os.path.exists(self.lockfile):
      with open(self.lockfile, 'r') as connectFile:
        h, p, d = connectFile.read().split(',')
        self.host = h
        self.port = int(p)
        self.database = int(d)

      logger.debug("Service already running on " + h)

    # Otherwise, start it locally as a daemon server process
    else:
      self.start()

    # Connect to redis as client
    try:
      pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.database)
      self.connection_pool = pool        

      if not self.ping():
        logger.error("ERROR connecting to redis service on %s", self.host)
    except redis.ConnectionError as ex:
      logger.error("ERROR connecting to redis service on %s", self.host)



  def exists(self):
    try:
      alive = self.ping()
      return os.path.exists(self.lockfile) and self.ping()
    except redis.ConnectionError as ex:
      return False

  def clear(self):
    self.flushdb()


  def start(self):
    self.host = socket.gethostname()
    with open(self.lockfile, 'w') as connectFile:
      connectFile.write('%s,%d,%d' % (self.host, self.port, self.database))

    # Start redis via suprocess
    err = proc.call(['redis-server', 'redis.conf'])
    if err:
      logger.error("ERROR starting local redis service on %s", self.host)    
    logger.debug('Started redis locally on ' + self.host)

  # TODO: Graceful shutdown and hand off -- will need to notify all clients
  def stop(self):
    self.save()
    self.shutdown()
    if os.path.exists(self.lockfile):
      os.remove(self.lockfile)




  def save(self, data):

    pipe = self.pipeline()
    for key, value in data.items():
      if key == 'JCQueue':
        print (key, type(key), value, type(value))
      if isinstance(value, list):
        tp = 'LIST'
        pipe.delete(key)
        if len(value) > 0:
          if key == 'JCQueue':
            print ("     SETTING:", value, type(value), len(value))
          pipe.rpush(key, *(tuple(value)))

      elif isinstance(value, dict):
        print ("     SETTING a dict for :" + key)
        for k, v in value.items():
          print("    ", k, v)
        pipe.hmset(key, value)
        tp = 'DICT'
      else:
        pipe.set(key, value)
        tp = 'VAL'
      logger.debug("Saving data elm  `%s` of type %s, `%s`" % (key, tp, type(data[key])))

      # TODO:  handle other datatypes beside list

    pipe.execute()


  # Retrieve data stored for each key in data & store into data 
  def load(self, data):

    # Support single item data retrieval:
    keys = data.keys()
    pipe = self.pipeline()
    for key in keys:
      tp = ''
      if isinstance(data[key], list):
        if key == 'JCQueue':
            print ("     GETTING:", data[key], type(data[key]))
        pipe.lrange(key, 0, -1)
        tp = 'LIST'
      elif isinstance(data[key], dict):
        pipe.hgetall(key)
        tp = 'DICT'
      else:
        pipe.get(key)
        tp = 'VAL'
      logger.debug("Loading data elm  `%s` of type %s, `%s`" % (key, tp, type(data[key])))

      # TODO:  handle other datatypes beside list

    vals = pipe.execute()

    for i, key in enumerate(keys):
      logger.debug('Caching:  ' + key)
      if isinstance(data[key], list):
        data[key] = [val.decode() for val in vals[i]]
      elif isinstance(data[key], dict):
        data[key] = {k.decode():v.decode() for k,v in vals[i].items()}
      elif isinstance(data[key], int):
        data[key] = int(vals[i].decode())
      elif isinstance(data[key], float):
        data[key] = float(vals[i].decode())
      else:
        data[key] = vals[i].decode()


  # Slice off data in-place. Asssume key stores a list
  def slice(self, key, num):
    data = self.lrange(key, 0, num-1)
    self.ltrim(key, num-1, -1)
    return [d.decode() for d in data]

  # Check if key exists in db
  def check(self, key):
    if self.type(key).decode() == 'none':
      return False
    else:
      return True




  # TODO:  Additional notification logic, as needed
  def notify(self, key, state):
    if state == 'ready':
      self.set(key, state)

