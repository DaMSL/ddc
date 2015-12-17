import redis
import socket
import sys
import time
import os
import subprocess as proc
import abc
import json

from common import *
from catalog import catalog
from threading import Thread, Event
from kvadt import kv2DArray

# import logging
# logger = setLogger()
logger=logging.getLogger('__name__')

terminationFlag = Event()


class DType:

  types = dict(int = int,
    float = float,
    num = float,
    list = list,
    dict = dict,
    str = str,
    ndarray = np.ndarray,
    matrix = kv2DArray)

  @classmethod
  def get(cls, t):
    if t not in cls.types:
      logging.error("Type `%s` not a valid type") 
      return None

    return cls.types[t]


  @classmethod
  def cmp(cls, this, other):
    return this.__name__ == other.__name__






def get2DKeys(key, X, Y):
  return ['key_%d_%d' % (x, y) for x in range(X) for y in range(Y)]


def infervalue(value):
  try:
    castedval = None
    if value.isdigit():
      castedval = int(value)
    else:
      castedval = float(value)
  except ValueError as ex:
    castedval = value
  return castedval


def decodevalue(value):
  data = None
  if isinstance(value, list):
    try:
      if len(value) == 0:
        data = []
      elif value[0].isdigit():
        data = [int(val) for val in value]
      else:
        data = [float(val) for val in value]
    except ValueError as ex:
      data = value
  elif isinstance(value, dict):
    # logging.debug("Hash Loader")
    data = {}
    for k,v in value.items():
      data[k] = infervalue(v)
  elif value is None:
    data = None
  else:
    data = infervalue(value)
  return data


class dataStore(redis.StrictRedis, catalog):
  def __init__(self, name, host='localhost', port=6379, db=0, persist=True, connect=True):

    redis.StrictRedis.__init__(self, host=host, port=port, decode_responses=True)

    self.lockfile = name + '.lock'
    self.config = name + '_db.conf'
    self.host = host
    self.port = int(port)
    self.database = int(db)
    self.name = name
    self.persist = persist

    self.terminationFlag = Event()

    if connect:
      self.conn()



  def conn (self, host='localhost'):

    # Set up service object thread to return, in case this client dual-acts as a service
    serviceThread = None

    # # Check if already connected to service
    if self.exists():
      logging.debug('Data Store, `%s` already connected on `%s`', self.name, self.host)
      return serviceThread

      # If already started by another node, get connection info
    try:
      with open(self.lockfile, 'r') as connectFile:
        h, p, d = connectFile.read().split(',')
        self.host = h
        self.port = int(p)
        self.database = int(d)
        logging.debug('Data Store Lock File, `%s` DETECTED on %s, port=%s', self.name, self.host, self.port)

      # Check connection string -- F/T in case connection dies, using loaded params & self as hostname
      self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.database, decode_responses=True)
      if self.exists():
        logging.debug('Data Store, `%s` ALIVE on %s, port=%s', self.name, self.host, self.port)
        serviceAlive = True
        return serviceThread
      else:
        logger.warning("WARNING: Redis Server locked, but not running. Removing file and running it locally")
        os.remove(self.lockfile)
    except FileNotFoundError as ex:
      logging.debug("No Lock file found. Assuming Data Store is not alive")

    # Otherwise, start it locally as a daemon server process
    serviceThread = self.start()

    # Connect to redis as client
    try:
      logging.debug("\nSetting new connection_pool for client: %s, %d, %d", self.host, self.port, self.database)
      self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.database, decode_responses=True)
      logging.debug("     ...Connection Updated... Trying to Ping!")

      if self.ping():
        logging.debug(" I can ping the server. It's up")
      else:
        logger.error("FAILED to connect to redis service on %s", self.host)
    except redis.ConnectionError as ex:
      logger.error("ERROR Raised connecting to redis service on %s", self.host)
      return None

    return serviceThread

  def exists(self):
    try:
      alive = self.ping()
      return os.path.exists(self.lockfile) and self.ping()
    except redis.ConnectionError as ex:
      return False

  def clear(self):
    self.flushdb()

  def redisServerMonitor(self, termEvent):

    # Start redis via suprocess  -- Threaded
    logger.debug('\n[Catalog Monitor]  Initiated')

    #  Connect a monitoring client (for now: check idle status)
    # TODO: here is where we can work migration support and monitoring for other things
    monitor = redis.StrictRedis()

    alive = False
    timeout = time.time() + DEFAULT.CATALOG_STARTUP_DELAY
    while not alive:
      if time.time() > timeout:
        logger.debug("[Catalog Monitor]  Timed Out waiting on the server")
        break
      try:
        alive = monitor.ping()
        logger.debug("[Catalog Monitor]  Redis Server is ALIVE locally on %s", str(socket.gethostname()))
      except redis.ConnectionError as ex:
        alive = False
        time.sleep(1)

    if not alive:
      logger.debug("[Catalog Monitor]  Redis Server Failed to Start/Connect. Exitting the monitor thread")
      return

    monitor.client_setname("monitor")

    while not termEvent.wait(DEFAULT.MONITOR_WAIT_DELAY):
      try:
        logger.debug('[Catalog Monitor]  On')
        idle = True

        for client in monitor.client_list():
          if client['name'] == 'monitor':
            continue
          if int(client['idle']) < DEFAULT.CATALOG_IDLE_THETA:
            idle = False
            break
        if idle:
          logger.debug('[Catalog Monitor]  Service was idle for more than %d seconds. Stopping.', DEFAULT.CATALOG_IDLE_THETA)
          self.stop()
  
      except redis.ConnectionError as ex:
        termEvent.set()
        logger.debug('[Catalog Monitor]  Connection error to the server. Terminating monitor')
  

    # Post-termination logic
    logger.debug('[Catalog Monitor]  Redis Service was shutdown. Exiting monitor thread.')

  def start(self):

    # Prepare 
    with open(DEFAULT.REDIS_CONF_TEMPLATE, 'r') as template:
      source = template.read()
      logging.info("SOURCE LOADED:")

    params = dict(localdir=DEFAULT.WORKDIR, port=self.port, name=self.name)

    self.host = socket.gethostname()

    # Write lock file for persistent server; otherwise, write to local configfile
    if self.persist:

      # Check to ensure lock is not already acquired
      try:
        lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock, bytes('%s,%d,%d' % (self.host, self.port, self.database), 'UTF-8'))
        os.close(lock)
      except FileExistsError as ex:
        logging.debug("Lock File exists (someone else has acquired it). Backing off 10 seconds and connecting....")
        time.sleep(10)
        with open(self.lockfile, 'r') as connectFile:
          h, p, d = connectFile.read().split(',')
          self.host = h
          self.port = int(p)
          self.database = int(d)
          logging.debug('Data Store Lock File, `%s` DETECTED on %s, port=%s', self.name, self.host, self.port)
        return None


    else:
      self.config = self.name + '-' + self.host + ".conf"

    with open(self.config, 'w') as config:
      config.write(source % params)
      logging.info("Data Store Config `%s` written to  %s", self.name, self.config)

    err = proc.call(['redis-server', self.config])
    if err:
      logger.error("ERROR starting local redis service on %s", self.host)    
      # Exit or Return ???
      sys.exit(0)

    pong = False
    timeout = time.time() + DEFAULT.CATALOG_STARTUP_DELAY
    ping_cmd = 'redis-cli -p %d ping' % self.port
    logger.debug("PING CMD: %s", ping_cmd)

    while not pong:
      if time.time() > timeout:
        logger.debug("[Catalog Monitor]  Timed Out waiting on the server")
        break

      time.sleep(1)
      check = executecmd(ping_cmd).strip()
      logger.debug("Ping from local redis server:  %s", check)
      pong = (check == 'PONG')


    logger.debug('Started redis locally on ' + self.host)


    logging.debug("Starting the Redis Server Monitor Thread")
    t = Thread(target=self.redisServerMonitor, args=(self.terminationFlag,))
    t.start()

    logging.debug("Monitor Daemon Started.")

    return t



  # TODO: Graceful shutdown and hand off -- will need to notify all clients
  def stop(self):
    # self.save()
    self.terminationFlag.set()
    self.shutdown()
    if os.path.exists(self.lockfile):
      os.remove(self.lockfile)


  def loadSchema(self):
    logging.debug("Loading system schema")

    self.schema = self.hgetall('META_schema')
    for k, v in self.schema.items(): print("  ",k, v)


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
      logger.debug("  Saving data elm  `%s` as %s ", key, type(data[key]))

    pipe.execute()

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
        logging.debug("Dynamically loaded %s. Updating schema", key)
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
        logging.error("BAD KEY:  %s", key)
        logging.error("Trace:  %s", str(ex))
        sys.exit(0)

    return data


  def append(self, data):
    print('CATALOG APPEND')
    deferredappend = []
    pipe = self.pipeline()
    for key, value in data.items():
      logger.debug("Appending data elm  `%s` of type, %s", key, type(data[key]))
      if key not in self.schema.keys():
        logging.warning("  KEY `%s` not found in local schema! Will try Dynamic Append")
        deferredsave.append(key)
      elif self.schema[key] == 'int':
        pipe.incr(key, value)
        logging.warning("  Increment `%s` as int by %d", key, value)
      elif self.schema[key] == 'float':
        pipe.incrbyfloat(key, value)
        logging.warning("  Increment `%s` as float by %d", key, value)
      elif self.schema[key] == 'list':
        for val in value:
          pipe.rpush(key, val)
        logging.warning("  Pushing onto list `%s`:  %s", key, str(value))
      elif self.schema[key] == 'dict':
        for k, v in value:
          pipe.hset(key, k, v)
        logging.warning("  Updaing hash `%s`:  %s", key, str(value.keys()))
      elif self.schema[key] in ['matrix', 'ndarray']:
        deferredappend.append(key)
      else:
        pipe.set(key, value)

    pipe.execute()

    for key in deferredappend:
      if self.schema[key] == 'ndarray':
        logging.warning(" NUMPY ARRAY MERGING NOT SUPPORTED")
        # self.storeNPArray(data[key])
      elif self.schema[key] == 'matrix':
        logging.warning("  Merging matrix `%s`:  %s", key, str(data[key]))
        matrix = kv2DArray(self, key)
        matrix.merge(data[key])


  # Slice off data in-place. Asssume key stores a list
  def slice(self, key, num):
    data = self.lrange(key, 0, num-1)
    self.ltrim(key, num, -1)
    return data

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
    print('NP KEY = ', key)
    elm = self.hgetall(key)
    if elm == {}:
      return None
    header = json.loads(elm['header'])
    arr = np.fromstring(elm['data'], dtype=header['dtype'])
    return arr.reshape(header['shape'])

  # TODO:  Additional notification logic, as needed
  def notify(self, key, state):
    if state == 'ready':
      self.set(key, state)





  # def append(self, data):
  #   """
  #   Append Only updates to the catalog data
  #   """
  #   pipe = self.pipeline()
  #   for key, value in data.items():
  #     #  Lists are removed and updated enmasse
  #     if isinstance(value, list):
  #       for val in value:
  #         pipe.rpush(key, val)
  #     elif isinstance(value, dict):
  #       pipe.hmset(key, value)
  #     elif isinstance(value, int):
  #       pipe.incr(key, value)
  #     elif isinstance(value, float):
  #       pipe.incrbyfloat(key, value)      
  #   logger.debug("  Saving data elm  `%s` of type `%s`" % (key, type(data[key])))

  #     # TODO:  handle other datatypes beside list

  #   pipe.execute()



  # # Retrieve data stored for each key in data & store into data 
  # def load(self, keys):

  #   defer = []
  #   # Support single item data retrieval:
  #   pipe = self.pipeline()
  #   for key in keys:
  #     if key not in self.schema:
  #       logging.error('KEY ERROR. %s not found in Data Store Schema', key)
  #     elif self.schema[key] == 'list':
  #       pipe.lrange(key, 0, -1)
  #     elif self.schema[key] == 'dict':
  #       pipe.hgetall(key)
  #     elif self.schema[key] == 'kv2DArray':
  #       defer.append(key)
  #     else:
  #       pipe.get(key)

  #   vals = pipe.execute()

  #   #  Data Conversion

  #   for i, key in enumerate(keys):
  #     try:
  #       if self.schema[key] == 'list':
  #         tmp = [val.decode() for val in vals[i]]
  #         try:
  #           if len(tmp) == 0:
  #             data[key] = []
  #           elif tmp[0].isdigit():
  #             data[key] = [int(val) for val in tmp]
  #           else:
  #             data[key] = [float(val) for val in tmp]
  #         except ValueError as ex:
  #           data[key] = tmp
  #       elif self.schema[key] == 'dict':
  #         for k,v in vals[i].items():
  #           try:
  #             subkey = k.decode()
  #             subval = v.decode()
  #             # logging.debug("   %s:  %s", k, (str(v)))
  #             if v.isdigit():
  #               data[key][subkey] = int(subval)
  #             else:
  #               data[key][subkey] = float(subval)
  #           except ValueError as ex:
  #             data[key][subkey] = subval
  #       elif self.schema[key] == 'kv2DArray':
          
  #       else:
  #         pipe.get(key)


  #       if isinstance(data[key], list):
  #       elif isinstance(data[key], dict):
  #         # logging.debug("Hash Loader")
  #       elif isinstance(data[key], int):
  #         data[key] = int(vals[i].decode())
  #       elif isinstance(data[key], float):
  #         data[key] = float(vals[i].decode())
  #       elif vals[i] is None:
  #         data[key] = None
  #       else:
  #         data[key] = vals[i].decode()
  #     except (AttributeError, KeyError) as ex:
  #       logging.error("BAD KEY:  %s", key)
  #       logging.error("Trace:  %s", str(ex))
  #       sys.exit(0)




    # def load(self, keyslisdt):

    # # Support single item data retrieval:
    # deferredload = []
    # keys = data.keys()
    # pipe = self.pipeline()
    # for key in keys:
    #   tp = ''
    #   if isinstance(data[key], list):
    #     pipe.lrange(key, 0, -1)
    #     tp = 'LIST'
    #   elif isinstance(data[key], dict):
    #     pipe.hgetall(key)
    #     tp = 'DICT'
    #   elif isinstance(data[key], np.ndarray):
    #     deferredload.append(key)
    #   else:
    #     pipe.get(key)
    #     tp = 'VAL'
    #   # logger.debug("Loading data elm  `%s` of type %s, `%s`" % (key, tp, type(data[key])))


    # vals = pipe.execute()

    # for key in deferredload:
    #   if isinstance(data[key], np.ndarray):
    #     logging.debug("Loading NDArray, %s", key)
    #     data[key] = self.loadNPArray(key)

    # #  Data Conversion
    # # TODO:  MOVE TO JSON BASED RETRIEVAL

    # i = 0
    # for key in keys:
    #   try:
    #     # logger.debug('Caching:  ' + key)
    #     if key in deferredload:
    #       continue
    #     if isinstance(data[key], list):
    #       tmp = [val.decode() for val in vals[i]]
    #       try:
    #         if len(tmp) == 0:
    #           data[key] = []
    #         elif tmp[0].isdigit():
    #           data[key] = [int(val) for val in tmp]
    #         else:
    #           data[key] = [float(val) for val in tmp]
    #       except ValueError as ex:
    #         data[key] = tmp
    #     elif isinstance(data[key], dict):
    #       # logging.debug("Hash Loader")
    #       for k,v in vals[i].items():
    #         try:
    #           subkey = k.decode()
    #           subval = v.decode()
    #           # logging.debug("   %s:  %s", k, (str(v)))
    #           if v.isdigit():
    #             data[key][subkey] = int(subval)
    #           else:
    #             data[key][subkey] = float(subval)
    #         except ValueError as ex:
    #           data[key][subkey] = subval
    #     elif isinstance(data[key], int):
    #       data[key] = int(vals[i].decode())
    #     elif isinstance(data[key], float):
    #       data[key] = float(vals[i].decode())
    #     elif vals[i] is None:
    #       data[key] = None
    #     else:
    #       data[key] = vals[i].decode()
    #     i += 1
    #   except (AttributeError, KeyError) as ex:
    #     logging.error("BAD KEY:  %s", key)
    #     logging.error("Trace:  %s", str(ex))
    #     sys.exit(0)