import redis
import socket
import sys
import time
import os
import subprocess as proc
import abc

from common import *
from catalog import catalog
from threading import Thread, Event

# import logging
# logger = setLogger()
logger=logging.getLogger('__name__')

terminationFlag = Event()



def get2DKeys(key, X, Y):
  return ['key_%d_%d' % (x, y) for x in range(X) for y in range(Y)]



class dataStore(redis.StrictRedis, catalog):
  def __init__(self, name, host='localhost', port=6379, db=0, persist=True, connect=True):

    redis.StrictRedis.__init__(self, host=host, port=port)

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
      self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.database)
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
      self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.database)
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



      # TODO:  Implementation of list vs queue (append vs update)



  def loadSchema(self, keys):
    self.schema = self.hgetall('META_schema')

  def save(self, data):
    pipe = self.pipeline()
    for key, value in data.items():
      #  Lists are removed and updated enmasse
      if isinstance(value, list):
        pipe.delete(key)
        for val in value:
          pipe.rpush(key, val)
      elif isinstance(value, dict):
        pipe.hmset(key, value)
      else:
        pipe.set(key, value)
      logger.debug("  Saving data elm  `%s` of type `%s`" % (key, type(data[key])))

      # TODO:  handle other datatypes beside list

    pipe.execute()




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

  # Retrieve data stored for each key in data & store into data 
  def load(self, data):

    # Support single item data retrieval:
    keys = data.keys()
    pipe = self.pipeline()
    for key in keys:
      tp = ''
      if isinstance(data[key], list):
        pipe.lrange(key, 0, -1)
        tp = 'LIST'
      elif isinstance(data[key], dict):
        pipe.hgetall(key)
        tp = 'DICT'
      else:
        pipe.get(key)
        tp = 'VAL'
      # logger.debug("Loading data elm  `%s` of type %s, `%s`" % (key, tp, type(data[key])))


    vals = pipe.execute()

    #  Data Conversion
    # TODO:  MOVE TO JSON BASED RETRIEVAL

    for i, key in enumerate(keys):
      try:
        # logger.debug('Caching:  ' + key)
        if isinstance(data[key], list):
          tmp = [val.decode() for val in vals[i]]
          try:
            if len(tmp) == 0:
              data[key] = []
            elif tmp[0].isdigit():
              data[key] = [int(val) for val in tmp]
            else:
              data[key] = [float(val) for val in tmp]
          except ValueError as ex:
            data[key] = tmp
        elif isinstance(data[key], dict):
          # logging.debug("Hash Loader")
          for k,v in vals[i].items():
            try:
              subkey = k.decode()
              subval = v.decode()
              # logging.debug("   %s:  %s", k, (str(v)))
              if v.isdigit():
                data[key][subkey] = int(subval)
              else:
                data[key][subkey] = float(subval)
            except ValueError as ex:
              data[key][subkey] = subval
        elif isinstance(data[key], int):
          data[key] = int(vals[i].decode())
        elif isinstance(data[key], float):
          data[key] = float(vals[i].decode())
        elif vals[i] is None:
          data[key] = None
        else:
          data[key] = vals[i].decode()
      except (AttributeError, KeyError) as ex:
        logging.error("BAD KEY:  %s", key)
        logging.error("Trace:  %s", str(ex))
        sys.exit(0)

  # Slice off data in-place. Asssume key stores a list
  def slice(self, key, num):
    data = self.lrange(key, 0, num-1)
    self.ltrim(key, num, -1)
    return [d.decode() for d in data]

  # Remove specific items from a list
  #  Given: a list and a set of indices into that list
  def removeItems(self, key, itemlist):
    nullvalue = getUID()

    pipe = self.pipeline()
    for index in itemlist:
      pipe.lset(key, index, nullvalue)

    pipe.lrem(key, 0, nullvalue)
    pipe.execute()


  def append(self, key, itemlist):
    self.rpush(key, *tuple(itemlist))

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

