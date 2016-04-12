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
from datetime import datetime as dt
import logging
import argparse
import sys
import subprocess as proc
import shlex
import json

import redis

from core.common import systemsettings, executecmd, executecmd_pid, getUID
from core.kvadt import kv2DArray, decodevalue
from core.slurm import slurm
from overlay.overlayService import OverlayService, OverlayServerStatus
from overlay.overlayException import OverlayNotAvailable

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


# EXPERIMENT_NUMBER = 1   # UNDERLAP (let it die)
EXPERIMENT_NUMBER = 2   # OVERLAP (let it die)





class RedisService(OverlayService):
  """
  """


  # REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'

  def __init__(self, name, port=6379, **kwargs):
    """
    """
    OverlayService.__init__(self, name, port, **kwargs)
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
    # if os.path.exists(self.lockfile):
    #   host, port, state = self.getconnection()
    #   self.shutdowncmd = 'redis-cli -h %s -p %s shutdown' % (host, port)

    # FOR IDLE REPORTING Using Command Count
    self.ping_count = 0
    self.last_cmd_count = 0
    self.last_cmd_ts = dt.now()
    self.total_idle_time = 0.
    self.current_idle_time = 0.

    # FOR IDLE REPORTING and Metric Recording
    self.persist = True
    self.idle_report = 0
    self.idle_actual = 0
    self.IDLE_REPORT_THETA = 60   # we will consider the service "idle" if it receives no request for 1 min

    # The Command monitor to track idle time
    self.cmd_mon = None
    self.cmd_mon_pid = None


    # External Logger to track metrics:
    logfile = os.path.join(os.environ['HOME'], 'ddc', 'results', 'redismon_%s.log'%name)
    self.monlog = logging.getLogger('redis_mon_%s'%name)
    self.monlog.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    self.monlog.addHandler(fh)
    self.monlog.propagate = False


    # For AUTO-HANDOVER OF SERVICE:

  def call_next(self, delay=.75):
    """Reschedules self for next job (if Slurm Job) with a delay
    percentage of current time. Delay is float between 0 and 1
    """

    if self.slurm_id is None:
      logging.warning("Skipping self-invoked Call Next (not a Slurm Job)")
    else:
      ts = int(dt.now().timestamp())
      total_time = int(self.ttl) - ts
      next_start_delay = round(delay * total_time) 
      logging.debug('TIMES')
      logging.debug('TTL %f', self.ttl)
      logging.debug('TS  %d', ts)
      logging.info('Redis Service will schedule next job to begin in %d seconds', next_start_delay)
      # for k, v in os.environ.items():
      #   print(k, ':     ', v)
      params = {}
      params['time']     = self.jobinfo['TimeLimit']
      params['exclude']  = self.jobinfo['NodeList']
      params['nodes']    = os.getenv('SLURM_JOB_NUM_NODES')
      params['cpus-per-task'] = os.getenv('SLURM_CPUS_PER_TASK')
      params['partition']= os.getenv('SLURM_JOB_PARTITION')
      params['job-name'] = os.getenv('SLURM_JOB_NAME')
      params['workdir']  = os.getcwd()
      params['begin'] = 'now+%d' % (next_start_delay)
      params['output'] = '/home-1/bring4@jhu.edu/ddc/osvc-redis-%%j.out'
      logging.debug('CALL NEXT for next Redis Server Handover:  %s', str(params))
      slurm.sbatch(taskid=params['job-name'],
              options = params,
              modules = set(['redis']),
              cmd = "src/overlay.py --name=%s redis start" % self._name_app)


  def ping(self, host='localhost', port=None):
    """Heartbeat to the local Redis Server
    """
    # TODO: Should wrap around try/catch and propagate an IO exception
    while True:
      if port is None:
        port = self._port
      ping_cmd = 'redis-cli -h %s -p %s ping' % (host, port)
      pong = executecmd(ping_cmd).strip()
      self.ping_count += 1
      if 'LOADING' in pong:
        time.sleep(1)
        continue
      break
    return (pong == 'PONG')



  def cmd_monitor(self):
    """ For running the proactive Redis Command Monitor
    Warning: this has a significant imact on performance (50%% reduction)
    """
    up = self.state_running.wait(self.SERVICE_STARTUP_DELAY)
    if not up:
      logging.warning('COMMAND Monitor never Started. Running state was never detected')
      return

    log = logging.getLogger('redis_mon')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler('redis_mon.log')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.propagate = False

    cmd = 'redis-cli monitor'
    task = proc.Popen(cmd, shell=True,
                                stdin=None,
                                stdout=proc.PIPE,
                                stderr=proc.STDOUT)

    # Poll process for new output until finished; buffer output
    # NOTE: Buffered output reduces message traffic via mesos
    BUFFER_SIZE = 4096
    last_ts = 0
    last = 0
    cmd_count = 0
    logbuffer = ''
    while not self.terminationFlag.is_set():
      output = task.stdout.readline()
      logbuffer += output.decode()
      if output == '' and task.poll() != None:
        break
      # If buffer-size is reached: send update message
      #  Where to put this info???
      if len(logbuffer) > BUFFER_SIZE:
        for line in logbuffer.split('\n'):
          elms = line.split()
          if len(elms) > 3:
            ts = int(elms[0].split('.')[0])
            cmd = elms[3].replace('"', '').upper()
            ignore = (cmd == 'PING')
            if ts == last:
              cmd_count += 1
            else:
              if not ignore:
                if cmd_count == 1:
                  log.info('%d %d', ts, cmd_count)
                else: 
                  log.info('%d %d', last, cmd_count-1)
              last = ts
              cmd_count = 1
        logbuffer = ""

    for line in logbuffer.split('\n'):
      elms = line.split()
      if len(elms) > 3:
        ts = int(elms[0].split('.')[0])
        cmd = elms[3].replace('"', '').upper()
        ignore = (cmd == 'PING')
        if ts == last:
          cmd_count += 1
        else:
          if not ignore:
            if cmd_count == 1:
              log.info('%d %d', ts, cmd_count)
            else: 
              log.info('%d %d', last, cmd_count-1)
          last = ts
          cmd_count = 1

    exitCode = task.returncode
    logging.info("Redis Monitor is Completed. Exit: %d", exitCode)      


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

    # Schedule My Successor (75% into the job):
    if EXPERIMENT_NUMBER == 2:
      self.call_next(.75)

    # FOR Active Monitoring:
    # self.cmd_mon = Thread(target=self.cmd_monitor)
    # self.cmd_mon.start()

    # LAUNCH THIS:
    #  redis monitor > redis_mon.log 2>&1 &

    # FOR Passive Idle Time tracking
    self.last_cmd_ts = dt.now()


  def idle(self):
    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      # self.connection.client_setname("monitor")

    server_reported_cmd_count = self.connection.info('stats')['total_commands_processed']
    self.last_cmd_count += (1 + self.ping_count)   # Cmd from above line and all recent pings
    ts = dt.now()

    delta_ts = (ts-self.last_cmd_ts).total_seconds()
    delta_cmd = server_reported_cmd_count - self.last_cmd_count

    if  delta_cmd == 0:
      self.total_idle_time += delta_ts
      self.current_idle_time += delta_ts
      logging.debug('IDLE: Server has been idle for %.1f sec', self.current_idle_time)
      self.monlog.info('%s,idle,%.1f', ts.strftime('%X'), delta_ts)
    else:
      logging.debug('CMDS: Server processed %d command(s) in the last %.1f secs', delta_cmd, delta_ts)
      self.monlog.info('%s,cmds,%d', ts.strftime('%X'), delta_cmd)
      self.current_idle_time = 0
      # TODO:  Track CMD processing rate and detect if the catalog is slowing down performance 
      #   Then, elastically grow the service on demand (or shrink # of servers)

    self.last_cmd_count = server_reported_cmd_count
    self.last_cmd_ts = ts
    self.ping_count = 0


    # TODO: NEXT IDLE DETECT & SHUTDOWN
    # return (self.current_idle_time > self.CATALOG_IDLE_THETA)
    return False

  def handover_to(self):
    """Invoked to handover services from this instance to a new master
    Called when a new slave is ready to receive as a new master
    """
    logging.info("[Overlay - %s] RECEIVIED Flag to Handover.", self._name_svc)
    if len(self.slavelist) == 0:
      logging.warning(" WARNING: Handover protocol started, but no slaved detected.")
    else:
      logging.info("Detected slave(s) from the following")
      for i in self.slavelist:
        logging.info('Slave detected starting up on %s. Current state = %s', i, self.slavelist[i])

    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")

    #  Wait until the slave connects to this master
    while not self.terminationFlag.is_set():
      info = self.connection.info()
      if info['connected_slaves'] == 0:
        logging.info("[Overlay - %s] Flagged to Handover. But no slaves are connected. Waiting for slave to come online.", self._name_svc)
        time.sleep(1)
      else:
        break

    while True:
      with open(self.lockfile, 'r') as conn:
        lines = conn.read().strip().split('\n')
      logging.info("[%s-Mon] Checking Lockfile shows %d entries", self._name_svc, len(lines))    
      if len(lines) > 1:
        for i in lines[1:]:
          logging.info('SLAVE detected: %s', i)
          data = i.split(',')
          self.slavelist[data[0]] = data[-1]
          logging.info("[%s-Mon]  Detected a slave on %s", self._name_svc, data[0])
      if len(self.slavelist) == 0:
        # TODO: Lost Slave while sync'ing -- revert back to Master here
        return None
      info = self.connection.info('replication')
      next_master = info['slave0']
      # logging.debug('INFO:')
      # for k,v in info.items():
      #   logging.debug('%12s  %s',k, v)
      sync_complete = False
      for k, v in self.slavelist.items():
        if v == 'SYNC':
          logging.info('Slave on %s is SYNC Complete', k)
          sync_complete = True
        else:
          logging.info('Slave on %s is still in state: %s', k, v)
      if sync_complete: 
        break
      time.sleep(0.5)

    # TODO: For multiple slave: iterate for each and find first one online.....
    info = self.connection.info()
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
      try:
        for client in self.connection.client_list():
          # Ignore other slave(s) 
          if client['name'] == 'monitor' or client['flags'] == 'S':
            continue
          if 'x' in client['flags'] or int(client['multi']) > 0:
            logging.debug('[Monitor - %s]  Found a client processing a pipeline. Waiting.', self._name_svc)
            active_clients = True
          # if int(client['idle']) < 3:
          #   logging.debug('[Monitor - %s]  Found a client idle for less than 3 second. Waiting to stop serving.', self._name_svc)
          #   active_clients = True
      except ConnectionError as e:
          logging.info('Monitor no longer tracking the Redis Servive. (Hence no active clients)')
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

    # Track size of DB (for metrics):
    meminfo = self.connection.info()['used_memory']
    self.stat.collect('MEMSIZE', meminfo)
    sync_start = dt.now()

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

    sync_time = (dt.now() - sync_start).total_seconds()
    self.stat.collect('REPLICA_SYNC', sync_time)
    # Flag master to go read only and then stop

    # FLAG MASTER
    with open(self.lockfile, 'a') as conn:
      ts = dt.now().timestamp()
      conn.write('%s,%d,%f,%f,%s\n' % (self._host, self._port, ts, self.ttl, 'SYNC'))

    # Replica prepared to begin as master. It will wait until notified by master to 
    #   Assume responsibility
    while True:
      try:
        info = self.connection.info('replication')
        logging.debug('REPLICATION INFO:')
        for k,v in info.items():
          logging.debug('%12s   %s', k, v)
        link_status = info['master_link_status']
        # last_link_time = info['master_link_down_since_seconds']
        if link_status != 'up':
          logging.debug("[Overlay - %s] Detected MASTER has gone READONLY.", self._name_svc)
          break
        logging.info('Current link status with master: %s.', link_status)
        time.sleep(1)
      except redis.RedisError as e:
        logging.info('Error connecting to Master (%s). Taking Control', e.__name__)
        break

    # Become the master
    logging.debug("[Overlay - %s] Assumed control as MASTER on %s.", self._name_svc, self._host)
    self.connection.slaveof()

  def tear_down(self):
    logging.info('IDLE_TOTAL,%.1f', self.total_idle_time)
    self.stat.collect('IDLE', self.total_idle_time)
    if self.cmd_mon_pid is not None:
      exectutecmd('kill -9 %d' % self.cmd_mon_pid)
    if self.cmd_mon is not None and self.cmd_mon.is_alive():
      logging.warning("WARNING. The contol Monitor thread was still alive (somehow)")
    logging.info("[Redis] Tear down complete.")

class RedisClient(redis.StrictRedis):

  def __init__(self, name):
    settings = systemsettings()
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, 'RedisService')
    self.isconnected = False
    self.host = None
    self.port = None
    self.pool = None

    print('LOCKFILE', self.lockfile)

    connect_wait = 0
    while True:
      try:
        with open(self.lockfile, 'r') as conn:
          conn_string = conn.read().split(',')
          if len(conn_string) < 5:
            raise OverlayNotAvailable
        host = conn_string[0]
        port = conn_string[1]
        ttl  = float(conn_string[3])
        state = conn_string[4]
        ts = dt.now()
        if ts.timestamp() > ttl:
          logging.warning("Overlay Service Master has expired. Using this at risk!")
        self.pool = redis.ConnectionPool(host=host, port=port, db=0, decode_responses=True)
        redis.StrictRedis.__init__(self, connection_pool=self.pool, decode_responses=True)
        logging.info('Connection is open... Checking to see if I can connect')
        self.host = host
        self.port = port
        # If set name suceeds. The Db is connected. O/W handle the error below accordingly
        self.client_setname(getUID())
        logging.info('[Redis Client] Connected as client to master at %s on port %s', host, port)
        self.isconnected = True
        break
      except FileNotFoundError:
        logging.warning('[Redis Client] No Lockfile Found. Service unavailable: %s', self.lockfile)
        raise OverlayNotAvailable
      except redis.ReadOnlyError as e:
        logging.warning('[Redis Client] Connecting as read only')
        self.isconnected = True
        break
      except redis.BusyLoadingError as e:
        logging.warning('[Redis Client] Current Master is starting up. Standing by.....')
        time.sleep(5)
        connect_wait += (dt.now()-start).total_seconds()
        continue
      except redis.RedisError:
        logging.warning('[Redis Client] Service is not running. Cannot get master from lockfile: %s', self.lockfile)
        raise OverlayNotAvailable


  def check_connection(self, timeout=30):
    """Blocks until the current connection is established
    or times out and return False
    """
    while timeout > 0:
      master = OverlayServerStatus(self.lockfile)
      if master.valid and master.avalable:
        try:
          if self.ping():
            return True
        except:
          logging.warning("Redis Client Connection Problem:  %s    (Will Timeout in %d secs)", sys.exc_info()[0])
          time.sleep(1)
          timeout -= 1
      return False

  def execute_command(self, *args, **options):
    """Execute a command and return a parsed response
    Catches connection errors and attempts to connect with correct master
    """
    initial_connect = True

    # For metrics:
    cmd_wait = 0.
    connect_wait = 0.
    while True:
      pool = self.connection_pool
      command_name = args[0]
      connection = pool.get_connection(command_name, **options)
      try:
          start = dt.now()
          connection.send_command(*args)
          cursor = self.parse_response(connection, command_name, **options)
          ts = (dt.now()-start).total_seconds()
          connection.disconnect()
          cmd_wait += ts
          if cmd_wait > 1:
            logging.debug("CLIENT_DELAY,CMD,%.1f",cmd_wait)
          if connect_wait > 1:
            logging.debug("CLIENT_DELAY,CONN,%.1f",connect_wait)
          return cursor
      except (ConnectionResetError, ConnectionAbortedError, redis.ReadOnlyError) as e:
        logging.warning('[Redis Client] Current Master is busy. It may be trying to shutdown. Wait and try again')
        time.sleep(5)
        connect_wait += (dt.now()-start).total_seconds()
        continue
      except (redis.BusyLoadingError) as e:
        logging.warning('[Redis Client] Current Master is starting up. Standing by.....')
        time.sleep(5)
        connect_wait += (dt.now()-start).total_seconds()
        continue
      except (redis.ConnectionError, redis.TimeoutError) as e:
        logging.warning('[Redis Client] Error connecting to %s', str(self.host))
        connect_wait += (dt.now()-start).total_seconds()
        start = dt.now()
      logging.info('[Redis Client] Rechecking lock')
      try:
        connection.disconnect()
        with open(self.lockfile) as master:
          config = master.read().split('\n')[0].split(',')
          if config[0] == self.host and config[1] == str(self.port):
            if initial_connect:
              initial_connect = False
              connect_wait += (dt.now()-start).total_seconds()
              continue
            else:
              logging.warning('[Redis Client] Lock exists, but cannot connect to the master on %s', str(self.host))
              logging.debug("CLIENT_DELAY,CONN,%.1f",connect_wait)
              time.sleep(5)
              connect_wait += (dt.now()-start).total_seconds()
              if connect_wait > 60:
                logging.warning('Timed Out waiting on the master.')
                raise OverlayNotAvailable
              else:
                continue
          self.host = config[0]
          self.port = config[1]
          self.connection_pool = redis.ConnectionPool(host=self.host, port=self.port, decode_responses=True)
          logging.info('[Redis Client] Changing over to new master on %s, port=%s', self.host, self.port)
          connect_wait += (dt.now()-start).total_seconds()
      except FileNotFoundError as ex:
        logging.error('[Client - Redis] ERROR. Service is not running.')
        logging.debug("CLIENT_DELAY,CONN,%.1f",connect_wait)
        break

  def execute(self, raise_on_error=True):
    "OVERRIDE The Pipeline execute function"
    stack = self.command_stack
    if not stack:
        return []
    if self.scripts:
        self.load_scripts()
    if self.transaction or self.explicit_transaction:
        execute = self._execute_transaction
    else:
        execute = self._execute_pipeline

    conn = self.connection
    if not conn:
        conn = self.connection_pool.get_connection('MULTI',
                                                   self.shard_hint)
        # assign to self.connection so reset() releases the connection
        # back to the pool after we're done
        self.connection = conn

    try:
        logging.debug('PIPELINE COMMAND:  Checking for connetion THEN executing')
        self.check_connection()
        return execute(conn, stack, raise_on_error)
    except (ConnectionError, TimeoutError) as e:
        conn.disconnect()
        if not conn.retry_on_timeout and isinstance(e, TimeoutError):
            raise
        # if we were watching a variable, the watch is no longer valid
        # since this connection has died. raise a WatchError, which
        # indicates the user should retry his transaction. If this is more
        # than a temporary failure, the WATCH that the user next issues
        # will fail, propegating the real ConnectionError
        if self.watching:
            raise WatchError("A ConnectionError occured on while watching "
                             "one or more keys")
        # otherwise, it's safe to retry since the transaction isn't
        # predicated on any state
        return execute(conn, stack, raise_on_error)
    finally:
        self.reset()

  def loadSchema(self):
    logging.debug("Loading system schema")
    self.schema = self.hgetall('META_schema')

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
    logging.debug('CATALOG APPEND: %s', str(data.keys()))
    deferredappend = []
    try:
      self.check_connection()
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
    except OverlayNotAvailable as e:
      logging.error('Redis Server is not available. Cannot run pipeline command')
      raise OverlayNotAvailable

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

  def lock_acquire(self, key, hold_time=30):
    """ Acquires a lock for the given key (concurrency control)
    """
    timeout = int(hold_time * 1.5)

    while True:
      lock = self.get(key + ':LOCK')
      if lock is None:
        logging.info('Acquiring Lock for %s', key)
        unique_key = getUID()
        self.set(key + ':LOCK', unique_key)
        self.expire(key + ':LOCK', hold_time)
        return unique_key   # Return unique key for this process
      timeout -= 1
      if timeout == 0:
        logging.warning('Timedout waiting to aqcuire lock on %s', key)
        break
      logging.info('Waiting to acquire Lock for %s.....', key)
      time.sleep(3)
    return None

  def lock_release(self, key, passcode):
    """ Releases a lock for the given key (concurrency control)
    """
    lock = self.get(key + ':LOCK')
    if lock == passcode:
      self.delete(key + ':LOCK')
      logging.info('Lock released for %s', key)
      return True
    else:
      logging.info('Wrong process tried to release lock on %s', key)
      return False




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




