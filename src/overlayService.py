#!/usr/bin/env python
"""Cache Implemenation for the Data Driven Control Project

    Cache is designed to hold high dimensional points. An abstract
    class is provides for future implementation using differing
    cache and storage policies
"""
import abc
import os
import redis
import time
import numpy as np
from collections import deque
from threading import Thread, Event
import logging
import socket
logging.basicConfig(level=logging.DEBUG)

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.0.1"
__email__ = "bring4@jhu.edu"
__status__ = "Development"

from common import *
from contextlib import closing

def check_socket(host='127.0.0.1', port='23'):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            print ("Port is open")
            return True
        else:
            print ("Port is not open")
            return False




class OverlayService(object):
  """Overlay services provide an abstracted layer of control to 
  allow an underlying implemented service to operate within
  a timedelayed HPC environment
  (TODO:  abstract to other environments)
  """
  __metaclass__ = abc.ABCMeta

  # Delay (in sec) to wait 
  SERVICE_MONITOR_DELAY = 5
  SERVICE_HEARTBEAT_DELAY = 5
  SERVICE_STARTUP_DELAY = 5
  MONITORING_PORT = 32023


  def __init__(self, name, port, **kwargs):
    self._host = socket.gethostname()
    self._port = port
    self._name_svc = type(self).__name__
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, self._name_svc)

    self._launchcmd = None
    self._pid = None

    self._state = 'INIT'
    self._role  = 'NONE'

    # TODO: All services' nodes assume to run on same port (different hosts)
    #  This the master hold only the host name
    self.master = None

    self.terminationFlag = Event()
    self.handoverFlag = Event()

  @abc.abstractmethod
  def ping(self, host='localhost'):
    """Method to check if service is running on the given host
    """
    pass

  def prepare_service(self):
    """Pre-execution processing (e.g. config file / env creation)
    """
    pass

  def idle(self):
    """To define an idle detection method and gracefully shutdown if the service
    is idle. If undefined, idle returns False and the system will always assume
    that is it never idle.
    """
    return False


  @abc.abstractmethod
  def handover_to(self):
    """Invoked to handover services from this instance to a new master
    """
    pass

  @abc.abstractmethod
  def handover_from(self):
    """Invoked as a callback when another master service is handing over
    service control (i.e. master duties) to this instance
    """
    pass

  def start(self, as_replica=True):
    """
    """
    # Check to ensure lock is not already acquired
    while True:
      try:
        lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock, bytes('%s,%d' % (self._host, self._port), 'UTF-8'))
        os.close(lock)
        self._state = 'START'
        self._role  = 'MASTER'
        self.master = self._host
        break
      except FileExistsError as ex:
        logging.debug("[Overlay - %s] Lock File exists (someone else has acquired it)", self._name_svc)
        with open(self.lockfile, 'r') as conn:
          conn_string = conn.read().split(',')
          host = conn_string[0]
          port = conn_string[1]
          logging.debug("[Overlay - %s] Checking if service is available on %s, %s", self._name_svc, host, port)

          if self.ping(host, port):
            if as_replica:

              # If starting up as a replica -- assume immediate replication and control
              #  of service operation. This will initiate this node to start as a 
              #  replica and initate replication/handover protocol. Future improvements
              #  should allow for the system to detect the # of replica (for FT) and 
              #  flag handover accordingly
              self._role  = 'REPLICA'
              self.master = host
              self.handoverFlag.set()
              break
            else:
              logging.debug('[Overlay - %s] Service already running on %s. To start as slave, call handover_slave', self._name_svc, host)
              return False
          else:
            logging.warning('[Overlay - %s] Service is NOT running. Will attempt to recover and start locally.', self._name_svc)
            os.remove(self.lockfile)
            time.sleep(1)

    # if not as_replica:
    #   logging.debug('[Overlay - %s] To start as slave, call handover_slave', self._name_svc, host)
    #   return False

    self.prepare_service()

    if self.launchcmd is None:
      logging.error("[Overlay - %s] Launch Command not set. It needs to be defined.", self._name_svc)
      return False

    # TODO: Check subproc call here -- should this also be threaded in a python wrap call?
    err = proc.call(self.launchcmd)
    if err:
      logging.error("[Overlay - %s] ERROR starting local service on %s", self._name_svc, self.host)    
      # Exit or Return ???
      return False

    # Ensure service has started locally
    svc_up = False
    timeout = time.time() + OverlayService.SERVICE_STARTUP_DELAY
    while not svc_up:
      if time.time() > timeout:
        logging.error("[Overlay - %s] Timed Out waiting on the server", self._name_svc)    
        break
      time.sleep(1)
      svc_up = self.ping()

    if not svc_up:
      logging.error("[Overlay - %s] Service never started. You may need to retry.", self._name_svc)    
      return False

    self._state = 'RUNNING'

    logging.info("[Overlay - %s] Service started on %s. Starting the local monitor.", self._name_svc, self._host)    

    t = Thread(target=self.monitor)
    t.start()
    return True

  def monitor(self):

    logging.info("[Overlay - %s] Monitor Daemon Started.", self._name_svc)
    logging.debug('\n[Monitor - %s]  Initiated', self._name_svc)

    # Redundant check to ensure service has started locally (it should be started already)
    alive = False
    timeout = time.time() + OverlayService.SERVICE_STARTUP_DELAY
    while not alive:
      if time.time() > timeout:
        logger.error("[Monitor - %s] Timed Out waiting on the server", self._name_svc)    
        break
      time.sleep(1)
      alive = self.ping()
    if not alive:
      logger.error("[Monitor - %s] Service never started. You may need to retry.", self._name_svc)    
      return

    logger.info("[Monitor - %s] Service is up locally. Running the event handler loop.", self._name_svc)    
    #  Blocking loop. Will only exit if the term flag is set or by a 
    #   in-loop check every heartbeat (via miss ping, lost pid, or idle timeout)
    while not self.terminationFlag.wait(OverlayService.SERVICE_HEARTBEAT_DELAY):
      # TODO:  try/catch service connection errors here

      # Heartbeat
      heartbeat = self.ping()
      if heartbeat:
        logger.debug("[Monitor - %s]  Heartbeat on %s. My role = %s", self._name_svc, self._host, self._role)
      else:
        logger.warning("[Monitor - %s]  MISSED Heartbeat on %s. Cannot ping the service", self._name_svc, self._host)
        # TODO:  Force/Hard Shurdown
        break

      # CHECK PID HERE

      # CHECK IDLE TIME
      # if self.idle():
      #   logger.info("[Monitor - %s] Service is idle. Initiate graceful shutdown (TODO).", self._name_svc)    
      #   break

      with open(self.lockfile, 'r') as conn:
        lines = conn.read().split('\n')
        if len(lines) > 1:
          # TODO: Multiple slave
          print (lines, len(lines))
          data = lines[1].split(',')
          logging.info("[Monitor - %s]  Detected a new slave on %s", self._name_svc, data[1])
          if self._role == 'MASTER':
            self.handoverFlag.set()

      if self.handoverFlag.is_set():
        # This service has been flagged to handover control to a new master or from an old one
        if self._role == 'MASTER':
          # Handover Control
          next_master = self.handover_to()
          logging.info("[Monitor - %s]  Handover complete. I am now terminating.", self._name_svc)
          # Terminate
          self.terminationFlag.set()

        elif self._role == 'REPLICA':
          # Initiate replica protocol and assume role as master
          self.handover_from()
          logging.info("[Monitor - %s]  Handover complete. I am now the master.", self._name_svc)

          # Re-set lock file (TODO: other slaves)
          self._role = 'MASTER'
          os.remove(self.lockfile)
          lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
          os.write(lock, bytes('%s,%d' % (self._host, self._port), 'UTF-8'))
          os.close(lock)

          self.handoverFlag.clear()
          logging.debug("[Monitor - %s] Handover status is %s.", self._name_svc, (self.handoverFlag.is_set()))

        else:
          logger.error("[Monitor - %s]  Flagged for handover, but cannot (I am neither a master or a replica -- what am I?", self._name_svc)
          self.terminationFlag.set()

    self.stop()
  
    # Post-termination logic
    logging.info("[Overlay - %s] Service is shutdown on %s. Monitor is terminating", self._name_svc, self._host)

  @abc.abstractmethod
  def stop(self):
    """
    """

    shutdown_cmd = 'redis-cli -p %d shutdown' % (self._port)
    pong = executecmd(shutdown_cmd).strip()


class RedisService(OverlayService):
  """
  """

  REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'

  def __init__(self, name, port=6379):
    """
    """
    OverlayService.__init__(self, name, port)
    self.connection = None

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
    with open(RedisService.REDIS_CONF_TEMPLATE, 'r') as template:
      source = template.read()
      logging.info("Redis Source Template loaded")

    # params = dict(localdir=DEFAULT.WORKDIR, port=self._port, name=self._name)
    params = dict(localdir='.', port=self._port, name=self._name_app)

    # TODO: This should be local
    self.config = self._name_app + "_db.conf"
    with open(self.config, 'w') as config:
      config.write(source % params)
      logging.info("Data Store Config written to  %s", self.config)

    self.launchcmd = ['redis-server', self.config]

  def idle(self):

    CATALOG_IDLE_THETA = 500

    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")

    for client in self.connection.client_list():
      if client['name'] == 'monitor':
        continue
      if int(client['idle']) < CATALOG_IDLE_THETA:
        logger.debug('[Monitor - %s]  Service was idle for more than %d seconds. Stopping.', self._name_svc, CATALOG_IDLE_THETA)
        return True
    return False

  def handover_to(self):
    """Invoked to handover services from this instance to a new master
    Called when a new slave is ready to receive as a new master
    """
    if self.connection is None:
      self.connection = redis.StrictRedis(
          host='localhost', port=self._port, decode_responses=True)
      self.connection.client_setname("monitor")

    # Check to ensure no clients actively using to the DB  -- hold on this
    # while True:
    #   all_idle = True
    #   for client in self.connection.client_list():
    #     if client['name'] == 'monitor':
    #       continue
    #     if int(client['idle']) == 0:
    #       logger.debug('[Overlay - %s]  A Client is connected. Waiting and rechecking before stopping.', self._name_svc)
    #       timmer.sleep(2)
    #       all_idle = False
    #       continue
    #   if all_idle:
    #     break

    info = self.connection.info()
    if info['connected_slaves'] == 0:
      logging.error("[Overlay - %s] ERROR. Flagged to Handover. But no slaves are connected.", self._name_svc)
      return

    # TODO: For multiple slave: iterate for each and find first one online.....
    next_master = info['slave0']
    if next_master['state'] != 'online':
      logging.error("[Overlay - %s] ERROR. Slave tabbed for next master `%s` is OFFLINE", self._name_svc, next_master['ip'])
      return

    # Become slave of next master (effectively goes READONLY on both ends)
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


  def stop(self):
    """
    """
    logging.info('Redis service is shutting down......')
    shutdown_cmd = 'redis-cli -p %d shutdown' % (self._port)
    executecmd(shutdown_cmd)

    while self.ping():
      logging.debug('Redis is still alive on %d', self._host)
      time.sleep(1)

    logging.info('Redis shutdown complete')


def testOverlay():
  serverA = RedisService('testoverlay')
  serverA.start()



if __name__ == "__main__":
  testOverlay()





    # ping_interval = OverlayService.SERVICE_HEARTBEAT_DELAY
    # timeout = time.time() + ping_interval
    # while not alive:
    #   if time.time() > timeout:
    #     logger.warning('\n[Monitor - %s]  Timeout waiting on the service', self._name_svc)
    #     break
    #   alive = self.ping()
    #   if alive
    #     logger.debug("[Monitor - %s]  Heartbeat local on %s", self._name_svc, self._host)
    #     # Default to 2 * heartbeat delay () (timeout is effectively two missed heartbeats)
    #     timeout = time.time() + Overlay.SERVICE_HEARTBEAT_DELAY * 2
    #     ping_interval = OverlayService.SERVICE_HEARTBEAT_DELAY
    #   else:
    #     # Ping every second unti connection is re-established or a timeout
    #     logger.debug("[Monitor - %s]  Heartbeat local on %s", self._name_svc, self._host)
    #     ping_interval = 1
    #   time.sleep(ping_interval)

    # if not alive:
    #   logger.warning("[Monitor - %s]  MISSED Heartbeat on %s. Cannot ping the service", self._name_svc, self._host)
    #   return