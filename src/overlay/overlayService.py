#!/usr/bin/env python
"""Cache Implemenation for the Data Driven Control Project

    Cache is designed to hold high dimensional points. An abstract
    class is provides for future implementation using differing
    cache and storage policies
"""
import abc
import os
import time
from threading import Thread, Event
import logging
import socket
import argparse
import sys
import subprocess as proc
import shlex
import signal

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

# Function to derive signal name from the signum
#  (This becomes obsolete in Python 3.5 when they are enums)
siglist = {eval('signal.%s' % s): s \
   for s in [i for i in dir(signal) \
   if i.startswith('SIG') and '_' not in i]}


class OverlayService(object):
  """Overlay services provide an abstracted layer of control to 
  allow an underlying implemented service to operate within
  a timedelayed HPC environment
  (TODO:  abstract to other environments)
  """
  __metaclass__ = abc.ABCMeta

  # Delay (in sec) to wait 
  # SERVICE_HEARTBEAT_DELAY = 5
  # SERVICE_STARTUP_DELAY = 5

  def __init__(self, name, port, **kwargs):
    self._host = socket.gethostname()
    self._port = port
    self._name_svc = type(self).__name__
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, self._name_svc)

    self.numslaves = kwargs.get('numslaves', 0)
    self.launchcmd = None
    self.shutdowncmd = None

    self._pid = None

    self._state = 'INIT'
    self._role  = kwargs.get('role', 'NONE')

    # TODO: All services' nodes assume to run on same port (different hosts)
    #  This the master hold only the host name
    self.master = kwargs.get('master', None)
    self.slavelist = {}

    self.terminationFlag = Event()
    self.handoverFlag = Event()
    self.shuttingdown = Event()

    # TODO: Should we require basic setting via systemsetting object???
    #   and thus, enforce this object on the abstract class?
    self.SERVICE_STARTUP_DELAY = 10
    self.SERVICE_HEARTBEAT_DELAY = 15

    # Register tty signaling to gracefully shutdown when halted externall7
    #   (e.g. from scancel)
    signal.signal(signal.SIGTERM, self.halt)
    signal.signal(signal.SIGINT, self.halt)
    signal.signal(signal.SIGQUIT, self.halt)
    signal.signal(signal.SIGHUP, self.halt)

  @abc.abstractmethod
  def ping(self, host='localhost'):
    """Method to check if service is running on the given host
    """
    pass

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

  def prepare_service(self):
    """Pre-execution processing (e.g. config file / env creation)
    """
    pass

  def tear_down(self):
    """Post-execution processing (e.g. remove temp files)
    """
    pass

  def idle(self):
    """To define an idle detection method and gracefully shutdown if the service
    is idle. If undefined, idle returns False and the system will always assume
    that is it never idle.
    """
    return False

  def launch_slave(self):
    """For multi-node services, the master node is responsible for starting/stopping
    all slave nodes. Launch slave is the method called when the master monitor
    needs to start up a new slave.
    """

  def stop_slave(self):
    """For multi-node services, the master node will invoke this to stop a 
    slave. 
    TODO: policy development for rotating slave nodes and ensuring they 
    are either not all stopping concurrently or do exactly that.
    """

  def start(self, as_replica=True):
    """
    """
    if self._role == 'SLAVE':
      logging.warning('[%s] Starting as a slave. ')
      slave_wait_ttl = self.SERVICE_STARTUP_DELAY * 4
      while slave_wait_ttl > 0:
        host, port = self.getconnection()
        if host is not None and self.ping(host):
          break
        logging.warning('[%s] My Master is Not alive. Host is %s.  Waiting...', self._name_svc, str(host))
        sleep(self.SERVICE_STARTUP_DELAY)
        slave_wait_ttl -= self.SERVICE_STARTUP_DELAY
        if slave_wait_ttl < 0: 
          logging.warning('[%s] Could not find a master. Slave is shutting down on %s.  Waiting...', self._name_svc, self._host)
          return False
      # Set the master 
      self.master = host
      #TODO: 
      # self.masterport = port

    # Check to ensure lock is not already acquired
    while self.master is None:
      try:
        lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock, bytes('%s,%d' % (self._host, self._port), 'UTF-8'))
        os.close(lock)
        logging.debug("[%s] Lock File not found (I am creating it from %s)", self._name_svc, self._host)
        self._state = 'START'
        self._role  = 'MASTER'
        self.master = self._host
        break
      except FileExistsError as ex:
        logging.debug("[%s] Lock File exists (someone else has acquired it)", self._name_svc)
        host, port = self.getconnection()
        logging.debug("[%s] Checking if service is available on %s, %s", self._name_svc, host, port)
        if self.ping(host, port):
          logging.debug("[%s] Running on %s", self._name_svc, host)
          if as_replica:
            logging.debug("[%s] Starting in replica -- will initiate master takeover control protocol", self._name_svc)
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
            logging.debug('[%s] Service already running on %s. To start as slave, call handover_slave', self._name_svc, host)
            return False
        else:
          logging.warning('[%s] Service is NOT running. Will attempt to recover and start locally.', self._name_svc)
          os.remove(self.lockfile)
          time.sleep(1)

    self.prepare_service()

    if self.launchcmd is None:
      logging.error("[%s] Launch Command not set. It needs to be defined.", self._name_svc)
      return False

    if self.shutdowncmd is None:
      logging.error("[%s] Shutdown Command not set. It needs to be defined.", self._name_svc)
      return False

    # TODO: Check subproc call here -- should this also be threaded in a python wrap call?
    logging.info("[%s] Launching following command %s", self._name_svc, self.launchcmd)    
    args = shlex.split(self.launchcmd)
    service = proc.Popen(args)

    if service.returncode is not None:
      logging.error("[%s] ERROR starting local service on %s", self._name_svc, self.host)    
      # Exit or Return ???
      return False

    # Ensure service has started locally
    svc_up = False
    timeout = time.time() + self.SERVICE_STARTUP_DELAY
    while not svc_up:
      if time.time() > timeout:
        logging.error("[%s] Timed Out waiting on the server", self._name_svc)    
        break
      time.sleep(1)
      svc_up = self.ping()

    if not svc_up:
      logging.error("[%s] Service never started. You may need to retry.", self._name_svc)    
      return False

    self._state = 'RUNNING'
    logging.info("[%s] My Service started local on %s. Starting the local monitor.", self._name_svc, self._host)    

    self.service = service

    t = Thread(target=self.monitor)
    t.start()
    return t

  def monitor(self):

    logging.info("[%s] Monitor Daemon Started.", self._name_svc)
    logging.debug('\n[%s-Mon]  Initiated', self._name_svc)

    # TODO: Move to settings
    missed_hb = 0
    allowed_misshb = 3

    # Redundant check to ensure service has started locally (it should be started already)
    alive = False
    timeout = time.time() + self.SERVICE_STARTUP_DELAY
    while not alive:
      if time.time() > timeout:
        logging.error("[%s-Mon] Timed Out waiting on the server", self._name_svc)    
        break
      time.sleep(1)
      alive = self.ping()
    if not alive:
      logging.error("[%s-Mon] Service never started. You may need to retry.", self._name_svc)    
      return

    # For multi-node services, the Master monitor process will start up all slaves
    #  as individual jobs
    self._state = 'SINGLE'
    if self.numslaves > 0:
      logging.info("[%s-Mon] Service is up locally on %s. Preparing to launch %d slaves.", self._name_svc, self.numslaves)    

      # TODO: Thread each slave with a separate monitor or launcher 
      for i in len(range(self.numslaves)):
        self.launch_slave()

    #  Blocking loop. Will only exit if the term flag is set or by a 
    #   in-loop check every heartbeat (via miss ping, lost pid, or idle timeout)
    logging.info("[%s-Mon] Service is Operational and READY. Running the event handler loop.", self._name_svc)    
    self._state = 'READY'
    while not self.terminationFlag.wait(self.SERVICE_HEARTBEAT_DELAY):

      if self.shuttingdown.is_set():
        logging.warning("[%s-Mon] Detectd Service was flagged to shutdown. Monitor is halting.", self._name_svc)
        break

      # TODO:  try/catch service connection errors here
      if self.service.poll():
        logging.warning("[%s-Mon] Local Service has STOPPED on %s.", self._name_svc, self._host)
        break

      # Heartbeat
      if not self.ping():
        missed_hb += 1
        logging.warning("[%s-Mon]  MISSED Heartbeat on %s. Lost communicate with service for %d seconds.", 
              self._name_svc, self._host, missed_hb * self.SERVICE_HEARTBEAT_DELAY)
        continue

      missed_hb = 0

      # CHECK IDLE TIME
      # if self.idle():
      #   logging.info("[%s-Mon] Service is idle. Initiate graceful shutdown (TODO).", self._name_svc)    
      #   break

      if self._role == 'MASTER':
        with open(self.lockfile, 'r') as conn:
          lines = conn.read().split('\n')
        if len(lines) > 1:
          # TODO: Multiple slave
          print (lines, len(lines))
          data = lines[1].split(',')
          logging.info("[%s-Mon]  Detected a new slave on %s", self._name_svc, data[1])
          self.handoverFlag.set()

      if self.handoverFlag.is_set():
        # This service has been flagged to handover control to a new master or from an old one
        if self._role == 'MASTER':
          # Handover Control
          next_master = self.handover_to()
          self._role = 'REPLICA'
          logging.info("[%s-Mon]  Handover complete. I am now terminating. My role is now %s.", self._name_svc, self._role)
          # Terminate
          self.terminationFlag.set()

        elif self._role == 'REPLICA':
          # Initiate replica protocol and assume role as master
          self.handover_from()
          logging.info("[%s-Mon]  Handover complete. I am now the master.", self._name_svc)

          # Re-set lock file (TODO: other slaves)
          self._role = 'MASTER'
          os.remove(self.lockfile)
          lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
          os.write(lock, bytes('%s,%d' % (self._host, self._port), 'UTF-8'))
          os.close(lock)

          self.handoverFlag.clear()
          logging.debug("[%s-Mon] Handover status is %s.", self._name_svc, (self.handoverFlag.is_set()))

        else:
          logging.error("[%s-Mon]  Flagged for handover, but cannot (I am neither a master or a replica -- what am I?", self._name_svc)
          self.terminationFlag.set()

    self.stop()
  
    # Post-termination logic
    logging.info("[%s] Service is shutdown on %s. Monitor is terminating", self._name_svc, self._host)

  def stop(self):
    """
    """
    if not self.shuttingdown.is_set():
      self.shuttingdown.set()
      logging.debug('MY ROLE = %s', self._role)
      if self._role == 'MASTER':
        logging.info('[%s] Last master shutdown. Removing lockfile', self._name_svc)
        host, port = self.getconnection()
        if host == self._host:
          os.remove(self.lockfile)
        else:
         logging.info('[%s] This master is shutting down, but detected someone else.', self._name_svc)

      logging.info('[%s] Service shutting down on %s.', self._name_svc, self._host)
      args = shlex.split(self.shutdowncmd)
      result = proc.call(args)
      logging.info("[%s] shutdown with return code: %s.  Tearing Down remaining work.", self._name_svc, str(result))

      self.tear_down()


  def halt(self, signum, frame):

    logging.warning('[%s] SIGNAL Received: %s', self._name_svc, siglist[signum])
    logging.warning('[%s] Setting Termination Flag and Gracefully stopping the service',self._name_svc)
    self.terminationFlag.set()
    self.stop()

  def getconnection(self):
    try:
      with open(self.lockfile, 'r') as conn:
        conn_string = conn.read().split(',')
        host = conn_string[0]
        port = conn_string[1]
      return host, port
    except FileNotFoundError:
      logging.waring("WARNING!  Lock file is missing.")
      return None, None


