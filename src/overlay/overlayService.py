#!/usr/bin/env python
"""Overlay Service Abstract Class
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
import json
from datetime import datetime as dt
from dateutil import parser as dtparser
from collections import deque

from core.slurm import slurm, systemsettings
from bench.stats import StatCollector

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

DEFAULT_TTL = 60*60  #1 Hour default timelimit (in sec)

class OverlayServerStatus:
  def __init__(self, file):
    self.host = None
    self.port = None
    self.ttl = None
    self.state = None
    self.valid = False
    self.available = False

  def getMaster (self, lockfile):
    if os.path.exists(lockfile):
      with open(lockfile) as master:
        config = master.read().split('\n')[0].split(',')
      if len(config) < 2:
        logging.error('Lockfile ERROR. Lock file is corrupt:  %s', lockfile)
        return
    self.host = config[0]
    self.port = config[1]
    starttime = int(config[2])
    self.ttl = int(config[3])
    self.state = config[4]
    ts = dt.now()
    if dt.now() > self.ttl:
      logging.warning("Overlay Service Master has expired. Check lockfile: %s", lockfile)
    else:
      self.valid = True
    if self.state == 'RUN':
      self.available = True


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

    settings = systemsettings()
    settings.applyConfig(name + '.json')

    # For Stat Collection
    self.stat = StatCollector('Overlay')
    self.stat.collect('name', name)

    # DEBUG
    self._host = socket.gethostname()
    self._port = port

    svc_name = kwargs.get('svc_name', type(self).__name__)
    self._name_svc = svc_name
    self._name_app = name
    self.lockfile = '%s_%s.lock' % (self._name_app, self._name_svc)
    logging.info('MY Lock file is: %s', self.lockfile)

    self.numslaves = kwargs.get('numslaves', 0)
    self.launchcmd = None
    self.shutdowncmd = None

    self._pid = None

    self._state = 'INIT'
    self._role  = kwargs.get('role', 'NONE')

    self.aggressive = kwargs.get('isaggressive', False)

    # Check if this is a Slurm job and get the settings
    self.slurm_id = os.getenv('SLURM_JOBID')
    if self.slurm_id is None:
      # Not running as a Slurm Job (set default TTL)
      self.ttl = dt.now().timestamp() + DEFAULT_TTL
      self.jobinfo = None
      self.stat.collect("slurmid","None")
    else:
      logging.info("SLURM JOB:  %s", str(self.slurm_id))
      self.jobinfo = slurm.jobinfo(int(self.slurm_id))
      endtime = dtparser.parse(self.jobinfo['EndTime'])
      self.ttl = endtime.timestamp()
      self.stat.collect("slurmid",self.slurm_id)
    self.stat.collect("ttl",self.ttl)


    # TODO: All services' nodes assume to run on same port (different hosts)
    #  This the master hold only the host name
    self.master = kwargs.get('master', None)
    self.slavelist = {}

    self.terminationFlag = Event()
    self.handoverFlag = Event()
    self.shuttingdown = Event()
    self.state_running = Event()

    # TODO: Should we require basic setting via systemsetting object???
    #   and thus, enforce this object on the abstract class?
    self.SERVICE_STARTUP_DELAY = 60
    self.SERVICE_HEARTBEAT_DELAY = 60

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

  def start(self):
    """
    """
    logging.info('Overlay Service is Starting: %s', self._name_svc)
    logging.info('  SERVICE_STARTUP_DELAY  = %d', self.SERVICE_STARTUP_DELAY)
    logging.info('  SERVICE_HEARTBEAT_DELAY  = %d\n', self.SERVICE_HEARTBEAT_DELAY)

    # Validate the initialization State. Services first start in a Slave State
    #  but change over to Master if no Master is detected. If the master is
    #  online, the service becomes a replica (future enhancements will allow
    #  disitrbuted and multiple slaves/replicas)
    if self._role == 'SLAVE':
      logging.info('[%s] Starting as a slave. ', self._name_svc)
      slave_wait_ttl = self.SERVICE_STARTUP_DELAY * 4
      while slave_wait_ttl > 0:
        host, port, state = self.getconnection()
        logging.debug('SLAVE Coming online: Current Conn  = %s, %s', str(host), str(state))
        if host is None:
          # No Master alive -- become the master
          logging.info('No Master Detected. Becoming the Master.')
          self._role = 'MASTER'
          break
        elif (state == 'RUN' or state == 'EXPIRED') and self.ping(host, port):
          # FLAG MASTER
          logging.info('[%s] Master is Alive. Starting as a slave. ', self._name_svc)
          with open(self.lockfile, 'a') as conn:
            ts = dt.now().timestamp()
            conn.write('%s,%d,%f,%f,%s\n' % (self._host, self._port, ts, self.ttl, 'SLAVE'))
          # Master is ALIVE and RUNNING
          host = None
          self._role = 'REPLICA'
          break
        elif state == 'EXPIRED':
          # Master is ALIVE, but detected as EXPIRED
          logging.info('[%s] Master at %s has expired. Becoming the Master.', self._name_svc, str(host))
          host = None
          self._role = 'MASTER'
          break
        else:
          # Master is either registered as START or it's unavailable. Wait
          logging.warning('[%s] My Master is Not alive. Host is %s.  Waiting to see if it will come online...', self._name_svc, str(host))
          time.sleep(self.SERVICE_STARTUP_DELAY)
          slave_wait_ttl -= self.SERVICE_STARTUP_DELAY
          if slave_wait_ttl < 0: 
            logging.warning('[%s] Could not find a master. Taking over as Master.', self._name_svc, self._host)
            if os.path.exists(self.lockfile):
              os.remove(self.lockfile)
            host = None
            self._role = 'MASTER'
            break
      #TODO: 
      # self.masterport = port

    # Check to ensure lock is not already acquired
    while self.master is None:
      logging.info('Checking on the master.... (MY ROLE = %s)', self._role)
      if self._role == 'MASTER':
        try:
          lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
          ts = dt.now().timestamp()
          os.write(lock, bytes('%s,%d,%f,%f,%s\n' % (self._host, self._port, ts, self.ttl, 'START'), 'UTF-8'))
          os.close(lock)
          logging.debug("[%s] Lock File not found (I am creating it from %s)", self._name_svc, self._host)
          self._state = 'START'
          self.master = self._host
          break
        except FileExistsError as ex:
          logging.debug("[%s] Lock File exists (someone else has acquired it)", self._name_svc)

      host, port, state = self.getconnection()
      logging.debug('Overlay Service State = %s', str(state))
      if state is None or state == 'EXPIRED':
        logging.error("[%s] BAD state from existing %s. Removing it retrying to lock it.", self._name_svc, self.lockfile)
        if os.path.exists(self.lockfile):
          os.remove(self.lockfile)
        self._role = 'MASTER'
        continue

      if state == 'START':
        logging.debug("[%s] SOMEONE ELSE is starting my service on %s.", self._name_svc, host)
        timeout = self.SERVICE_STARTUP_DELAY
        while timeout > 0 and state == 'START':
          time.sleep(1)
          host, port, state = self.getconnection()
          timeout -= 1
        if state is not None and state == 'RUN':
          logging.info('Found a NEW master on %s.', host)
        else:
          logging.debug("[%s] Detected zombie master. Killing the lockfile and retrying to start locally.", self._name_svc)
          self._role  = 'MASTER'
          if os.path.exists(self.lockfile):
            os.remove(self.lockfile)
          continue

      if state == 'RUN':
        logging.debug("[%s] Checking if service is available on %s, %s", self._name_svc, host, port)
        if self.ping(host, port):
          logging.debug("[%s] Service is ALIVE on %s", self._name_svc, host)
          if self.aggressive:
            logging.debug("[%s] Starting AGGRESSIVE -- will initiate master takeover control protocol", self._name_svc)
            # If starting up as an aggressive replica -- assume immediate replication and control
            #  of service operation. This will initiate this node to start as a 
            #  replica and initate replication/handover protocol. Future improvements
            #  should allow for the system to detect the # of replica (for FT) and 
            #  flag handover accordingly
            self._role  = 'REPLICA'
            self.master = host
            self.handoverFlag.set()
            break
          else:
            logging.debug('[%s] Service already running on %s. I am TIMID -- to take over, start me in aggressive mode', self._name_svc, host)
            return None
        else:
          logging.warning('[%s] Service is NOT running. Will attempt to recover and start locally.', self._name_svc)
          self._role  = 'MASTER'
          if os.path.exists(self.lockfile):
            os.remove(self.lockfile)

    self.prepare_service()

    if self.launchcmd is None:
      logging.error("[%s] Launch Command not set. It needs to be defined.", self._name_svc)
      return None

    if self.shutdowncmd is None:
      logging.error("[%s] Shutdown Command not set. It needs to be defined.", self._name_svc)
      return None

    # TODO: Check subproc call here -- should this also be threaded in a python wrap call?
    logging.info("[%s] Launching following command %s", self._name_svc, self.launchcmd)    
    args = shlex.split(self.launchcmd)
    service = proc.Popen(args)

    if service.returncode is not None:
      logging.error("[%s] ERROR starting local service on %s", self._name_svc, self.host)    
      # Exit or Return ???
      return None

    # Ensure service has started locally
    svc_up = None
    timeout = time.time() + self.SERVICE_STARTUP_DELAY
    while not svc_up:
      if time.time() > timeout:
        logging.error("[%s] Timed Out waiting on the server", self._name_svc)    
        break
      time.sleep(1)
      svc_up = self.ping()

    if not svc_up:
      logging.error("[%s] Service never started. You may need to retry.", self._name_svc)    
      return None

    self._state = 'RUN'
    self.state_running.set()
    logging.info("[%s] My Service started local on %s (ROLE=%s). Starting the local monitor.", self._name_svc, self._role, self._host)    
    ts = dt.now().timestamp()

    if self._role  == 'MASTER':
      if os.path.exists(self.lockfile):
        logging.info("[%s] Removing and creating the Lockfile.", self._name_svc)    
        os.remove(self.lockfile)
      else:
        logging.info("[%s] Creating a new Lockfile.", self._name_svc)    
      lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
      os.write(lock, bytes('%s,%d,%f,%f,%s\n' % (self._host, self._port, ts, self.ttl, 'RUN'), 'UTF-8'))
      os.close(lock)

    self.service = service

    t = Thread(target=self.monitor)
    t.start()
   
    return t

  def monitor(self):

    MISS_HB_TO = 5  # 5 min timeot

    logging.info("[%s] Monitor Daemon Started.", self._name_svc)
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

    # TODO:  For multi-node services, the Master monitor process will start up all slaves
    #  as individual jobs  -- FUTURE
    # if self.numslaves > 0:
    #   logging.info("[%s-Mon] Service is up locally on %s. Preparing to launch %d slaves.", self._name_svc, self.numslaves)    

    #   # TODO: Thread each slave with a separate monitor or launcher 
    #   for i in len(range(self.numslaves)):
    #     self.launch_slave()

    #  Blocking loop. Will only exit if the term flag is set or by a 
    #   in-loop check every heartbeat (via miss ping, lost pid, or idle timeout)
    logging.info("[%s-Mon] Service is Operational and READY. Running the event handler loop.", self._name_svc)    
    if self.jobinfo is not None:
      load_time = (dt.now() - dtparser.parse(self.jobinfo['StartTime'])).total_seconds()
      logging.info('Load Time:\nLOAD_TIME,%f', load_time)
      self.stat.collect('LOAD', load_time)

    while not self.terminationFlag.wait(self.SERVICE_HEARTBEAT_DELAY):
      logging.warning("[%s-Mon] Heartbeat. My ROLE = %s", self._name_svc, self._role)
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
        if missed_hb * self.SERVICE_HEARTBEAT_DELAY > 60 * MISS_HB_TO:
          logging.warning("[%s-Mon]  Lost commo with Server for over %d minutes. Shutting Down",self._name_svc, MISS_HB_TO)
          self.terminationFlag.set()
        continue

      missed_hb = 0

      # CHECK IDLE TIME
      if self.idle():
        logging.info("[%s-Mon] Service is idle. Initiate graceful shutdown.", self._name_svc)    
        break

      if self._role == 'MASTER':
        with open(self.lockfile, 'r') as conn:
          lines = conn.read().strip().split('\n')
        logging.info("[%s-Mon] Checking Lockfile shows %d hosts online", self._name_svc, len(lines))    
        if len(lines) > 1:
          # TODO: Multiple slave
          for i in lines[1:]:
            logging.info('SLAVE detected: %s', i)
            data = i.split(',')
            self.slavelist[data[0]] = data[-1]
            logging.info("[%s-Mon]  Detected a new slave on %s", self._name_svc, data[0])
          self.handoverFlag.set()

      if self.handoverFlag.is_set():
        handover_begin = dt.now()
        # This service has been flagged to handover control to a new master or from an old one
        if self._role == 'MASTER':
          # Handover Control
          next_master = self.handover_to()
          if next_master is None:
            self.handoverFlag.clear()
            continue
          self._role = 'REPLICA'
          logging.info("[%s-Mon]  Handover complete. I am now terminating. My role is now %s.", self._name_svc, self._role)
          # Terminate
          handover_time = (dt.now() - handover_begin).total_seconds()
          logging.info('handover Time:\nHANDOVER_TIME_TO,%f', handover_time)
          self.stat.collect('HANDOVER_TO', handover_time)

          self.terminationFlag.set()

        elif self._role == 'REPLICA':
          # Initiate replica protocol and assume role as master
          self.handover_from()
          logging.info("[%s-Mon]  Handover complete. I am now the master.", self._name_svc)

          # Re-set lock file (TODO: other slaves)
          self._role = 'MASTER'
          ts = dt.now().timestamp()
          ttl = ts + self.ttl
          os.remove(self.lockfile)
          lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
          os.write(lock, bytes('%s,%d,%f,%f,%s\n' % (self._host, self._port, ts, ttl, 'RUN'), 'UTF-8'))
          os.close(lock)

          handover_time = (dt.now() - handover_begin).total_seconds()
          logging.info('handover Time:\nHANDOVER_TIME_FROM,%f', handover_time)
          self.stat.collect('HANDOVER_FROM', handover_time)

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
      if self._role == 'MASTER' and len(self.slavelist) == 0:
        logging.info('[%s] Last master shutdown. Veryfying I am the last one', self._name_svc)
        host, port, state = self.getconnection()
        if host == self._host:
          os.remove(self.lockfile)
        else:
          logging.info('[%s] New service detected on %s.', self._name_svc, host)
      else:
        logging.info('[%s] This master is shutting down, but detected someone else.', self._name_svc)

      logging.info('[%s] Service shutting down on %s.', self._name_svc, self._host)
      args = shlex.split(self.shutdowncmd)
      result = proc.call(args)
      logging.info("[%s] shutdown with return code: %s.  Tearing Down remaining work.", self._name_svc, str(result))

      self.tear_down()
      self.stat.show()


  def halt(self, signum, frame):

    logging.warning('[%s] SIGNAL Received: %s', self._name_svc, siglist[signum])
    logging.warning('[%s] Setting Termination Flag and Gracefully stopping the service',self._name_svc)
    self.terminationFlag.set()
    self.stop()

  def getconnection(self):
    try:
      with open(self.lockfile, 'r') as conn:
        nodes = conn.read().strip().split('\n')
        if len(nodes) == 0:
          return None, None, None
        conn_string = nodes[0].split(',')
        if len(conn_string) < 5:
          return None, None, None
        host = conn_string[0]
        port = conn_string[1]
        ttl  = float(conn_string[3])
        state = conn_string[4].strip()
        ts = dt.now()
        if ts.timestamp() > ttl:
          logging.warning("Overlay Service Master has expired. Check lockfile: %s", self.lockfile)
          state = 'EXPIRED'
      return host, port, state
    except FileNotFoundError:
      logging.warning("WARNING!  Lock file is missing.")
      return None, None, None






#OLD START:
  # def start(self, as_replica=True):
  #   """
  #   """

  #   logging.info('Overlay Service is Starting: %s', self._name_svc)
  #   logging.info('  SERVICE_STARTUP_DELAY  = %d', self.SERVICE_STARTUP_DELAY)
  #   logging.info('  SERVICE_HEARTBEAT_DELAY  = %d\n', self.SERVICE_HEARTBEAT_DELAY)


  #   if self._role == 'SLAVE':
  #     logging.warning('[%s] Starting as a slave. ', self._name_svc)
  #     slave_wait_ttl = self.SERVICE_STARTUP_DELAY * 4
  #     while slave_wait_ttl > 0:
  #       host, port, state = self.getconnection()
  #       if host is not None and state == 'RUN' and self.ping(host):
  #         break
  #       elif host is not None and state == 'EXPIRED':
  #         logging.info('[%s] Master at %s has expired. Becoming Master....', self._name_svc, str(host))
  #         self.master = None
  #         self._role = 'MASTER'
  #       logging.warning('[%s] My Master is Not alive. Host is %s.  Waiting...', self._name_svc, str(host))
  #       sleep(self.SERVICE_STARTUP_DELAY)
  #       slave_wait_ttl -= self.SERVICE_STARTUP_DELAY
  #       if slave_wait_ttl < 0: 
  #         logging.warning('[%s] Could not find a master. Slave is shutting down on %s.  Waiting...', self._name_svc, self._host)
  #         return None
  #     # Set the master 
  #     self.master = host
  #     #TODO: 
  #     # self.masterport = port

  #   # Check to ensure lock is not already acquired
  #   while self.master is None:
  #     try:
  #       lock = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
  #       os.write(lock, bytes('%s,%d' % (self._host, self._port), 'UTF-8'))
  #       os.close(lock)
  #       logging.debug("[%s] Lock File not found (I am creating it from %s)", self._name_svc, self._host)
  #       self._state = 'START'
  #       self._role  = 'MASTER'
  #       self.master = self._host
  #       break
  #     except FileExistsError as ex:
  #       logging.debug("[%s] Lock File exists (someone else has acquired it)", self._name_svc)

  #     host, port, state = self.getconnection()
  #     if state is None:
  #       logging.error("[%s] BAD state from existing %s. Sleeping and retrying to lock it.", self._name_svc, self.lockfile)
  #       time.sleep(1)
  #       continue

  #     if state == 'START':
  #       logging.debug("[%s] SOMEONE ELSE is starting my service on %s.", self._name_svc, host)
  #       try:

  #       timeout = self.SERVICE_STARTUP_DELAY
  #       while timeout > 0 and state == 'START':
  #         time.sleep(1)
  #         host, port, state = self.getconnection()
  #       if state is None:
  #         continue

  #     if state == 'START':




  #         logging.debug('[%s] Service already running on %s. To start as slave, call handover_slave', self._name_svc, host)
  #         return None

  #       logging.debug("[%s] Checking if service is available on %s, %s  (", self._name_svc, host, port)
  #       if self.ping(host, port):
  #         logging.debug("[%s] Running on %s", self._name_svc, host)
  #         if as_replica:
  #           logging.debug("[%s] Starting in replica -- will initiate master takeover control protocol", self._name_svc)
  #           # If starting up as a replica -- assume immediate replication and control
  #           #  of service operation. This will initiate this node to start as a 
  #           #  replica and initate replication/handover protocol. Future improvements
  #           #  should allow for the system to detect the # of replica (for FT) and 
  #           #  flag handover accordingly
  #           self._role  = 'REPLICA'
  #           self.master = host
  #           self.handoverFlag.set()
  #           break
  #         else:
  #           logging.debug('[%s] Service already running on %s. To start as slave, call handover_slave', self._name_svc, host)
  #           return None
  #       else:
  #         logging.warning('[%s] Service is NOT running. Will attempt to recover and start locally.', self._name_svc)
  #         os.remove(self.lockfile)
  #         time.sleep(1)

  #     if verifymaster:





  #   self.prepare_service()

  #   if self.launchcmd is None:
  #     logging.error("[%s] Launch Command not set. It needs to be defined.", self._name_svc)
  #     return None

  #   if self.shutdowncmd is None:
  #     logging.error("[%s] Shutdown Command not set. It needs to be defined.", self._name_svc)
  #     return None

  #   # TODO: Check subproc call here -- should this also be threaded in a python wrap call?
  #   logging.info("[%s] Launching following command %s", self._name_svc, self.launchcmd)    
  #   args = shlex.split(self.launchcmd)
  #   service = proc.Popen(args)

  #   if service.returncode is not None:
  #     logging.error("[%s] ERROR starting local service on %s", self._name_svc, self.host)    
  #     # Exit or Return ???
  #     return None

  #   # Ensure service has started locally
  #   svc_up = None
  #   timeout = time.time() + self.SERVICE_STARTUP_DELAY
  #   while not svc_up:
  #     if time.time() > timeout:
  #       logging.error("[%s] Timed Out waiting on the server", self._name_svc)    
  #       break
  #     time.sleep(1)
  #     svc_up = self.ping()

  #   if not svc_up:
  #     logging.error("[%s] Service never started. You may need to retry.", self._name_svc)    
  #     return None

  #   self._state = 'RUNNING'
  #   self.state_running.set()
  #   logging.info("[%s] My Service started local on %s. Starting the local monitor.", self._name_svc, self._host)    

  #   self.service = service

  #   t = Thread(target=self.monitor)
  #   t.start()

    
  #   return t