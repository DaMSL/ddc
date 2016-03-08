#!/usr/bin/env python
"""Alluxio Overlay Service
    Implementation for the aluxio (f.k.a Tachyon) distrinbuted
    in-memory file service
"""
import os
import tempfile
from threading import Thread, Event
import logging

import time
import numpy as np
from collections import deque
import sys
import subprocess as proc
import shlex
import shutil
import json

# import redis

from core.common import systemsettings, executecmd
# from core.kvadt import kv2DArray, decodevalue
from overlay.overlayService import OverlayService

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

class AlluxioService(OverlayService):
  """
  """

  DEFAULT_PORT = 19999

  def __init__(self, name, role):
    """
    """

    OverlayService.__init__(self, name, AlluxioService.DEFAULT_PORT)
    self.connection = None
    self._role = role

    # # Check if a connection exists to do an immediate shutdown request
    # if os.path.exists(self.lockfile):
    #   host, port = self.getconnection()
    #   self.shutdowncmd = 'redis-cli -h %s -p %s shutdown' % (host, port)

  def ping(self, host='localhost', port=None):
    """Heartbeat to the local Redis Server
    """
    # TODO: curl cmd to web U/I
    return True

  def prepare_service(self):

    config = systemsettings()
    if not config.configured():
      # For now assume JSON file
      config.applyConfig(self._name_app + '.json')

    # Default Settings
    home = os.getenv('HOME')
    alluxio_home = os.path.join(home, 'pkg', 'alluxio-1.0.0')
    self.workdir   = config.WORKDIR  #ini.get('workdir', '.')
    self.ramdisk = tempfile.mkdtemp()
    os.environ['ALLUXIO_HOME'] = alluxio_home
    if self._role == 'SLAVE':
      os.environ['ALLUXIO_MASTER_ADDRESS'] = self.master
      self.launchcmd = 'alluxio-start.sh worker Mount'
    else:
      os.environ['ALLUXIO_MASTER_ADDRESS'] = 'localhost'
      self.launchcmd = 'alluxio-start.sh local'

    os.environ['DEFAULT_LIBEXEC_DIR'] = os.path.join(alluxio_home, 'libexec')
    os.environ['ALLUXIO_RAM_FOLDER'] = self.ramdisk
    os.environ['ALLUXIO_UNDERFS_ADDRESS'] = config.ALLUXIO_UNDERFS
    os.environ['ALLUXIO_WORKER_MEMORY_SIZE'] = config.ALLUXIO_WORKER_MEM

    self.MONITOR_WAIT_DELAY    = config.MONITOR_WAIT_DELAY #ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = config.CATALOG_IDLE_THETA #ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = config.CATALOG_STARTUP_DELAY #ini.get('catalog_startup_delay', 10)

    logging.debug("Checking ENV:")
    logging.debug('  ALLUXIO_HOME=%s', executecmd('echo $ALLUXIO_HOME'))
    logging.debug('  ALLUXIO_MASTER_ADDRESS=%s', executecmd('echo $ALLUXIO_MASTER_ADDRESS'))
    logging.debug('  ALLUXIO_RAM_FOLDER=%s', executecmd('echo $ALLUXIO_RAM_FOLDER'))


    self.shutdowncmd = 'alluxio-stop.sh all'

  def idle(self):
    return False

  def handover_to(self):
    """Invoked to handover services from this instance to a new master
    Called when a new slave is ready to receive as a new master
    """
    logging.info("[%s] RECEVIED Flag to Handover <Not Implemented>", self._name_svc)
    return None

  def handover_from(self):
    """Invoked as a callback when another master service is handing over
    service control (i.e. master duties) to this instance
    """
    logging.info("[%s] INITIATED Handover <Not Implemented>", self._name_svc)

  def tear_down(self):
    logging.info("[%s] Removing the ramdisk", self._name_svc)
    shutil.rmtree(self.ramdisk)

  def launch_slave(self):
    """This is used for the Alluxio Master to launch subsequent worker nodes
    in a rolling succession
    """
    taskid = 'osvc-alx'
    params = {'time':'4:0:0', 
              'nodes':1, 
              'cpus-per-task':1, 
              'partition':'debug', 
              'job-name':taskid,
              'workdir' : os.getcwd()}
    environ = {'ALLUXIO_HOME': os.getenv['ALLUXIO_HOME'],
               'ALLUXIO_MASTER_ADDRESS': self._host,
               'DEFAULT_LIBEXEC_DIR':  os.getenv['DEFAULT_LIBEXEC_DIR'],
               'ALLUXIO_RAM_FOLDER': '/tmp/alluxio'}
    cmd = '\nmkdir -p /tmp/alluxio'


class AlluxioClient(object):
  """Alluxio Client interface for performing POSIX-like operations on the 
  Alluxio distributed in-memory file system. Client object uses the 
  Alluxio Shell executable command (via terminal command line); 

  TODO: Future implementation should implement via jython or jcc
  wrapped over the Java AlluxioShell Class
  """

  def __init__(self, home=None, master=None):

    config = systemsettings()
    # Require alluxio home to be defined
    if home is None and os.getenv('ALLUXIO_HOME') is None:
      logging.error('[AlluxioClient] ALLUXIO_HOME is not set or provided. Cannot create shell client')
      return
    elif home is not None:
      self._home = home
      os.environ['ALLUXIO_HOME'] = home
    else:
      self._home = os.getenv('ALLUXIO_HOME')

    # Find master using AlluxioService lock file 
    if master is None:
      lockfile = '%s_%s.lock' % (config.name, 'AlluxioService')
      if not os.path.exists(lockfile):
        logging.error('[AlluxioClient] Alluxio service is not running.')
        return
      with open(lockfile, 'r') as conn:
        conn_string = conn.read().split(',')
        master = conn_string[0]
    self.set_master(master)

  def set_master(self, master):
    self._master = master
    os.environ['ALLUXIO_MASTER_ADDRESS'] = master

  # Copy local file into Alluxio
  def put(self, localfile, destdir=''):
    if not os.path.exists(localfile):
      logging.error('[AlluxioClient] File does not exist: %s', localfile)
      return
    cmd = 'alluxio fs copyFromLocal %s /%s' % (localfile, destdir)
    out = executecmd(cmd)

  # Retrieve file from Alluxio
  def get(self, destdir):
    cmd = 'alluxio fs cat %s /%s' % (localfile, destdir)
    contents = executecmd(cmd)
    return contents

  # Retrieve file from Alluxio
  def cp(self, remotefile, destdir):
    cmd = 'alluxio fs copyToLocal /%s %s/%s' % (remotefile, destdir, remotefile)
    logging.debug('[AlxClient] cp cmd: %s', cmd)
    contents = executecmd(cmd)
    logging.debug('CP results: %s', contents)
    return contents



  @classmethod
  def package_cmd(cls, cmd, args=[]):
    cmd_line = ['alluxio', 'fs', cmd]
    cmd_line.extend(args)
    return cmd_line

  @classmethod
  def ls(cls, destdir=''):
    args = ['/' + destdir]
    cmd = package_alluxio_cmd('ls', args)
    out = executecmd(cmd)


    return out

