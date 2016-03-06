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

logging.basicConfig(level=logging.DEBUG)

class AlluxioService(OverlayService):
  """
  """

  def __init__(self, name, port=6379):
    """
    """
    OverlayService.__init__(self, name, port)
    self.connection = None

    config = systemsettings()
    if not config.configured():
      # For now assume JSON file
      config.applyConfig(name + '.json')

    # Default Settings
    # TODO: Set envars via os
    home = os.getenv('HOME')
    alluxio_home = os.path.join(home, 'pkg', 'alluxio-1.0.0')
    self.workdir   = config.WORKDIR  #ini.get('workdir', '.')
    self.ramdisk = tempfile.mkdtemp()
    os.environ['ALLUXIO_HOME'] = alluxio_home
    os.environ['ALLUXIO_MASTER_ADDRESS'] = 'localhost'
    os.environ['DEFAULT_LIBEXEC_DIR'] = os.path.join(alluxio_home, 'libexec')
    os.environ['ALLUXIO_RAM_FOLDER'] = self.ramdisk
    self.MONITOR_WAIT_DELAY    = config.MONITOR_WAIT_DELAY #ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = config.CATALOG_IDLE_THETA #ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = config.CATALOG_STARTUP_DELAY #ini.get('catalog_startup_delay', 10)

    logging.debug("Checking ENV:")
    logging.debug('  ALLUXIO_HOME=%s', executecmd('echo $ALLUXIO_HOME'))

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
    # Prepare 

    # System Environment Settings
    #  READ & SET for EACH 
    # with open(self.redis_conf_template, 'r') as template:
    #   source = template.read()
    #   logging.info("Redis Source Template loaded")

    # params = dict(localdir=DEFAULT.WORKDIR, port=self._port, name=self._name)
    # params = dict(localdir=self.workdir, port=self._port, name=self._name_app)

    # # TODO: This should be local
    # self.config = self._name_app + "_db.conf"
    # with open(self.config, 'w') as config:
    #   config.write(source % params)
    #   logging.info("Data Store Config written to  %s", self.config)

    self.launchcmd = 'alluxio-start.sh local'
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



