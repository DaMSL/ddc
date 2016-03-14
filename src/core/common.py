#!/usr/bin/env python
"""Common definitions and methods
"""  
import logging
import os
import shutil
import uuid
import subprocess as proc
import numpy as np
import sys
import random
import string
import json
import pickle
import datetime as dt
from collections import OrderedDict

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

def executecmd(cmd):
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode()


class microbench:
  def __init__(self):
    self._begin = None
    self.tick = OrderedDict()
  def start(self):
    self._begin = dt.datetime.now()
    self.tick['START'] = self._begin
  def mark(self, label=None):
    if label is None:
      label = 'mark-%02d' % len(self.tick.keys())
    self.tick[label] = dt.datetime.now()
  def show(self):
    timelist = []
    for label, ts in self.tick.items():
      t = (ts-self._begin).total_seconds()
      if t < 10.:
        timelist.append('##    %7.2f  %s' % (t, label))
      else:
        timelist.append('##    %4d       %s' % (t, label))
    print ('##    TIME   EVENT')
    print ('  ------  ---------')
    for t in timelist:
      print(t)

def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args)
        return instances[cls]
    return getinstance

@singleton
class systemsettings:
  def __init__(self, confile=None):
    self._configured = False
    self._confile = confile

  def configured(self):
    return self._configured

  def applyConfig(self, ini_file=None):

    if ini_file is not None:
      self._confile = ini_file
    logging.info("Applying System Settings from inifile:  %s", self._confile)   

    with open(self._confile) as f:
      defaults = json.loads(f.read())

    ini = defaults['settings']

    # System Environment Settings
    #  READ & SET for EACH 
    application = os.path.basename(self._confile).split('.')[0]
    self.APPL_LABEL  = application
    self.WORKDIR     = ini.get('workdir', '.')

    self.LOGDIR = os.path.join(self.WORKDIR, 'log', application)
    self.JOBDIR = os.path.join(self.WORKDIR, 'jc', application)
    self.DATADIR = os.path.join(self.WORKDIR, 'data', application)

    # Overlay Service generic (or should this unique to ea. service????)
    self.MONITOR_WAIT_DELAY    = ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = ini.get('catalog_startup_delay', 10)
    self.SERVICE_HEARTBEAT_DELAY = ini.get('catalog_heartbeat_delay', 15)

    # Redis Service specific settings
    self.REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'

    # Alluxio Service specific settings
    # FOR shared Lustre (not working!)
    # self.ALLUXIO_UNDERFS = os.path.join(self.WORKDIR, 'alluxio', application)
    # FOR local UFS (will need to clean up!)
    self.ALLUXIO_UNDERFS = '/tmp/alluxio'
    self.ALLUXIO_WORKER_MEM = '20GB'

    # <DEPPRICATE>
    self.catalogConfig  = dict(
        name=ini.get('catalog_name', application),
        port=ini.get('catalog_port', '6379') )
    self.archiveConfig  = dict(
        name=ini.get('archive_name', 'archive'),
        port=ini.get('archive_port', '6380') )
    self.HASH_NAME             = ini.get('hash_name', 'rbphash')  #TODO CHANGE NAME


    # Filter Options: {‘all’, ‘alpha’, ‘minimal’, ‘heavy’, ‘water’}
    # atom_filter = ini.get('atom_filter', 'heavy')
    # self.ATOM_SELECT_FILTER = lambda x: x.top.select_atom_indices(selection=atom_filter)

    # Analysis Setting
    self.MAX_RESERVOIR_SIZE = 1000

    # Controller Settings
    # self.CANDIDATE_POOL_SIZE = ini.get('candidate_pool_size', 100)
    self.MAX_JOBS_IN_QUEUE   = ini.get('max_jobs_in_queue', 100)
    self.MAX_NUM_NEW_JC      = ini.get('max_num_new_jc', 10)
  
    # Potentailly Dynamic
    self.MANAGER_RERUN_DELAY = ini.get('manager_rerun_delay', 60)
    self.PARTITION = ini.get('partition', 'shared')

    #  PARAMS TO BE SET:
    self.OBS_NOISE = ini.get('obs_noise', 10000)
    self.RUNTIME_FIXED = ini.get('runtime', 100000)
    self.DCDFREQ = ini.get('dcdfreq', 500)


    # Config Schema -- placed here for now (TODO: Split????)    
    # SCHEMA
    #  For now defined as a dict, for simplicity, with thread state receiving a 
    #     copy of this and caching locally only what it needs as defined in setState()
    #  TODO:  Each item in the schema should be traced obj w/getter/setter attrib to
    #     trace dirty flagging (which reduces I/O) and provides a capability to 
    #     synchronize between threads thru the catalog

    self.schema = defaults['schema']
    self.init = defaults['init']

    # TODO: Migration to system settings accessors....
    self.name = self.APPL_LABEL

    # make_config_file = 'default.conf'

    # if make_config_file:
    #   data = {'settings': ini, 'schema': self.schema}
    #   with open(make_config_file, 'w') as f:
    #     f.write(json.dumps(data, sort_keys=True, indent=4))

    self._configured = True



  # INDEX_LOCKFILE = os.path.join(WORKDIR, 'index.lock')


  # Catalog Params
  # TODO: Move this from a file to the archive!
  
  

  def setnum_pc(self, n=3):
    self.NUM_PCOMP = n

  def setnum_var(self, n=454):
    self.NUM_VAR = n
  
  def envSetup(self):
    for d in [self.JOBDIR, self.LOGDIR, self.DATADIR, self.ALLUXIO_UNDERFS]:
      if not os.path.exists(d):
        os.mkdir(d)

    checkpath = lambda k, x: print('%-10s: %s.....%s ' % (k, x, ('ok' if os.path.exists(x) else " DOES NOT EXIST")))

    checkpath('WORKDIR', self.WORKDIR)
    checkpath('JOBDIR', self.JOBDIR)
    checkpath('LOGDIR', self.LOGDIR)
    checkpath('REDIS_CONF', self.REDIS_CONF_TEMPLATE)
    # checkpath('TOPO', self.TOPO)
    # checkpath('PARM', self.PARM)


  @classmethod
  def getEmptyHash(cls):
    # return UniBucket(DEFAULT.HASH_NAME)
    return RandomBinaryProjections(None, None)
    # return PCABinaryProjections(None, None, None)


@singleton
def setLogger(name='', logfile=None):
  # global logger

  # if logger is None:
    # log_fmt = logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
  log_fmt = logging.Formatter(fmt= '[%(module)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger(name)
  log_console = logging.StreamHandler()
  log_console.setFormatter(log_fmt)
  logger.setLevel(logging.DEBUG)
    # logger.addHandler(log_console)
  logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
  logging.info("LOGGING IS SET UP")

  return logger


def getUID():
  chrid = random.choice(string.ascii_lowercase + string.ascii_lowercase)
  unique = str(uuid.uuid1()).split('-')[0]

  return chrid + unique


def wrapKey(prefix, key):
  return "%s_%s" % (prefix, key) if not key.startswith('%s_' % prefix) else key

def unwrapKey(key):
  return key[key.find('_')+1:] if '_' in key else key




DEFAULT = systemsettings()
logger=setLogger('')  # Set Top Level Logger For formatting


