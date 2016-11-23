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


# Hard Coded Feature Set (pre-calculated for now)
FEATURE_SET = [2, 52, 56, 60, 116, 258, 311, 460, 505, 507, 547, 595, 640, 642, 665, 683, 728, 767, 851, 1244, 1485, 1629, 1636]
FEATURE_SET_RESID = [3, 4, 9, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50]

def executecmd(cmd):
  """ Execute single shell command. Block on output and return STDOUT """
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode()


def executecmd_pid(cmd):
  """ Execute single shell command. Block on output and return STDOUT & PID """
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode(), task.pid

def gettempdir():
  """ Returns path to local shared mem directory (standardizes name) """
  ramdisk = '/dev/shm/ddc/'
  # ramdisk = '/tmp/ddc/'
  if not os.path.exists(ramdisk):
    os.mkdir(ramdisk)
  return ramdisk



def singleton(cls):
  """ The Singleton Design Pattern. Each execution of the application can
    only create one instance of this class """
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

  def applyConfig(self, ini_file=None, force_config=False):

    if self.configured() and not force_config:
      return

    if ini_file is not None:
      self._confile = ini_file

    logging.info("Applying System Settings from inifile:  %s", self._confile)   

    with open(self._confile) as f:
      defaults = json.loads(f.read())

    ini = defaults['settings']

    # System Environment Settings
    self.EXPERIMENT_NUMBER    = ini.get('experiment_number', -1)

    # For now exit to remind me to set this
    if self.EXPERIMENT_NUMBER < 0:
      logging.error('NO Experiment Number set in JSON. Please set it.')
      sys.exit(1)

    #  READ & SET for EACH 
    # Application name is the name of the .ini file 
    application = os.path.basename(self._confile).split('.')[0]
    self.name = self.APPL_LABEL = application   #Alias
    self.workdir = self.WORKDIR = ini.get('workdir', '.')

    # Associated sub-dirs are automatically created in the working dir
    self.logdir = self.LOGDIR = os.path.join(self.WORKDIR, 'log', application)
    self.jobdir = self.JOBDIR = os.path.join(self.WORKDIR, 'jc', application)
    self.datadir = self.DATADIR = os.path.join(self.WORKDIR, 'data', application)

    # Overlay Service generic (or should this be unique to ea. service????)
    self.MONITOR_WAIT_DELAY    = ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = ini.get('catalog_startup_delay', 10)
    self.SERVICE_HEARTBEAT_DELAY = ini.get('catalog_heartbeat_delay', 15)

    # Redis Service specific settings
    self.REDIS_CONF_TEMPLATE = ini.get('redis_conf_template', 6379)
    self.CATALOG_PORT = ini.get('catalog_port', 6379)

    # CACHE SERVICE PARAMS  (in GB)
    self.CACHE_CAPACITY = ini.get('cache_capacity', 80)

    # Alluxio Service specific settings
    # FOR shared Lustre (not working!)
    # self.ALLUXIO_UNDERFS = os.path.join(self.WORKDIR, 'alluxio', application)
    # FOR local UFS (will need to clean up!)
    # self.ALLUXIO_UNDERFS = '/tmp/alluxio'
    # self.ALLUXIO_WORKER_MEM = '20GB'


    # Analysis Setting
    self.MAX_RESERVOIR_SIZE = 1000

    # Controller Settings
    self.SIMULATE_RATIO = ini.get('simulation_ratio', 1)   # FOR DEBUGGING
  
    # Potentailly Dynamic
    self.PARTITION = ini.get('partition', 'shared')

    #  PARAMS TO BE SET:
    self.RMSD_THETA = ini.get('rmsd_theta', .33)   # RMS Transition detetion Sensitivity
    self.RMSD_CENTROID_FILE = ini.get('rmsd_centroid_file', 'data/gen-alpha-cartesian-centroid.npy')
    self.PCA_VECTOR_FILE = ini.get('pca_vector_file', 'data/pca_comp.npy')
    self.PCA_NUMPC = ini.get('pca_numpc', 3)

    # self.OBS_NOISE = ini.get('obs_noise', 10000)
    # self.RUNTIME_FIXED = ini.get('runtime', 100000)
    # self.DCDFREQ = ini.get('dcdfreq', 500)
    # self.SIM_STEP_SIZE = 2   #FIXED at 2 fs per timestep


    # Config Schema -- placed here for now (TODO: Split????)    
    # SCHEMA
    #  For now defined as a dict, for simplicity, with thread state receiving a 
    #     copy of this and caching locally only what it needs as defined in setState()
    #  TODO:  Each item in the schema should be traced obj w/getter/setter attrib to
    #     trace dirty flagging (which reduces I/O) and provides a capability to 
    #     synchronize between threads thru the catalog


    # schema : data types to support K-V operations (place holder for now)
    self.schema = defaults['schema']

    # 
    self.state = defaults['state']
    self.sim_params = defaults['simulation']


    # make_config_file = 'default.conf'

    # if make_config_file:
    #   data = {'settings': ini, 'schema': self.schema}
    #   with open(make_config_file, 'w') as f:
    #     f.write(json.dumps(data, sort_keys=True, indent=4))

    self._configured = True

  def manualConfig(self, application):
    if self.configured():
      return

    logging.info("Applying System Settings from local map")   
    # System Environment Settings
    self.EXPERIMENT_NUMBER    = -1

    #  READ & SET for EACH 
    self.APPL_LABEL  = application
    self.name = application   #Alias
    self.WORKDIR     = os.path.join(os.getenv('HOME'), 'work')

    self.LOGDIR = os.path.join(self.WORKDIR, 'log', application)
    self.JOBDIR = os.path.join(self.WORKDIR, 'jc', application)
    self.DATADIR = os.path.join(self.WORKDIR, 'data', application)

    # Redis Service specific settings
    self.REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'
    self._configured = True

  def setnum_pc(self, n=3):
    self.NUM_PCOMP = n

  def setnum_var(self, n=454):
    self.NUM_VAR = n
  
  def envSetup(self):
    for d in [self.JOBDIR, self.LOGDIR, self.DATADIR]:
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
  log_fmt = logging.Formatter(fmt= '[%(module)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger(name)
  log_console = logging.StreamHandler()
  log_console.setFormatter(log_fmt)
  logger.setLevel(logging.DEBUG)
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



# Create the Singleton Instances

DEFAULT = systemsettings()
logger=setLogger('')  # Set Top Level Logger For formatting


