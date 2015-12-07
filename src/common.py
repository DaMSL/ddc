import logging

import os
import shutil
import uuid
import subprocess as proc
import nearpy
from nearpy.hashes import RandomBinaryProjections, PCABinaryProjections, UniBucket
from nearpy.distances import EuclideanDistance
from nearpy.storage import RedisStorage
import sys
import random
import string
import json

from collections import namedtuple


def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args)
        return instances[cls]
    return getinstance


logger = None


# archiveConfig = dict(name='archive', port=6380)


@singleton
class systemsettings:
  def __init__(self, confile=None):
    self._configured = False
    self._confile = confile

  def applyConfig(self, ini_file=None):

    if ini_file is not None:
      self._confile = ini_file

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

    self.REDIS_CONF_TEMPLATE = 'templates/redis.conf.temp'
    self.MONITOR_WAIT_DELAY    = ini.get('monitor_wait_delay', 30)
    self.CATALOG_IDLE_THETA    = ini.get('catalog_idle_theta', 300)
    self.CATALOG_STARTUP_DELAY = ini.get('catalog_startup_delay', 10)

    self.catalogConfig  = dict(
        name=ini.get('catalog_name', application),
        port=ini.get('catalog_port', '6379') )



    # Remainder COULD all move to catalog
    self.archiveConfig  = dict(
        name=ini.get('archive_name', 'archive'),
        port=ini.get('archive_port', '6380') )
    self.HASH_NAME             = ini.get('hash_name', 'rbphash')  #TODO CHANGE NAME

    # Simulation & Analysis Protein Settings
    raw = ini.get('raw_archive','bpti')
    self.RAW_ARCHIVE = raw if raw.startswith('/') else os.path.join(self.WORKDIR, raw)
    pdb = ini.get('pdb','bpti-all.pdb')
    self.PDB_FILE = pdb if pdb.startswith('/') else os.path.join(self.RAW_ARCHIVE, pdb)
    self.TOPO       = ini.get('topo') 
    self.PARM       = ini.get('parm') 
    self.NUM_PCOMP  = ini.get('num_pcomp', 3)
    self.NUM_VAR    = ini.get('num_var', 454)  # TODO: Set during Init
    self.RUNTIME    = ini.get('runtime', 51000)

    # Filter Options: {‘all’, ‘alpha’, ‘minimal’, ‘heavy’, ‘water’}
    atom_filter = ini.get('atom_filter', 'heavy')
    self.ATOM_SELECT_FILTER = lambda x: x.top.select_atom_indices(selection=atom_filter)

    # Controller Settings
    self.CANDIDATE_POOL_SIZE = ini.get('candidate_pool_size', 100)
    self.MAX_JOBS_IN_QUEUE   = ini.get('max_jobs_in_queue', 100)
    self.MAX_NUM_NEW_JC      = ini.get('max_num_new_jc', 10)
  
    # Potentailly Dynamic
    self.MANAGER_RERUN_DELAY = ini.get('manager_rerun_delay', 60)


    # Config Schema -- placed here for now (TODO: Split????)    
    # SCHEMA
    #  For now defined as a dict, for simplicity, with thread state receiving a 
    #     copy of this and caching locally only what it needs as defined in setState()
    #  TODO:  Each item in the schema should be traced obj w/getter/setter attrib to
    #     trace dirty flagging (which reduces I/O) and provides a capability to 
    #     synchronize between threads thru the catalog

    self.schema = defaults['schema']

    # make_config_file = 'default.conf'

    # if make_config_file:
    #   data = {'settings': ini, 'schema': self.schema}
    #   with open(make_config_file, 'w') as f:
    #     f.write(json.dumps(data, sort_keys=True, indent=4))


    candidPoolKey = lambda x, y: 'candidatePool_%d_%d' % (x, y)
    for i in range(5):
      for j in range(5):
        self.schema[candidPoolKey(i,j)] = []

    self._configured = True



  # INDEX_LOCKFILE = os.path.join(WORKDIR, 'index.lock')


    # DATA_LABEL_FILE = os.path.join(os.getenv('HOME'), 'ddc', 'bpti_labels_ms.txt')

  # Catalog Params
  # TODO: Move this from a file to the archive!
  
  

  def setnum_pc(self, n=3):
    self.NUM_PCOMP = n

  def setnum_var(self, n=454):
    self.NUM_VAR = n
  
  def envSetup(self):
    if not os.path.exists(self.JOBDIR):
      os.makedirs(self.JOBDIR)

    if not os.path.exists(self.LOGDIR):
      os.mkdir(self.LOGDIR)

    checkpath = lambda k, x: print('%-10s: %s.....%s ' % (k, x, ('ok' if os.path.exists(x) else " DOES NOT EXIST")))

    checkpath('WORKDIR', self.WORKDIR)
    checkpath('JOBDIR', self.JOBDIR)
    checkpath('LOGDIR', self.LOGDIR)
    checkpath('REDIS_CONF', self.REDIS_CONF_TEMPLATE)
    checkpath('TOPO', self.TOPO)
    checkpath('PARM', self.PARM)


  @classmethod
  def getEmptyHash(cls):
    # return UniBucket(DEFAULT.HASH_NAME)
    return RandomBinaryProjections(None, None)
    # return PCABinaryProjections(None, None, None)


DEFAULT = systemsettings()


def setLogger(name=""):
  global logger

  if logger is None:
    # log_fmt = logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
    log_fmt = logging.Formatter(fmt= '[%(module)s] %(message)s',datefmt='%H:%M:%S')
    logger = logging.getLogger("")
    log_console = logging.StreamHandler()
    log_console.setFormatter(log_fmt)
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(log_console)

  return logger


def getUID():
  chrid = random.choice(string.ascii_lowercase + string.ascii_lowercase)
  unique = str(uuid.uuid1()).split('-')[0]

  return chrid + unique


def wrapKey(prefix, key):
  return "%s_%s" % (prefix, key) if not key.startswith('%s_' % prefix) else key

def unwrapKey(key):
  return key[key.find('_')+1:] if '_' in key else key


def chmodX(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2 
    os.chmod(path, mode)


def executecmd(cmd):
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode()




label =namedtuple('window', 'time state')

def loadLabels(fn=None):
  if fn is None:
    fn = os.path.join(os.getenv('HOME'), 'ddc', 'bpti_labels_ms.txt')
  label =namedtuple('window', 'time state')
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

def getLabelList(labels):
  labelset = set()
  for lab in labels:
    labelset.add(lab.state)
  return sorted(list(labelset))


def getNearpyEngine(archive, indexSize):
  redis_storage = RedisStorage(archive)
  config = redis_storage.load_hash_configuration('rbphash')
  if not config:
    logging.error("LSHash not configured")
    sys.exit(0)

  logging.debug("CONFIG:")
  for k,v in config.items():
    logging.debug("%s,  %s", str(k), str(v))

  # Create empty lshash and load stored hash
  eucl = EuclideanDistance()
  lshash = DEFAULT.getEmptyHash()
  lshash.apply_config(config)

  if config['dim'] is None:
    logging.debug("NO DIM SET IN HASH. RESETTING TO 10")
    lshash.reset(10)
    redis_storage.store_hash_configuration(lshash)
    logging.debug("HASH SAVED")


  logging.debug("INDEX SIZE = %d:  ", indexSize)
  engine = nearpy.Engine(indexSize, distance=eucl, lshashes=[lshash], storage=redis_storage)

  return engine




