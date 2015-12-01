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

from collections import namedtuple



logger = 0


archiveConfig = dict(name='archive', port=6380)
#archiveConfig = dict(name='testarch', port=6381)


class DEFAULT:

  MANAGER_RERUN_DELAY = 60

  #  TODO:  Set up with Config File
  WORKDIR = os.path.join(os.environ['HOME'], 'work')
  LOG_DIR  = os.path.join(WORKDIR, 'log')
  PSF_FILE = os.path.join(WORKDIR, 'bpti_gen.psf')
  PDB_FILE = os.path.join(WORKDIR, 'bpti', 'bpti-all.pdb')
  RAW_ARCHIVE = os.path.join(WORKDIR, 'bpti')
  # FFIELD   = '/home-1/bring4@jhu.edu/namd/toppar/par_all36_prot.rtf'

  TOPO = os.path.join(os.environ['HOME'], 'bpti', 'amber', 'top_all22_prot.inp')
  PARM = os.path.join(os.environ['HOME'], 'bpti', 'amber', 'par_all22_prot.inp')

  # FFIELD   = '/home-1/bring4@jhu.edu/bpti/toppar/top_all22_prot.prm'
  HIST_FILE_DIR  = os.path.join(WORKDIR, 'bpti')
  JOB_DIR = os.path.join(WORKDIR, 'jc')

  # INDEX_LOCKFILE = os.path.join(WORKDIR, 'index.lock')
  CONFIG_DIR     = WORKDIR
  NUM_PCOMP = 3
  NODES = 1
  CPU_PER_NODE = 24

  HIST_SLIDE  = 50
  HIST_WINDOW = 100

  HASH_NAME = 'rbphash'

  SIM_CONF_TEMPLATE = 'src/sim_template.conf'
  REDIS_CONF_TEMPLATE = 'src/redis.conf.temp'

  PARTITION = 'parallel'

  # Catalog Params
  MONITOR_WAIT_DELAY = 30
  CATALOG_IDLE_THETA = 300
  CATALOG_STARTUP_DELAY = 10

  # TODO: Move this from a file to the archive!
  DATA_LABEL_FILE = os.path.join(os.getenv('HOME'), 'ddc', 'bpti_labels_ms.txt')
  CANDIDATE_POOL_SIZE = 100

  MAX_NUM_NEW_JC = 10
  MAX_JOBS_IN_QUEUE = 100
  
  # TODO:  Make this a functional filter ILO string filter
  ATOM_SELECT_FILTER = lambda x: x.top.select_atom_indices(selection='heavy')
  NUM_VAR = 454  # TODO: Set during Init

  @classmethod
  def envSetup(cls):
    if not os.path.exists(cls.JOB_DIR):
      os.makedirs(cls.JOB_DIR)

    if not os.path.exists(cls.LOG_DIR):
      os.mkdir(cls.LOG_DIR)

  @classmethod
  def getEmptyHash(cls):
    return UniBucket(DEFAULT.HASH_NAME)
    return RandomBinaryProjections(None, None)
    # return PCABinaryProjections(None, None, None)




def setLogger(name=""):
  global logger
  if not logger:
    # log_fmt = logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
    log_fmt = logging.Formatter(fmt= '[%(module)s] %(message)s',datefmt='%H:%M:%S')
    logger = logging.getLogger("")
    log_console = logging.StreamHandler()
    log_console.setFormatter(log_fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_console)
  return logger


def getUID():
  return str(uuid.uuid1()).split('-')[0]


# TODO:  Encode/Decode Wrapper Class via functional methods
def encodeLabel(window, seqnum):
  return "%04d_%03d" % (int(window), int(seqnum))

def decodeLabel(label):
  win, seq = label.split("_")
  return int(win), int(seq)

def getJC_Key(uid):
  return "jc_%s" % str(uid) if not uid.startswith('jc_') else uid

def getJC_UID(jckey):
  return jckey[3:] if jckey.startswith('jc_') else jckey

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

def loadLabels(fn=DEFAULT.DATA_LABEL_FILE):
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
    # Connect Storage & nearpy engine
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


#  PROGRAM DEFAULTS FOR INITIALIZATION
#   TODO:  Consolidate & dev config file



schema = dict(  
        JCQueue = [],
        JCComplete = 0,
        JCTotal = 1,
        simSplitParam =  4, 
        anlSplitParam =  4,
        ctlSplitParam =  4,
        dcdFileList =  [], 
        processed =  0,
        indexSize = DEFAULT.NUM_VAR*DEFAULT.NUM_PCOMP,
        LDIndexList = [],
        converge =  0.,
        weight_alpha = .4,
        weight_beta = .6,
        observation_counts = [])

candidPoolKey = lambda x, y: 'candidatePool_%d_%d' % (x, y)

for i in range(5):
  for j in range(5):
    schema[candidPoolKey(i,j)] = []






def initialize(catalog, archive, flushArchive=False):


  #  Create a "seed" job
  logging.debug("Loading schema and setting initial job")
  jcuid = 'SEED'

  seedJobCandidate = dict(
    workdir = str(os.path.join(DEFAULT.JOB_DIR,  jcuid)),
    psf     = jcuid + '.psf',
    pdb     = jcuid + '.pdb',
    parm    = DEFAULT.PARM,
    name    = jcuid,
    temp    = 310,
    runtime = 25000)


  key = getJC_Key(jcuid)
  initParams = {key:seedJobCandidate}

  # Load schema and insert the start job into the queue
  startState = dict(schema)
  startState['JCQueue'] = [key]
  startState['JCTotal'] = 1

  executecmd("shopt -s extglob | rm !(SEED.p*)")

  # TODO:  Job ID Management
  ids = {'id_' + name : 0 for name in ['sim', 'anl', 'ctl']}

  logging.debug("Catalog found on `%s`. Clearing it.", catalog.host)
  catalog.clear()

  logging.debug("Loading initial state into catalog.")
  catalog.save(ids)
  catalog.save(startState)
  catalog.save(initParams)

  for k, v in initParams.items():
      logging.debug("    %s: %s" % (k, v))
  

  # Load DEShaw data
  histo = np.load('histogram.npy')

  # TO INIT FATIGUE VALS:



  logging.debug("Stopping the catalog.")
  catalog.stop()
  if os.path.exists('catalog.lock'):
    os.remove('catalog.lock')



  logging.debug("Archive found on `%s`. Stopping it.", archive.host)

  if flushArchive:
    archive.clear()

    # Create redis storage adapter
    redis_storage = RedisStorage(archive)

    # Create Hash
    # lshash = RandomBinaryProjections(DEFAULT.HASH_NAME, 3)

    # Assume vects is 
    # pcahash = PCABinaryProjections('pcahash', 10, [v[0] for v in vects])
    # redis_storage.store_hash_configuration(pcahash)
    # eng2 = Engine(454, lshashes=[pcahash], storage=redis_storage)
    # for v in vects:
    #   eng2.store_vector(v[0], v[1])



    # Store hash configuration in redis for later use
    logging.debug('Storing Hash in Archive')
    redis_storage.store_hash_configuration(lshash)

    # TODO:  Automate Historical Archive (Re)Loading


  archive.stop()
  if os.path.exists('archive.lock'):
    os.remove('archive.lock')


  logging.debug("Initialization complete\n")
