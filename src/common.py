import logging
import os
import shutil
import uuid
import subprocess as proc


logger = 0


archiveConfig = dict(name='archive', port=6380)


class DEFAULT:
  #  TODO:  Set up with Config File
  WORKDIR = os.path.join(os.environ['HOME'], 'work')
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
  NUM_PCOMP = 2
  NODES = 1
  CPU_PER_NODE = 24

  HIST_SLIDE  = 50
  HIST_WINDOW = 100

  HASH_NAME = 'lshash'

  SIM_CONF_TEMPLATE = 'src/sim_template.conf'
  REDIS_CONF_TEMPLATE = 'src/redis.conf.temp'

  # NEWSIM_PARAM = dict(topo=TOPO, bpti_prot=SRC_PROT, bpti_water=SRC_WATER,
  #   forcefield=FFIELD)



  # NUM_NEIGH = 1000
  # MDS_DIM = 250
  # NUM_NEIGH = 100
  # MDS_DIM = 500
  # MDS_START = 50
  # MDS_STEP = 25
  # Noise Params
  # NOISE_CUT_START = 0.0
  # NOISE_CUT_STEP  = 0.01
  # NOISE_CUT_NUM   = 9


  @classmethod
  def envSetup(cls):
    if not os.path.exists(cls.JOB_DIR):
      os.mkdir(cls.JOB_DIR)





def setLogger():
  global logger
  if not logger:
    # logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
    logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
    logger = logging.getLogger("")
    log_console = logging.StreamHandler()
    # log_console.setFormatter(log_fmt)
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
  return "jc_%s" % str(uid) if not jckey.startswith('jc_') else uid

def getJC_UID(jckey):
  return jckey[3:] if jckey.startswith('jc_') else jckey



def chmodX(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2 
    os.chmod(path, mode)


def executecmd(cmd):
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode()




#  PROGRAM DEFAULTS FOR INITIALIZATION
#   TODO:  Consolidate & dev config file

schema = dict(  
        JCQueue = [],
        JCComplete = 0,
        JCTotal = 1,
        simSplitParam =  1, 
        anlSplitParam =  1,
        dcdFileList =  [], 
        processed =  0,
        indexSize = 852*DEFAULT.NUM_PCOMP,
        LDIndexList = [],
        converge =  0.)


def initialize(catalog, archive, flushArchive=False):


  #  Create a "seed" job
  logging.debug("Loading schema and setting initial job")
  jcuid = 'SEED'

  seedJobCandidate = dict(
    psf     = jcuid + '.psf',
    pdb     = jcuid + '.pdb',
    parm    = DEFAULT.PARM,
    name    = jcuid,
    temp    = 310,
    runtime = 200000)

  key = getJC_Key(jcuid)
  initParams = {key:seedJobCandidate}

  # Load schema and insert the start job into the queue
  startState = dict(schema)
  startState['JCQueue'] = [key]
  startState['JCTotal'] = 1


  threadnames = ['simmd', 'anlmd', 'ctlmd']

  # TODO:  Job ID Management
  ids = {'id_' + name : 0 for name in threadnames}

  logging.debug("Catalog found on `%s`. Clearing it.", catalog.host)
  catalog.clear()

  logging.debug("Loading initial state into catalog.")
  catalog.save(ids)
  catalog.save(startState)
  catalog.save(initParams)


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
    lshash = RandomBinaryProjections(DEFAULT.HASH_NAME, 3)

    # Store hash configuration in redis for later use
    logging.debug('Storing Hash in Archive')
    redis_storage.store_hash_configuration(lshash)

    # TODO:  Automate Historical Archive (Re)Loading


  archive.stop()
  if os.path.exists('archive.lock'):
    os.remove('archive.lock')


  logging.debug("Initialization complete\n")
