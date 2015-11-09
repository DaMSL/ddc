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
  PSF_FILE = os.path.join(WORKDIR, 'bpti.psf')
  PDB_FILE = os.path.join(WORKDIR, 'bpti', 'bpti-all.pdb')
  RAW_ARCHIVE = os.path.join(WORKDIR, 'bpti')
  FFIELD   = '/home-1/bring4@jhu.edu/bpti/toppar/par_all22_prot.inp'
  HIST_FILE_DIR  = os.path.join(WORKDIR, 'bpti')
  COORD_FILE_DIR = os.path.join(WORKDIR, 'jc')
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
    if not os.path.exists(cls.COORD_FILE_DIR):
      os.mkdir(cls.COORD_FILE_DIR)





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
  return "jc_%s" % str(uid)

def getJC_UID(jckey):
  return jckey[3:]



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

jcuid = getUID()
jcuid = '1ab98866'
startCoord = jcuid + '.pdb'
shutil.copyfile('bpti.pdb', startCoord)


sampleSimJobCandidate = dict(
  psf     = DEFAULT.PSF_FILE,
  pdb     = startCoord,
  forcefield = DEFAULT.FFIELD,
  runtime = 200000)

jcKey = getJC_Key(jcuid)

initParams = {jcKey:sampleSimJobCandidate}

schema = dict(  
      JCQueue = list(initParams.keys()),
      JCComplete = 0,
      JCTotal = len(initParams),
      simSplitParam =  1, 
      dcdFileList =  [], 
      processed =  0,
      anlSplitParam =  1,
      indexSize = 852*DEFAULT.NUM_PCOMP,
      LDIndexList = [],
      omega =  [0, 0, 0, 0],
      omegaMask = [False, False, False, False],
      converge =  0.)

threadnames = ['simmd', 'anlmd', 'ctlmd']