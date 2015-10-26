import logging
import os
import uuid
import subprocess as proc


logger = 0


class DEFAULT:
  NODES = 1
  CPU_PER_NODE = 24
  
  WORKDIR = os.environ['HOME'] + '/bpti/'

  #  TODO:  Set up with Config File
  PSF_FILE = '/home-1/bring4@jhu.edu/bpti/bpti.psf'
  PDB_FILE = '/home-1/bring4@jhu.edu/bpti/bpti.pdb'
  FFIELD   = '/home-1/bring4@jhu.edu/bpti/toppar/par_all22_prot.inp'

  # NUM_NEIGH = 1000
  # MDS_DIM = 250
  NUM_NEIGH = 2000
  MDS_DIM = 500
  MDS_START = 50
  MDS_STEP = 25

  # Noise Params
  NOISE_CUT_START = 0.0
  NOISE_CUT_STEP  = 0.01
  NOISE_CUT_NUM   = 9


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

def chmodX(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2 
    os.chmod(path, mode)


def executecmd(cmd):
  task = proc.Popen(cmd, shell=True,
          stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = task.communicate()
  return stdout.decode()

