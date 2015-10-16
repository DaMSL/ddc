import logging
import os
import uuid

logger = 0

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