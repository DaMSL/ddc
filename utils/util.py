import numpy as np
from numpy import linalg as LA
from collections import namedtuple
from nearpy import Engine
from nearpy.hashes import *  # Pick one, eventually
from nearpy.distances import *
from nearpy.storage import *
import redis
import datetime as dt
import os
import sys
import mdtraj as md
import argparse
import logging
import math
from random import randint
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


archive = None
catalog = None
winsize = 50
slide = 25
home = os.getenv('HOME')
num_pc = 3
label =namedtuple('window', 'time state')


def setArchive(_archive):
  global archive
  archive = _archive

def setCatalog(_catalog):
  global catalog
  catalog = _catalog


def showvals(k):
  for key in sorted(archive.keys(k+'*')):
    print(key.decode(), '%8.4f' % float(archive.get(key).decode()))

def showlist(key):
  for elm in sorted(catalog.lrange(key, 0, -1)):
    print(elm.decode())

def makeUnique(key):
  bag = set()
  for elm in sorted(catalog.lrange(key, 0, -1)):
    bag.add(elm.decode())
  catalog.delete(key)
  for elm in bag:
    catalog.rpush(key, elm)
    print(elm)



def loadLabels(fn):
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

def dcdfile(n):
  return (home + '/work/bpti/bpti-all-' + ('%03d' if n < 1000 else '%04d') + '.dcd') % n


def startEngine():
  archive = redis.StrictRedis(host='login-node03', port=6380)
  redis_storage = RedisStorage(archive)
  config = redis_storage.load_hash_configuration('pcahash')
  if not config:
    logging.error("LSHash not configured")
    sys.exit(0)
  #TODO: Gracefully exit
  # lshash = RandomBinaryProjections(None, None)
  lshash = PCABinaryProjections(None, None, None)
  lshash.apply_config(config)
  eng = Engine(num_pc*454, lshashes=[lshash], storage=redis_storage)
  return eng



setArchive(redis.StrictRedis(port=6380))
setCatalog(redis.StrictRedis())
eng = startEngine()