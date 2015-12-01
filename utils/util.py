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


def setArchive():
  global archive
  with open('archive.lock') as a:
    config = a.read().split(',')
  archive = redis.StrictRedis(host=config[0], port=config[1])

def setCatalog():
  global catalog
  with open('catalog.lock') as a:
    config = a.read().split(',')
  catalog = redis.StrictRedis(host=config[0], port=config[1])


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


setArchive()
setCatalog()
eng = startEngine()



SEED = {}
SEED['JCQueue'] = ['2d2839f8', 'dcf5f2a4', '2b40464e', '856b9862', '33dfde68', '830dd954', '8a6480d4', 'db57e966', 'e50cb9d4', '8898c416', '2eebef48', '2c070b80', '7c34b38c', '7e694368', 'e38457a2', '8a6e4d7e', '88dce416', '899b6b20', '34ab7de8', 'd5dadc90', '32d96818', '2dfd9cf6', '894c9b24', '25d9f028', '8bf2104c', '3c95a28e', '8b2b3846', '887e1f2e', '7cee286c', '3bc51380', 'dc285a74', '7cbeb03e', '7d8d71f8', '7e889432', 'd7dc1f58', 'd89abda0', '322231de', '26a8c966', 'da535562', 'd50bdb52', '8a1d4daa', 'd78ef878', '251c8088', '8cc9c83e', 'd97e04a2', '872b1276', '849e3c32', '3a31eb74', 'd6a8684a', '7bfc8932', '2d62cbf6', '87d2b29e', '83d1c652', '3af780d2', '2fbb7cd6', '2e265468', 'e5e12188', 'da95fcfc', 'e44549d0', '8b46ebac', '7db7b574', '277a715a']
SEED['dcdFileList'] = ['/home-1/bring4@jhu.edu/work/jc/d6a8684a/d6a8684a.dcd', '/home-1/bring4@jhu.edu/work/jc/89fe661a/89fe661a.dcd', '/home-1/bring4@jhu.edu/work/jc/2c070b80/2c070b80.dcd', '/home-1/bring4@jhu.edu/work/jc/a3394c6c/a3394c6c.dcd', '/home-1/bring4@jhu.edu/work/jc/5c9d4be6/5c9d4be6.dcd', '/home-1/bring4@jhu.edu/work/jc/d78ef878/d78ef878.dcd', '/home-1/bring4@jhu.edu/work/jc/7cee286c/7cee286c.dcd', '/home-1/bring4@jhu.edu/work/jc/25d9f028/25d9f028.dcd', '/home-1/bring4@jhu.edu/work/jc/dcf5f2a4/dcf5f2a4.dcd', '/home-1/bring4@jhu.edu/work/jc/d50bdb52/d50bdb52.dcd', '/home-1/bring4@jhu.edu/work/jc/2d2839f8/2d2839f8.dcd', '/home-1/bring4@jhu.edu/work/jc/a3fdd0c8/a3fdd0c8.dcd', '/home-1/bring4@jhu.edu/work/jc/da95fcfc/da95fcfc.dcd', '/home-1/bring4@jhu.edu/work/jc/a6229fc8/a6229fc8.dcd', '/home-1/bring4@jhu.edu/work/jc/251c8088/251c8088.dcd', '/home-1/bring4@jhu.edu/work/jc/7db7b574/7db7b574.dcd', '/home-1/bring4@jhu.edu/work/jc/8863b454/8863b454.dcd', '/home-1/bring4@jhu.edu/work/jc/87d2b29e/87d2b29e.dcd', '/home-1/bring4@jhu.edu/work/jc/8898c416/8898c416.dcd', '/home-1/bring4@jhu.edu/work/jc/8966d4c8/8966d4c8.dcd', '/home-1/bring4@jhu.edu/work/jc/87051a14/87051a14.dcd', '/home-1/bring4@jhu.edu/work/jc/2dfd9cf6/2dfd9cf6.dcd', '/home-1/bring4@jhu.edu/work/jc/d5dadc90/d5dadc90.dcd', '/home-1/bring4@jhu.edu/work/jc/5e62cd98/5e62cd98.dcd', '/home-1/bring4@jhu.edu/work/jc/db57e966/db57e966.dcd', '/home-1/bring4@jhu.edu/work/jc/5bc63336/5bc63336.dcd', '/home-1/bring4@jhu.edu/work/jc/8acffa18/8acffa18.dcd', '/home-1/bring4@jhu.edu/work/jc/26a8c966/26a8c966.dcd', '/home-1/bring4@jhu.edu/work/jc/7e889432/7e889432.dcd', '/home-1/bring4@jhu.edu/work/jc/dc285a74/dc285a74.dcd', '/home-1/bring4@jhu.edu/work/jc/a5415158/a5415158.dcd', '/home-1/bring4@jhu.edu/work/jc/2b40464e/2b40464e.dcd', '/home-1/bring4@jhu.edu/work/jc/7c34b38c/7c34b38c.dcd', '/home-1/bring4@jhu.edu/work/jc/8930ae5a/8930ae5a.dcd', '/home-1/bring4@jhu.edu/work/jc/277a715a/277a715a.dcd']
SEED['LDIndexList'] = ['89fe661a', '8863b454', '87d2b29e', '8930ae5a', '5e62cd98', '8acffa18', '87051a14', '8966d4c8', '5bc63336', '5c9d4be6']

def delQueues():
  for q in SEED.keys():
    catalog.delete(q)

def seedQeueus():
  for q in SEED.keys():
    catalog.rpush(q, *tuple(SEED[q]))


def loadDEShawTraj(start, end=-1):
  if end == -1:
    end = start +1
  trajectory = None
  bptidir = DEFAULT.RAW_ARCHIVE + '/'
  for dcdfile in range(start, end):
    f = 'bpti-all-%03d.dcd' % dcdfile if dcdfile<1000 else 'bpti-all-%04d.dcd' % dcdfile
    if not os.path.exists(bptidir + f):
      logging.info('%s   File not exists. Continuing with what I got', f)
      break
    logging.info("LOADING:  %s", f)
    traj = md.load(bptidir + f, top=DEFAULT.PDB_FILE)
    filt = traj.top.select_atom_indices(selection='heavy')
    traj.atom_slice(filt, inplace=True)
    trajectory = trajectory.join(traj) if trajectory else traj
  return trajectory



def covmatrix(traj, numpc=0):
  n_frames = traj.shape[0]
  N_atoms = traj.shape[1]*3
  if numpc == 0:
    numpc = N_atoms
  A = traj.reshape(n_frames, N_atoms)
  a = A - np.mean(A, axis=0)
  cov = np.dot(a.T, a)/n_frames
  return cov

