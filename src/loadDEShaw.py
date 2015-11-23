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


winsize = 50
slide = 25
home = os.getenv('HOME')
num_pc = 3

label =namedtuple('window', 'time state')

def loadLabels(fn):
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

def dcdfile(n):
  return (home + '/work/bpti/bpti-all-' + ('%03d' if n < 1000 else '%04d') + '.dcd') % n

# Eigen Calc
def eigenDecomB(traj):
  n_frames = traj.shape[0]
  n_atoms  = traj.shape[1]
  mean = np.mean(traj, axis=0)
  dist = np.zeros(shape = (n_atoms, n_atoms), dtype=np.float32)
  for A in range(n_atoms):
    # if A % 100 == 0:
    #   logging.info("Atom # %d" % A)
    for B in range(A, n_atoms):
      delta = LA.norm(mean[A] - mean[B])
      dist[A][B] = delta
      dist[B][A] = delta
  logging.info("  Calculating Eigen")
  return LA.eigh(dist)


def pclist2vector(eg, ev, numpc):
  """
  Convert set of principal components into a single vectors
  """
  index = np.zeros(shape=(numpc, len(ev[0])), dtype=ev.dtype)
  for pc in range(numpc):
    np.copyto(index[pc], ev[-pc-1] * eg[-pc-1])
  return index.flatten()

# Split windows & process eigens:
def geteig(num, traj, win, winsize=100):
  logging.info("Window %d - %04d" % (num, win))
  eg, ev = eigenDecomB(traj.xyz[win:win+winsize])
  eg /= LA.norm(eg)
  ev = np.transpose(ev)   # Transpose eigen vectors
  return pclist2vector(eg, ev, num_pc)


def geteigens(num, traj, winsize=100, slide=50):
  result = {}
  for win in range(0, len(traj.xyz) - winsize+1, slide):
    ev = geteig(num, traj, win, winsize=winsize)
    key = '%04d' % num + ':' + '%04d' % win
    result[key] = ev
  return result


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



def probe(eng, n):
  for i in eng.neighbours(n):
    dist = i[2]
    nn = i[1]
    logging.info("%s   %s  state= %s", dist, nn, win[int(nn[:4])].state)
  for i in eng.neighbours(n):
    print(str(i[1]), str(i[2]))

mdload = lambda x: md.load(dcdfile(x), top=home+'/work/bpti/bpti-all.pdb')
loadmd = lambda x: md.load(home+'/work/jc/'+x+'/'+x+'.dcd', top=home+'/work/jc/'+x+'/'+x+'.pdb')

mdslice = lambda x: x.atom_slice(x.top.select_atom_indices(selection='heavy'), inplace=True)
win = loadLabels(home + '/ddc/bpti_labels_ms.txt')
win.append(label(1.031125, 2))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--init', action='store_true')
  parser.add_argument('--check',  action='store_true')
  parser.add_argument('start', nargs='?', type=int)
  parser.add_argument('-n', '--num', type=int, default=50)
  parser.add_argument('--histocalc', action='store_true')
  parser.add_argument('--makebins', action='store_true')
  args = parser.parse_args()

  if args.init:
    logging.info("INITIALIZING. Loading Training Data")

    test_data = []
    for i in range(4100, 4125):
      logging.info('=============     %04d' % i)
      tr = mdload(i)
      mdslice(tr)
      idx = geteigens(i, tr)
      test_data.append(idx)
    # FOR TESTING....

    vects=[]
    for d in test_data:
      vects.extend(d.values())

    pcahash = PCABinaryProjections('pcahash', 10, vects)
    archive = redis.StrictRedis(host='login-node04', port=6380)
    archive.flushall()
    redis_storage = RedisStorage(archive)
    logging.debug('Storing Hash in Archive')
    redis_storage.store_hash_configuration(pcahash)
    sys.exit(0)

  eng = startEngine()

  if args.histocalc:
      archive = redis.StrictRedis(host='login-node03', port=6380)
      redis_storage = RedisStorage(archive)

      # Get all indices
      idx_us = []
      for i in range(2**10):
        idx_us.extend(redis_storage.get_bucket('pcahash', '%010d' % int(bin(i)[2:])))

      idx = sorted(idx_us, key=lambda x: x[1])

      logging.debug('LOADED %d Indices' % len(idx))

          # Initialize Data
      histo = [0]*6
      confInterval = {}
      statesseen = set()
      for index in idx:
        logging.info(index[1])
        statesseen.clear()
        stateCount = [0] * 5
        neigh = eng.neighbours(index[0])
        idx_state = win[int(index[1][:4])].state
        if len(neigh) == 0:
          logging.info("Index `%s` has no neighbours" % index[1])
        elif idx_state != win[int(neigh[0][1][:4])].state:
          logging.info("Index `%s` IS NOT its own NN" % index[1])
        for n in neigh:
          nn_state = win[int(n[1][:4])].state
          statesseen.add(nn_state)
          stateCount[nn_state] += 1
        confInterval[index[1]] = stateCount
        histo[len(statesseen)] += 1

      np.save('histogram', np.array(histo))
      with open('stateObservations', 'w') as obs:
        for key in sorted(confInterval.keys()):
          val = confInterval[key]
          obs.write(key+','+','.join((str(x) for x in val))+'\n')



  if args.makebins:
    theta = 0.1 * math.sqrt(2)
    unit = [[(1 if x==y else 0) for x in range(5)] for y in range(5)]
    conf = {}
    with open('stateObservations') as obs:
      elm = obs.read().split('\n')
      for row in elm:
        d = row.split(',')
        conf[d[0]] = [int(x) for x in d[1:]]
    src = np.zeros(shape=(4125, 5))
    for key in sorted(conf.keys()):
      if len(key) == 0:
        continue
      src[int(key[:4])] = src[int(key[:4])] + conf[key]
    bins = {key: [] for key in [(x,y) for x in range(5) for y in range(5)]}     
    for w, st in enumerate(src):
      # st_norm = st / LA.norm(st)
      s = sum(st)
      if s == 0:
        continue
      # dist = [LA.norm(st_norm - u) for u in unit]
      dist = st / s
      stateA = np.argmax(dist)
      if max(dist) > 0.8: #< theta:
        bins[(stateA, stateA)].append(w)
      else:
        dist[stateA] = 0        
        stateB = np.argmax(dist)
        bins[(stateA, stateB)].append(w)

    # Save Bins for later
    archive = redis.StrictRedis(host='login-node04', port=6380)
    archive.set('transitionBins_magnitude', 5)
    for a in range(5):
      for b in range(5):
        key = 'transitionBins_%d_%d' % (a, b)
        archive.delete(key)
        if len(bins[(a,b)]) > 0:
          archive.rpush(key, *tuple(bins[(a,b)]))

        transitionBins = {}
        for a in range(5):
          # transitionBins[(a, a)] = bins[(a,a)]
          for b in range(5):
            transitionBins[(a, b)] = bins[(a,b)] + bins[(b,a)]
            #TODO:  Store in Archive


  if args.check:
    samp = randint(0, 4124)
    traj = mdload(samp)
    mdslice(traj)
    wind = randint(0, 18) * 50
    key = '%04d' % samp + ':' + '%03d' % wind
    res = geteig(samp, traj, wind)
    for i in eng.neighbours(res):
      dist = i[2]
      nn = i[1]
      logging.info("%s   %s  state= %s", dist, transinn, win[int(nn[:4])].state)
      sys.exit(0)


  if not args.start and args.start != 0:
    logging.debug("No Start Frame")
    sys.exit(0)

  start = args.start
  for i in range(start, start+args.num):
    logging.info('=============     %04d' % i)
    tr = mdload(i)
    mdslice(tr)
    idx = geteigens(i, tr)
    logging.info("Num indices to store = %d", len(idx.keys()))
    for k, v in idx.items():
      eng.store_vector(v, k)


