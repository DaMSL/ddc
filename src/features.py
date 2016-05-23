#!/usr/bin/env python

# from simmd import *
# from anl import *
# from ctl import *
import argparse
import math
import json
import bisect
import datetime as dt


import mdtraj as md
import numpy as np
from numpy import linalg as LA


from core.common import *
import mdtools.deshaw as deshaw
from overlay.redisOverlay import RedisClient
import core.ops as op
from core.kvadt import kv2DArray
from core.slurm import slurm
from core.kdtree import KDTree
import core.ops as ops
import datatools.datareduce as datareduce
from datatools.rmsd import *
from mdtools.simtool import generateNewJC
import mdtools.deshaw as deshaw
import bench.db as db
import plot as P
import redis

DO_COV_MAT = False

DESHAW_LABEL_FILE = 'data/deshaw_labeled_bins.txt'




class ExprAnl:
  def __init__(self, host='localhost', port=6379, adaptive_cent=False):
    self.r = redis.StrictRedis(port=port, host=host, decode_responses=True)
    # self.rms = [np.fromstring(i) for i in self.r.lrange('subspace:rms', 0, -1)]
    self.seq = ['jc_'+x[0] for x in sorted(self.r.hgetall('anl_sequence').items(), key=lambda x:x[1])]
    self.conf=[self.r.hgetall(i) for i in self.seq]
    self.trlist = {}
    self.wells = [[] for i in range(5)]
    self.rmsd = {}
    self.rmsd15 = {}
    self.feal_list = None
    self.checked_rmsd = []
    cent = np.load('../data/init-centroids.npy')
    # cent = np.load('../data/gen-alpha-cartesian-centroid.npy')
    self.centroid = cent
    self.cent15 = []
    for a in range(5):
      for b in range(a, 5):
        if a == b:
          self.cent15.append(np.array(cent[a]))
        else:
          self.cent15.append((np.array(cent[a]) + np.array(cent[b]))/2)
    self.cw = [1., 1., 1., 1., 1.]  
    # self.cw = [.92, .92, .96, .99, .99]  

    if adaptive_cent:
      pass

  def loadtraj(self, tr, first=None):
    if isinstance(tr, list):
      trlist = tr
    else:
      trlist = [tr]
    for t in trlist:
      traj = md.load(self.conf[t]['dcd'], top=self.conf[t]['pdb'])
      # traj.center_coordinates()
      if first is not None:
        traj = traj.slice(np.arange(first))
      self.trlist[t] = datareduce.filter_alpha(traj)

  def ld_wells(self):
    for x, i in enumerate(self.conf):
      if i['origin'] == 'deshaw':
        A, B = eval(i['src_bin'])
        if A == B:
          traj = md.load(self.conf[A]['dcd'], top=self.conf[A]['pdb'])
          traj.center_coordinates()
          alpha = dr.filter_alpha(traj)
          maxf = min(1000, alpha.n_frames)
          for i in alpha.xyz[:maxf]:
            self.wells[A].append(i)

  def load(self, num, first=None):
    for i in range(num):
      _ = self.rms(i, first)

  def rms(self, trnum, noise=False, force=False, first=None):
    if trnum not in self.trlist.keys():
      self.loadtraj(trnum, first)
    if trnum not in self.rmsd.keys() or force:
      if noise:
        # With Noise Filter
        noise = int(self.r.get('obs_noise'))
        dcdfreq = int(self.r.get('dcdfreq'))
        stepsize = int(self.r.get('sim_step_size'))
        nwidth = noise//(2*stepsize)
        noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
        source_pts = np.array([noisefilt(self.trlist[trnum].xyz, i) for i in range(self.trlist[trnum].n_frames)])
      else:
        source_pts = self.trlist[trnum].xyz
      rmsd = [[self.cw[i]*LA.norm(self.centroid[i]-pt) for i in range(5)] for pt in source_pts]
      # rmsd = [[LA.norm(c-i) for c in self.centroid] for i in self.trlist[trnum].xyz]
      self.rmsd[trnum] = rmsd
    return self.rmsd[trnum]

  def rms15(self, trnum):
    if trnum not in self.trlist.keys():
      self.loadtraj(trnum)
    if trnum not in self.rmsd.keys():
      rmsd = [[LA.norm(c-i) for c in self.cent15] for i in self.trlist[trnum].xyz]
      self.rmsd[trnum] = rmsd
    return self.rmsd[trnum]

  def feature_landscape(self, window, var=False):
    """ FEATURE LANDSCAPE Calculation for traj f data pts
    """
    log_prox = op.makeLogisticFunc(1., .5, .5)
    log_reld = op.makeLogisticFunc(1., -1, .5)
    counts = [0 for i in range(5)]
    tup_list = []
    for rms in window:
      counts[np.argmin(rms)] += 1
      tup = []

      # Proximity
      for n, dist in enumerate(rms):
        tup.append(log_prox(dist))
        # tup.append(max(11.34-dist, 0))

      # Additional Feature Spaces
      for a in range(4):
        for b in range(a+1, 5):
          rel_dist = rms[a]-rms[b]
          tup.append(log_reld(rel_dist))
      tup_list.append(tup)

    # Normalize Count
    landscape = [c/sum(counts) for c in counts]

    # Average over the window
    landscape.extend(np.mean(tup_list, axis=0))
    if var:
      variance = [0 for i in range(5)]
      variance.extend(np.std(tup_list, axis=0))
      return np.array(landscape), np.array(variance)
    else:
      return np.array(landscape)

  def feal(self, trnum, winsize=None, var=False):
    feal_list = []
    var_list = []
    N = len(self.rmsd[trnum])
    wsize = N if winsize is None else winsize
    if trnum not in self.rmsd.keys():
      _ = self.rms(trnum)
    for idx in range(0, N, wsize):
      window = self.rmsd[trnum][idx:min(N,idx+wsize)]
      if var:
        f, v = self.feature_landscape(window, var=True)
        var_list.append(v)
      else:
        f = self.feature_landscape(window)
      feal_list.append(f)
    if var:
      return np.array(feal_list), np.array(var_list)
    return np.array(feal_list)

  def feal_atemp(self, rms, scaleto=10):
    """Atemporal (individual frame) featue landscape
    """
    log_prox = op.makeLogisticFunc(scaleto, .5, .5)
    log_reld = op.makeLogisticFunc(scaleto, -3, 0)

    fealand = [0 for i in range(5)]
    fealand[np.argmin(rms)] = scaleto
    tup = []
    # Proximity
    for n, dist in enumerate(rms):
      # tup.append(log_prox(dist))
      maxd = 10.  #11.34
      # tup.append(scaleto*max(maxd-dist, 0)/maxd)
      tup.append(max(maxd-dist, 0))

    # Additional Feature Spaces
    for a in range(4):
      for b in range(a+1, 5):
        rel_dist = rms[a]-rms[b]
        tup.append(log_reld(rel_dist))

    fealand.extend(tup)
    return np.array(fealand)   # Tuple or NDArray?

  def all_feal(self, force=False):
    if self.feal_list is None or force:
      self.feal_list = op.flatten([[self.feal_atemp(i) for i in self.rmsd[tr]] for tr in self.rmsd.keys()])    
    return self.feal_list

  def feal_global(self):
    flist = self.all_feal()
    return np.mean(flist, axis=0)


  def bootstrap(self, size):
    feal = self.all_feal()
    i = 0
    boot = []
    while i+size < len(feal):
      print(i)
      boot.append(op.bootstrap_block(feal[:i+size], size))
      i += size
    return boot

  def draw_feal(self, trnum=None, norm=10):
    if trnum is None:
      flist = self.all_feal()
      agg = np.mean(flist, axis=0)
      P.feadist(agg, 'feal_global_%s' % self.r.get('name'), norm=norm)
    else:
      flist = [self.feal_atemp(i, scaleto=norm) for i in self.rmsd[trnum]]
      agg = np.mean(flist, axis=0)
      P.feadist(agg, 'feal_global_%s_%d' % (self.r.get('name'), trnum), norm=norm)

  def kdtree(self, leafsize, depth, method):
    self.index = []
    allpts = []
    # Recalc for indexing
    flist = [[self.feal_atemp(i) for i in self.rmsd[tr]] for tr in self.rmsd.keys()]
    for trnum, f in enumerate(flist):
      for i, tup in enumerate(f):
        allpts.append(tup[5:])
        self.index.append((trnum, i))
    self.kd = KDTree(leafsize, depth, np.array(allpts), method)
    self.hc = self.kd.getleaves()

  def hcmean(self, hckey=None):
    if hckey is None:
      flist = [[self.feal_atemp(i) for i in self.rmsd[tr]] for tr in self.rmsd.keys()]      
      result = {}
      for k, v in self.hc.items():
        hc_feal = []
        for idx in v['elm']:
          trnum, frame = self.index[int(idx)]
          hc_feal.append(flist[trnum][frame])
        result[k] = np.mean(hc_feal, axis=0)
      return result
    else:
      flist = []
      for idx in v['elm']:
        trnum, frame = self.index[idx]
        flist.append(self.feal_list[trnum][frame])
      return np.mean(flist, axis=0)

    

def calcrmslabel(exp, centroids, load=False):
  cw = [.92, .94, .96, .99, .99]
  rmslabel = []
  if load:
    pipe = exp.r.pipeline()
  for n, tr in sorted(exp.trlist.items()):
    numConf = tr.n_frames
    nwidth = 10
    noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)
    rms_filtered = np.array([noisefilt(tr.xyz, i) for i in range(numConf)])
    # Notes: Delta_S == rmslist
    rmslist_sv = calc_rmsd(rms_filtered, centroids, weights=cw)
    for rms in rmslist_sv:
      A, B = np.argsort(rms)[:2]
      delta = np.abs(rms[B] - rms[A])
      if delta < 0.33:
        sub_state = B
      else:
        sub_state = A
      if load:
        pipe.rpush('label:rms', (A, sub_state))
      rmslabel.append((A, sub_state))
  if load:
    pipe.execute()
  return rmslabel



def draw_windows(rmslist, title='feal_', winsize=10, slide=10):
  feallist = []
  N = len(rmslist)
  fnum=0
  for idx in range(0, N, slide):
    f = get_feal(rmslist[idx:min(N,idx+winsize)])
    P.feadist(f, title+'_%03d' % fnum)
    fnum += 1

def draw_win_flist(feal_list, title='feal_', winsize=10, slide=10):
  N = len(feal_list)
  fnum=0
  for idx in range(0, N, slide):
    f = np.mean(feal_list[idx:min(N,idx+winsize)], axis=0)
    P.feadist(f, title+'_%03d' % fnum)
    fnum += 1

def get_feal(rmslist):
  counts = [0 for i in range(5)]
  tup_list = []
  for rms in rmslist:
    counts[np.argmin(rms)] += 1
    tup = []
    for n, val in enumerate(rms):
      tup.append(max(11.34-val, 0))
    for a in range(4):
      for b in range(a+1, 5):
        tup.append(rms[a]-rms[b])
    tup_list.append(tup)
  landscape = [6*c/sum(counts) for c in counts]
  landscape.extend(np.mean(tup_list, axis=0))
  return np.array(landscape)

def feal_atemp(rms, scaleto=10):
  """Atemporal (individual frame) featue landscape
  """
  log_reld = op.makeLogisticFunc(scaleto, -3, 0)

  fealand = [0 for i in range(5)]
  fealand[np.argmin(rms)] = scaleto
  tup = []
  # Proximity
  for n, dist in enumerate(rms):
    # tup.append(log_prox(dist))
    maxd = 10.  #11.34
    # tup.append(scaleto*max(maxd-dist, 0)/maxd)
    tup.append(max(maxd-dist, 0))

  # Additional Feature Spaces
  for a in range(4):
    for b in range(a+1, 5):
      rel_dist = rms[a]-rms[b]
      tup.append(log_reld(rel_dist))

  fealand.extend(tup)
  return np.array(fealand)   # Tuple or NDArray?

def get_feal_var(rmslist):
  counts = [0 for i in range(5)]
  tup_list = []
  for rms in rmslist:
    counts[np.argmin(rms)] += 1
    tup = []
    for n, val in enumerate(rms):
      tup.append(max(11.34-val, 0))
    for a in range(4):
      for b in range(a+1, 5):
        tup.append(rms[a]-rms[b])
    tup_list.append(tup)
  landscape = [6*c/sum(counts) for c in counts]
  variance = [0 for i in range(5)]
  variance.extend(np.std(tup_list, axis=0))
  landscape.extend(np.mean(tup_list, axis=0))
  return np.array(landscape), variance

def plot_seq(rms_list, title, wsize=10, slide=10, witherror=False):
  N = len(rms_list)
  fnum = 0
  for i in range(0, len(rms_list), slide):
    if witherror:
      feal, var = get_feal_var(rms_list[i:min(i+slide,N)])
      P.feadist(feal, title+'_%04d' % fnum, err=var)
    else:
      feal = get_feal(rms_list[i:min(i+slide,N)])
      P.feadist(feal, title+'_%04d' % fnum)
    fnum += 1

def make_kdtree(feal_list):
  kdtree1 = KDTree(50, maxdepth=4, data=feal_list, method='median')
  hc1 = kdtree1.getleaves()
  for k, v in hc1.items():
    src_pts = []
    for i in v['elm']:
      a, b = tidx[i]
      src_pts.append(rms_val[a][b])
    print(k, np.mean(src_pts, axis=0))


  # a, b = binlist[i]
# for i in range(25):
#   P.plot_seq(frms[i], '/seq%d/24ns_10ps_%d_%d', 10, 10)


def centroid_bootstrap(catalog):
  centfile = settings.RMSD_CENTROID_FILE
  centroid = np.load(centfile)
  cent_npts = [1, 1, 1, 1, 1]  # TBD
  numLabels = len(centroid)
  binlist = [(a, b) for a in range(numLabels) for b in range(numLabels)]
  logging.info("Loaded Starting Centroids from %s", centfile)

  name = catalog.get('name')
  if name is None:
    logging.info('Name not configured in this catalog. Set it and try again')
    return

  # Load/Set initial (current) Configs from Catalog
  if catalog.exists('thetas'):
    thetas = catalog.loadNPArray('thetas')
  else:
    thetas = np.zeros(shape=(numLabels, numLabels))
    thetas[:] = 0.25

  if catalog.exists('transition_sensitivity'):
    trans_factor = catalog.loadNPArray('transition_sensitivity')
  else:
    trans_factor = 0.2
    
  use_gradient = True
  obs_count = {ab: 0 for ab in binlist}
  C_delta = []
  T_delta = []

  # Configure Noise Filter
  noise = int(catalog.get('obs_noise'))
  dcdfreq = int(catalog.get('dcdfreq'))
  stepsize = int(catalog.get('sim_step_size'))
  nwidth = noise//(2*stepsize)
  noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)


  # Get previously Labeled data (or label data IAW current settings)
  eid = db.get_expid(name)
  obslist = [i[0] for i in db.runquery('SELECT obs FROM obs WHERE expid=%d' % eid)]
  jobs = [i[0] for i in sorted(catalog.hgetall('anl_sequence').items(), key=lambda x: x[1])]
  shape = None

  # Initialize lists for pair-wise distances (top 2 nearest centroids)
  diffList  = {}
  transList = {}
  scatPlot  = {}
  for A in range(0, numLabels-1):
    for B in range(A+1, numLabels):
      diffList[(A, B)]  = []
      transList[(A, B)] = []
      scatPlot[(A, B)]  = []
  allScat = []
  # Load trajectories & filter
  obs_global = []

  # Process learning in batches (static batch size to start)
  batch_size = 25
  max_obs = 150
  batch = 0
  while batch <= max_obs:
    logging.info("Procssing Jobs %d - %d", batch, batch+batch_size)
    exec_sim = []
    obs_list = []
    for job in jobs[batch:batch+25]:
      conf = catalog.hgetall('jc_' + job)
      traj = md.load(conf['dcd'], top=conf['pdb'])
      alpha = datareduce.filter_alpha(traj)
      conf['alpha'] = alpha.xyz
      exec_sim.append(conf)
      if shape is None:
        shape = conf['alpha'].shape[1:]

      # xyz_filtered = np.array([noisefilt(alpha.xyz, i) for i in range(alpha.n_frames)])
      rmslist = calc_rmsd(alpha, centroid)
      labels = []
      for rms in rmslist:
        # [cw[i]*LA.norm(pt - centroid[i]) for i in range(5)]
        A, B = np.argsort(rms)[:2]
        delta = np.abs(rms[B] - rms[A])
        if delta < thetas[A][B]:
          sub_state = B
        else:
          sub_state = A
        classify = (A, sub_state)
        labels.append(classify)
        obs_count[classify] += 1

        # For globally updating Thetas
        obs_global.append(classify)
        if A < B:
          diffList[(A, B)].append(rms[A] - rms[B])
        else:
          diffList[(B, A)].append(rms[B] - rms[A])

        for a in range(0, numLabels-1):
          for b in range(a+1, numLabels):
            transList[(a, b)].append(rms[a] - rms[b])
            if (a, a) == classify or (b, b) == classify:
              c = 'b'
            elif (a, b) == classify or (b, a) == classify:
              c = 'g'
            elif a == A or b == A:
              c = 'r'
            else:
              c = 'black'
            scatPlot[(a, b)].append((rms[a] - rms[b], c))
      obs_list.append(labels)

    logging.info('Bin Distribution:')
    grpby = {}
    for llist in obs_list:
      for l in llist:
        if l not in grpby:
          grpby[l] = 0
        grpby[l] += 1
    for k in sorted(grpby.keys()):
      logging.info('%s:  %5d', k, grpby[k])
    for A in range(0, numLabels-1):
      for B in range(A+1, numLabels):
        d = diffList[(A, B)]
        logging.info('Diff list for %d,%d:  %d, %5.2f, %5.2f', A, B, len(d), min(d), max(d))


    # # 6. Apply Heuristics Labeling
    # # logging.debug('Applying Labeling Heuristic. Origin:   %d, %d', srcA, srcB)
    # rmslabel = []
    # 
    # label_count = {ab: 0 for ab in binlist}
    # groupbystate = [[] for i in range(numLabels)]
    # groupbybin = {ab: [] for ab in binlist}


    # For each frame in each traj: ID labeled well pts & build avg op
    logging.info('Selecting observed Well States')
    coor_sum = {i: np.zeros(shape=shape) for i in range(numLabels)}
    coor_tot = {i: 0 for i in range(numLabels)}
    for job, obslist in zip(exec_sim, obs_list):
      # offset = int(job['xid:start'])
      # for i, frame in enumerate(job['alpha']):
      for frame, label in zip(job['alpha'], obslist):
        # A, B = eval(obslist[offset+i])
        A, B = label
        if A != B:
          continue
        coor_sum[A] += frame
        coor_tot[A] += 1

    logging.info('Calculating Avg from following stats:')
    logging.info('   Total Frames: %d', sum([len(sim['alpha']) for sim in exec_sim]))

    # Calculate New Centroids (w/deltas)
    delta = []
    for S in range(numLabels):
      if coor_tot[S] == 0:
        logging.info("   State: %d --- NO OBSERVATIONS IN THIS WELL STATE", S)
        continue
      cent_local = coor_sum[S] / coor_tot[S]
      diff_local = LA.norm(centroid[S] - cent_local)
      update = ((centroid[S] * cent_npts[S]) + (cent_local * coor_tot[S])) / (cent_npts[S] + coor_tot[S])
      delta.append(LA.norm(update - centroid[S]))
      logging.info('   State %d:  NewPts=%5d   Delta=%5.2f   LocalDiff=%5.2f', 
        S, coor_tot[S], delta[-1], diff_local)
      centroid[S] = update
      cent_npts[S] += coor_tot[S]
    centroid_change = np.mean(delta)
    if len(C_delta) > 1:
      rel_change = np.abs((centroid_change - C_delta[-1]) / C_delta[-1])
      logging.info('Centroid Change:  %5.2f   (%5.2f%%)', centroid_change, 100*rel_change)
    C_delta.append(centroid_change)
    batch += batch_size


    # Update Thetas (usig global data ?????)
    delta = []
    for A in range(0, numLabels-1):
      for B in range(A+1, numLabels):
        X = sorted(diffList[(A, B)])
        if len(X) < 100:
          logging.info('Lacking data on %d, %d', A, B)
          continue
        # logging.info('  Total # Obs: %d', len(X))
        crossover = 0
        for i, x in enumerate(X):
          if x > 0:
            crossover = i
            break
        # logging.info('  Crossover at Index: %d', crossover)
        if crossover < 50 or (len(X)-crossover) < 50:
          logging.info('  Lacking local data skipping.')
          continue

        # Find local max gradient  (among 50% of points)
        
        if use_gradient:
          thetas_updated = np.copy(thetas)
          zoneA = int((1-trans_factor) * crossover)
          zoneB = crossover + int(trans_factor * (len(X) - crossover))
          gradA = zoneA + np.argmax(np.gradient(X[zoneA:crossover]))
          gradB = crossover + np.argmax(np.gradient(X[crossover:zoneB]))
          thetaA = X[gradA]
          thetaB = X[gradB]
          thetas_updated[A][B] = np.abs(thetaA)
          thetas_updated[B][A] = np.abs(thetaB)
          tdeltA = np.abs(thetas_updated[A][B] - thetas[A][B])
          tdeltB = np.abs(thetas_updated[B][A] - thetas[B][A])
          delta.append(tdeltA)
          delta.append(tdeltB)
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', A, B, tdeltA, (100*tdeltA/thetas[A][B]))
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', B, A, tdeltB, (100*tdeltB/thetas[B][A]))
          thetas[A][B] = thetas_updated[A][B]
          thetas[B][A] = thetas_updated[B][A]
        else:
          # Classify Fixed Percent of observations as Transitional
          thetas_updated = np.copy(thetas)
          transitionPtA = int((1-trans_factor) * crossover)
          transitionPtB = crossover + int(trans_factor * (len(X) - crossover))
          thetaA = X[transitionPtA]
          thetaB = X[transitionPtB]
          thetas_updated[A][B] = np.abs(thetaA)
          thetas_updated[B][A] = np.abs(thetaB)
          tdeltA = np.abs(thetas_updated[A][B] - thetas[A][B])
          tdeltB = np.abs(thetas_updated[B][A] - thetas[B][A])
          delta.append(tdeltA)
          delta.append(tdeltB)
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', A, B, tdeltA, (100*tdeltA/thetas[A][B]))
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', B, A, tdeltB, (100*tdeltB/thetas[B][A]))
          thetas[A][B] = thetas_updated[A][B]
          thetas[B][A] = thetas_updated[B][A]

    T_delta.append(np.mean(delta))
  P.line(np.array(C_delta), 'Avg_CHANGE_Centroid_Pos_%s' % name)
  P.line(np.array(T_delta), 'Avg_CHANGE_Theta_Val_%s' % name)
  P.bargraph_simple(obs_count, 'Final_Histogram_%s' % name)
  # for k, X in diffList.items():
  #   A, B = k
  #   P.transition_line(sorted(X), A, B, title='-X', trans_factor=.5)
  # for k, X in transList.items():
  #   A, B = k
  #   P.transition_line(sorted(X), A, B, title='-ALL', trans_factor=.5)
  for k, X in scatPlot.items():
    collab = {'b': 'Well', 'g': 'Trans', 'r': 'Primary', 'brown': 'Secondary', 'black': 'None'}
    ptmap = {k: [] for k in collab.keys()}
    ordpts = sorted(X, key = lambda x : x[0])
    for i, tup in enumerate(ordpts):
      y, c = tup
      ptmap[c].append((i, y))
      # if c == 'b' or c == 'g':
      #   ptmap[c].append((i, y))
      # else:
      #   ptmap[c].append((i, 0))
    A, B = k
    P.scat_Transtions(ptmap, title='-%d_%d'%(A,B), size=1, labels=collab)



def load_PCA_Subspace(catalog):

  # HCube leaf size of 500 points
  settings = systemsettings()
  vectfile = settings.PCA_VECTOR_FILE

  logging.info("Loading PCA Vectors from %s", vectfile)
  pc_vect = np.load(vectfile)
  max_pc = pc_vect.shape[1]
  num_pc = min(settings.PCA_NUMPC, max_pc)
  pc = pc_vect[:num_pc]
  logging.info("Storing PCA Vectors to key:  %s", 'pcaVectors')
  catalog.storeNPArray(pc, 'pcaVectors')

  logging.info("Loading Pre-Calculated PCA projections from Historical BPTI Trajectory")
  pre_calc_deshaw = np.load('data/pca_applied.npy')

  # Extract only nec'y PC's
  pts = pre_calc_deshaw.T[:num_pc].T

  pipe = catalog.pipeline()
  for si in pts:
    pipe.rpush('subspace:pca', bytes(si))
  pipe.execute()
  logging.debug("PCA Subspace stored in Catalog")

  logging.info('Creating KD Tree')
  kd = KDTree(500, maxdepth=8, data=pts)
  logging.info('Encoding KD Tree')
  packaged = kd.encode()
  encoded = json.dumps(packaged)
  logging.info('Storing in catalog')
  catalog.delete('hcube:pca')
  catalog.set('hcube:pca', encoded)
  logging.info('PCA Complete')

def labelDEShaw_rmsd(store_to_disk=False):
  """label ALL DEShaw BPTI observations by state & secondary state (A, B)
  Returns frame-by-frame labels  (used to seed jobs)
  """
  settings = systemsettings()
  logging.info('Loading Pre-Calc RMSD Distances from: %s   (For initial seeding)','bpti-rmsd-alpha-dspace.npy')
  rms = np.load('bpti-rmsd-alpha-dspace.npy')
  prox = np.array([np.argsort(i) for i in rms])
  theta = 0.27
  logging.info('Labeling All DEShaw Points.')
  rmslabel = []
  # Only use N-% of all points
  # for i in range(0, len(rms), 100):
  for i in range(len(rms)):
    A = prox[i][0]
    proximity = abs(rms[i][prox[i][1]] - rms[i][A])    #abs
    B = prox[i][1] if proximity < theta else A
    rmslabel.append((A, B))
  if store_to_disk:
    with open(DESHAW_LABEL_FILE, 'w') as lfile:
      for label in rmslabel:
        lfile.write('%s\n' % str(label))
  return rmslabel



def resetAnalysis(catalog):
  """Removes all analysis data from the database
  """
  settings = systemsettings()

  keylist0 =['completesim',
            'label:rms',
            'observe:count',
            'rsamp:rms:_dtype',
            'rsamp:rms:_shape',
            'subspace:rms',
            'xid:filelist',
            'xid:reference  ',
            'subspace:covar:fidx',
            'subspace:covar:pts',
            'subspace:covar:xid']
  keylist1 =['subspace:pca:%d',
            'subspace:pca:kernel:%d',
            'subspace:pca:updates:%d',
            'rsamp:rms:%d',
            'rsamp:rms:%d:full',
            'rsamp:rms:%d:spill']
  keylist2 =['observe:rms:%d_%d']

  logging.info("Clearing the database of recent data")
  count = 0
  for key in keylist0:
    count += catalog.delete(key)

  for key in keylist1:
    for A in range(5):
      count += catalog.delete(key % A)

  for key in keylist2:
    for A in range(5):
      for B in range(5):
        count += catalog.delete(key % (A, B))

  logging.info('Removed %d keys', count)

  jobs = catalog.hgetall('anl_sequence')
  logging.info('RE-RUN THESE JOBS')
  orderedjobs = sorted(jobs.items(), key=lambda x: x[1])
  seqnum = 1
  fileseq = 0
  jobfile = open('joblist.txt', 'w')
  for k, v in orderedjobs:
    if seqnum == 100:
      fileseq += 1
      seqnum = 1
    outline = 'src/simanl.py -a --useid="sw-%04d.%02d" -c %s -w %s' % \
      (fileseq,seqnum,settings.name, k)
    print(outline)
    jobfile.write(outline + '\n')
    seqnum += 1
  jobfile.close()

    
  # def feal_atemp(self, trnum):
  #   """Atemporal (individual frame) featue landscape
  #   """
  #   if trnum in self.feal_list:
  #     return self.feal_list[trnum]
  #   feal_list = []
  #   if trnum not in self.rmsd.keys():
  #     _ = self.rms(trnum)
  #   for rms in self.rmsd[trnum]:
  #     landscape = [0 for i in range(5)]
  #     landscape[np.argmin(rms)] = 1
  #     tup = []
  #     for n, val in enumerate(rms):
  #       tup.append(max(11.34-val, 0))
  #     for a in range(4):
  #       for b in range(a+1, 5):
  #         tup.append(rms[a]-rms[b])
  #     landscape.extend(tup)
  #     feal_list.append(np.array(landscape))
  #   self.feal_list[trnum] = np.array(feal_list) 
  #   return self.feal_list[trnum]


#############################


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', default='default')
  parser.add_argument('--centroid', action='store_true')
  args = parser.parse_args()

  confile = args.name + '.json'
  settings = systemsettings()
  settings.applyConfig(confile)
  catalog = RedisClient(args.name)

  # TO Recalculate PCA Vectors from DEShaw (~30-40 mins at 10% of data)
  # calcDEShaw_PCA(catalog)
  # sys.exit(0)

  if args.centroid:
    centroid_bootstrap(catalog)



# #================
# hcf = {k: np.array([np.array(feal[i]) for i in v['elm']]) for k,v in hc5.items()}
# fmean = {k: np.mean(v, axis=0) for k,v in hcf.items()}

# def find_hc(hclist, index):
#   for k, v in hclist.items():
#     if int(index) in v['elm']:
#       return k
#   return None

# def get_traj(expr, srcindex):
#   for i in expr.conf:
#     if int(i['src_index']) == int(srcindex):
#       return i

# def get_traj(expr, srcindex):
#   for i, con in enumerate(expr.conf):
#     if int(con['src_index']) == int(srcindex):
#       return i


# srcidx = [i['src_index'] for i in unif.conf]
# candid = [i for i in srcidx if i < 91985]
# hc_c = [find_hc(hc5, i) for i in candid]

# hc_map = {}
# for h,i in zip(hc_c, candid):
#   if h not in hc_map:
#     hc_map[h] = []
#   hc_map[h].append(i)

# gh = {}
# for i, h in enumerate(hc_start):
#   if h not in gh:
#     gh[h] = []
#   gh[h].append(i)

# HCCube -> Src traj
# '11100111': [27, 28, 29, 33, 36, 42]




# trlist = [get_traj(unif, i) for i in hc_map['01000011']]

# for k, v in fmean:
#   P.feadist(v, 'feal_hc/hc_'+k)