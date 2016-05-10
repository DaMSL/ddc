#!/usr/bin/env python


from collections import OrderedDict
import bench.db as db
import plot as P
import os
import logging
import numpy as np
import mdtools.deshaw as deshaw
import datatools.datareduce as datareduce
import mdtraj as md
import redis

from datatools.pca import PCAnalyzer, PCAKernel
from core.kdtree import KDTree
import datatools.datareduce as datareduce
import datatools.datacalc as datacalc
import mdtools.deshaw as deshaw




logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

binlist = [(a, b) for a in range(5) for b in range(5)]
sbinlist = [str(i) for i in binlist]
sslist = ['%d_%d'%i for i in binlist]

def getobs(name):
  eid = db.get_expid(name)
  t = db.runquery('select idx, obs from obs where expid=%d order by idx'%eid)
  return [i[1] for i in t]

def by_src_hcube(name):
  expid = db.get_expid(name)
  obslist = getobs(name)
  obs = {}
  jc = db.runquery('select bin, hcube, start, end from jc where expid=%d'%expid)
  for b, h, s, e in jc:
    if b not in obs:
      obs[b] = {}
    if h not in obs[b]:
      obs[b][h] = []
    obs[b][h].extend(obslist[s:e])
  D = {k: {h: {s: 0 for s in sbinlist} for h in obs[k].keys()} for k in obs.keys()}
  for k, v1 in obs.items():
    for h, v2 in v1.items():
      for i in v2:
        if i == '0-D':
          continue
        D[k][h][i] += 1
  return D


def by_src_bin(name):
  expid = db.get_expid(name)
  obslist = getobs(name)
  obs = {}
  jc = db.runquery('select bin, start, end from jc where expid=%d'%expid)
  for b, s, e in jc:
    if b not in obs:
      obs[b] = []
    obs[b].extend(obslist[s:e])
  D = {k: {str((a,b)): 0 for a, b in binlist} for k in obs.keys()}
  total = {k: 0 for k in obs.keys()}
  for k, v in obs.items():
    for i in v:
      total[k] += 1
      D[k][i] += 1
  return D



def src_bin_mat(name):
  data = by_src_bin(name)
  dd = np.zeros(shape=(25,25))
  for s in range(25):
    for d in range(25):
      dd[s][d] = data[sslist[s]][sbinlist[d]]
  return dd

def norm_by_col(dd):
  return np.nan_to_num(dd / np.linalg.norm(dd, axis=-1)[:, np.newaxis]).T

def norm_by_row(dt):
  dd = dt.T
  return np.nan_to_num(dd / np.linalg.norm(dd, axis=-1)[:, np.newaxis])

#P.heatmap(norm_c, sslist, sslist, 'bybin_%s'%name)


def print_bin_table(DD):
  for s in sbinlist:
    print(s, ' '.join(['%5d'%DD[e]['0_4'][s] for e in elist]))

def print_hc_table(DD):
  for s in sbinlist:
    print(s, ' '.join(['%5d'%DD[e]['0_4'][s] for e in elist]))


def high_low_check(r, tbin='(0, 4)'):
  print('Pulling data...')
  obslist=rb.lrange('label:rms', 0, -1)
  ob04 = [i for i, o in enumerate(obslist) if o == tbin]
  traj = backProjection(r, ob04)
  alpha = datareduce.filter_alpha(traj)
  print('Kpca')
  kpca1 = PCAKernel(6, 'sigmoid')
  kpca1.solve(alpha.xyz)
  X = kpca1.project(alpha.xyz)
  print('KDTree1')
  kdtree1 = KDTree(50, maxdepth=4, data=X, method='median')
  hc1 = kdtree1.getleaves()
  print('KDTree2')
  Y = alpha.xyz.reshape(alpha.n_frames, 174)
  kdtree2 = KDTree(50, maxdepth=4, data=Y, method='median')
  hc2 = kdtree2.getleaves()
  hc1k = sorted(hc1.keys())
  hc2k = sorted(hc2.keys())
  s1 = [set(hc1[k]['elm']) for k in hc1k]
  s2 = [set(hc2[k]['elm']) for k in hc2k]
  dd = np.zeros(shape=(len(s1), len(s2)))
  print('     ', ' '.join(hc1k))
  for i, a in enumerate(s1):
    print('  ' +hc1k[i], end=' ')
    for j, b in enumerate(s2):
      n = len(a & b)
      print('%4d'%n, end=' ')
      dd[i][j] = n
    print('\n', end=' ')
  
P.heatmap(dd, hc1k, hc2k, title='High_Low_unif_unbal2', xlabel='High Dim (x,y,z) KD Tree Leaves', ylabel='Low Dim (KPCA) KD Tree Leaves')

  return dd




def microb():
  traj = OrderedDict()
  trans = OrderedDict()
  THETA = 100   # ps 

  print (' BIN', '0000', '0101', '1100', '1111')
  for bb in binlist:
    ab = str(bb)
    print(ab, '%4d'%D['0000'][ab], '%4d'%D['0101'][ab], '%4d'%D['1100'][ab], '%4d'%D['1111'][ab])

  for k, v in D.items():
    for i, n in v.items():
      print()

  last = conf['src_bin']
  last='(1, 2)'
  n = 0

  max_n = (0, last)
  for i, obs in enumerate(tr):
    if obs == last:
      n += 1
      if n > max_n[0]:
        max_n = (n, obs)
    else:
      if n > THETA:
        trn.append((last, obs, i-n, n, max_n[1], max_n[0]))
        # trans[job].append((i-n, n, last, obs))
      n = 0
      last=obs


def backProjection(r, index_list):
        """Perform back projection function for a list of indices. Return a list 
        of high dimensional points (one per index). Check cache for each point and
        condolidate file I/O for all cache misses.
        """

        logging.debug('--------  BACK PROJECTION:  %d POINTS ---', len(index_list))

        # reverse_index = {index_list[i]: i for i in range(len(index_list))}

        source_points = []

        pipe = r.pipeline()
        for idx in index_list:
          # Negation indicates  historical index:
          index = int(idx)
          if index < 0:
            continue
          else:
            pipe.lindex('xid:reference', index)

        # Load higher dim point indices from catalog
        generated_framelist = pipe.execute()

        ref = deshaw.topo_prot  # Hard coded for now

        # Group all Generated indidces by file index 
        groupbyFileIdx = {}
        for i, idx in enumerate(generated_framelist):
          file_index, frame = eval(idx)
          if file_index not in groupbyFileIdx:
            groupbyFileIdx[file_index] = []
          groupbyFileIdx[file_index].append(frame)

        # Dereference File index to filenames
        generated_frameMask = {}
        generated_filemap = {}
        for file_index in groupbyFileIdx.keys():
          filename = r.lindex('xid:filelist', file_index)
          if filename is None:
            logging.error('Error file not found in catalog: %s', filename)
          else:
            key = os.path.splitext(os.path.basename(filename))[0]
            generated_frameMask[key] = groupbyFileIdx[file_index]
            generated_filemap[key] = filename

        # Check cache for generated data points
        bplist = []
        for filename, frames in generated_frameMask.items():
          bplist.append(('sim', generated_filemap[filename], frames))

        source_points = []
        logging.debug('Sequentially Loading %d trajectories', len(bplist))
        for ftype, fileno, framelist in bplist:
          traj = datareduce.load_trajectory(fileno)
          selected_frames = traj.slice(framelist)
          source_points.extend(selected_frames.xyz)

        logging.debug('All Uncached Data collected Total # points = %d', len(source_points))
        source_traj = md.Trajectory(np.array(source_points), ref.top)

        logging.info('--------  Back Projection Complete ---------------')
        return source_traj