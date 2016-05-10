#!/usr/bin/env python


from collections import OrderedDict
import bench.db as db
import plot as P

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