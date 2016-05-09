import plot as P
import numpy as np
import redis
import os
import mdtraj as md
import pickle
from overlay.cacheOverlay import CacheClient
import mdtools.deshaw as deshaw
import datatools.datareduce as datareduce
from datatools.pca import PCAnalyzer, PCAKernel, PCAIncremental
from core.kdtree import KDTree


# For scraping in the ctl log file (post-parsed) and pre-processing data

#  FOR SCRAPPING post-processed BiPartite Output (TODO: Pull direct from log)
def get_global():
  with open('../ctl.log') as ctllog:
    lines = ctllog.read().strip().split('\n')
  raw = [k.split(',') for k in lines]
  gkeys = raw[0][2:]
  gcounts = raw[1][2:]
  return {gk: gc for gk, gc in zip(gkeys, gcounts)}

def get_global_full():
  with open('../ctl.log') as ctllog:
    lines = ctllog.read().strip().split('\n')
  raw = [k.split(',') for k in lines]
  glob={}
  for i in raw[:4]:
    if i[0] =='global':
      glob[i[1]] = i[2:]
  return glob

def get_local():
  with open('../ctl.log') as ctllog:
    lines = ctllog.read().strip().split('\n')
  raw = [k.split(',') for k in lines]
  local = {}
  for r in raw:
    if r[0] == 'local':
      if r[1] not in local:
        local[r[1]] = {}
      local[r[1]][r[2]] = r[3:]
  return local

def get_local_full():
  with open('../ctl.log') as ctllog:
    lines = ctllog.read().strip().split('\n')
  raw = [k.split(',') for k in lines]
  local={}
  for i in raw:
    if i[0] !='local':
      continue
    if i[1] not in local:
      local[i[1]] = {}
    local[i[1]][i[2]] = i[3:]
  return local



def get_edges():
  local = get_local()
  with open('../ctl.log') as ctllog:
    lines = ctllog.read().strip().split('\n')
  raw = [k.split(',') for k in lines]
  gkeys = raw[0][2:]
  data = {}
  for l in local.keys():
    data[l] = []
    for k in gkeys:
      data[l].append([0 for i in range(len(local[l]['keys']))])
  gmap = {k: i for i, k in enumerate(gkeys)}
  lmap = {b: {k: i for i, k in enumerate(local[b]['keys'])} for b in local.keys()}
  for r in raw:
    if r[0] == 'edge':
      _, b, l, g, n = r
      data[b][gmap[g]][lmap[b][l]] += int(n)
  return data

def plotheatmaps(data, title=''):
  local = get_local_full()
  glob = get_global_full()
  gden = [('%4.0f'%float(i)).lstrip('0') for i in glob['density']]
  gcnt = [int(i) for i in glob['count']]
  max_gden = max([float(i) for i in glob['density']])
  for tbin in data.keys():
    c = np.array(data[tbin])
    # gcnt = np.sum(c, axis=1)
    # lcnt = np.sum(c, axis=0)
    lcnt = [int(i) for i in local[tbin]['count']]
    lden = [float(i) for i in local[tbin]['density']]
    lden_norm = [i / sum(lden) for i in lden]
    lden_scaled = [i * max_gden for i in lden_norm]
    denlab = [('%3.0f'%i) for i in lden_scaled]
    print(local[tbin]['volume'])
    glabels = ['%4d/%4s' % i for i in zip(gcnt,gden)]
    llabels = ['%4d/%4s' % i for i in zip(lcnt,denlab)]
    norm_c = np.nan_to_num(c / np.linalg.norm(c, axis=-1)[:, np.newaxis]).T
    P.heatmap(norm_c, glabels, llabels, title+tbin+'_col')
    d = c.T
    norm_r = np.nan_to_num(d / np.linalg.norm(d, axis=-1)[:, np.newaxis])
    P.heatmap(norm_r, glabels, llabels, title+tbin+'_row')

    combined = (norm_c + norm_r) / 2
    P.heatmap(combined, glabels, llabels, title+tbin+'_combined')
    print(combined)


# To reproject previous sampled data points onto newer reweighted graphs
def reproject_oldata():
  r1 = redis.StrictRedis(port=6390, decode_responses=True)
  cache = redis.StrictRedis(host='bigmem0006', port=6380, decode_responses=True)
  execlist = r1.hgetall('anl_sequence')
  keyorder = ['jc_'+i[0] for i in sorted(execlist.items(), key=lambda x:x[1])]
  # skip first 100 (non-sampled)
  pts = []
  bad_ref = 0
  miss = 0
  for key in keyorder:
    conf = r1.hgetall(key)
    src = int(conf['src_index'])
    ref = r1.lindex('xid:reference', src)
    if ref is not None:
      fileno, frame = eval(ref)
      ckey = 'sim:%s' % conf['name']
      xyz = cache.lindex(ckey, frame)
      if xyz is not None:
        pts.append(pickle.loads(xyz))
      else:
        tr = md.load_frame(conf['dcd'], frame, top=conf['pdb'])
        if len(tr.xyz) == 0:
          miss += 1
        else:
          pts.append(tr.xyz[0])
    else:
      bad_ref += 1
  traj = md.Trajectory(pts, deshaw.topo_prot.top)
  alpha = datareduce.filter_alpha(traj)
  return alpha


# for tbin in binlist:
def reproj_distro():
  local = get_local()
  data  = get_edges()
  r2 = redis.StrictRedis(port=6385, decode_responses=True)
  for tbin in [(2,4), (2,2), (4,4), (4,2), (4,1), (3,1)]:
    print('Processing:', tbin)
    tkey = '%d_%d' % tbin
    # Get Kernel
    kpca_key = 'subspace:pca:kernel:' + tkey
    kpca = PCAnalyzer.load(r2, kpca_key)
    # Get Training Data
    data_raw = r2.lrange('subspace:pca:' + tkey, 0, -1)
    pca_pts = np.array([np.fromstring(x) for x in data_raw])
    kdtree = KDTree(200, maxdepth=8, data=pca_pts, method='middle')
    proj_pts = kpca.project(alpha.xyz)
    biased_hcubes = []
    for i, pt in enumerate(proj_pts):
      biased_hcubes.append(kdtree.probe(pt, probedepth=9))
    if len(data) == 0:
      print('No Raw PCA data points for bin %s.... Going to next bin', str(tbin))
      continue
    counts = {}
    for i in biased_hcubes:
      if i not in counts:
        counts[i] = 0
      counts[i] += 1
    for i in local[tkey]['keys']:
      if i not in counts:
        counts[i] = 0
    print('check')
    cvect = [counts[i] for i in local[tkey]['keys']]
    d = np.array(cvect)/sum(cvect)
    c = np.array(data[tkey])
    lcnt = np.sum(c, axis=0)
    gcnt = np.sum(c, axis=1)
    norm = np.nan_to_num(c / np.linalg.norm(c, axis=-1)[:, np.newaxis])
    # Add biased data as a col
    kpca_cnt = np.array([int(i) for i in local[tkey]['count']])
    kpca_cnt_norm = kpca_cnt / np.sum(kpca_cnt)
    arr = np.vstack((norm, kpca_cnt_norm, d)).T
    rowlist = tuple(gcnt) + ('localKPCA', 'biased',)
    P.bargraph((np.mean(norm, axis=0), d), tkey, ['Reweight', 'Biased'])

def plot_heatmaps():
  P.heatmap(arr, rowlist, lcnt, tkey+'_reproj')
  P.heatmap(norm.T, gcnt, lcnt, tbin+'_Norm_by_Col')
  P.heatmap(d.reshape(53,1), ['biased'], local['2_0']['count'], 'testbiased')

