import mdtraj as md
import numpy as np
import logging
import math
import redis

from numpy import linalg as LA
from scipy import ndimage

import datatools.datareduce as dr
import mdtools.deshaw as deshaw

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

np.set_printoptions(precision=3, suppress=True)
logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


def calc_rmsd(traj, centroid, space='cartesian', title=None, top=None, weights=None):
  """Calculate the RMSD from each point in traj to each of the centroids
  Input passed can be either a trajectory or an array-list object
  Title, if defined, will be used in the output filename
  """
  cw = [1 for i in range(len(centroid))] if weights is None else weights
  # TODO: Check dimenstionality of source points and centroid for given space
  observations = traj.xyz if isinstance(traj, md.Trajectory) else traj
  rmsd = np.zeros(shape=(len(observations), len(centroid)))
  for n, pt in enumerate(observations):
    # TODO:  Check Axis here
    rmsd[n] = np.array([cw[i]*LA.norm(pt - centroid[i]) for i in range(5)])
  return rmsd





def calc_deshaw_centroid_alpha_cartesian(ptlist=None):
  """ Calc RMSD from list of trajectories
  """
  if ptlist is None:
    pts = deshaw.loadpts(skip=100, filt=deshaw.FILTER['alpha'])
  else:
    pts = ptlist
  # sums = np.zeros(shape=(5, 58, 3))
  # cnts = [0 for i in range(5)]
  groupby = [[] for i in range(5)]
  label = deshaw.loadlabels_aslist()
  for idx, pt in enumerate(pts):
      # idx = math.floor(i/10)
      try:
        state = label[idx]
        if state == label[idx-2] == label[idx-1] == label[idx+1] == label[idx+2]:
          # sums[state] += pt
          # cnts[state] += 1
          groupby[state].append(pt)
      except IndexError as err:
        pass # ignore idx errors due to tail end of DEShaw data

  # cent = [sums[i] / cnts[i] for i in range(5)]
  cent = np.zeros(shape=(5, 58,3))
  for i in range(5):
    cent[i] = ndimage.measurements.center_of_mass(np.array(groupby[i]))
  return np.array(cent)



def calc_bpti_centroid(traj_list):
  """Given a trajectory list of frames corresponding to the pre-labeled DEShaw
  dataset, calcualte centroids:
    - Groups frames by states and calcuate average (x,y,z) for all atoms
    This wil exclude any near-transition states and does a best fit of 
    using only in-state points (non-transition ones)
  """
  # Assuming distance space (with alpha-filter), hence 1653 dimensions
  sums = np.zeros(shape=(5, 1653))
  cnts = [0 for i in range(5)]
  for n, traj in enumerate(prdist):
    for i in range(0, len(traj), 40):
      try:
        idx = (n*400)  + (i // 1000)
        state = label[idx]
        # Exclude any near transition frames
        if idx < 3 or idx > 4121:
          continue
        if state == label[idx-2] == label[idx-1] == label[idx+1] == label[idx+2]:
          sums[state] += traj[i]
          cnts[state] += 1
      except IndexError as err:
        pass # ignore idx errors due to tail end of DEShaw data
  cent = [sums[i] / cnts[i] for i in range(5)]
  return (np.array(cent))


def check_bpti_rms(traj_list, centroid, skip=40):
  hit = 0
  miss = 0
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      predicted_state = np.argmin(dist)
      if labeled_state == predicted_state:
        hit += 1
      else:
        miss += 1
  print ('Hit rate:  %5.2f  (%d)' % ((hit/(hit+miss)), hit))
  print ('Miss rate: %5.2f  (%d)' % ((miss/(hit+miss)), miss))


def check_bpti_rms_trans(traj_list, centroid, skip=40):
  traj_list = prdist
  centroid = cent_d
  skip=100
  exact_hit = 0
  total_miss = 0
  det_trans = 0
  total = 0
  tdist = 0.
  wdist = 0.
  tlist = []
  theta = 0.15
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      prox = np.argsort(dist)
      A = prox[0]
      B = prox[1]
      delta = dist[B] - dist[A]
      if delta > theta:
        B = A
      if labeled_state == A == B:
        exact_hit += 1
        wdist += delta
      elif labeled_state == A or labeled_state == B:
        det_trans += 1
        tdist += delta
        tlist.append((idx, (A, B), labeled_state))
      else:
        total_miss += 1
      total += 1


def transplot(r, StateA, StateB):
  if StateA == StateB:
    print("Transitions ONLY")
    return
  if StateA > StateB:
    print("SANITY CHECK: A < B")
    tmp = StateA
    StateA = StateB
    StateB = tmp
  rmslist = [np.fromstring(i) for i in r.lrange('subspace:rms', 0, -1)]
  obslist = r.lrange('label:rms', 0, -1)
  translist=[str((a,b)) for a in [StateA, StateB] for b in [StateA, StateB]]
  plotlist = {t: [] for t in translist}
  potential_trans = []
  # diff = []
  # ratio = []
  # invrat = []
  # adjrat = []
  for rms, obs in zip(rmslist, obslist):
    if obs not in translist:
      continue
    proxA, proxB = np.argsort(rms)[:2]
    d = rms[StateA] - rms[StateB]
    if proxA in [StateA, StateB] and proxB in [StateA, StateB]:
      potential_trans.append(d)      
    # diff.append(d)
    # a = min(rms[StateA],rms[StateB])
    # b = max(rms[StateA],rms[StateB])
    # polarity = -1 if d < 0 else 1
    # ratio.append(rms[StateA]/rms[StateB])
    # adjrat.append(polarity * (a/b))
    # invrat.append(polarity * (b/a))
    # plotlist[obs].append((rms[StateA], rms[StateB]))
  return sorted(potential_trans)

  # P.line(np.array(sorted(ratio)), 'trans_2_3_ratio')
  # P.line(np.array(sorted(adjrat)), 'trans_2_3_adjrat')
  # P.line(np.array(sorted(invrat)), 'trans_2_3_invrat')
  # P.line(np.array(sorted(diff)), 'trans_2_3_diff')

  # P.scats(plotlist, 'RMSD_2_3')    

# for a in range(4):
#   for b in range(a+1, 5):
#     D = rmsd.transplot(r, a, b)
#     P.transition_line(np.array(D), a, b)

def show(x, n=10):
  for i in x[:n]:
    print (' '.join(['%7.2f'%p for p in i]))

# centroid = np.load('../data/gen-alpha-cartesian-centroid.npy')

# cent = [np.mean(np.array(i), axis=0) for i in well]

# Adaptive Centroids using modified KMeans like algorithm
def adapt_cent(pts, labels, init_cent=None, init_wgt=None, n_iter=50):
  N = len(pts)
  K = max(labels) + 1
  rms = np.zeros(shape=(N, K))
  k_count = np.bincount(labels)
  cur_score = np.zeros(K)
  if init_cent is None:
    grouped = [[] for i in range(K)]
    for i, L in enumerate(labels):
      grouped[L].append(pts[i])
    centroid = [np.mean(P, axis=0) for P in grouped]
  else:
    centroid = init_cent
  if init_wgt is None:
    weight = np.ones(K)
  else:
    weight = init_wgt
  for i in range(n_iter):
    score = np.zeros(K)
    cent_adj = [[] for i in range(K)]
    for n, x in enumerate(pts):
      for k, c in enumerate(centroid):
        rms[n][k] = weight[k] * LA.norm(x-c)
      k_near = np.argmin(rms[n])
      k_actual = labels[n]
      if k_near == k_actual:
        score[k_actual] += 1
      else:
        # Interpolate to adjust cent' for this pt
        d_near = rms[n][k_near]
        d_actual = rms[n][k_actual]
        cent_p = x + (d_near / d_actual) * (centroid[k_actual] - x)
        cent_adj[k_actual].append((d_actual, cent_p))
    for k in range(K):
      new_centlist = sorted(cent_adj[k], key=lambda x: x[0])
      candid = [c[1] for c in new_centlist[:10]]
      centroid[k] = np.mean(candid, axis=0)
    if i % 1 == 0 or i == n_iter-1:
      print ('%4d -'%i, ' '.join(['%4.0f' % i for i in score] ),  '[%d]' % sum(score))
  return centroid


def adapt_wgts(pts, labels, init_cent=None, init_wgt=None, n_iter=10):
  N = len(pts)
  K = max(labels) + 1
  if init_cent is None:
    grouped = [[] for i in range(K)]
    for i, L in enumerate(labels):
      grouped[L].append(pts[i])
    centroid = [np.mean(P, axis=0) for P in grouped]
  else:
    centroid = init_cent
  if init_wgt is None:
    weight = np.ones(K)
  else:
    weight = init_wgt
  k_count = np.bincount(labels)
  for i in range(n_iter):
    score = np.zeros(K)
    wgt_adj = []
    # cent_adj = [[] for i in range(K)]
    for n, x in enumerate(pts):
      w_n = np.ones(K)
      rms_actual = np.array([LA.norm(x-c) for c in centroid])
      rms_adj = rms_actual * weight
      k_near = np.argmin(rms_adj)
      k_actual = labels[n]
      if k_near == k_actual:
        score[k_actual] += 1
      else:
        d_near = rms_actual[k_near]
        d_actual = rms_actual[k_actual]
        w_n[k_actual] = weight[k_actual] * d_near / d_actual
      wgt_adj.append(w_n)  
    weight = np.mean(wgt_adj, axis=0)
    if i % 1 == 0 or i == n_iter-1:
      print ('%4d -'%i, ' '.join(['%4.0f' % i for i in score] ),  '[%d]' % sum(score))
  print('Final weights:')
  print (' '.join(['%4.3f' % i for i in weight] ))
  return weight


def adapt_combine(pts, labels, n_iter=10):
  N = len(pts)
  K = max(labels) + 1
  grouped = [[] for i in range(K)]
  for i, L in enumerate(labels):
    grouped[L].append(pts[i])
  centroid = [np.mean(P, axis=0) for P in grouped]
  weight  = np.ones(K)
  k_count = np.bincount(labels)
  for i in range(n_iter):
    score = np.zeros(K)
    wgt_adj = [[] for i in range(K)]
    cent_adj = [[] for i in range(K)]
    w_nk = np.ones(shape=(N,K))
    for n, x in enumerate(pts):
      rms_actual = np.array([LA.norm(x-c) for c in centroid])
      rms_adj = rms_actual * weight
      k_near = np.argmin(rms_adj)
      k_actual = labels[n]
      if k_near == k_actual:
        score[k_actual] += 1
      else:
        d_near = rms_actual[k_near]
        d_actual = rms_actual[k_actual]
        wgt_adj[k_actual].append(d_near / d_actual)
        cent_adj[k_actual].append(x + (d_near / d_actual) * (centroid[k_actual] - x))
      # wgt_adj.append(w_n)
    centroid = [np.mean(C, axis=0) for C in cent_adj]
    weight = [np.mean(i, axis=0) for i in wgt_adj]
    print (' '.join(['%4.0f' % i for i in score] ))
  print('Final weights:')
  print (' '.join(['%4.3f' % i for i in weight] ))


def adapt_centroid(pts, labels, init_cent=None, init_wgt=None, n_iter=50):
  N = len(pts)
  K = max(labels) + 1
  rms = np.zeros(shape=(N, K))
  k_count = np.bincount(labels)
  cur_score = np.zeros(K)
  if init_cent is None:
    grouped = [[] for i in range(K)]
    for i, L in enumerate(labels):
      grouped[L].append(pts[i])
    centroid = [np.mean(P, axis=0) for P in grouped]
  else:
    centroid = init_cent
  if init_wgt is None:
    weight = np.ones(K)
  else:
    weight = init_wgt
  for i in range(n_iter):
    score = np.zeros(K)
    group = [[] for i in range(K)]
    for n, x in enumerate(pts):
      rms = weight * np.array([LA.norm(x-c) for c in centroid])
      k_near = np.argmin(rms)
      k_actual = labels[n]
      if k_near == k_actual:
        score[k_actual] += 1
        group[k_actual].append((rms[k_near], x))
    for k in range(K):
      new_centlist = sorted(group[k], key=lambda x: x[0])
      candid = [c[1] for c in new_centlist]
      centroid[k] = np.mean(candid, axis=0)
    if i % 1 == 0 or i == n_iter-1:
      print ('%4d -'%i, ' '.join(['%4.0f' % i for i in score] ),  '[%d]' % sum(score))
  return centroid


# centtraj = [md.Trajectory(c, alpha_pdb.top) for c in cent]

# mdrms = np.array([md.lprmsd(a, c) for c in centtraj]).T
# mind = [np.argmin(i) for i in mdrms]
# np.bincount(mind)

# for a in range(5):
#   for b in range(5):
#     d = LA.norm(cent[a]-cent[b])
#     print(a, b, '%5.2f'%d)
# rms_a=[]
# for v in conf:
#   for i in rms_all[int(v['xid:start'])+1:int(v['xid:end'])+1]:
#     rms_a.append(i)

# rms_b=[]
# for t in alpha:
#   for i in t.xyz:
#     rms_b.append([LA.norm(i-c) for c in centroid])

# rms_c = []
# for t in alpha:
#   t.center_coordinates(mass_weighted=False)
#   for i in t.xyz:
#     rms_c.append([LA.norm(i-c) for c in centroid])

# with open('foo2', 'w') as w:
#   for x in rms_c:
#     _=w.write(' '.join(['%7.2f'%k for k in x]) + '\n')

# rms_e = []
# for t in tr1:
#   # t.center_coordinates(mass_weighted=True)
#   a = dr.filter_alpha(t)
#   for i in a.xyz:
#     rms_e.append([LA.norm(i-c) for c in centroid])

# with open('foo_e', 'w') as w:
#   for x in rms_d:
#     _=w.write(' '.join(['%7.2f'%k for k in x]) + '\n')

# cent15 = []
# for a in range(4):
#   for b in range(a, 5):
#     if a == b:
#       cent15.append(np.array(cent[a]))
#     else:
#       cent15.append((np.array(cent[a]) + np.array(cent[b]))/2)


if False:
  exp = rmsd.ExprAnl(port=6382)
  exp.loadtraj(list(range(0, 200)))
  rmslist = []
  for k in exp.trlist.keys():
    for i in exp.trlist[k].xyz:
      rmslist.append([LA.norm(c-i) for c in exp.cent15])

  # rmsnorm = []
  # for rms in rmslist:
  #   rmsnorm.append(np.array(rms)/np.sum(rms))
  feal = np.array([[max(0, 16 - i)/16 for i in rms] for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_dif16_15')
  feal = np.array([[max(0, 11.34 - i)/11.34 for i in rms] for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_dif11_15')
  feal = np.array([np.array(rms)/sum(rms) for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_norm_15')
  feal = np.array([.33-np.array(rms)/sum(rms) for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_ninv_15')
  feal = np.array([[max(0, 11.34 - i)/11.34 for i in rms] for rms in rmslist[:12000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr12_dif11_15')

  feal = np.array([.33-np.array(rms)/sum(rms) for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_ninv_15')

  feal = np.array([[i*i for i in rms] for rms in rmslist[:1000]])
  P.feadist(list(np.mean(feal, axis=0)), 'feal_tr0_sqr_15')

  feal = np.array([.33-np.array(rms)/sum(rms) for rms in rmslist[:1000]])
  P.feadist(list(np.mean(reld[:10000], axis=0)), 'feal_tr10_pw')

  for x in range(5):
    feal = np.array([[max(0, np.power((11.34 - i), x)/np.power(11.34,x)) for i in rms] for rms in rmslist[:12000]])
    P.feadist(list(np.mean(feal, axis=0)), 'feal_tr12_p%dinvn_15'%x)

  rmslist = []
  for k in exp.trlist.keys():
    for i in exp.trlist[k].xyz:
      rmslist.append([LA.norm(c-i) for c in c15])


  rms15 = [exp.get_rms(i) for i in range(200)]
  rms5  = [exp.get_rms5(i) for i in range(200)]

  rms_val = rms5
  nCent = len(rms_val[0])

  w=[]
  tidx=[]
  window_size = 800
  for tnum, traj in enumerate(rms5):
      N = len(traj)
      if N < window_size:
        continue
      for i in range(0, N, window_size):
        x = np.array(traj[i:i+window_size])
        if len(x) == window_size:
          w.append(x)
          tidx.append((tnum, i))

  wind = np.array(w)

  kdtree1 = KDTree(50, maxdepth=4, data=wind, method='median')
  hc1 = kdtree1.getleaves()
  for k, v in hc1.items():
    src_pts = []
    for i in v['elm']:
      a, b = tidx[i]
      src_pts.append(rms_val[a][b])
    print(k, np.mean(src_pts, axis=0))


  var=[]
  for traj in rms5:
      N = len(traj)
      for i in range(0, N, 100):
        end = min(i+100, N)
        var.append(np.std(traj[i:end], axis=0))

  kdtree2 = KDTree(50, maxdepth=4, data=np.array(var), method='median')
  hc2 = kdtree2.getleaves()
  for k, v in hc2.items():
    print(k, np.mean([var[i] for i in v['elm']], axis=0))

  from sklearn.cluster import KMeans
  km1 = KMeans(5)
  km1.fit([np.mean(i, axis=0) for i in w])
  L=km1.predict(np.mean(i, axis=0))
  X = np.mean(i, axis=0)


  for k in diff.keys():
    reld2 = np.array([v for k,v in sorted(diff.items())])
  s = np.random.randint(140000) + 25000; showhist(reld2[s:s+100], title='win_100')
  showhist(reld2[s-50:s-49], title='win_f-50')
  showhist(reld2[s+100:s+151], title='win_f+50')
  for i in range(0, 22000, 5000):
    showhist(reld2[i:i+1000], title='seq24/src4_1_fr_%d'% (i//5000))
  for i in range(0, 400, 10):
    showhist(reld2[s2+i:s+i+10], title='seq1/framestep_10_%03d'%(i//10))
  for i in range(0, 1000, 10):
    showhist(reld2[i:i+10], title='seq1/framestep_10_%03d'%(i//10))
  for i in range(0, 500, 1):
    showhist(reld2[i:i+1], title='seq24/src4_1_fr_%03d'%(i))
  diff = {}
  reld2 = []
  for a in range(4):
    for b in range(a+1, 5):
      diff['%d-%d'%(a,b)] = []
  for a in range(5):
    diff['-%d-'%(a)] = []

def showhist(data, agg=np.mean, title=''):
    P.feadist(counts, title)

def draw_windows(rmslist, title='feal_', winsize=10, slide=10):
  feallist = []
  N = len(rmslist)
  fnum=0
  for start in range(0, N, slide):
    f = get_feal(rmslist[start:min(N,start+slide)])
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
  landscape = [5*c/sum(counts) for c in counts]
  landscape.extend(np.mean(tup_list, axis=0))
  return np.array(landscape)

