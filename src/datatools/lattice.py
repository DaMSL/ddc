import itertools as it
import string
import numpy as np
import numpy.linalg as LA
import scipy.stats as stats
import copy
import threading
import logging
import sys
 
from datetime import datetime as dt
from sortedcontainers import SortedSet, SortedList
from collections import OrderedDict, deque, defaultdict

ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
k_domain = label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek
 
frommask = lambda obs: ''.join([label_domain[i] for i, x in enumerate(obs) if x])
tomask = lambda key, size: [(1 if i in key else 0) for i in domain[:size]]
toidx = lambda key: [i for i,f in enumerate(k_domain) if f in key]
tok   = lambda x: ''.join(sorted([k_domain[i] for i in x]))
tofz  = lambda x: frozenset([k_domain.index(i) for i in x])
# toidx = lambda x: [ord(i)-97 for i in x]
fromk = lambda x: np.array([1 if i in x else 0 for i in k_domain])
fromm = lambda x: ''.join(sorted([k_domain[i] for i,m in enumerate(x) if m == 1]))
 

key_meet = lambda a, b: ''.join(sorted(set(a).intersection(set(b))))
key_join = lambda a, b: ''.join(sorted(set(a).union(set(b))))

deref = lambda T, key: np.where(np.logical_and(T[:,key], True).all(1))[0]


factorial = lambda x: 1 if x==1 else x * factorial(x-1)


class ProgressBar(object):
  def __init__(self, total, interval=100):
    self.total    = total
    self.interval = interval
    self.cnt      = 0
    self.DEBUG    = False
    try:
      if sys.ps1:
        self.DEBUG = True
        print ('\r[{0:<{width}}] {1:4.1f}%%'.format('', 0, width=self.interval), end='')
    except AttributeError as nointerp:
      self.interval = 4

    self.tick = max(1, total//self.interval)
    self.start = dt.now()


  def incr(self):
    self.cnt += 1
    if self.cnt % self.tick == 0:
      tottime = (dt.now() - self.start).total_seconds()
      prog = 100*self.cnt/self.total
      remain = (((100*tottime) / prog) - tottime) / 60.
      if self.DEBUG:
        print ('\r[{0:<{width}}] {1:4.1f}%%  ({2:5.2f} min remain)'.format('#'*int(prog), prog, remain, width=self.interval), end='')
      else:
        logging.info('  Progress: {0:4.1f}%%  ({1:5.2f} min remain)'.format(prog, remain))

class Lattice(object):

  nbins    = 20
  brange = (4, 8)

  def __init__(self, event, feature_set, cutoff=7, support=50):
    self.E = event
    self.Kr = feature_set
    self.support = support
    self.cutoff = cutoff

  def set_dlat(self, dlat, Ik):  
    self.dlat = dlat
    self.Ik = Ik

  def set_fis(self, mfis, lfis):
    self.max_fis = mfis
    self.low_fis = lfis

  def _CM(self):
    try:
      return (self.E[:,self.Kr] < self.cutoff)
    except IndexError as err:
      print('KR: ', self.Kr, type(self.Kr))
      print('E : ', self.E.shape)
      raise

  def maxminer(self):
    cm = self._CM()
    self.max_fis, self.low_fis = maxminer(cm, self.support)

  def derive_lattice(self):
    cm = self._CM()
    self.dlat, self.Ik = derived_lattice(self.max_fis, self.E[:,self.Kr], cm)


  def merge(self, other):
    N, K = len(self.E), len(self.Kr)
    delta_CM = other._CM()
    CM = self._CM()

    # childlist = defaultdict(list)
    # for n, plist in dlat.items():
    #   for p in plist.keys():
    #     childlist[p].append(n)
    
    # 1. Expand all MFIS in delta lattice and flag each existing itemset for update
    logging.info('Expand delta Max-Frequent Itemsets')
    progress = ProgressBar(len(other.max_fis))
    nodelist = set()
    updatelist = defaultdict(set)
    cag = set()
   
    for mfis in other.max_fis:
      # Start with each MF itemset as the root
      cag.add(mfis)

      # Expand and add new subnodes to candidate group
      while len(cag) > 0:
        node = cag.pop()
        if node in nodelist:
          continue

        # Add to list of FIS nodes
        nodelist.add(node)

        # Stop at keylen of 1
        if len(node) == 1:
          continue

        # Flag all children for this node to update EMD calculations
        for i in node:
          cag.add(node - {i})

      progress.incr()

    # 2. Update itemset (or add if new) from bot to top
    logging.info('\nMerge all delta event lists for observed frequent itemsets.')
    progress = ProgressBar(len(nodelist))
    for node in sorted(nodelist, key=lambda x: len(x)):
      nkey = tok(node)

      # Update index numbers for delta
      delta_iset = [N + i for i in other.Ik[nkey]]

      # Node already exists: Flag all parent edges for EMD update
      if nkey in self.Ik:
        cur_iset = self.Ik[nkey]
        for par in self.dlat[nkey].keys():
          updatelist[node].add(par)
          # if max(cur_iset) >= 90050:
          #   logging.error('\nERROR on cur___ISET: ', 
          #     nkey, type(cur_iset), type(delta_iset), max(cur_iset), max(delta_iset),
          #     CM.shape, self.E.shape)
          #   return

      # New node: get items and identify its children for EMD update
      else:
        cur_iset = np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]
        for k in node:
          child = node - {k}
          if tok(child) in self.Ik:
            updatelist[child].add(nkey)

      # Merge itemsets
      self.Ik[nkey] = np.concatenate((cur_iset, delta_iset))
      # if max(self.Ik[nkey]) >= 90050:
      #   logging.error('\nERROR on ISET Merge: ', 
      #     max(delta_iset), nkey, type(cur_iset), type(delta_iset), max(cur_iset), max(delta_iset),
      #     CM.shape, self.E.shape)
      #   return
      progress.incr()      

    # 3. Merge event lists & update contact matrix
    logging.info('\nMerge Event Matrix.')
    self.E = np.vstack((self.E, other.E))
    logging.info('Complete: %s', str(self.E.shape))
    CM = self._CM()

    # 4. Merge Low Support & check threshold
    logging.info('Merge Low count itemsets and identify newly supported frequent itemsets.')
    progress = ProgressBar(len(other.low_fis))
    for node, z in sorted(other.low_fis.items(), key=lambda x: len(x)):
      if node not in self.low_fis:
        self.low_fis[node] = z
      else:
        self.low_fis[node] += z

      if self.low_fis[node] >= self.support:
        self.Ik[nkey] = np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]  
        for k in node:
          child = node - {k}
          if tok(child) in Ik:
            updatelist[child].add(nkey)

        nodelist.add(node)
        del self.low_fis[node]
      progress.incr()      

    # 5. Pre-Process all 1D histograms
    Ik_update = {}
    for k, v in updatelist.items():
      key = tok(k)
      if key not in Ik_update:
        Ik_update[key] = self.Ik[key]
      for p in v:
        if p not in Ik_update:
          Ik_update[p] = self.Ik[p]

    H = histograms(Ik_update, self.E[:,self.Kr], Lattice.nbins, Lattice.brange)

    # 5. Update EMD for all flagged edges:
    logging.info('\nUpdate EMD for all flagged nodes')
    prog = ProgressBar(len(updatelist))
    for node, parentlist in updatelist.items():
      nkey = tok(node)

      #  Calculated distribution delta along every edge
      for parent in parentlist:
        flow = np.zeros(len(node))

        # Calculate bin-by-bin difference
        delta = H[nkey] - H[parent][[parent.find(i) for i in nkey]]

        # Flow is tracked along each dimension separately
        flow = np.zeros(len(node))
        for k in range(len(delta)):

          # Calculate flow from 0 to n-1 and add/subtract flow to neighboring bin
          flow_k = 0
          for i in range(Lattice.nbins-1):
            flow_k     += delta[k][i]
            delta[k][i+1] += delta[k][i]
          flow[k] = flow_k

        # EMD is sqrt of sums of squares for each 1D distribution delta
        self.dlat[nkey][parent] = np.sqrt(np.sum(flow**2))
      prog.incr()

  def solve_cluster(self):
    cm = self._CM()
    self.cluster = itermergecluster(self.dlat, cm, self.E[:,self.Kr], self.Ik)

  def sample(self):
    # TODO: Use whole distro or only part(?)
    N = len(self.E)
    clusters = self.cluster
    centroid = {k: self.E[v].mean(0) for k,v in clusters.items()}
    variance = {}
    for k,v in clusters.items():
        cov = np.cov(self.E[v][:,self.Kr].T)
        ew, ev = LA.eigh(cov)
        variance[k] = np.sum(ew)

    score = [None for i in range(N)]
    samplist = []

    clusterlist = []
    clusterscore = np.zeros(len(clusters))
    elmscore = [[] for i in range(len(clusters))]
    print("  TOTAL # CLUSTERS   :   ", len(clusters))
    total_var = np.sum([k for k in variance.values()])
    for clnum, (k, iset) in enumerate(clusters.items()):

      # THE CLUSTER SCORE
      sc_var  = 1 / np.sqrt(variance[k])
      sc_size = 1 - (len(iset) / N)
      clscore = sc_var + sc_size


      clusterscore[clnum] = max(0, clscore)
      elmlist = []

      for i in iset:
        dist_cent = LA.norm(self.E[i] - centroid[k])
        accuracy  = variance[k]
        rarity    = len(iset)

        # ELEMENT SCORE
        score[i] = (k, N / (dist_cent * accuracy * rarity))
        elmlist.append((i, dist_cent))

      elmscore[clnum] = sorted(elmlist, key= lambda x: x[1])

      clusterlist.append((clnum, k, len(iset), variance[k], clscore, sc_size, sc_var))

    for i in clusterlist: #sorted(clusterlist, key =lambda x : x[2], reverse=True):
      print('%3d.  %-18s%5d  %6.2f : %7.2f  (%5.2f %5.2f)' % i)

    sampidx = np.zeros(len(clusters), dtype=np.int16)
    pdf = clusterscore / np.sum(clusterscore)
    print('\nPDF:  ', pdf)
    print('SAMPLE OF 20 CANDIDATES.....')
    for i in range(20):
      clidx = int(np.random.choice(len(pdf), p=pdf))
      elm, dist = elmscore[clidx][sampidx[clidx]]
      print(' %2d.  Clu  ( %d )       evidx %4d' % (i, clidx, elm))
      sampidx[clidx] += 1    


class LatticeNode:
  def __init__ (self, iset, plist=set()):
    self.iset = iset
    self.parents = plist

  def add_parent (self, key):
    self.parents.keys.add(key)


def get_iset (key, cIk):
  iset = set([i for i in cIk[key].iset])
  if len(cIk[key].parents) == 0:
    return iset
  nlist = defaultdict(set)
  cur_len = len(key) + 1
  nlist[cur_len] =set([i for i in cIk[key].parents])
  while len(nlist) > 0:
    if len(nlist[cur_len]) == 0:
      del nlist[cur_len]
      cur_len += 1
    else:
      parent = nlist[cur_len].pop()
      for i in cIk[parent].parents:
        nlist[cur_len+1].add(i)
      iset |= cIk[parent].iset
  return iset


def compress_lattice(dlat, Ik):
  keylist = sorted(Ik, key=lambda x: (len(x), x), reverse=True)
  compIk = {}
  for key in keylist:
    if len(dlat[key]) == 0:
      compIk[key] = LatticeNode(set(Ik[key]))
    else:
      plist = set(dlat[key].keys())
      iset  = set(it.chain(*[Ik[i] for i in plist]))
      compIk[key] = LatticeNode(set(Ik[key]) - iset, plist)
  return compIk



 
def unique_events(A):
  '''Groups all rows by similarity. Accepts a 1-0 Contact Matrix and 
  returns a mapping from alpha-key to a list of indices with the corresponding
  key'''
  U = defaultdict(list)
  for idx, row in enumerate(A):
    U[frommask(row)].append(idx)
  return U

def find_supersets(A):  
  ''' A is assumed to be a list of keys (TODO: perform check)'''
  klist = [set(i) for i in sorted(A, key=lambda x: (len(x), x))]
  subsets = []
  N = len(klist)
  for i in range(N-1):
    for j in range(i+1, N):
      if klist[i] < klist[j]:
        subsets.append(i)
        break
  supersets = [i for i in range(N) if i not in subsets]
  return supersets
 
def reduced_feature_set(A, theta=.02):
  '''Reduces te feature set (columns) by eliminating all "trivial" features.
  Accepts a 1-0 Contact Matrix and identifies all columns whose values are
  (1-theta)-percent 1 or 0. Theta represents a noise threshold. E.g. if theta
  is .02, returned feature set will include column k, if more than 2% of the 
  values are 0 and more than 2% are 1.'''
  (N, K), count  = A.shape, A.sum(0)
  T = theta*N
  trivial = set(it.chain(np.where(count < T)[0], np.where(count > 2*(N-T))[0]))
  first_order = sorted(set(range(K)) - trivial)

  pair_wise = set()
  for i in range(len(first_order)-1):
    for j in range(i+1, len(first_order)):
      a, b = first_order[i], first_order[j]
      corr = (A[:,a]==A[:,b]).sum()
      if corr < T or corr > (N-T):
        pair_wise.add(a)
        pair_wise.add(b)
  return sorted(set(first_order) - pair_wise) 


def reduced_feature_set2(D, cutoff=7, theta=.05, maxk=25):
  '''Reduces te feature set (columns) by eliminating all "trivial" features.
  Accepts a 1-0 Contact Matrix and identifies all columns whose values are
  (1-theta)-percent 1 or 0. Theta represents a noise threshold. E.g. if theta
  is .02, returned feature set will include column k, if more than 2% of the 
  values are 0 and more than 2% are 1.'''

  N, K = D.shape
  T = theta*N
  low_cut, hi_cut = T,  N - 2*T

  A = (D < cutoff).astype(int)
  
  ranged = D.max(0) - D.min(0)
  low_range = np.where(ranged < 3)[0]  # MIN 3 Angstrom separation
  print('Low range  ', len(low_range), min(ranged))
  prune = set(low_range)

  #  Prune features with too little or too many cutoff elms
  count  = A.sum(0)
  conn_low = np.where(count < low_cut)[0]
  conn_hi  = np.where(count > hi_cut)[0]
  print('Low Count  ', len(conn_low), min(count))
  print('Hi  Count  ', len(conn_hi))

  prune |= set(conn_low)
  prune |= set(conn_hi)

  # prune Correlating features
  correlated = set()
  kr = sorted(set(range(K)) - prune)
  for i in range(len(kr)-1):
    a = kr[i]
    for j in range(i+1, len(kr)):
      b = kr[j]
      corr = np.logical_xor(A[:,a], A[:,b]).sum()
      if corr < 2*low_cut:
        # print('corr: ',a, b, corr)
        correlated.add(a)
  print('Correlated  ', len(correlated))
  kr = sorted(set(kr) - correlated)
  print('Kr reduced to :', len(kr))

  count = A[:,kr].sum(0)
  frequency = [kr[i] for i in np.argsort(count)]
  rlow, rhigh = [np.sum(A[:,i]) for i in [frequency[0], frequency[maxk]]]
  print('Return %d   (%d - %d)' % (maxk, rlow, rhigh))

  return frequency[:maxk]

## FIND item sets from node keys
def get_itemsets(nodelist):
  if isinstance(nodelist[0], string):
    keylist = nodelist
  elif isinstance(nodelist[0], frozenset):
    keylist = [tok(node) for node in nodelist]
  else:
    print("ERROR! Nodelist elms must be either key-strings or index-frozensets")
    return None

  print('Selecting events (basins) for each itemset node')
  progress = ProgressBar(len(keylist))
  Ik = {}
  for key in keylist:
    Ik[key] = np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]
    progress.incr
  return Ik
 
## MAX MINER
def maxminer_initgrps (T, epsilon):
  N, K = T.shape
  C = []
  F_init = [k for k in range(K) if len(np.where(T[:,k]==1)[0] > epsilon)]
  for k in F_init[:-1]:
    hg = [k] #frozenset({k})
    tg = [j for j in range(k+1, K)]  #frozenset([j for j in range(k+1, K)])
    C.append((hg, tg))
  F = [[] for i in range(len(F_init)+1)]
  F[1].append(frozenset((F_init[-1],))) 
  return C, F

def maxminer_subnodes(g, T, C, epsilon):
  hg, tg = g #g[0], list(g[1])
  i = len(tg)-1
  while i >= 0:
    support = np.sum(np.logical_and(T[:,hg + [tg[i]]], True).all(1))
    if support < epsilon:
      tg.pop(i)
    i -= 1
  if len(tg) > 0:
    for i in range(len(tg)-1):
      C.append((hg + [tg[i]], [j for j in tg[i+1:]]))
    return C, hg + [max(tg)]  #hg.union({max(tg)})
  return C, hg

def maxminer(T, epsilon):
  ''' Max-Miner Algorithm Implemenation. Epsilon is the lowest bond support value
  and T is the 1-0 Input matrix of events (row) by attribute obsevations (cols)'''
  N, K = T.shape
  start = dt.now()

  # 1. INIT C and F
  C, F = maxminer_initgrps(T, epsilon)
  Mtrack = defaultdict(int)
  n_iter = 0
  times = None
  subtimes = None

  # 2. Loop until no candidates
  logging.info(' #. CumulatTime   /  Scan T  Freq IS  EnumSubN Prune-F Prune-C')
  while len(C) > 0:
    ts = []
    t0 = dt.now()

    # 3. Scan T to count support groups
    cag = [hg+tg for hg,tg in C]
    C_spt = [np.sum(np.logical_and(T[:,g], True).all(1)) for g in cag]
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #1

    # 4. Add frequent itemsets
    for g, spt in zip(cag, C_spt):
      if spt >= epsilon:
        F[len(g)].append(frozenset(g))
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #2

    # 5. Init new candidate group list
    C_new = []

    # 6. For each infrequent itemset, enumerate subnodes and append max
    for (hg, tg), g, spt in zip(C, cag, C_spt):
      if spt < epsilon:
        Mtrack[frozenset(g)] = spt
        C_new, F_new = maxminer_subnodes((hg, tg), T, C_new, epsilon)
        F[len(F_new)].append(frozenset(F_new))
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #3

    # 7. Update candidate list
    C = C_new

    # 8. Prune F: remove itemsets with a superset in F
    # print('   F: %6d'%len(F), end='')
    x0, x1, x2 = 0,0,0
    for k in range(len(F)-1, 1, -1):
      for superset in F[k]:
        for j in range(k-1, 0, -1):
          for f_s in range(len(F[j])-1, -1, -1):
            if F[j][f_s] < superset:
              F[j].pop(f_s)
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #4
 
    # NOT THREADED 
    cidx = len(C)-1
    while cidx > 0:
      hg, tg = C[cidx]
      g = hg + tg
      glen = len(g)
      fs_g = frozenset(g)
      prune = False
      for k in range(len(g)+1, len(F)):
        for f in F[k]:
          if f < fs_g:
            C.pop(cidx)
            prune = True
            break
        if prune:
          break
      cidx -= 1    

    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #5

    # Benchmarking
    if times is None:  times = np.zeros(len(ts))
    times += ts
    n_iter += 1
    tot = (dt.now()-start).total_seconds()
    logging.info('%2d. TOT: %5.1f   / '%(n_iter, tot) + ('%5.2f s  ' * len(times)) % tuple(ts) + 'F: %6d   C: %6d' % (sum([len(i) for i in F]), len(C)))


  end = dt.now()
  logging.info('BENCH TIMES:     %s', ('%5.2f s  ' * len(times)) % tuple(times))
  logging.info('TOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))
  F_agg = []
  for k in range(len(F)-1, 0, -1):
    F_agg.extend(F[k])
  return F_agg, Mtrack

 
##  THE DERIVED LATTICE
def enum_iset(iset):
  cag = set({iset})
  nodelist = set()
  dlat = defaultdict(dict)
  while len(cag) > 0:
    node = cag.pop()
    if node in nodelist:
      continue
    nodelist.add(node)
    if len(node) == 1:
      continue
    for i in node:
      child = node - {i}
      dlat[child][node] = 0
      cag.add(child)
  return dlat

def derived_lattice(F, D, CM, nbins=20, brange=(4,8)):

  max_len = max([len(i) for i in F])
  dlat = defaultdict(dict)
  times = np.zeros(2)
  subtimes = None
  redund = 0
  t0 = start = dt.now()
  n_iter = 0

  logging.info('Build Lattice Structure from Max-Frequent Itemsets')
  progress = ProgressBar(len(F))
  # Expand all max freq itemsets
  nodelist = set()
  cag = set()
  for mfis in F:

    # Start with each MF itemset as the root
    cag.add(mfis)

    # Expand and add new subnodes to candidate group
    while len(cag) > 0:
      node = cag.pop()
      if node in nodelist:
        continue
      nodelist.add(node)
      if len(node) == 1:
        continue
      nkey = tok(node)
      for i in node:
        child = node - {i}
        dlat[tok(child)][nkey] = 0
        cag.add(child)

    progress.incr()

  logging.info('\nLattice Nodes:  %d', len(nodelist))

  # Build itemsets
  logging.info('Selecting events (basin) for each itemset node')
  Ik = {}
  progress = ProgressBar(len(nodelist))
  for node in nodelist:
    key = tok(node)

    # Add freq itemset
    Ik[key] = np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]
    progress.incr()

  # t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1

  logging.info('\nIntermediate Lattice Complete.  %5.1f sec' % (dt.now()-start).total_seconds())
  # Add last itemset
  # Ik[tok(F[-1])] = np.where(np.logical_and(CM[:,sorted(F[-1])], True).all(1))[0]

  # Calculate Distribution delta for every node
  logging.info('Precalculate all Histograms (1D) for all keys...')
  t0 = dt.now()
  H = histograms(Ik, D, nbins, brange)
  logging.info('\nHistograms pre-processed.  %5.1f sec' % (dt.now()-t0).total_seconds())

  logging.info('Calculating Derived Lattice:')
  progress = ProgressBar(len(dlat))

  # Remove null set
  if '' in dlat:
    del dlat['']

  # Calculate all node to parent edges
  for node, parentlist in dlat.items():

    #  Calculated distribution delta along every edge
    for parent in parentlist.keys():
      flow = np.zeros(len(node))

      # Calculate bin-by-bin difference
      delta = H[node] - H[parent][[parent.find(i) for i in node]]

      # Flow is tracked along each dimension separately
      flow = np.zeros(len(node))
      for k in range(len(delta)):

        # Calculate flow from 0 to n-1 and add/subtract flow to neighboring bin
        flow_k = 0
        for i in range(nbins-1):
          flow_k     += delta[k][i]
          delta[k][i+1] += delta[k][i]
        flow[k] = flow_k

      # EMD is sqrt of sums of squares for each 1D distribution delta
      dlat[node][parent] = np.sqrt(np.sum(flow**2))
      progress.incr()

  end = dt.now()

  # print('BENCH TIMES:     ', ('%5.1f s  ' * len(times)) % tuple(times))
  logging.info('\nTOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))
  return dlat, Ik

def histograms(Ik, D, nbins=20, binrange=(4,8)):
  N, K = D.shape
  H = {}
  col = {k: toidx(k) for k in k_domain}

  tot_iter = len(Ik)
  tick = max(1, tot_iter//100)
  prog_cnt = 0

  logging.info('\nPreProcessing Histograms (1D)')
  progress = ProgressBar(len(Ik))
  for key, iset in Ik.items():
    hist = np.zeros(shape=(len(key), nbins))
    distr = D[iset]
    for i, k in enumerate(key):
      hist[i] = np.histogram(distr[:,col[k]], nbins, binrange)[0]
    H[key] = hist / hist.sum(1)[:,None][0]
    progress.incr()

  return H

def cheapEMD(d1, d2, nbins=20, binrange=None):
  ''' Performs a cheap EMD using 1D histograms. "dirt" is only
  moved to adjacent bins along each dim and is cacluated iteratively 
  from the lowest to highest bin. Simulated "scraping dirt/anti-dirt"
  from bin to bin. The returned val is the sqrt of the sum of squares for
  all dimensions'''
  N1, K1 = d1.shape
  N2, K2 = d2.shape
  flow = np.zeros(K1)
  brange = (4,8) if binrange is None else binrange
  # Create normalized Histograms with different distributions (same bin #/sizes)
  for k in range(K1):
    ha = np.histogram(d1[:,k], nbins, brange)[0] / N1
    hb = np.histogram(d2[:,k], nbins, brange)[0] / N2
    flow_k = 0
    # Calculate bin-by-bin difference
    delta = ha - hb
    # calculate flow from 0 to n-1 and add/subtract flow to neighboring bin
    for i in range(nbins-1):
      flow_k     += delta[i]
      delta[i+1] += delta[i]
      # Note: there is a assumed 'delta[i] -= delta[i]'

    # Normalize the result by returning absolute delta and dividing by
    # number of "moves" (n-1)
    flow[k] = flow_k / (nbins-1)

  # Aggregate dimensions
  return np.sqrt(np.sum(flow**2))


# def get_itemset(keylist, CM):
  

def extend_dlat(dlat, Ik, D, V, epsilon, nbins=20, binrange=(4,8)):

  # IDX or Alpha??? OR is this inherently determined by poset
  print('Initializing new lattice')
  N, K = D.shape
  nodeitem = frozenset({K-1})
  new_key_char = tok(nodeitem)
  nodeitemset = np.where(V)[0]

  # 1. Set candidate flag
  candidate = {k: True for k in Ik.keys()}

  # 2. sort low to high
  keylist = deque(sorted(Ik.keys(), key=lambda x: (len(x), x)))
  new_node_list = []
  update_list  = defaultdict(list)

  # 3. Base case
  Ik[tok([K-1])] = nodeitemset
  nodeitemset = set(nodeitemset)

  # 4. Pop and merge 
  print('Merging into all existing keys: ', len(keylist))
  progress = ProgressBar(len(keylist))
  while len(keylist) > 0:
    key = keylist.popleft()
    if candidate[key]:
      itemset = set(Ik[key]).intersection(nodeitemset)
    
      # Check for min support
      if len(itemset) > epsilon:
        # Accept the merge
        new_node = frozenset(toidx(key)) | nodeitem
        new_key  = tok(new_node)
        Ik[new_key] = np.array(list(itemset))
        new_node_list.append(new_node)
        update_list[key].append(new_key)
        for parent in dlat[key].keys():
          update_list[new_key].append(parent + new_key_char)

      else:
        # Reject and flag all subsequent nodes
        reject_list = deque([key])
        while len(reject_list) > 0:
          reject = reject_list.popleft()
          if candidate[reject]:
            candidate[reject] = False
            for parent in dlat[reject].keys():
              reject_list.append(parent)

    progress.incr()


  # 5. Calculate new EMD values
  print('\nCalculating all EMD values for new keys:', len(update_list))
  progress = ProgressBar(len(update_list))
  for nkey, parent_list in update_list.items():
    node = toidx(nkey)
    d1 = D[Ik[nkey]][:,node]
    for parent in parent_list:
      if parent not in Ik:
        continue
      d2 = D[Ik[parent]][:,node]
      dlat[nkey][parent] = cheapEMD(d1, d2, nbins, binrange)
    progress.incr()

  # return update lattice and itemsets
  print('\n All Done!')
  return dlat, Ik

def update_lattice(dlat, Ik, D_old, CM, Mtrack, D_new, M_new, M_delta, epsilon, nbins=20, binrange=(4,8)):

  N, K = D_old.shape
  update_list = defaultdict(list)
  new_node_list = []

  # 1. Run maxminer on new data
  # M_new, M_delta = maxminer( )

  # 2. Expand all MFIS and flag each itemset for update
  print('Build Lattice Structure from Max-Frequent Itemsets')
  progress = ProgressBar(len(M_new))
  nodelist = set()
  cag = set()
 
  for mfis in M_new:
    # Start with each MF itemset as the root
    cag.add(mfis)

    # Expand and add new subnodes to candidate group
    while len(cag) > 0:
      node = cag.pop()
      if node in nodelist:
        continue
      nodelist.add(node)
      if len(node) == 1:
        continue
      nkey = tok(node)
      for i in node:
        child = node - {i}
        update_list[tok(child)].append(nkey)
        cag.add(child)

    progress.incr()

  # 3. Search list of potential max-freq itemsets and add new keys if discovered
  for node, z in M_delta.items():
    key = tok(node)
    if key in Ik:
      # node already exists
      continue
    elif node in Mtrack:
      # node partially observed previously
      Mtrack[node] += z
      if Mtrack[node] > epsilon:
        new_node_list.append(node)
      del Mtrack[node]
    else:
      # Newly observed max-frequent node
      Mtrack[node] = z
      # TODO:  DO I NEED TO CHECK SUBSETs

  # 4. Explicit add for new nodes
  all_keys = frozenset(range(K))
  new_node_list = sorted(new_node_list, key=lambda x: len(x), reverse=True)
  print('\nAdd new keys:', len(new_node_list))
  progress = ProgressBar(len(new_node_list))
  for node in new_node_list:
    key = tok(node)
    Ik[key] = [N + i for i in np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]]
    dlat[key] = {}
    for k in all_keys - node:
      parent = tok(node | {k})
      if parent in Ik:
        update_list[key].append(parent)
    progress.incr()

   # 5. Calculate new EMD values
  D = np.vstack((D_old, D_new))
  print('\nCalculating all EMD values for new keys:', len(update_list))
  progress = ProgressBar(len(update_list))
  for nkey, parent_list in update_list.items():
    node = toidx(nkey)
    d1 = D[Ik[nkey]][:,node]
    for parent in parent_list:
      if parent not in Ik:
        continue
      d2 = D[Ik[parent]][:,node]
      dlat[nkey][parent] = cheapEMD(d1, d2, nbins, binrange)
    progress.incr()

  return D, dlat, Ik, Mtrack

## CLUSTER AND SAMPLE
def botup_collapse(dlat, theta=.9):
  # Can be optimized if needed
  dlw = {k: {i: 1-j for i, j in v.items() if j<1} for k,v in dlat.items()}
  # get smallest keys
  klist = sorted(dlw.keys(), key=lambda x: (len(x), x))

  C = defaultdict(dict)
  for node in klist:
    if len(dlw[node]) == 0:
      continue
    for parent, emd in dlw[node].items():
      if emd < theta:
        continue
      C[parent][node] = emd
      for child, w in C[node].items():
        w_t = w * emd
        if w_t > theta:
          C[parent][child] = w_t
    if node in C:
      del C[node]

  setlist, assignment = {}, {}
  for key, members in C.items():
    setlist[key] = set(members.keys())
    for node in members.keys():
      assignment[node] = key

  return setlist, assignment

def clusterlattice(dlat, CM, D, Ik, theta=.9, minclusize=0, bL=None):
  ''' USes full distrubution of clustered items to merge clusters based 
  on eucidean distance to centroid '''
  N, K = D.shape
  global_var = D.std(0)


  logging.info('CLUSTERING the derived Lattice')

  # gs, ga = topdown_group_single(dlat, theta)
  gs, ga = botup_collapse(dlat, theta)

  logging.info("  TOTAL # of Itemset-Groups   :   %d", len(gs))

  grp, nogrp = {}, []

  clusters = defaultdict(list)
  ### OPTION A:  Map to corr feature set (for unique key)
  # 3.  For each item: map to a group:   <map>
  for i in range(N):
    k = fromm(CM[i])
    if len(k) == 0:
      continue
    if k in ga:
      #  Set its group 
      grp[i] = ga[k]
      clusters[ga[k]].append(i)
    else:
    # Add all off-by-1 
      added = False
      immed_lower = [''.join(sorted(i)) for i in it.combinations(k, len(k)-1)]
      for n in immed_lower:
        if n in ga:
          clusters[ga[n]].append(i) 
          added = True

    #   # Keep track of unassigned nodes (TODO: is this just noise?)
      if not added:
        nogrp.append(i)

  logging.info("  # Initial Clust   :   %d", len(clusters))

  ## OPTION B:  Assigned to longest key:
  # print('Iteratively assigning event-items to clusters (via longest observered key)')
  # keylist = sorted(Ik.keys(), key=lambda x: (len(x), x), reverse=True)
  # for i in range(N):
  #   k = fromm(CM[i])
  #   stoplen = 0
  #   for key in keylist:
  #     if len(key) <= stoplen:
  #       break
  #     if i in Ik[key]:
  #       grp[i] = ga[key]
  #       clusters[ga[key]].append(i)
  #       stoplen = len(key) - 1

  #   if stoplen == 0:
  #     nogrp.append(i)

  maxsize = max([len(v) for v in clusters.values()])
  minsize = min([len(v) for v in clusters.values() if len(v) > 0])

  logging.info("  TOTAL # Clusters   :   %d", len(clusters))

  # remove single cluster nodes:
  keylist = [i[0] for i in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]  
  for k in keylist:
    if len(clusters[k]) == 0:
      logging.info('EMPTY CLUSTER')
    if len(clusters[k]) == 1:
      for i in clusters[k]:
        nogrp.append(i)
      del clusters[k]
  logging.info("  Pruned Clusters    :   %d", len(clusters))
  logging.info("  Events w/NO grp    :   %d", len(set(nogrp)))
  keylist = [i[0] for i in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]  

  # Calc centroids
  # centroid = {k: Dr[clu[k]][:,toidx(k)].mean(0) for k in keylist}
  centroid = {k: D[v].mean(0) for k,v in clusters.items()}
  variance = {}
  for k,v in clusters.items():
    cov = np.cov(D[v].T)
    ew, ev = LA.eigh(cov)
    variance[k] = np.sum(ew)
  # k: D[v].std(0)/global_var for k,v in clusters.items()}
  
  Nc = len(clusters)
  G = np.zeros(shape=(Nc, Nc))
  for i in range(Nc-1):
    for j in range(i+1, Nc):
      G[i][j] = G[j][i] = LA.norm(centroid[keylist[i]] - centroid[keylist[j]])

  inum = 0
  expelled = []
  maxval = G.max()
  logging.info('Running the merge loop')
  while True:
  # while minsize < minclu:
    Nc = len(clusters)
    sigma = np.array([variance[k] for k in keylist])
    mean_sig, var_sig, sum_sig, max_sig = sigma.mean(), sigma.std(), sigma.sum(), sigma.max()

    np.fill_diagonal(G, G.max())
    minval = G.min()
    minidx = np.argmin(G)
    row, col  = minidx // Nc, minidx % Nc
    np.fill_diagonal(G, 0)

    Gv = G[np.triu_indices(len(G), 1)]
    gmean, gsum, gvar = Gv.mean(), Gv.sum(), Gv.std()

    # Internal Cluster Metric (how well defined is the least defined cluster)
    m_highvar = np.argmax([variance[k] for k in keylist])
    k_highvar = keylist[m_highvar]
    int_score = np.abs(variance[k_highvar] - mean_sig) / var_sig

    # External Cluster Metric (how separate are the closest two clusters)
    m_attr  = row if len(clusters[keylist[row]]) < len(clusters[keylist[col]]) else col
    m_donor = col if m_attr == row else row
    k_attr, k_donor = keylist[m_attr], keylist[m_donor]
    ext_score = np.abs(minval - gmean) / gvar

    logging.info ('%4d.'%inum + 'Gmean %4.2f (%4.2f) MinG %4.2f  / TotV %4.2f  MeanV %4.2f (%4.2f) MaxV %4.2f /  %4.2f v %4.2f ' \
      % (gmean, gvar, minval, sum_sig/K, mean_sig, var_sig, max_sig, int_score,  ext_score) + ' {%2d-%4d}' % (minsize, maxsize))

      # Find node attactor (larger) & donor (smaller)

    # logging.info('MERGE   dis= %5.2f  var=%5.2f' % (minval, variance[k_donor]))
    for elm in clusters[k_donor]:
      clusters[k_attr].append(elm)

    for i in range(Nc):
      if i in [m_attr, m_donor]:
        continue
      G[m_attr][i] = G[i][m_attr] = LA.norm(centroid[k_attr] - centroid[keylist[i]])

    centroid[k_attr] = D[clusters[k_attr]].mean(0)
    ew, ev = LA.eig(np.cov(D[clusters[k_attr]].T))
    variance[k_attr] = np.sum(ew)



    # Update Data Stucts
    keylist.pop(m_donor)
    del clusters[k_donor]
    del centroid[k_donor]
    del variance[k_donor]
    G = np.delete(G, m_donor, 0)
    G = np.delete(G, m_donor, 1)

    # Add in all elements with no clusters
    # while len(nogrp) > 0:
    #   i = nogrp.pop()
    #   delta = {k: LA.norm(D[i] - v) for k,v in centroid.items()}
    #   c = min(delta.items(), key=lambda x: x[1])[0]
    #   clusters[c].append(i)

    # Update all Centroids and variances
    centroid = {k: D[v].mean(0) for k,v in clusters.items()}
    variance = {}
    for k,v in clusters.items():
      cov = np.cov(D[v].T)
      ew, ev = LA.eigh(cov)
      variance[k] = np.sum(ew)

    maxsize = max([len(v) for v in clusters.values()])
    minsize = min([len(v) for v in clusters.values() if len(v) > 0])

    inum += 1
    if len(clusters) < 12:
      break

  logging.info("  TOTAL # CLUSTERS   :   %d", len(clusters))
  if bL is not None:
    n = 0
    clusterlist = []
    for idx, (k,v) in enumerate(clusters.items()):
      bc  = np.bincount([bL[i] for i in v], minlength=5)
      state = np.argmax(bc)
      stperc = 100*bc[state] / sum(bc)
      elms = (n, k, len(v), variance[k].mean(), state, stperc, bc)
      clusterlist.append(elms)
      # print('%2d.'%n, '%-15s'%k, '%4d '%len(v), 'State: %d  (%4.1f%%)' % (state, stperc))
      n += 1
    logging.info(' #.     Key              Size      Var  /  State  (Percent)      /    BinCount')
    for i in sorted(clusterlist, key =lambda x : x[2], reverse=True):
      print('%3d.  %-18s%7d  %6.3f /  State: %d  (%5.1f%%)   /    %s' % i)

  logging.info('\nReassiging {0} groupless events'.format(len(nogrp)))
  for i in nogrp:
    delta = {k: LA.norm(D[i] - v) for k,v in centroid.items()}
    c = min(delta.items(), key=lambda x: x[1])[0]
    clusters[c].append(i)

 
  # FOR SCORING
  # Eigen Weights for variance (internal cluster metric)
  logging.info('Ensure Unique Values in each cluster')
  for k in clusters.keys():
    clusters[k] = sorted(set(clusters[k]))

  centroid = {k: D[v].mean(0) for k,v in clusters.items()}
  variance = {}
  for k,v in clusters.items():
      cov = np.cov(D[v].T)
      ew, ev = LA.eigh(cov)
      variance[k] = np.sum(ew)

  samplist = []

  clusterlist = []
  clusterscore = np.zeros(len(clusters))
  elmscore = [[] for i in range(len(clusters))]
  logging.info("  TOTAL # CLUSTERS   :   %d", len(clusters))
  total_var = np.sum([k for k in variance.values()])

  low_var = min(variance.values())
  MIN_EV = .0005 * N
  for clnum, (k, iset) in enumerate(clusters.items()):

    # THE CLUSTER SCORE
    sc_var  = low_var / variance[k]
    sc_size = 1 - (2 * len(clusters)*len(iset) / (N))
    if len(iset) < MIN_EV:
      sc_size = -2
    clscore = sc_var + sc_size


    clusterscore[clnum] = max(0, clscore)
    elmlist = []

    for i in iset:
      dist_cent = LA.norm(D[i] - centroid[k])

      # ELEMENT SCORE
      elmlist.append((i, dist_cent))

    elmscore[clnum] = sorted(elmlist, key= lambda x: x[1])

    if bL is not None:
      bc  = np.bincount([bL[i] for i in iset], minlength=5)
      state = np.argmax(bc)
      stperc = 100*bc[state] / sum(bc)
      clusterlist.append((clnum, k, len(iset), variance[k], clscore, sc_size, sc_var, state, stperc, bc))

  if bL is not None:
    logging.info(' #.     Key              Size    Var  :  Score: (size + var)  /   State (percent)   BinCount')
    for i in clusterlist: #sorted(clusterlist, key =lambda x : x[2], reverse=True):
      logging.info('%3d.  %-18s%5d  %6.2f : %7.2f  (%5.2f %5.2f) /  State: %d (%5.1f%%) %s' % i)


   # FOR SAMPLING
  if bL is not None:
    sampidx = np.zeros(len(clusters), dtype=np.int16)
    pdf = clusterscore / np.sum(clusterscore)
    logging.info('\nPDF:  %s', str(pdf))
    logging.info('SAMPLE OF 20 CANDIDATES.....')
    for i in range(20):
      clidx = int(np.random.choice(len(pdf), p=pdf))
      elm, dist = elmscore[clidx][sampidx[clidx]]
      logging.info(' %2d.  Clu#   %2d        idx %5d    State= %d ' % (i, clidx, elm, bL[elm]))
      sampidx[clidx] += 1

  return clusters, clusterscore, elmscore








def dlattice_meet(F, D, CM, nbins=20, brange=(4,8)):

  max_len = max([len(i) for i in F])
  dlat = defaultdict(dict)
  Ik   = {}
  times = np.zeros(2)
  subtimes = None
  redund = 0
  t0 = start = dt.now()
  n_iter = 0

  # 1. Find all meet points for all F
  tot_iter = ((len(F)**2)-len(F))/2
  tick = max(1, tot_iter//100)
  prog_cnt = 0
  print('Tracking Lattice Build Progress:')
  for i in range(len(F)):

    # Add max freq itemset
    lkey = tok(F[i])
    if lkey not in Ik:
      Ik[lkey] = np.where(np.logical_and(CM[:,list(F[i])], True).all(1))[0]

    # Add empty node
    dlat[lkey] = {}

    for j in range(i+1, len(F)):
      prog_cnt += 1

      # Find and add meet point
      meet = F[i] & F[j]
      mkey = tok(meet)
      if mkey not in Ik:
        Ik[mkey] = np.where(np.logical_and(CM[:,list(meet)], True).all(1))[0]

      # Enmerate All intermediate nodes
      for top in [F[i], F[j]]:
        internodes = [meet]
        while len(internodes) > 0:
          node = internodes.pop()
          nkey = tok(node)
          if nkey not in Ik:
            Ik[nkey] = np.where(np.logical_and(CM[:,list(node)], True).all(1))[0]
          for k in top - node:
            n_iter += 1
            parent = node | {k}
            pkey = tok(parent)
            if pkey not in dlat[nkey]:
              internodes.append(parent)
              dlat[nkey][pkey] = 0
      if prog_cnt % tick == 0:
        prog = 100*prog_cnt/tot_iter
        print ('\r[%-100s] %4.1f%%' % ('#'*int(prog), prog), end='')


  # t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1

  print('\nIntermediate Lattice Complete.  %5.1f sec' % (dt.now()-start).total_seconds())
  # Add last itemset
  # Ik[tok(F[-1])] = np.where(np.logical_and(CM[:,sorted(F[-1])], True).all(1))[0]

  # Calculate Distribution delta for every node
  print('Precalculate all Histograms (1D) for all keys...')
  t0 = dt.now()
  H = histograms(Ik, D, nbins, brange)
  print('\nHistograms pre-processed.  %5.1f sec' % (dt.now()-t0).total_seconds())

  print('Calculating Derived Lattice:')
  tot_iter = sum([len(v) for v in dlat.values()])
  prog_cnt, tick = 0, max(1, tot_iter // 100)
  print ('\r[%-100s] %4.1f%%' % ('', 0), end='')

  # Remove null set
  if '' in dlat:
    del dlat['']

  # Calculate all node to parent edges
  for node, parentlist in dlat.items():

    #  Calculated distribution delta along every edge
    for parent in parentlist.keys():
      flow = np.zeros(len(node))

      # Calculate bin-by-bin difference
      delta = H[node] - H[parent][[parent.find(i) for i in node]]

      # Flow is tracked along each dimension separately
      flow = np.zeros(len(node))
      for k in range(len(delta)):

        # Calculate flow from 0 to n-1 and add/subtract flow to neighboring bin
        flow_k = 0
        for i in range(nbins-1):
          flow_k     += delta[k][i]
          delta[k][i+1] += delta[k][i]
        flow[k] = flow_k

      # EMD is sqrt of sums of squares for each 1D distribution delta
      dlat[node][parent] = np.sqrt(np.sum(flow**2))

    # idx = toidx(node)
    # d1 = D[Ik[node]][:,idx]
    # for parent in dlat[node]:
    #   d2 = D[Ik[parent]][:,idx]
    #   dlat[node][parent] = cheapEMD(d1, d2, 20, (4,8))
      prog_cnt += 1
      if prog_cnt % tick == 0:
        prog = 100*prog_cnt/tot_iter
        print ('\r[%-100s] %4.1f%%' % ('#'*int(prog), prog), end='')

  # t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
  end = dt.now()

  # print('BENCH TIMES:     ', ('%5.1f s  ' * len(times)) % tuple(times))
  print('\nTOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))
  return dlat, Ik

def dlattice_mm_2(F, D, CM, epsilon=20):

  times, subtimes = None, None
  start = dt.now()
  Lk = [[]]
  dlattice = defaultdict(dict)

  max_depth = max([len(i) for i in F])
  cag = set()
  for iset in F:
    for ca in it.combinations(iset, 1):
      cag.add(frozenset(ca))
  Lk.append(cag)

  depth = 2
  while depth <= max_depth:
    print('processing: depth=', depth, '  Lk[d-1]=',len(Lk[depth-1]), '   dlat=', len(dlattice))
    ts, t0 = [], dt.now()

    # Get all itemset and distributions for child nodes
    Ik, Dk, childnodes = {}, {}, {}
    for k in Lk[depth-1]:
      key = tok(k)
      idx = sorted(k)
      itemset = np.where(np.logical_and(CM[:,idx], True).all(1))[0]
      if len(itemset) > epsilon//2:
        childnodes[key] = k
        Ik[key] = itemset
        Dk[key] = D[Ik[key]][:,idx]
    print('Ch-nodes:', len(Ik), end='  /  ')
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1

    # Enumerate all candidate nodes at current depth from max-frequent itemsets 
    C = set()
    for iset in F:
      if len(iset) >= depth:
        for ca in it.combinations(iset, depth):
          C.add(frozenset(ca))
    print('Pa-cand:', len(C))
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1

    # Calculate all distribution deltas for child-parent edges
    subtimes = np.zeros(3)
    for ca in C:

      pkey     = tok(ca)
      idx     = sorted(ca)
      itemset = np.where(np.logical_and(CM[:,idx], True).all(1))[0]
      for ckey, child in childnodes.items():
        ts2, t2 = [], dt.now()
        if child < ca:
          t3 = dt.now(); ts2.append((t3-t2).total_seconds()); t2=t3
          d2 = D[itemset][:,idx]
          t3 = dt.now(); ts2.append((t3-t2).total_seconds()); t2=t3
          dlattice[ckey][pkey] = cheapEMD(Dk[ckey], d2, nbins=24, binrange=(4, 8))
          t3 = dt.now(); ts2.append((t3-t2).total_seconds()); t2=t3
        else:
          t3 = dt.now(); ts2.append((t3-t2).total_seconds()); t2=t3
          ts2.append(0)
          ts2.append(0)
        subtimes += ts2
    print('SUB TIMES:     ', ('%5.2f s  ' * len(subtimes)) % tuple(subtimes))
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
  
    # TODO: Prune C
    Lk.append(C)
    depth += 1
    if times is None:  times = np.zeros(len(ts))
    times += ts
    print('%2d  - times: '%depth, ('%5.2f s  ' * len(subtimes)) % tuple(ts2))


  end = dt.now()
  print('BENCH TIMES:     ', ('%5.2f s  ' * len(times)) % tuple(times))
  print('TOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))

  return dlattice
    # Retain only clustering parent nodes
    # nodelist = set()
    # for child in Lk[depth-1]:
    #   min_node, min_val = min(dlattice[child].items(), key=lambda x: x[1])
    #   nodelist.add(min_node)
    # cag_nodes = set()

def dlattice_mm_1(F, D, CM):
  t0 = dt.now()
  nnodes, nedges = 0, 0

  dlattice = defaultdict(dict)
  Fk = sorted([tok(k) for k in F], key=lambda x: (len(x), x), reverse=True)
  Fs = sorted(F, key=lambda x: len(x), reverse=True)

  Lk = [set() for i in range(len(Fs[0])+1)]
  for k in Fs: 
    Lk[len(k)].add(frozenset(k))
    nnodes += 1 

  grpassign = {}  #[Lk[-1]]
  for d in range(len(Lk)-1, 0, -1):
    print("Processing depth = ", d)
    # Add all top nodes' immediate children for keylen==d
    for node in Lk[d]:
      for ch in it.combinations(node, d-1):
        Lk[d-1].add(frozenset(ch))

    nnodes += len(Lk[d-1])
 
    # Get all child itemsets
    ch_Ik = {tok(idx): np.where(np.logical_and(CM[:,list(idx)], True).all(1))[0] for idx in Lk[d-1]}

    # Calc EMD distr delta for all child-parent edges
    emd = defaultdict(dict)
    for ch in Lk[d-1]:
      ch_key = tok(ch)
      ch_idx = list(ch)
      d1 = D[ch_Ik[ch_key]][:,ch_idx]
      for parent in Lk[d]:
        if ch < parent:
          pkey = tok(parent)
          d2 = D[Ik[pkey]][:,ch_idx]
          dlattice[ch_key][pkey] = emd[ch_key][pkey] = cheapEMD(d1, d2, nbins=24, binrange=(4, 8))
          nedges += 1

    # Select at most 1 parent for each child
    for ch, parents in emd.items():
      min_emd, min_val = min(parents.items(), key=lambda x: x[1])
      if min_emd in grpassign:  # and min_val < theta
        grpassign[ch] = grpassign[min_emd]
      else:
        grpassign[ch] = min_emd

  tottime = 0.

    # print('  TIMES:  #Edges= %7d'% nedges_k, '  Avg EMD= %6.4f  '%np.mean(times),
    #   '  TotTime= %6.3f'% np.sum(times))

    # print('   Bench Time:  %7.4f    %7.4f    %7.4f     %7.4f' % 
    #   (sum(md1), sum(md2), sum(md3), sum(nt)))
    # tottime += sum(times)
  nnodes = len(dlattice)
  tottime = (dt.now()-t0).total_seconds()
  print('\n Total Time = %7.2f   Nodes = %d   Edges = %d' % (tottime, nnodes, nedges))
  return dlattice, grpassign

def dlattice_mm(F, D, CM, epsilon):
  dlat = defaultdict(dict)
  Ik   = {}
  times = np.zeros(2)
  subtimes = None
  redund = 0
  start = dt.now()
  n_iter = 0
  print('Running DLat...')
  for n, node in enumerate(F[::-1]):
    if tok(node) in Ik:
      print('      Node exists in lattice: ', tok(node))
      continue
    nstart = dt.now()
    print('%3d.  Node:  %12s   ' % (n, tok(node)), end='')
    for i in range(len(node)+1):
      for k in it.combinations(node, i):
        key = tok(k)
        if key in Ik:
          redund += 1
          continue
        n_iter += 1
        ts = []
        t0 = dt.now()
        idx = sorted(k)
        eventlist = np.where(np.logical_and(CM[:,idx], True).all(1))[0]
        t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
        Ik[key] = len(eventlist)
        # lab = np.bincount([bL[i] for i in eventlist], minlength=5)
        for p in it.combinations(node, i+1):
          if set(k) < set(p):
            d1 = D[eventlist][:,idx]
            eset2 = np.where(np.logical_and(CM[:,p], True).all(1))[0]
            if len(eset2) <= 1:
              continue
            d2 = D[eset2][:,idx]
            dlat[key][tok(p)] = cheapEMD(d1, d2, 20, (4, 8))
        t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
      times += ts
    nend = dt.now()
    print('dlat:%7d    Time: %5.1f s' % (len(dlat), (nend-nstart).total_seconds()))

  end = dt.now()

  print('BENCH TIMES:     ', ('%5.1f s  ' * len(times)) % tuple(times))
  print('TOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))
  return dlat, Ik

def lattice_enum(key):
  keylist = [[]]
  for i in range(1, len(key)):
    keylist.append(list(it.combinations(key, i)))
  return keylist

def cluster_mm(F, CM, D, minclu=3, incldist=True, bL=None):
  
  Fk = [tok(k) for k in F]
  Ik, centroid = {}, {}
  for k, idx in zip(Fk, F):
    itemset = np.where(np.logical_and(CM[:,list(idx)], True).all(1))[0]
    if itemset is not None:
      Ik[k]        = itemset
      centroid[k]  = D[itemset][:,list(idx)].mean(0)
    print('NCent=', len(centroid))

    clusters = defaultdict(list)
    nogrp = []
    ga = []
    keylist = list(Ik.keys())
    setlist = [set(k) for k in keylist]
    for i in range(len(CM)):
      item = fromm(CM[i])
      if item in keylist:
        clusters[item].append(i)
        continue
      neigh = {}
      for k, cent in centroid.items():
        if set(item) <= set(k):
          try:
            neigh[k] = LA.norm(D[i][toidx(k)] - centroid[k])
          except IndexError as err:
            print(i, k, idx)
            return
      if len(neigh) == 0:
        nogrp.append(i)
      else:
        sel_clu = min(neigh.items(), key=lambda x: x[1])[0]
        clusters[sel_clu].append(i)

  print("  TOTAL # of Clusters :   ", len(clusters))
  print("  Items w/no groups   :   ", len(nogrp))

  # FOR DEBUG/DISPLAY
  if bL is not None:
    printclu(clusters, bL, minclu=minclu, incldist=True)    
  return clusters

def deref(T, key):
  deref = lambda T, key: np.where(np.logical_and(T[:,key], True).all(1))[0]

def maxminer_orig(T, epsilon):
  N, K = T.shape
  start = dt.now()

  # 1. Init C and F
  C, F = maxminer_initgrps(T, epsilon)
  n_iter = 0
  times = None
  subtimes = None

  # 2. Loop until no candidates
  while len(C) > 0:
    ts = []
    t0 = dt.now()
    C_new = []
    for hg, tg in C:
      ts2 = []
      s0 = dt.now()
      # g = hg.union(tg)
      g = hg + tg 
      if len(g) <= 1:
        continue
      # tupg = list(g)
      # mask = np.ones(K)
      # mask[tupg] = 0
      idx = T[:,g]
      s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1   #s1
      support = np.sum(np.logical_and(idx, True).all(1))
      # support = len(np.where(np.logical_and(idx, True).all(1))[0])
      # support = np.sum(np.logical_or(T, mask).all(1))
      s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1     #s2
      if support >= epsilon:
        F.append(g)
        s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1   #s3
        ts2.append(0)
        # ts2.append(0)
        # ts2.append(0)
      else:
        C_new, gn = maxminer_subnodes((hg,tg), T, C_new, epsilon)
        ts2.append(0)
        s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1    #s4

        # i = len(tg)-1
        # while i > 0:
        #   support = np.sum(np.logical_and(T[:,hg + [tg[i]]], True).all(1))
        #   # print(i, len(tg), end='  / ')
        #   if support < epsilon:
        #     tg.pop(i)
        #   i -= 1
        # s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1    #s5
        # if len(tg) > 0:
        #   for i in range(len(tg)-1):
        #     C.append((hg + [i], [j for j in tg[i+1:]]))
        #   C_new, gn = C, (hg + [max(tg)])  #hg.union({max(tg)})
        # else:
        #   C_new, gn = C, hg
        # s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1    #s6

        F.append(gn)
      if subtimes is None:  subtimes = np.zeros(len(ts2))
      subtimes += ts2
    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
    C = C_new
    f = 0
    while f < len(F):
      for i in range(f+1, len(F)):
        if F[f] < F[i]:
          del F[f]
          break
      f += 1

    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
    c = 0
    while c < len(C):
      hg, tg = C[c]
      # g = hg.union(tg)
      g = hg + tg
      for f in F:
        if g < f:
          del C[c]
          break
      c += 1

    t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1
    if times is None:  times = np.zeros(len(ts))
    times += ts
    n_iter += 1
    print('%2d  - subtimes: '%n_iter, ('%5.2f s  ' * len(subtimes)) % tuple(subtimes))
  print(ts)
  # times /= n_iter
  end = dt.now()
  print('BENCH TIMES:     ', ('%5.2f s  ' * len(times)) % tuple(times))
  print('BENCH SUBTIMES:  ', ('%5.2f s  ' * len(subtimes)) % tuple(subtimes))
  print('TOTAL TIME: %5.1f sec   /  %4d loops' % ((end-start).total_seconds(), n_iter))
  return F

## APRIORI
def apriori(M, epsilon):
  N, K = M.shape
  L, I = [[]], OrderedDict()
  k = 2
  Lk = []
 
  # Base Case (k == 1)
  k1key = []
  start = dt.now()
  Vk = set()
  P=SortedSet()
  prMap = {}
  for i in range(K):
    k1key.append(frozenset({i}))
    Ik = np.where(M[:,i]==1)[0]
    prMap[i] = set()
    if len(Ik) > epsilon:
      I[k1key[i]] = SortedSet(Ik)
      Lk.append(k1key[i])
      Vk.add(i)
    else:
      prMap[i].add(k1key[i])
  L.append(Lk)

  # Build lattice from bottom to top
  while len(L[k-1]) > 0:
    ts0 = dt.now()
    Lk, Ck = [], []
    Ik = OrderedDict()
    Cd = set()

    # Map all k-size candidate keys from L[k-1] elements
    # for i in range(0, len(L[k-1])-1):
    #   for j in range(i+1, len(L[k-1])):
    #     ca = L[k-1][i].union(L[k-1][j])
    #     if len(ca) == k: # and ca not in Ck:
    #       # Ik.append((I[L[k-1][i]], I[L[k-1][j]]))
    #       # Ck.append(SortedSet(ca))
    #       Ik[ca] = (I[L[k-1][i]], I[L[k-1][j]])
    for i in L[k-1]:
      for j in Vk:
        if j in i:
          continue
        # added_key = frozenset({j})
        ca = i.union(k1key[j])
        Ik[ca] = (i, j)
    ts1 = dt.now()
    print('%2d Candidates: %7d' % (k, len(Ik)), '  P=', len(P), '   Ck=', len(Ck))

    # Prune:  Remove keys which have subsets that are not
    # keys in any lower tiers
    Ck = SortedList(Ik.keys())
    prunelist = []
    for i, c in reversed(list(enumerate(Ck))):
      for p in P:
        if p < c:
          prunelist.append(i)
          del Ik[Ck[i]]
          del Ck[i]
          break
    ts1a = dt.now()
    # for p in reversed(prunelist):
    #   del Ik[Ck[p]]
    #   del Ck[p]
    ts2 = dt.now()
    print('   Pruned to:  %7d' % len(Ik))

    # Reduce: candidates' itemsets and accept/reject based on epsilon
    for c, (sA, b) in Ik.items():
      # ia = I[sA].intersection(I[k1key[b]])
      ia = np.where(np.logical_and(M[:,tuple(c)], True).all(1))[0]

      if len(ia) > epsilon:
        Lk.append(c)
        I[c] = ia
      else:
        P.add(c)
    L.append(Lk)
    Vk = set(it.chain(*Lk))
    ts3 = dt.now()
    print('  TIME:  Map: %5.2f s   Prune: %5.2f / %5.2f s    Reduce: %5.2f s   #Keys: %d' % 
      ((ts1-ts0).total_seconds(), (ts1a-ts1).total_seconds(), (ts2-ts1a).total_seconds(), (ts3-ts2).total_seconds(), len(Lk)))
    k += 1

  print('Total Time:  %6.2f sec' % (dt.now()-start).total_seconds(),  
    '    N= %d  K= %d  # Nodes= %d' % (N, K, len(I)))
  return L, I

def compdist(k1, k2, I, D, usekeys=None):
  ''' Compares distributions of key 1 to key 2 using keys in KEY 1'''
  # keys = [ord(i)-97 for i in (k1 if len(k1)>len(k2) else k2)]
  keys = tofz(k1) if usekeys is None else usekeys
  return cheapEMD(D[I[k1]][:,keys], D[I[k2]][:,keys])

def mcl_clustering(G, n=1, pow=2, inf=2):
  N = len(G)
  theta = 1/N
  M=G
  M = M/M.sum(0)
  for i in range(n):
    M = LA.matrix_power(M, pow)
    for elm in np.nditer(M, op_flags=['readwrite']):  
      elm[...] = elm ** inf
    M = M/M.sum(0)
  return M

def inter_clu(M):
  assigned = [-1 for i in range(466)]
  connlist  = [set() for i in range(N)]
  connlist2  = [set() for i in range(N)]
  follow = np.where(M.sum(1) < 1)[0].astype(int)
  attrac = np.where(M.sum(1) >= 1)[0].astype(int)
  for n in range(466):
      for i in np.where(M[:,n] > 0)[0]:
        connlist[n].add(i)
      for i in np.where(M[n] > 0)[0]:
        connlist2[n].add(i)
  newgrp = 0

def kstest(D):
  ''' Implements Kolmogorov-Smirnov test on the provided distribution to test how
  close D is to gaussian norm. Uses mean and std of the distribution to create the 
  norm CDF and then performs a step-wise difference for each point in D. 
  D must be a 1-D ndarray'''
  mu, sigma = D.mean(), D.std()
  gaus = stats.norm(loc=mu, scale=sigma)
 

def parent_map(L, bylevel=False):
  pmap = []
  ncount = 0
  for k in range(len(L)-1):
    Lk = OrderedDict()
    n = 0
    st = dt.now()
    for child in L[k]:
      nodes = []
      for idx, parent in enumerate(L[k+1]):
        if child < parent:
          nodes.append(idx)
          n += 1
      Lk[child] = nodes
    pmap.append(Lk)
    ncount += n
    print('L=%2d'%k, 'Child-ParNodes: %6d /%6d'%(len(L[k]), len(L[k+1])),
      '  #edges: %6d'%n, '  Time: ', (dt.now()-st).total_seconds())
  print("TOTAL # Edges:  ", ncount)
  return pmap
 
def child_map(L, bylevel=False):
  cmap = [{} for i in range(len(L))] if bylevel else OrderedDict()
  ncount = 0
  for k in range(len(L)-1, 0, -1):
    if bylevel:
      Lk = OrderedDict()
    n = 0
    st = dt.now()
    for parent in L[k]:
      nodes = []
      for child in L[k-1]:
        if child < parent:
          nodes.append(tok(child))
          n += 1
      if bylevel:
        Lk[tok(parent)] = nodes
      else:
        cmap[tok(parent)] = nodes
    if bylevel:
      cmap[k] = Lk
    ncount += n
    print('L=%2d'%k, 'Child-ParNodes: %6d /%6d'%(len(L[k-1]), len(L[k])),
      '  #edges: %6d'%n, '  Time: ', (dt.now()-st).total_seconds())
  print("TOTAL # Edges:  ", ncount)
  return cmap

def child_parent_maps(L, haskey=False):
  cmap, pmap = OrderedDict(), OrderedDict()
  ncount = 0
  # Initialize
  for Lk in L:
    for node in sorted(tok(i) for i in Lk):
      cmap[node] =  []
      pmap[node] =  []
  for k in range(len(L)-1, 0, -1):
    n=0
    st = dt.now()
    for parent in L[k]:
      pkey = tok(parent)
      for child in L[k-1]:
        if child < parent:
          ckey = tok(child)
          cmap[pkey].append(ckey)
          pmap[ckey].append(pkey)
          n += 1
    print('L=%2d'%k, 'Child-ParNodes: %6d /%6d'%(len(L[k-1]), len(L[k])),
      '  #edges: %6d'%n, '  Time: ', (dt.now()-st).total_seconds())
    ncount += n
  print("TOTAL # Edges:  ", ncount)
  return cmap, pmap

def derived_lattice_simple(E, L, I, nbins=20, binrange=None):
  dlattice = {}
  tottime = 0.
  nnodes, nedges = 0, 0
  print("Finding all Parent-Child Edges...")
  pmap = parent_map(L)
  for k in range(1, len(L)-1):
    print('Processing %d nodes with features sets of len: %d' % (len(L[k]), k))
    times = []
    emdtot, emdtime = 0, 0.
    md1, md2, md3, nt = [], [], [], []
    nedges_k = 0
    for child, parent_list in pmap[k].items():
      nnodes += 1
      flist = sorted(child)
      d1 = E[I[child]][:,flist]
      child_key = tok(child)
      dlattice[child_key] = {}
      for pidx in parent_list:
        parent =  L[k+1][pidx]
        start = m1 = dt.now()
        if child < parent:
          m2 = dt.now()
          d2 = E[I[parent]][:,flist]
          m3 = dt.now()
          dlattice[child_key][tok(parent)] = cheapEMD(d1, d2, nbins=nbins, binrange=binrange)
          nedges_k += 1
          m4 = dt.now()
          md1.append((m2-m1).total_seconds())
          md2.append((m3-m2).total_seconds())
          md3.append((m4-m3).total_seconds())
        else:
          nt.append((dt.now()-m1).total_seconds())
        times.append((dt.now()-start).total_seconds())
    nedges += nedges_k

    print('  TIMES:  #Edges= %7d'% nedges_k, '  Avg EMD= %6.4f  '%np.mean(times),
      '  TotTime= %6.3f'% np.sum(times))

    print('   Bench Time:  %7.4f    %7.4f    %7.4f     %7.4f' % 
      (sum(md1), sum(md2), sum(md3), sum(nt)))
    tottime += sum(times)
  print('\n Total Time = %7.2f   Nodes = %d   Edges = %d' % (tottime, nnodes, nedges))
  return dlattice

def dlat_trace(Ik, D, key, pmap):
  dlattice = {}
  idx = toidx(key)
  for child, items in Ik.items():
    if key not in child or len(items) < 5:
      continue
    d1 = D[items][:,idx]
    dlattice[child] = {}
    for parent in pmap[child]:
      if key not in parent or not(set(child) < set(parent)) or len(Ik[parent]) < 5:
        continue
      d2 = D[Ik[parent]][:,idx]
      dlattice[child][parent] = cheapEMD(d1, d2)
  return dlattice

def dlat_by_depth(dlat):
  max_k = max([len(k) for k in dlat.keys()])
  L = [{}]
  for k in range(1, max_k):
    ca = [i for i in dlat.keys() if len(i) == k]
    L.append({key: dlat[key] for key in sorted(ca)})
  return L

def sublattice(L, key):
  apex = set(key)
  sublat_keys = [k for k in L.keys() if set(k) <= apex]
  sublat = {}
  for fs in sublat_keys:
    node = {k: v for k,v in L[fs].items() if set(k) <= apex}
    sublat[fs] = node
  sublat[key] = {}
  return sublat

def enumpaths(a, b, dlat):
  if len(a) + 1 == len(b):
    return [[(a,b,dlat[a][b])]]
  subpaths = []
  for p in dlat[a]:
    if set(p) < set(b):
      for sp in enumpaths(p, b, dlat):
        subpaths.append([(a,p,dlat[a][p])] + sp)
  return subpaths

def enumpaths_distr(a, b, dlat):
  if len(a) + 1 == len(b):
    return [[dlat[a][b]]]
  subpaths = []
  for p in dlat[a]:
    if set(p) < set(b):
      for sp in enumpaths_distr(p, b, dlat):
        subpaths.append([dlat[a][p]] + sp)
  return subpaths

def getpaths(a, b, dlat):
  nodelist = {a, b}
  for klen in range(len(a)+1, len(b)):
    for ca in it.combinations(b, klen):
      if a in ca:
        nodelist.add(''.join(sorted(ca)))
  nodelist = sorted(nodelist, key=lambda x: (len(x), x))
  for c in nodelist:
    if c not in dlat:
      continue
    for p in nodelist:
      if p  in dlat[c]:
        print('%6s%6s%7.3f' % (c,p,dlat[c][p]))
  return nodelist


def botup_cluster(dlat, D, Ik, theta1=.95, theta2=1, minclu=3, bL=None):
  
  N, K = D.shape

  grpset = botup_collapse(dlat, theta=theta1)

  clusters = {k: set(Ik[k]) for k in grpset.keys()}
  for k, v in grpset.items():
    for i in v: clusters[k] |= set(Ik[i])

  # Convert to list (for indexing)
  for k in clusters.keys():
    clusters[k] = sorted(clusters[k])

  #  TODO:  Is full distribution and/or centroid methods applicable?
  keylist = [i[0] for i in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]
  Nc = len(keylist)
  maxsize = max([len(v) for v in clusters.values()])
  minsize = min([len(v) for v in clusters.values()])

  # Calc centroid & variance
  centroid, variance = np.zeros(shape=(Nc, K)), np.zeros(shape=(Nc, K))
  for i, k in enumerate(keylist):
    centroid[i] = D[clusters[k]].mean(0)
    variance[i] = D[clusters[k]].std(0)

  # TODO: Optimize with vectorization (if needed)
  G = np.ones(shape=(Nc, Nc))
  for i in range(Nc-1):
    for j in range(i+1, Nc):
      G[i][j] = G[j][i] = LA.norm(centroid[i] - centroid[j])

  nnlist = [np.argmin(G[i]) for i in range(Nc)]
  
  inum = 0
  expelled = []
  maxval = G.max()
  while np.min(G) < theta2:
  # while minsize < minclu:
    keylist = [i[0] for i in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]
    Nc = len(clusters)
    minidx = np.argmin(G)
    row = minidx // Nc
    col = minidx % Nc
    np.fill_diagonal(G, 0)
    print ('%4d.'%inum, '%5.2f / %5.2f'% (np.mean(G), np.min(G)), '%4d->%4d' % (row, col), ' Sizes: %4d to %4d' % (minsize, maxsize),
      'extMEAN= %5.2f' % G.mean(), 'intVAR= %5.2f' % variance.mean())
    np.fill_diagonal(G, 1)
    m_attr  = row if len(clusters[keylist[row]]) < len(clusters[keylist[col]]) else col
    m_donor = col if m_attr == row else row
    k_attr, k_donor = keylist[m_attr], keylist[m_donor]
    for elm in clusters[k_donor]:
      clusters[k_attr].append(elm)
    G[m_donor] = maxval
    G[:,m_donor] = maxval
    centroid[m_attr] = D[clusters[k_attr]].mean(0)
    variance[m_attr] = D[clusters[k_attr]].std(0)
    for i in range(Nc):
      if i == m_attr:
        continue
      G[m_attr][i] = G[i][m_attr] = LA.norm(centroid[m_attr] - centroid[i])
    del clusters[k_donor]
    G = np.delete(G, m_donor, 0)
    G = np.delete(G, m_donor, 1)
    maxsize = max([len(v) for v in clusters.values()])
    minsize = min([len(v) for v in clusters.values() if len(v) > 0])
    inum += 1

  for k in [keylist[i] for i in expelled]:
    del clusters[k]

  if bL is not None:
    printclu(clusters, bL, minclu, incldist=True)

  return clusters


def cluster_lattice(clu):
  keylist = sorted(clu.keys(), key=lambda x: (len(x), x))


def cluster(dlat, Ik, CMr, D, theta=.9, minclu=5, bL=None):
  ''' Clustering algorithm which assigns each item (event/basin) to a 
  grouping in the lattice'''
  
  N, K = D.shape

  # 1.  Build feature based group sets in the lattice
  # gs, ga = topdown_group_single(dlat, theta)
  gs = botup_group(dlat, theta)

  # 2.  Create list of items for each group
  Ig = {k: sorted(it.chain(*[Ik[i] for i in v])) for k,v in gs.items()}
  keylist = sorted(Ig.keys(), key=lambda x: (len(x), x))

  # 3. Calc centroids for each grouping based on included events for itemset
  print('Calculating Centroids...')
  centroid = {k: np.mean(D[sorted(v)], axis=0) for k,v in Ig.items()}

  # 4. Find euclidean distance for each event to each centroid
  print('Event to centroid distances...')
  eucl = np.zeros(shape=(N, len(Ig)))
  for i, key in enumerate(keylist):
    eucl[:,i] = LA.norm(D - centroid[key], axis=1)

  # 5. Assign clusters
  clusters = defaultdict(list)
  for i, delta in enumerate(eucl):
    idx = np.argmin(delta)
    clusters[keylist[idx]].append(i)

  print("  TOTAL # of GROUP   :   ", len(gs))
  print("  TOTAL # Clusters   :   ", len(clusters))

  # FOR DEBUG/DISPLAY
  if bL is not None:
    printclu(clusters, bL, minclu=minclu, incldist=True)    
  return clusters

def cluster_1(dlat, Ik, CMr, D, theta=.5, minclu=5, bL=None):
  ''' Clustering algorithm which assigns each item (event/basin) to a 
  grouping in the lattice'''
  
  # 1.  Build feature based group sets in the lattice
  gs, ga = topdown_group_single(dlat, theta)

  # # 2.  Create list of items for each group
  # Ig = {k: sorted(it.chain(*[Ik[i] for i in v])) for k,v in gs.items()}

  grp, nogrp = {}, []

  ### OPTION A:  Map to corr feature set (for unique key)
  # 3.  For each item: map to a group:   <map>
  clusters = defaultdict(list)
  for i in range(len(CMr)):
    k = fromm(CMr[i])
    #  Set its group 
    if k in ga:
      grp[i] = ga[k]
      clusters[ga[k]].append(i)
    # Keep track of unassigned nodes (TODO: is this just noise?)
    else:
      nogrp.append(i)

  print("  TOTAL # of GROUP   :   ", len(gs))
  print("  Items w/no group   :   ", len(nogrp))


  # ### OPTION B:  For all nodes wo/grp: Map to "bestfit" grouping based on RMS to group centroid
  # centroid = {k: D[v].mean(0) for k,v in clusters.items()}
  # # 4.  For each item: map to a group:   <map>
  # for i in nogrp:
  #   item  = D[i]
  #   bestfit, min_d = '', 100000
  #   for k, cent in centroid.items():
  #     delta = LA.norm(item - cent)
  #     if delta < min_d:
  #       bestfit, min_d = k, delta
  #   grp[i] = bestfit
  #   clusters[bestfit].append(i)

  # 5. Reorganize (group-by) element  <reduce>
  # clu = {k: [] for k in set(grp.values())}
  # for k,v in grp.items(): 
  #   clu[v].append(k)

  # FOR DEBUG/DISPLAY
  if bL is not None:
    printclu(clusters, bL, minclu=minclu, incldist=True)    
  return clusters, nogrp

def recluster(clu, Dr, theta= .5, minclu=3, bL=None):
  keylist = sorted(clu.keys(), key=lambda x: (len(x), x))
  min_k, max_k = [f([len(i) for i in keylist]) for f in [min, max]]
  Lc = [[i for i in keylist if len(i) == k] for k in range(min_k, max_k+1)]
  lat = {k: {} for k in keylist}
  I = {k: sorted(clu[k]) for k in keylist}

  for k in range(len(Lc)-1, 0, -1):
    for parent in Lc[k]:
      for child in Lc[k-1]:
        common_keys = set(child) & set(parent)
        union_keys = set(child) | set(parent)
        if len(common_keys) > 0:
          try:
            keyset = toidx(sorted(union_keys))
            d_c = Dr[I[child]][:,keyset]
            d_p = Dr[I[parent]][:,keyset]
            lat[child][parent] = cheapEMD(d_c, d_p)        
          except IndexError as err:
            print ('ERROR:   ', child, parent, clu[child], clu[parent], keyset)
            raise IndexError

  gs, ga = topdown_group_single(lat, theta)
  clusters = {k: set(it.chain(*[clu[i] for i in v])) for k,v in gs.items()}

  if bL is not None:
    n = 0
    print("TOTAL # of CLUSTERS:   ", len(clusters))
    clusterlist = []
    for idx, (k,v) in enumerate(clusters.items()):
      if len(v) > minclu: 
        bc  = np.bincount([bL[i] for i in v], minlength=3)
        state = np.argmax(bc)
        stperc = 100*bc[state] / sum(bc)
        clusterlist.append((k, len(v), state, stperc))
        # print('%2d.'%n, '%-15s'%k, '%4d '%len(v), 'State: %d  (%4.1f%%)' % (state, stperc))
        n += 1
    for i in sorted(clusterlist, key =lambda x : x[1], reverse=True):
      print('%-15s%5d  /  %d - (%4.1f%%)' % i)

  return clusters

def samplecluster(clu, Dr, num):
  centroid = {k: Dr[v].mean(0) for k,v in clu.items()}
  prox = {k: {i: LA.norm(Dr[i] - centroid[k]) for i in v} for k,v in clu.items()}
  # itemcnt={k:sorted((k, len(v)) for k,v in clu.items()}
  {k: [i + (bL[i[0]],) for i in sorted(prox[k].items(), key=lambda x: x[1])] for k in prox.keys()}
  ['%4d (%5.2f) '%i + '%d ' %bL[i[0]] for i in sorted(prox['bcdfikmprtuv'].items(), key=lambda x: x[1])]

def printclu(clusters, bL, minclu=3, incldist=False, variance=None):
  print("  TOTAL # CLUSTERS   :   ", len(clusters))
  if bL is not None:
    n = 0
    clusterlist = []
    for idx, (k,v) in enumerate(clusters.items()):
      if len(v) > minclu: 
        bc  = np.bincount([bL[i] for i in v], minlength=5)
        state = np.argmax(bc)
        stperc = 100*bc[state] / sum(bc)
        elms = (n, k, len(v), state, stperc, bc) if incldist else (n, k, len(v), state, stperc)
        clusterlist.append(elms)
        # print('%2d.'%n, '%-15s'%k, '%4d '%len(v), 'State: %d  (%4.1f%%)' % (state, stperc))
        n += 1
    for i in sorted(clusterlist, key =lambda x : x[2], reverse=True):
      if incldist:
        print('%3d.  %-17s%5d  /  State: %d  (%5.1f%%)   %s' % i)
      else:
        print('%3d.  %-17s%5d  /  State: %d  (%5.1f%%)' % i)

def mergecluster(clu, Dr, theta=1., minclu=3, bL=None):
  ''' USes full distrubution of clustered items to merge clusters based 
  on eucidean distance to centroid '''

  #  TODO:  Is full distribution and/or centroid methods applicable?

  keylist = [i[0] for i in sorted(clu.items(), key=lambda x: len(x[1]), reverse=True)]
  # Make a copy
  clusters = copy.deepcopy(clu)

  # Calc centroids
  centroid = {k: Dr[clu[k]][:,toidx(k)].mean(0) for k in keylist}

  Nc = len(keylist)
  G = np.ones(shape=(Nc, Nc))
  for i in range(Nc):
    a = keylist[i]
    kidx = toidx(a)
    for j in range(Nc):
      if i == j:
        G[i][i] = theta
        continue
      b = keylist[j]
      G[i][j] = LA.norm(Dr[clu[a]][:,kidx].mean(0) - Dr[clu[b]][:,kidx].mean(0))

  nnlist = [np.argmin(G[i]) for i in range(Nc)]

  for i in range(Nc-1, 0, -1):
    # Do not merge this
    if nnlist[i] > i:
      continue
    # Break up the cluster & reassign
    if nnlist[i] == i:
      for elm in clusters[keylist[i]]:
        bestfit, bestclu = 100000, -1
        for j in range(i+1):
          keyset = toidx(keylist[i])
          d = LA.norm(Dr[j][keyset] - centroid[keylist[i]])
          if d < bestfit:
            bestclu = j
            bestfit = d
        if bestclu == 1:
          continue
        clusters[keylist[bestclu]].append(elm)
        clusters[keylist[i]].remove(elm)
        # Should we wait to recalc this???
        centroid[keylist[bestclu]] = Dr[clusters[keylist[bestclu]]][:,toidx(keylist[bestclu])].mean(0)
        centroid[keylist[i]] = Dr[clusters[keylist[i]]][:,toidx(keylist[i])].mean(0)
      if len(clusters[keylist[i]]) == 0:
        del clusters[keylist[i]]
    # Merge to nn
    else:
      nnkey = keylist[nnlist[i]]
      for elm in clusters[keylist[i]]:
        clusters[nnkey].append(elm)
      centroid[nnkey] = Dr[clusters[nnkey]][:,toidx(nnkey)].mean(0)
      del clusters[keylist[i]]

  print("  TOTAL # CLUSTERS   :   ", len(clusters))
  if bL is not None:
    n = 0
    clusterlist = []
    for idx, (k,v) in enumerate(clusters.items()):
      if len(v) > minclu: 
        bc  = np.bincount([bL[i] for i in v], minlength=3)
        state = np.argmax(bc)
        stperc = 100*bc[state] / sum(bc)
        var = Dr[v][:,toidx(k)].std(0).mean()
        clusterlist.append((n, k, len(v), state, stperc, var))
        # print('%2d.'%n, '%-15s'%k, '%4d '%len(v), 'State: %d  (%4.1f%%)' % (state, stperc))
        n += 1
    for i in sorted(clusterlist, key =lambda x : x[2], reverse=True):
      print('%3d.  %-15s%5d  /  State: %d  (%5.1f%%)   /   %5.2f' % i)
  return clusters

def reverse_dlat(dlat):
  rd = {k: {} for k in dlat.keys()}
  for child, parentlist in dlat.items():
    for parent in parentlist:
      rd[parent][child] = dlat[child][parent]
  return rd


def botup_cluster_1(dlat, CMr, theta=.02, minclu=3, bL=None):
  # get smallest keys
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x))
  setlist = defaultdict(set)
  for node in klist:
    if len(dlat[node]) == 0:
      continue
    parent, value = min(dlat[node].items(), key = lambda x: x[1])
    if value <= theta:
      setlist[parent] |= setlist[node].union({parent})
      del setlist[node]

  nogrp = []

  ### OPTION A:  Map to corr feature set (for unique key)
  # 3.  For each item: map to a group:   <map>
  clusters = defaultdict(list)
  for i in range(len(CMr)):
    k = fromm(CMr[i])
    found = False
    #  TODO: optimize (linear searches) 
    for grp, nodelist in setlist.items():
      if k in nodelist:
        clusters[grp].append(i)
        found = True
        break
    # Keep track of unassigned nodes (TODO: is this just noise?)
    if not found:
      nogrp.append(i)

  print("  TOTAL # of GROUP   :   ", len(setlist))
  print("  Items w/no group   :   ", len(nogrp))
  # FOR DEBUG/DISPLAY
  if bL is not None:
    printclu(clusters, bL, minclu=minclu, incldist=True)    
  return clusters

def topdown_group(dlat, theta=.02, pmap=None):
  ''' Single-Pass lattice clustering. Iterates through each node from 
  top to bottom and assigns each node to a group. Grouping is based solely
  on the child-parent relationship:
      If none:  Create a new group
      Else:     Assign to parent node with min EMD provided emd < theta
         if min(emd) > theta:  create a new group'''
  # 1. Build list of keys from top to bottom
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True)
  
  # # 1a. Groupby level
  bylevel = [[] for k in range(len(klist[0]))]
  for k in klist:
    bylevel[len(k)].append(k)

  grpsets    = {}
  assignment = {}
  notop = 0
  # 2. For each key: assign to a group or start a new one
  for level in bylevel:
    topnodes = {}
    for k in level:
      # Node has parents:  check if it should be assigned 
      if len(dlat[k]) > 0:
        # Get min (emd)
        key, emd = min(dlat[k].items(), key = lambda x: x[1])
        # assign to smallest emd (if < theta)
        if emd < theta:
          grpsets[assignment[key]].add(k)
          assignment[k] = assignment[key]
          continue
      else:
        topnodes[k] = {}

    # Create join points for each pairing for all top nodes and 
    ntop = len(topnodes)
    supernodes = []
    for i, na in enumerate(ntop[:-1]):
      for nb in ntop[i+1:]:
        supernodes.append(''.join(set(na)|set(nb)))

    # Calc EMD to all parent
      # Create a new group
      grpsets[k]    = set({k})
      assignment[k] = k
  print(' NO TOP=', notop)
  return grpsets, assignment

def topdown_group_single(dlat, theta=.02):
  ''' Single-Pass lattice clustering. Iterates through each node from 
  top to bottom and assigns each node to a group. Grouping is based solely
  on the child-parent relationship:
      If none:  Create a new group
      Else:     Assign to parent node with min EMD provided emd < theta
         if min(emd) > theta:  create a new group'''
  # 1. Build list of keys from top to bottom
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True)
  
  grpsets    = {}
  assignment = {}
  notop = 0

  # 2. For each key: assign to a group or start a new one
  for k in klist:
    # Node has parents:  check if it should be assigned 
    if len(dlat[k]) > 0:
      # Get min (emd)
      key, emd = min(dlat[k].items(), key = lambda x: x[1])
      # assign to smallest emd (if < theta)
      if key in assignment and emd < theta:
        grpsets[assignment[key]].add(k)
        assignment[k] = assignment[key]
        continue
    else:
      notop += 1
    # Create a new group
    grpsets[k]    = set({k})
    assignment[k] = k
  print(' NO TOP=', notop)
  return grpsets, assignment



def topdown_group_multi(dlat):
  ''' Multi-Pass lattice clustering. Iterates through each node from 
  top to bottom and assigns each node to a group. Grouping is based solely
  on the child-parent relationship:
      If none:  Create a new group
      Else:     Assign to parent node with min EMD provided emd < theta
         if min(emd) > theta:  create a new group'''
  # 1. Build list of keys from top to bottom
  dlw = {k: {i: 1-j for i, j in v.items() if j<1} for k,v in dlat.items()}
  klist = sorted(dlw.keys(), key=lambda x: (len(x), x), reverse=True)

  bylevel = [[] for k in range(len(klist[0])+1)]
  for k in klist:
      bylevel[len(k)].append(k)

  # Add Supernodes for all keys with no parents
  notop = [k for k,v in dlw.items() if len(v) == 0]
  while len(notop) > 1000:
    node = notop.pop()
  nkeys = set(node)
  for k in bylevel[len(node)]:
    plist = [''.join(sorted(nkeys.union(set(k)))) for k in bylevel[len(node)]]

  klist = sorted(dlw.keys(), key=lambda x: (len(x), x), reverse=True)
  
  grpsets    = {}
  assignment = {}
  notop = 0
  theta = 1.
  # 2. For each key: assign to a group or start a new one
  for k in klist:
    # Node has parents:  check if it should be assigned 
    if len(dlat[k]) > 0:
      # Get max (emd)
      key, emd = max(dlw[k].items(), key = lambda x: x[1])
      # assign to smallest emd (if < theta)
      if key in assignment and emd < theta:
        grpsets[assignment[key]].add(k)
        assignment[k] = assignment[key]
        continue
    else:
      notop += 1
    # Create a new group
    grpsets[k]    = set({k})
    assignment[k] = k
  print(' NO TOP=', notop)
  return grpsets, assignment



def topdown_group_uni(dlat, L, theta=.02, cmap=None):
  # get smallest keys
  if cmap is None:
    cmap = child_map(L)
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True)
  grpsets = defaultdict(set)
  assigned = {k: False for k in klist}
  stnodes = {klist[0]}
  while len(stnodes) > 0:
    rootnode = stnodes.pop()
    if assigned[rootnode]:
      continue
    curtree = {rootnode}
    while len(curtree) > 0:
      node = curtree.pop()
      if assigned[node]:  
        continue
      grpsets[rootnode].add(node)
      assigned[node] = True
      for child in cmap[node]:
        if assigned[child]:
          continue
        if dlat[child][node] <= theta:
          curtree.add(child)
        else:
          stnodes.add(child)
  return grpsets

def topdown_group_bidir(dlat, L, theta=.02):
  # get smallest keys
  cmap = child_map(L)
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True)
  grpsets = defaultdict(set)
  assigned = {k: False for k in klist}
  stnodes = {klist[0]}
  while len(stnodes) > 0:
    node = stnodes.pop()
    if assigned[node]:
      continue
    curtree = {node}
    while len(curtree) > 0:
      parent = curtree.pop()
      if assigned[parent]:
        continue
      grpsets[node].add(parent)
      assigned[parent] = True
      for child in cmap[parent]:
        if assigned[child]:
          continue
        if dlat[child][parent] <= theta:
          curtree.add(child)
        else:
          stnodes.add(child)
  return grpsets

def botup_group(dlat, theta=.5):
  
  # Change from delta to similarity metric
  dlw = {k: {i: 1-j for i, j in v.items() if j<1} for k,v in dlat.items()}
  
  # get shorter keys
  klist = sorted(dlw.keys(), key=lambda x: (len(x), x))
  setlist = defaultdict(set)
  for node in klist:
    if len(dlw[node]) == 0:
      continue

    #  Assign ea node to at most one parent
    parent, value = max(dlw[node].items(), key = lambda x: x[1])
    if value >= theta:
      setlist[parent] |= setlist[node].union({parent})
      del setlist[node]
  return setlist


def botup_merge(dlat, Ik, theta=.9):
  # Can be optimized if needed
  dlw = {k: {i: 1-j for i, j in v.items() if j<1} for k,v in dlat.items()}
  # I = {k: set(i) for k,i in Ik.items()}
  I = {}
  klist = sorted(dlw.keys(), key=lambda x: (len(x), x))

  prog_cnt, tot_iter, tick = 0, len(klist), len(klist)//100
  print ('\r[%-100s] %4.1f%%' % ('', 0), end='')
  for i, node in enumerate(klist):
    if len(dlw[node]) == 0:
      continue
    I[node] = set(Ik[node])
    for parent, emd in dlw[node].items():
      if emd < theta:
        continue
      I[node] -= set(Ik[parent])
      if len(I[node]) == 0:
        del I[node]
        break
    prog_cnt += 1
    if prog_cnt % tick == 0:
      prog = 100*prog_cnt/tot_iter
      print ('\r[%-100s] %4.1f%%' % ('#'*int(prog), prog), end='')

  return I



def assign_mfis(CMr, MFIS, dlat):
  dlw = {k: {i: 1-j for i, j in v.items() if j<1} for k,v in dlat.items()}
  mk = [tok(k) for k in MFIS]
  clusters = defaultdict(list)

  for i, d in enumerate(CMr):
    key = fromm(d)
    node = key
    while node not in mk and len(dlw[node]) > 0:
      node = max(dlw[node].items(), key=lambda x: x[1])[0]
    clusters[node].append(i)

  return clusters

# dlat3 = {key: {k: np.sum(np.abs(list(v.values()))) for k,v in distr.items()} for key, distr in dlat2.items()}

def edgelist(dlat):
  elist = {}
  klist = list(dlat.keys())
  kmap = {k: i for i, k in enumerate(klist)}
  for node, distr in dlat.items():
    nidx = kmap[node]
    for parent, val in distr.items():
      elist[(nidx, kmap[parent])] = val
  return elist, klist


merge_theta = .05
def merge_children(cmap, dlat, node):
  k=len(node)
  if cmap[k][node] == []:
    return set()
  nodeset = set({node}) 
  for child in cmap[k][node]:
    if dlat[child][node] < merge_theta and child not in nodeset:
      nodeset |= merge_children(cmap, dlat, child)
  return nodeset

def group_lattice_nodes(L, dlat, theta = .05, cmap=None, pmap=None):
  if cmap==None:
    cmap, pmap = child_parent_maps(L)
  grpmap  = {k: -2 for k in cmap.keys()}
  group_list = []
  next_grp_num = 0
  group_neighbors = []
  ca_node_list = deque(sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True))
  while len(ca_node_list) > 0:
    nextnode = ca_node_list.popleft()
    new_group = [nextnode]
    tba_list = deque(new_group)
    grpmap[nextnode] = -1
    neighbors = set()
    while len(tba_list) > 0:
      node = tba_list.popleft()
      for child in cmap[node]:
        if grpmap[child] == -2 and dlat[child][node] < theta:
          grpmap[child] = -1
          ca_node_list.remove(child)
          tba_list.append(child)
          new_group.append(child)
        elif grpmap[child] >= 0 and dlat[child][node] < theta:
          neighbors.add(grpmap[child])
      for parent in pmap[node]:
        if grpmap[parent] == -2 and dlat[node][parent] < theta:
          grpmap[parent] = -1
          ca_node_list.remove(parent)
          tba_list.append(parent)
          new_group.append(parent)
        elif grpmap[parent] >= 0 and dlat[node][parent] < theta:
          neighbors.add(grpmap[parent])
    print("Assigning Group # %3d  to %5d  Nodes.   Unassigned:  %6d" % (next_grp_num, len(new_group), len(ca_node_list)))
    for node in new_group:
      grpmap[node] = next_grp_num
    group_list.append(new_group)
    group_neighbors.append(neighbors)
    next_grp_num += 1
  return grpmap, group_list, group_neighbors

def group_lattice_nodes_edges(dlat):
  elist, klist = edgelist(dlat)
  sorted_edges = sorted(elist.items(), key=lambda x: x[1])
  Ne = len(sorted_edges)
  grp_map = {}
  grp_list = []
  enum = 0
  next_group_num = 0
  while len(grp_map) < len(dlat) and enum < Ne:
    (ch, par), val = sorted_edges[enum]
    if ch not in grp_map:
      if par not in grp_map:
        grp_map[ch] = grp_map[par] = next_group_num
        grp_list.append({ch, par})
        next_group_num += 1
      else:
        grp_map[ch] = grp_map[par]
    else:
      if par not in grp_map:
        grp_map[par] = grp_map[ch]
      else:
        if grp_map[par] != grp_map[ch]:
          print('Merge: ', par, ch, grp_map[par], grp_map[ch], len(grp_list), grp_list[grp_map[ch]])
          grp_list[grp_map[ch]] |= grp_list[grp_map[par]]
          for n in grp_list[grp_map[par]]:
            grp_map[n] = grp_map[ch]
          del grp_list[grp_map[par]]
          # print("Group Assignement Mismatch: ", 
          #   klist[ch], grp_map[ch], klist[par], grp_map[par])
    enum += 1
    if enum % 1000 == 0:
      prog = 100*enum/Ne
      print ('\r[%20s] %4.2f%%' % ('#'*int(prog/5), prog), end='')
      # print '\r[{0}] {1}%'.format('#'*(progress/10), progress)
  return grp_list

def edgemap(L, I, E):
  edgemap = defaultdict(list)
  for k in range(len(L)-1):
    for fs in L[k]:
      for parent in L[k+1]:
        if fs < parent:

          s = E[I[fs]][:,5]
          cdf[tok(fs)] = sorted(s)
          m, sd  = s.mean(0), s.std(0)
          print('%7s'%tok(fs),'%5d'%len(s), '%5.2f    %6.3f' % (m, sd))

          edgemap[fs].append(parent)
  return edgemap



# def apriori(M, epsilon):
#   N, K = M.shape
#   T = [np.where(e==1) for e in M]
#   base_itemsets = [set(np.where(CM[:,i]==1)[0]) for i in range(K)]
#   event_map = [{tuple(k): I for k, I in enumerate(base_itemsets) if len(I) > epsilon}]
#   L = [[set(i) for i in event_map.keys()]]
#   k = 2
#   while len(L[-1]) > 0:
#     Ck = combinatorics of all sets from L[k-1]
#     Ck = set()
#     for a in L[-1]:
#       Ck.update([a | set(b) for b in range(K) if b not in a])

  # base_itemsets = [frozenset(np.where(M[:,i]==1)[0]) for i in range(K)]
  # L = [frozenset([k]) for k, I in enumerate(base_itemsets) if len(I) > epsilon]
    # Ck = [frozenset(set.union(*i)) for i in it.combinations(L[k-1], k)]


def apriori_orig(M, epsilon):
  N, K = M.shape
  T = [frozenset(np.where(e==1)[0]) for e in M]
  base_itemsets = [SortedSet(np.where(M[:,i]==1)[0]) for i in range(K)]
  L = [[], [frozenset({k}) for k, icount in enumerate(M.sum(0)) if icount >= epsilon]]
  P = [[], [frozenset({k}) for k, icount in enumerate(M.sum(0)) if icount < epsilon]]
  k = 2
  I = {frozenset({i}): base_itemsets[i] for i in range(K)}
  # I = OrderedDict()
  # for i in range(len(L[1])):
  #   I[(1, i)] = base_itemsets[i]
  while len(L[k-1]) > 0:
    ts0 = dt.now()
    Lk, Pk, Ck, Ik = [], [], [], []
    Ca = it.chain(*[[a.union({b}) for b in range(K) if b not in a] for a in L[k-1]])
    Ia = it.chain(*[[(a,b) for b in range(K) if b not in a] for ai, a in enumerate(L[k-1])])
    Pk = [frozenset(i) for i in it.chain(*[[a|{b} for b in range(K) if b not in a] for a in P[k-1]])]
    # prune:  Ca - {c | s < c and s not in L[k-1]}
    ts1 = dt.now()
    mark = np.zeros(3)
    n = 0
    for c, ai in zip(Ca, Ia):
      mt = [dt.now()]
      prune = False
      for s in P[k-1]:
        if s < c:
          Pk.append(c)
          prune = True
          break
      mt.append(dt.now())
      if not prune:
        Ck.append(frozenset(c))
        Ik.append(ai)
    ts2 = dt.now()
    count = defaultdict(int)
    itemsets = {}
    mark = np.zeros(3)
    for c, (sub_k, merge_k) in zip(Ck, Ik):
      mt = [dt.now()]
      # m = list(c)
      subset = np.array(I[sub_k])
      mask = np.ones(K);                            
      m = list(c)
      mask[m] = 0;                                  mt.append(dt.now())
      # itemset = [i for i in M if (M[i][m]==1).all()];   mt.append(dt.now())
      # idxlist = np.where(M[subset][:,merge_k])[0];  mt.append(dt.now())
      # itemsets[c] = [subset[i] for i in idxlist];   mt.append(dt.now())
      # idxlist = np.where(np.logical_or(mask, M[subset]).all(1))[0];     mt.append(dt.now())
      idxlist = np.where(M[subset][:,merge_k])[0];     mt.append(dt.now())
      itemsets[c] = subset[idxlist];   mt.append(dt.now())
      # itemsets[c] = itemset;     mt.append(dt.now())
      for i in range(len(mt)-1):
        mark[i] += (mt[i+1] - mt[i]).total_seconds()
    print('MARK: ', mark)

    # for t in T:
    #   for c in Ck:
    #     if c <= t:
    #       count[c] += 1
    ts2a = dt.now()
    for c, elm in itemsets.items():
      if len(elm) >= epsilon:
        I[c] = elm
        Lk.append(c)
      else:
        Pk.append(c)
    L.append(Lk)
    P.append(Pk)
    ts3 = dt.now()
    print(k, 'Ca: %5.2f    Prune: %5.2f    Merge: %5.2f  Count: %5.2f' % 
      ((ts1-ts0).total_seconds(), (ts2-ts1).total_seconds(), (ts2a-ts2).total_seconds(),(ts3-ts2a).total_seconds()))
    k += 1
  return L, I



def itermergecluster_1(clu, Dr, theta=1., minclusize=3, bL=None):
  ''' USes full distrubution of clustered items to merge clusters based 
  on eucidean distance to centroid '''

  #  TODO:  Is full distribution and/or centroid methods applicable?

  keylist = [i[0] for i in sorted(clu.items(), key=lambda x: len(x[1]), reverse=True)]
  # Make a copy
  clusters = copy.deepcopy(clu)

  maxsize = max([len(v) for v in clusters.values()])
  minsize = min([len(v) for v in clusters.values()])

  # Calc centroids
  # centroid = {k: Dr[clu[k]][:,toidx(k)].mean(0) for k in keylist}
  centroid = {k: Dr[clu[k]].mean(0) for k in keylist}



  Nc = len(keylist)
  # G = np.ones(shape=(Nc, Nc))
  # for i in range(Nc):
  #   a = keylist[i]
  #   kidx = toidx(a)
  #   for j in range(Nc):
  #     if i == j:
  #       G[i][i] = theta
  #       continue
  #     b = keylist[j]
  #     G[i][j] = LA.norm(Dr[clu[a]][:,kidx].mean(0) - Dr[clu[b]][:,kidx].mean(0))
      # G[i][j] = LA.norm(Dr[clu[a]].mean(0) - Dr[clu[b]].mean(0))


  G = np.ones(shape=(Nc, Nc))
  for i in range(Nc-1):
    for j in range(i+1, Nc):
      G[i][j] = G[j][i] = LA.norm(centroid[i] - centroid[j])

  nnlist = [np.argmin(G[i]) for i in range(Nc)]
  
  inum = 0
  expelled = []
  maxval = G.max()
  while True:
  # while minsize < minclu:
    N = len(clusters)
    minidx = np.argmin(G)
    row = minidx // N
    col = minidx % N
    print ('%4d.'%inum, '%5.2f / %5.2f'% (np.mean(G), np.min(G)), '%4d->%4d' % (row, col), ' Sizes: %4d to %4d' % (minsize, maxsize))
    try:
      m_attr  = row if len(clusters[keylist[row]]) < len(clusters[keylist[col]]) else col
      m_donor = col if m_attr == row else row
    except KeyError as err:
      print(row, col, expelled)
    k_attr, k_donor = keylist[m_attr], keylist[m_donor]
    for elm in clusters[k_donor]:
      clusters[k_attr].append(elm)
    clusters[k_donor] = []  # place holder
    G[m_donor] = maxval
    G[:,m_donor] = maxval
    expelled.append(m_donor)
    ka_idx = toidx(k_attr)
    centroid[k_attr] = Dr[clusters[k_attr]][:,ka_idx].mean(0)
    for i in range(Nc):
      if i in expelled or i == m_attr:
        continue
      G[m_attr][i] = LA.norm(Dr[clusters[k_attr]][:,ka_idx].mean(0) - Dr[clusters[keylist[i]]][:,ka_idx].mean(0))
      ki_idx = toidx(keylist[i])
      G[i][m_attr] = LA.norm(Dr[clusters[k_attr]][:,ki_idx].mean(0) - Dr[clusters[keylist[i]]][:,ki_idx].mean(0))
    maxsize = max([len(v) for v in clusters.values()])
    minsize = min([len(v) for v in clusters.values() if len(v) > 0])
    inum += 1
    if minsize >= minclusize:
      break

  for k in [keylist[i] for i in expelled]:
    del clusters[k]

  if bL is not None:
    printclu(clusters, bL, minclu)

  return clusters




  #### MAXMINER HOLD:
      # x0, x1, x2 = 0,0,0
    # fidx = 0
    # while fidx < len(F)-1:
    #   prune = False
    #   for i in range(fidx+1, len(F)):
    #     xa = dt.now()
    #     if F[fidx] < F[i]:
    #       xb = dt.now(); x0+=(xb-xa).total_seconds(); xa=xb  #4
    #       F.pop(fidx)
    #       prune = True
    #       break
    #     else:

    #   if not prune:
    #     fidx += 1
    # # print('  ---> %6d'%len(F), end='   ,   ')
    # t1 = dt.now(); ts.append((t1-t0).total_seconds()); t0=t1  #4


    # 9. Prune C: remove candidates with a superset in F
    # TO THREAD Thread this part

    # prunelist = deque()
    # def worker(idxlist):
    #   print("S%d" % idxlist[0], len(idxlist))
    #   for cidx in idxlist:
    #     hg, tg = C[cidx]
    #     g = hg + tg
    #     glen = len(g)
    #     for f in F:
    #       if glen <= len(f) and set(g) <= f:
    #         prunelist.append(cidx)
    #         break
    #   print("D-%d" % idxlist[0], end=', ')

    # print("\nThreading...")
    # threads = []
    # ncores = 20
    # for para in range(ncores):
    #   shuffle = [i for i in range(len(C)) if i % ncores == para]
    #   t = threading.Thread(target=worker, args=(shuffle,))
    #   threads.append(t)
    #   t.setDaemon(True)
    #   t.start()

    # for t in threads:
    #   t.join()

    # print('All Threads complete!  # to Prune: ', len(prunelist))
    # print('   C: ', len(C), end='')
    # for cidx in sorted(prunelist)[::-1]:
    #   C.pop(cidx)