import itertools as it
import string
import numpy as np
import numpy.linalg as LA
import scipy.stats as stats
import copy

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

def unique_events(A):
  '''Groups all rows by similarity. Accepts a 1-0 Contact Matrix and 
  returns a mapping from alpha-key to a list of indices with the corresponding
  key'''
  U = defaultdict(list)
  for idx, row in enumerate(A):
    U[frommask(row)].append(idx)
  return U

def reduced_feature_set(A, theta=.02):
  '''Reduces te feature set (columns) by eliminating all "trivial" features.
  Accepts a 1-0 Contact Matrix and identifies all columns whose values are
  (1-theta)-percent 1 or 0. Theta represents a noise threshold. E.g. if theta
  is .02, returned feature set will include column k, if more than 2% of the 
  values are 0 and more than 2% are 1.'''
  (N, K), count  = A.shape, A.sum(0)
  T = theta*N
  trivial = set(it.chain(np.where(count < T)[0], np.where(count > (N-T))[0]))
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
 
## MAX MINER
def maxminer_initgrps (T, epsilon):
  N, K = T.shape
  C = []
  F = [k for k in range(K) if len(np.where(T[:,k]==1)[0] > epsilon)]
  for k in F[:-1]:
    hg = [k] #frozenset({k})
    tg = [j for j in range(k+1, K)]  #frozenset([j for j in range(k+1, K)])
    C.append((hg, tg))
  return C, [[F[-1]]]
  # return C, [frozenset({F[-1]})]

def maxminer_subnodes(g, T, C, epsilon):
  hg, tg = g #g[0], list(g[1])
  i = len(tg)-1
  while i > 0:
    # support = len(np.where(np.logical_and(T[:,list(hg.union({tg[i]}))], True).all(1))[0])
    support = np.sum(np.logical_and(T[:,hg + [tg[i]]], True).all(1))
    if support < epsilon:
      tg.pop(i)
    i -= 1
  if len(tg) > 0:
    for i in range(len(tg)-1):
      # C.append((hg.union({tg[i]}), frozenset([j for j in tg[i+1:]])))
      C.append((hg + [i], [j for j in tg[i+1:]]))
    return C, hg + [max(tg)]  #hg.union({max(tg)})
  return C, hg

def maxminer(T, epsilon):
  N, K = T.shape
  start = dt.now()
  C, F = maxminer_initgrps(T, epsilon)
  n_iter = 0
  times = None
  subtimes = None
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
      s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1
      support = np.sum(np.logical_and(idx, True).all(1))
      # support = len(np.where(np.logical_and(idx, True).all(1))[0])
      # support = np.sum(np.logical_or(T, mask).all(1))
      s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1
      if support >= epsilon:
        F.append(g)
        s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1
        ts2.append(0)
      else:
        C_new, gn = maxminer_subnodes((hg,tg), T, C_new, epsilon)
        F.append(gn)
        ts2.append(0)
        s1 = dt.now(); ts2.append((s1-s0).total_seconds()); s0=s1
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
  keys = [ord(i)-97 for i in (k1 if usekeys is None else usekeys)]
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

def cheapEMD(d1, d2, nbins=20, binrange=None):
  ''' Performs a cheap EMD using 1D histograms. "dirt" is only
  moved to adjacent bins along each dim and is cacluated iteratively 
  from the lowest to highest bin. Simulated "scraping dirt/anti-dirt"
  from bin to bin. The returned val is the sqrt of the sum of squares for
  all dimensions'''
  N1, K1 = d1.shape
  N2, K2 = d2.shape
  flow = np.zeros(K1)
  brange = (3.5, 7) if binrange is None else binrange
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

def derived_lattice(E, L, I, nbins=20, binrange=None):
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



def cluster_lattice(clu):
  keylist = sorted(clu.keys(), key=lambda x: (len(x), x))

def cluster(dlat, Ik, CMr, D, theta=.5, minclu=5, bL=None):
  ''' Clustering algorithm which assigns each item (event/basin) to a 
  grouping in the lattice'''
  
  # 1.  Build feature based group sets in the lattice
  gs, ga = topdown_group_single(dlat, theta)

  # 2.  Create list of items for each group
  Ig = {k: sorted(it.chain(*[Ik[i] for i in v])) for k,v in gs.items()}

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
  return clusters

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

def itermergecluster(clu, Dr, theta=1., minclu=3, bL=None):
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
      # G[i][j] = LA.norm(Dr[clu[a]].mean(0) - Dr[clu[b]].mean(0))

  nnlist = [np.argmin(G[i]) for i in range(Nc)]
  
  inum = 0
  expelled = []
  while np.min(G) < theta:
    N = len(clusters)
    minidx = np.argmin(G)
    row = minidx // N
    col = minidx % N
    # print (inum, np.mean(G), np.min(G), row, col)
    try:
      m_attr  = row if len(clusters[keylist[row]]) < len(clusters[keylist[col]]) else col
      m_donor = col if m_attr == row else row
    except KeyError as err:
      print(row, col, expelled)
    k_attr, k_donor = keylist[m_attr], keylist[m_donor]
    for elm in clusters[k_donor]:
      clusters[k_attr].append(elm)
    # del clusters[k_donor]
    G[m_donor] = theta
    G[:,m_donor] = theta
    expelled.append(m_donor)
    ka_idx = toidx(k_attr)
    centroid[k_attr] = Dr[clusters[k_attr]][:,ka_idx].mean(0)
    for i in range(Nc):
      if i in expelled or i == m_attr:
        continue
      G[m_attr][i] = LA.norm(Dr[clusters[k_attr]][:,ka_idx].mean(0) - Dr[clusters[keylist[i]]][:,ka_idx].mean(0))
      ki_idx = toidx(keylist[i])
      G[i][m_attr] = LA.norm(Dr[clusters[k_attr]][:,ki_idx].mean(0) - Dr[clusters[keylist[i]]][:,ki_idx].mean(0))
    inum += 1

  for k in [keylist[i] for i in expelled]:
    del clusters[k]

  if bL is not None:
    printclu(clusters, bL, minclu)

  return clusters

def samplecluster(clu, Dr, num):
  centroid = {k: Dr[v].mean(0) for k,v in clu.items()}
  prox = {k: {i: LA.norm(Dr[i] - centroid[k]) for i in v} for k,v in clu.items()}
  # itemcnt={k:sorted((k, len(v)) for k,v in clu.items()}
  {k: [i + (bL[i[0]],) for i in sorted(prox[k].items(), key=lambda x: x[1])] for k in prox.keys()}
  ['%4d (%5.2f) '%i + '%d ' %bL[i[0]] for i in sorted(prox['bcdfikmprtuv'].items(), key=lambda x: x[1])]

def printclu(clusters, bL, minclu=3, incldist=False):
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
        print('%3d.  %-15s%5d  /  State: %d  (%5.1f%%)   %s' % i)
      else:
        print('%3d.  %-15s%5d  /  State: %d  (%5.1f%%)' % i)

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


def botup_cluster(dlat, CMr, theta=.02, minclu=3, bL=None):
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
  # bylevel = [[] for k in range(len(klist[0]))]
  # for k in klist:
  #   bylevel[len(k)].append(k)

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

def topdown_group_single(dlat, theta=.02, pmap=None, topdown=True):
  ''' Single-Pass lattice clustering. Iterates through each node from 
  top to bottom and assigns each node to a group. Grouping is based solely
  on the child-parent relationship:
      If none:  Create a new group
      Else:     Assign to parent node with min EMD provided emd < theta
         if min(emd) > theta:  create a new group'''
  # 1. Build list of keys from top to bottom
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=topdown)
  
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
      if emd < theta:
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

