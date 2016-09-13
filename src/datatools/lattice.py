import itertools as it
import string
import numpy as np
import numpy.linalg as LA

from datetime import datetime as dt
from sortedcontainers import SortedSet, SortedList
from collections import OrderedDict, deque, defaultdict

ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek

frommask = lambda obs: ''.join([label_domain[i] for i, x in enumerate(obs) if x])
tomask = lambda key, size: [(1 if i in key else 0) for i in domain[:size]]
toidx = lambda key: [i for i,f in enumerate(K_domain) if f in key]


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
      ia = I[sA].intersection(I[k1key[b]])
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

def child_parent_maps(L):
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



tok   = lambda x: ''.join(sorted([chr(97+i) for i in x]))
def derived_lattice(E, L, I):
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
          dlattice[child_key][tok(parent)] = cheapEMD(d1, d2)
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

# def botup_group(dlat, theta=.02):
#   # get smallest keys
#   klist = sorted(dlat.keys(), key=lambda x: (len(x), x))
#   minlen = len(klist[0])
#   stnodes = set([k for k in klist if len(k) == 1])
#   grpsets = {}
#   for node in stnodes:
#     grpsets[node] = set()
#     sset = {node}
#     while len(sset) > 0:
#       child = sset.pop()
#       grpsets[node].add(child)
#       for parent, val in dlat[child].items():
#         if val > theta or parent in grpsets[node]:
#           continue
#         sset.add(parent)
#   return grpsets

def botup_group(dlat, theta=.02):
  # get smallest keys
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x))
  minlen = len(klist[0])
  stnodes = set([k for k in klist if len(k) == 1])
  grpsets = defaultdict(set)
  assigned = {k: False for k in klist}
  while len(stnodes) > 0:
    node = stnodes.pop()
    if assigned[node]:
      continue
    curtree = {node}
    while len(curtree) > 0:
      child = curtree.pop()
      if assigned[child]:
        continue
      grpsets[node].add(child)
      assigned[child] = True
      for parent, val in dlat[child].items():
        if assigned[parent]:
          continue
        if val <= theta:
          curtree.add(parent)
        else:
          stnodes.add(parent)
  return grpsets



def topdown_group(dlat, L, theta=.02, pmap=None):
  klist = sorted(dlat.keys(), key=lambda x: (len(x), x), reverse=True)
  grpsets    = {}
  assignment = {}
  notop = 0
  for k in klist:
    if len(dlat[k]) > 0:
      key, emd = min(dlat[k].items(), key = lambda x: x[1])
      if emd < theta:
        grpsets[assignment[key]].add(k)
        assignment[k] = assignment[key]
        continue
    else:
      notop += 1
    grpsets[k]    = set({k})
    assignment[k] = k
  print(' NO TOP=', notop)
  return grpsets, assignment






  N = len(G)
  theta = 1/N
  M=G
  M = M/M.sum(0)
  for i in range(n):
    M = LA.matrix_power(M, pow)
    for elm in np.nditer(M, op_flags=['readwrite']):  
      elm[...] = elm ** inf
    M = M/M.sum(0)



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

