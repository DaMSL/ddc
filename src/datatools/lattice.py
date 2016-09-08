import itertools as it
import string
import numpy as np

from datetime import datetime as dt
from sortedcontainers import SortedSet, SortedList
from collections import OrderedDict, deque, defaultdict

ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek

frommask = lambda obs: ''.join([label_domain[i] for i, x in enumerate(obs) if x])
tomask = lambda key, size: [(1 if i in key else 0) for i in domain[:size]]
toidx = lambda key: [i for i,f in enumerate(K_domain) if f in key]


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
  start = dt.now()
  Vk = set()
  for i in range(K):
    Ik = np.where(M[:,i]==1)[0]
    if len(Ik) > epsilon:
      key = frozenset({i})
      I[key] = SortedSet(Ik)
      Lk.append(key)
      Vk.add(i)
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
    Ck = set()
    for i in L[k-1]:
      for j in Vk:
        if j in i:
          continue
        added_key = frozenset({j})
        ca = i.union(added_key)
        Ck.add(ca)
        Ik[ca] = (i, added_key)
    ts1 = dt.now()
    print('Potential Candidate Nodes (Len %d): %d' % (k, len(Ik)))

    # Reduce: candidates' itemsets and accept/reject based on epsilon
    # for c, (sA, sB) in zip(Ck, Ik):
    for c, (sA, sB) in Ik.items():
      ia = I[sA].intersection(I[sB])
      if len(ia) > epsilon:
        Lk.append(c)
        I[c] = ia
    L.append(Lk)
    ts2 = dt.now()
    Vk = set(it.chain(*Lk))
    ts3 = dt.now()
    print('%2d'%k, 'Map: %5.2f s   Reduce: %5.2f s    Prune: %5.2f s   #Keys: %d' % 
      ((ts1-ts0).total_seconds(), (ts2-ts1).total_seconds(), (ts3-ts2).total_seconds(), len(Lk)))
    k += 1

  print('Total Time:  %6.2f sec' % (dt.now()-start).total_seconds(),  
    '    N= %d  K= %d  # Nodes= %d' % (N, K, len(I)))
  return L, I



def cheapEMD(d1, d2, nbins=20):
  ''' Performs a cheap EMD using 1D histograms. "dirt" is only
  moved to adjacent bins along each dim and is cacluated iteratively 
  from the lowest to highest bin. Simulated "scraping dirt/anti-dirt"
  from bin to bin. The returned val is the sqrt of the sum of squares for
  all dimensions'''
  N1, K1 = d1.shape
  N2, K2 = d2.shape
  flow = np.zeros(K1)
  # Create normalized Histograms with different distributions (same bin #/sizes)
  for k in range(K1):
    ha = np.histogram(d1[:,k], nbins, (3.5, 7))[0] / N1
    hb = np.histogram(d2[:,k], nbins, (3.5, 7))[0] / N2
    flow_k = 0
    # Calculate bin-by-bin difference
    delta = ha - hb
    # calculate flow from 0 to n-1 and add/subtract flow to neighboring bin
    for i in range(nbins-1):
      flow_k     += delta[i]
      delta[i+1] += delta[i]
      # Note: there is a assumed 'delta[i] -= delta[i]'

    flow[k] = flow_k / (nbins-1)

  # Normalize the result by returning absolute delta and dividing by
  # number of "moves" (n-1)
  return np.sqrt(np.sum(flow**2))


def parent_map(L):
  pmap = []
  for k in range(1, len(L)-1):
    Lk = []
    n = 0
    st = dt.now()
    for child in L[k]:
      nodes = []
      for idx, parent in enumerate(L[k+1]):
        if child < parent:
          nodes.append(idx)
          n += 1
      Lk.append({child: nodes})
    pmap.append(Lk)
    print('L=', k, ' #edges: ', n, '  Time: ', (dt.now()-st).total_seconds())
  return pmap

tok   = lambda x: ''.join(sorted([chr(97+i) for i in x]))
def derived_lattice(E, L, I):
  dlattice = {}
  tottime = 0.
  nnodes, nedges = 0, 0
  for k in range(1, len(L)-1):
    print('Processing %d nodes with features sets of len: %d' % (len(L[k]), k))
    times = []
    emdtot, emdtime = 0, 0.
    md1, md2, md3, nt = [], [], [], []
    for child in L[k]:
      nnodes += 1
      flist = list(child)
      d1 = E[I[child]][:,flist]
      child_key = tok(child)
      dlattice[child_key] = {}
      for parent in L[k+1]:
        start = m1 = dt.now()
        if child < parent:
          m2 = dt.now()
          d2 = E[I[parent]][:,flist]
          m3 = dt.now()
          dlattice[child_key][tok(parent)] = cheapEMD(d1, d2)
          nedges += 1
          m4 = dt.now()
          md1.append((m2-m1).total_seconds())
          md2.append((m3-m2).total_seconds())
          md3.append((m4-m3).total_seconds())
        else:
          nt.append((dt.now()-m1).total_seconds())
        times.append((dt.now()-start).total_seconds())

    print('  TIMES:  #Edges= %7d'% len(times), '  Avg EMD= %6.4f  '%np.mean(times),
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



# dlat3 = {key: {k: np.sum(np.abs(list(v.values()))) for k,v in distr.items()} for key, distr in dlat2.items()}



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

