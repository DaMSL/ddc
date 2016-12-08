""" FORMAL COMPONENT ANALYSIS """

import itertools
import string
import numpy as np

ascii_greek = ''.join([chr(i) for i in itertools.chain(range(915,930), range(931, 938), range(945, 969))])
label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek

frommask = lambda obs: ''.join([label_domain[i] for i, x in enumerate(obs) if x])
tomask = lambda key, size: [(1 if i in key else 0) for i in domain[:size]]
toidx = lambda key: [i for i,f in enumerate(K_domain) if f in key]

 
def clarify_row(A):
  U = {}
  for idx, row in enumerate(A):
    key = frommask(row)
    if key not in U:
      U[key] = []
    U[key].append(idx)
  return U

def unique_event_subsets(event_list):
  unique_events = []
  all_events = set(range(len(event_list)))
  while len(all_events) > 0:
    uidx = all_events.pop()
    substate_list = [uidx]
    for ev in list(all_events):
      if (event_list[ev] == event_list[uidx]).all():
        all_events.remove(ev)
        substate_list.append(ev)
    unique_events.append(substate_list)
  return unique_events


def nextClosure (G, M, I, Y):
  enum_set = [set([range(i)]) for i in len(M)]
  lecticOrder  = lambda Y1, Y2: True if Y1 == Y2 else min(Y1 ^ Y2) in Y1
  generativeOp = lambda Y, i: (Y & enum_set[i-1]) | i
  formalConcepts = []
  Y = Y_last = set()
  while Y != Y_last:  
    for i in range(len(M), -1, -1):
      p = M[i]
      if p not in Y:
        candidate = generativeOp(Y, p)
        if lecticeOrder(candidate, Y):
          Y = candidate
          break
    formalConcepts.append(Y)
  return formalConcepts



class parallelFCA(object):
  def __init__(self, data_matrix):
    N, M = data_matrix.shape
    self.data = data_matrix
    self.rows = [set() for m in range(M)]
    for n, x in enumerate(self.data):
      for m in range(M):
        if x[m]:
          self.rows[m].add(n)

  def computeClosure(self, A, B, y):
    N, M = self.data.shape
    C = np.zeros(N)
    D = np.ones(M)
    for i in set(np.where(A)[0]) & self.rows[y]:
      C[i] = 1
      for j in range(M):
        if self.data[i,j] == 0:
          D[j] = 0
    return C, D


  def generateFrom(self, A, B, y):
    N, M = self.data.shape
    print ('Extent:', list(np.where(A)[0]), frommask(B), y)
    if (B==1).all() or y >= M:
      return
    for j in range(y, M):
      if B[j] == 0:
        C, D = self.computeClosure(A, B, j)
        skip = False
        for k in range(0, j-1):
          if D[k] != B[k]:
            skip = True
            break
        print('   Clo:', list(np.where(C)[0]), frommask(D), j, skip)
        if not skip:
          self.generateFrom(C, D, j+1)



import networkx as nx
import matplotlib.pyplot as plt

def runme(data):
  plt.cla()
  N, M = data.shape
  U = clarify_row(data)
  S = sorted(U.keys())
  G = nx.Graph()
  for i, s in enumerate(S):
    G.add_node(s)
  edgelist = set()
  prev = frommask(data[0])
  for i in range(1, N):
    cur = frommask(data[i])
    if prev != cur:
      edgelist.add((min(prev,cur), max(prev,cur)))
    prev = cur
  for i, j in edgelist:
    G.add_edge(i, j)
  nx.draw_spring(G, with_labels=True, node_size=600, node_color='y', fontsize=10)
  plt.savefig('../graph/alan_graph_sprg.png')
  return G




# 0 ad 3 [1, 1, 1]
# 1 abce 3 [0, 0, 0]
# 2 e 3 [0, 0, 0]
# 3  1 [0]
# 4 abd 2 [1, 1]
# 5 ce 2 [2, 0]
# 6 bce 4 [0, 0, 0, 0]
# 7 b 3 [0, 0, 0]
# 8 be 2 [0, 0]
# 9 ae 2 [2, 0]
# 10 c 3 [2, 0, 1]
# 11 cd 4 [1, 1, 1, 1]
# 12 bd 10 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 13 ace 3 [2, 2, 0]
# 14 d 7 [1, 1, 1, 1, 1, 1, 1]
# 15 abc 1 [0]
# 16 abe 2 [0, 0]
# 17 bc 5 [2, 0, 0, 0, 0]