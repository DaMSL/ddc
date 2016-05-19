import numpy as np
from collections import deque

# 5.5175  5.3481  6.4806  5.1262  4.5091  5.2002  5.0034  6.3407  4.9641  4.4467
# Desired Landscape (a.k.a Knapsack, Ck  k in 1..m)  m=25

# CountsMax [C]:  [ 7.6706  7.6722  8.8404  8.4105  7.4063]

# StateDist [S]:  [ 8.3175  8.1733  8.2895  8.0771  8.1928]   m0..4  (0,0 .. 4,4)

# RelDist [A-B]:  [ 4.7868  4.7478  3.6659  5.1053  5.1326  4.3439  4.9957  3.7475  5.2856  5.8622]



def convert_knap(feal, scale=10, M=25):
  item = np.zeros(M)
  item[:5] = feal[5:10]    # S0-S4
  for i, k in enumerate(feal[10:]):
    item[5+i*2]   = 10-k
    item[5+i*2+1] = k
  return item


from core.kdtree import KDTree
kd = KDTree(100, 15, np.array(flist), 'median')
hc = kd.getleaves()
global_landscape = np.mean(flist, axis=0)
desired = 10-global_landscape
hcWgt = {k: np.mean([feallist[i] for i in v['elm']], axis=0) for k, v in hc.items()}
hcidx = [i for i in hcWgt.keys()]
hcWgt_list = [hcWgt[i] for i in hcidx]
numresources = 25
M=25


def knapsack_pack(desired, itemlist, numresources, M=25):
I = len(hcWgt_list)
X = np.zeros(shape=(I, M))  
capacity = numresources * convert_knap(desired)
itemlist = np.array([convert_knap(i) for i in hcWgt_list])

featureQ = [deque() for k in M]
for k, feature in enumerate(itemlist.T):
    bestfit_k = np.argsort(feature)
    weight = 0
    for i in bestfit_k:
      featureQ[k].append(i)
      if feature[i] + weight < capacity[k]:
        weight += feature[i]
        X[i][k] = 1
      # TODO: Prune here to optimize

score = np.sum(X, axis=1)
bff_M = np.argsort(score)[::-1]
bestfit = [itemlist[i] for i in bff_M]

  # TODO: Accept/Reject preprocess 2 partial loops to optimize ILO full linear scan
  accept = []
  for i in bestfit_M:
    if int(score[i]) == M:
      accept.append(i)
    elif int(score[i]) == 0:
      reject.append(i)

  weight = np.zeros(M)

  """ Greedy (with replacement)"""
space  = np.copy(capacity)   # GOAL: MIN(ABS(space))
knapsack = np.zeros(M)
available = deque(bff_M)
for i in range(M):
  nextitem_idx = available_items.popleft()
  knapsack[i] = itemlist[nextitem_idx]

#Stategic Oscillation
for i in range(100):
  score = capacity - np.sum(knapsack, axis=0)
  print('Score = ', score)
  min_k = np.argmin(score)   # Most penalized feature (most value if removed)
  max_k = np.argmax(score)   # Least gain (most value if added)
  if min_k > 0:
    break  # found solution (how good?)
  # min_k is negative --> largest penalized features
  most_penalized_item = np.argmax(knapsack[:,min_k])
  most_value_added = featureQ[k].popleft()
  # TODO: Reqeu the item???
  knapsack[most_penalized_item] = itemlist[most_value_added]
  print('Removing KP item # %d  -- Replace with ItemIdx # %d' % (most_penalized_item, most_value_added))

  # Global Relacement
cur_score = np.sum(space)
for i in range(100):    # of times to try replacing or find better strategy
    swapped = False
    for j in range(24, 0, -1):
      swap_candid_score = np.sum(itemlist[knapsack[j]])
      for n in range(len(available_items)):
        swap_score = cur_score - swap_candid_score + np.sum(itemlist[available_items[n]])
        if np.abs(swap_score) < np.abs(cur_score):
          available_items.append(knapsack[j])
          del knapsack[j]
          knapsack.append(available_items[n])
          del available_items[n]
          cur_score = swap_score
          swapped = True
          break
    if not swapped:
      print('found best fit')
      break

    # TODO:  Local Replacement within HCubes



def fill_knap_dp(capacity, slots, items):
  """ Dynamic Programming Approach --- inefficient
  """
  score = np.sum(capacity)
  if slots == 0 or len(items) < slots or score < 0 or np.sum(items[0]) < 100:
    print ('Item list size = %d    Score = %f   No More progress to make'% (len(items), score))
    return (score, [])
  item = items[0]
  with_item = fill_knap(capacity - item, slots-1, items[1:])
  without_item = fill_knap(capacity, slots, items[1:])
  if with_item[0] < without_item[0]:
    return (with_item[0] + np.sum(item), with_item[1] + [item])
  else:
    return without_item



