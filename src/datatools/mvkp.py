import numpy as np
from collections import deque
from core.kdtree import KDTree


def convert_knap(feal, scale=10, M=25):
  item = np.zeros(M-5)
  # item[:5] = feal[5:10]    # S0-S4
  for i, k in enumerate(feal[10:]):
    # item[5+i*2]   = 10-k
    # item[5+i*2+1] = k
    item[i*2]   = 10-k
    item[i*2+1] = k
  return item

def kp_setup(feallist, kd):
  # kd = KDTree(100, 15, np.array(feallist), 'median')
  hc = kd.getleaves()
  global_landscape = convert_knap(np.mean(feallist, axis=0))
  desired = 10-global_landscape
  hcWgt = {k: np.mean([feallist[i] for i in v['elm']], axis=0) for k, v in hc.items()}
  hcidx = [i for i in hcWgt.keys()]
  hcWgt_list = [hcWgt[i] for i in hcidx]
  itemlist = np.array([convert_knap(i) for i in hcWgt_list])
  return itemlist, desired

def knapsack(desired, itemlist, n_slots=25, n_iter=1000, R=5):
  I = len(itemlist)   # size of 'inventory'
  M = len(desired)      # num of feateres
  X = np.zeros(shape=(I, M))  # 0-1 matrix  (0-1 vect per k)
  # R = 3  # recency factor det. what is/is not recent
  capacity = n_slots * desired

  # Break up MV problem in k-one variance 0-1 KP problems
  for k, feature in enumerate(itemlist.T):
      bestfit_k = np.argsort(feature)
      weight = 0
      # Find best fit for each k & score each item on 0-1 scale
      for i in bestfit_k:
        if feature[i] + weight < capacity[k]:
          weight += feature[i]
          X[i][k] = 1
        # TODO: Prune here to optimize

  # Determine an initial best fit ranking of all items from X 
  score       = np.sum(X, axis=1)
  bestfit_idx = np.argsort(score)[::-1]
  bestfit     = [itemlist[i] for i in bestfit_idx]

  # Fill Knapsack greedily with first M-best options
  knapsack = np.zeros(shape=(n_slots, M))
  available_items = deque(bestfit_idx)
  for i in range(M):
    nextitem_idx = available_items.popleft()
    knapsack[i] = itemlist[nextitem_idx]

  #Stategic Oscillation
  # Sort items by k
  inventory = [deque(np.argsort(k)) for k in itemlist.T]

  # Initialize the recency queue with R most likely features
  # recent_list = deque(np.argsort([np.abs(np.sum(knapsack, axis=0))])[::-1][:R])
  recent_add = []
  recent_drop = []
  best_solution = (score, i, knapsack)
  best_score = 100000
  for i in range(n_iter):
    score = capacity - np.sum(knapsack, axis=0)
    total_score = np.sum(np.abs(score))
    ranking = np.argsort(score)
    min_k = ranking[0]
    max_k = ranking[M-1]

    # idxmin = 0
    # # Find Most penalized feature (most value if removed) -- but ignore recently dropped items
    # while True:
    #   if len(recent_drop) == 0 or ranking[idxmin] not in recent_drop or idxmin == R or \
    #     (recent_drop[0] == ranking[idxmin] and len(recent_drop) == R):
    #     min_k = ranking[idxmin]
    #     break
    #   idxmin += 1
    #   if idxmin == R:
    #     print('Set idx 0')
    #     min_k = ranking[0]
    #     break

    # idxmax = M - 1
    # while True:
    #   max_k = ranking[idxmax]
    #   if (len(recent_add) == 0 or max_k not in recent_add or \
    #        (recent_add[0] == max_k and len(recent_add) == R)) and \
    #      (len(recent_drop) == 0 or max_k not in recent_drop or \
    #        (recent_drop[0] == max_k and len(recent_drop) == R)):
    #     # print (recent_drop, 'max_k=', max_k, '  idxmax=', idxmax, max_k in recent_drop, max_k ==12, 12 in recent_drop)
    #     break
    #   idxmax -= 1
    #   if idxmax == M-R:
    #     print('Set idx M')
    #     max_k = ranking[M-1]
    #     break

    # Least gain k (also: k with most value if added)
    most_value_added = inventory[max_k].popleft()
    inventory[max_k].append(most_value_added)

    if np.abs(score[min_k]) > np.abs(score[max_k]):
      # Destructive Choice
      # cost of penalized items outweights unfilled space: replace it
      most_penalized_item = np.argmax(knapsack[:,min_k])
      knapsack[most_penalized_item] = itemlist[most_value_added]
      # if min_k in recent_drop:
      #   recent_drop.remove(min_k)
      # recent_drop.append(min_k)
      if i % 1 == 0:
        print('[%4d]  Score = %7.2f :  Destruct on k= %2d    Add MaxValue on k=%2d'% (i, np.sum(np.abs(score)), 
            min_k, max_k), recent_drop, ranking)
    else:
      # Constructive Choice
      # unfilled space has more to gain than removing a penalized item 
      least_value_added = np.argmin(knapsack[:,max_k])
      knapsack[least_value_added] = itemlist[most_value_added]
      if i % 1 == 0:
        print('[%4d]  Score = %7.2f : Construct on k= %2d    Add MaxValue on k=%2d'% (i, np.sum(np.abs(score)), 
          max_k, max_k), recent_drop, ranking)
      # if max_k in recent_drop:
      #   recent_drop.remove(max_k)
      # recent_drop.append(max_k)
    # if max_k in recent_add:
    #   recent_add.remove(max_k)
    # recent_add.append(max_k)
    total_score = np.sum(np.abs(score))
    if total_score < best_score:
      print('   NEW LOW SCORE ****:  ')
      best_score = total_score
      best_solution = (score, i, knapsack)


  print('CAPACITY:', capacity)
  print('KNAPSACK:', np.sum(knapsack, axis=0))
  print('   SCORE:', score)
  print('LOWEST SCORE:', best_solution[1], best_score)
  return knapsack
  # print('Removing KP item # %d  -- Replace with ItemIdx # %d' % (most_penalized_item, most_value_added))


  # TODO: Accept/Reject preprocess 2 partial loops to optimize ILO full linear scan
  # accept = []
  # for i in bestfit_M:
  #   if int(score[i]) == M:
  #     accept.append(i)
  #   elif int(score[i]) == 0:
  #     reject.append(i)



  # Global Relacement
  # cur_score = np.sum(space)
  # for i in range(100):    # of times to try replacing or find better strategy
  #   swapped = False
  #   for j in range(24, 0, -1):
  #     swap_candid_score = np.sum(itemlist[knapsack[j]])
  #     for n in range(len(available_items)):
  #       swap_score = cur_score - swap_candid_score + np.sum(itemlist[available_items[n]])
  #       if np.abs(swap_score) < np.abs(cur_score):
  #         available_items.append(knapsack[j])
  #         del knapsack[j]
  #         knapsack.append(available_items[n])
  #         del available_items[n]
  #         cur_score = swap_score
  #         swapped = True
  #         break
  #   if not swapped:
  #     print('found best fit')
  #     break

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



