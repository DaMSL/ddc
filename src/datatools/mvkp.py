"""
Multi-Variate Knapsack Algorithm
"""
import numpy as np
import datetime as dt
from collections import deque
from core.kdtree import KDTree

import logging

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

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
  slotlist = [-1 for i in range(n_slots)]

  # Break up MV problem in k-one variance 0-1 KP problems
  st = dt.datetime.now()
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
  for i in range(n_slots):
    nextitem_idx = available_items.popleft()
    knapsack[i] = itemlist[nextitem_idx]
    slotlist[i] = nextitem_idx

  #Stategic Oscillation
  # Sort items by k
  inventory = [deque(np.argsort(k)) for k in itemlist.T]

  # Initialize the recency queue with R most likely features
  # recent_list = deque(np.argsort([np.abs(np.sum(knapsack, axis=0))])[::-1][:R])
  recent_add = []
  recent_drop = []
  best_solution = (score, i, knapsack)
  best_score = 100000
  display_interval = n_iter // 100
  progress_change = 0
  last_score = 0
  for i in range(n_iter):
    score = capacity - np.sum(knapsack, axis=0)
    total_score = np.sum(np.abs(score))
    ranking = np.argsort(score)
    min_k = ranking[0]
    max_k = ranking[M-1]

    # Least gain k (also: k with most value if added)
    most_value_added = inventory[max_k].popleft()
    inventory[max_k].append(most_value_added)

    if np.abs(score[min_k]) > np.abs(score[max_k]):
      # Destructive Choice
      # cost of penalized items outweights unfilled space: replace it
      most_penalized_item = np.argmax(knapsack[:,min_k])
      knapsack[most_penalized_item] = itemlist[most_value_added]
      slotlist[most_penalized_item] = most_value_added
      # if i % 1 == 0:
      #   print('[%4d]  Score = %7.2f :  Destruct on k= %2d    Add MaxValue on k=%2d'% (i, np.sum(np.abs(score)), 
      #       min_k, max_k), recent_drop, ranking)
    else:
      # Constructive Choice
      # unfilled space has more to gain than removing a penalized item 
      least_value_added = np.argmin(knapsack[:,max_k])
      knapsack[least_value_added] = itemlist[most_value_added]
      slotlist[least_value_added] = most_value_added
      # if i % 1 == 0:
      #   print('[%4d]  Score = %7.2f : Construct on k= %2d    Add MaxValue on k=%2d'% (i, np.sum(np.abs(score)), 
      #     max_k, max_k), recent_drop, ranking)
    total_score = np.sum(np.abs(score))
    progress_change += np.abs(last_score - total_score)
    last_score = total_score

    if total_score < best_score:
      best_score = total_score
      best_solution = (score, i, slotlist)
      ts = dt.datetime.now()
      logging.info('   **********   NEW LOW SCORE : %7.2f  [idx=%7d]  /  %6.1f sec' % (best_score, i, (ts-st).total_seconds()))
    if i % display_interval == 0:
      ts =(dt.datetime.now()-st).total_seconds()
      logging.info('[%7d %4.1fsec] cur=%7.2f  best=%7.2f'% (i, ts, total_score, best_score))
      if progress_change == 0:
        logging.info('ZERO change in last %d intervals. Stoppping', display_interval)
        break
      progress_change = 0

  ts =(dt.datetime.now()-st).total_seconds()
  logging.info('CAPACITY: %s', str(capacity))
  logging.info('KNAPSACK: %s', str(np.sum(knapsack, axis=0)))
  logging.info('   SCORE: %s', str(score))
  logging.info('LOWEST SCORE:  %7.2f   at index= %d   TotalTime=%6.1f' % (best_score, best_solution[1], ts))
  return slotlist


def find_replace(desired, source_bucket, knap, target_list):
  cur_knap = np.copy(knap)
  n_slots = len(knap)
  capacity = n_slots * desired
  init_score = capacity - np.sum(cur_knap, axis=0)
  total_score = np.sum(np.abs(init_score))
  print('Solved best fit score: ', total_score)
  selected = []
  # TODO:  handle case of len(source_bucket) <= target_list
  for index in target_list:
    best_score = 100000
    best_idx = -1
    for idx in range(len(source_bucket)):
      if idx in selected:
        continue
      # TODO: Source Conversion
      source_pt = source_bucket[idx]
      cur_knap[index] = source_pt[10:]
      cur_score = capacity - np.sum(cur_knap, axis=0)
      total_score = np.sum(np.abs(cur_score))
      if total_score < best_score:
        best_score = total_score
        best_idx = idx
        print('Improved index found: %4d   Score=%f' % (best_idx, best_score))
    selected.append(best_idx)
  return selected


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



