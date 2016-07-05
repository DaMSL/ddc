#!/usr/bin/env python

"""Cache Implemenation for the Data Driven Control Project

    Cache is designed to hold high dimensional points. An abstract
    class is provides for future implementation using differing
    cache and storage policies

  STATUS:  Not Integrated into the project
"""
import abc
import os
import redis
import numpy as np
from collections import deque
import logging

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


class CacheStorage(object):
  """Storage for cache is assumed to always grow to capacity.
    Ergo: there is no remove options. It is a dumb storage as
    implementors must track what items to evict and where to 
    put them (only get/set are provided)
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def size(self):
    """Returns current size of the cache
    """
    pass

  @abc.abstractmethod
  def set(self, slot, value):
    """Set the location in the cache with the value
    """
    pass

  @abc.abstractmethod
  def get(self, slot):
    """Retrieve the value in the cache at location slot
    """
    pass


class CacheStoreNDArray(CacheStorage):
  """Implements a cache using a fixed size NDArray
  """

  def __init__(self, store_file, shape=None):
    """Creates a new cache using total shape of 'shape'
      First axis in shape is the effective slot capacity
      all remaining axis define the block size
      Size is stored in the 1st col of 1st row
    """

    # Allocate in memory
    if os.path.exists(store_file):
      # self._store = np.memmap(store_file, dtype='float32', mode='r+')
      self._store = np.load(store_file, mmap_mode='r+')
      self._max_size = len(self._store)
      self._size = 0
      logging.info('[CACHE] Cache loaded from ')
    elif shape==None:
      logging.error('[CACHE] Error creating new cache. No data shape provided.')
      return
    else:
      # self._store = np.memmap(store_file, dtype='float32', mode='w+', shape=shape)
      # TODO: RELOAD Cache settings (size)
      self._store = np.zeros(dtype='float32', shape=shape)
      np.save(store_file, self._store)
      self._store = np.load(store_file, mmap_mode='r+')
      self._max_size = shape[0]
      self._size = 0

  def size(self):
    return self._size

  def set(self, slot, value):
    """Set the location in the array with the value using copyto
    """
    self._store[slot] = value
    if self._size < self._max_size:
      self._size += 1
    self._store.flush()

  def get(self, slot):
    """Retrieve the value"""
    return self._store[slot]


class Cache(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def insert(self, index, value):
    """Insert a new item into the cache
    """
    pass


  @abc.abstractmethod
  def request(self):
    """Retrieve a value from the cache, if present
       Otherwise: return None (cache miss) 
    """
    pass


class TwoQCache(Cache):
  """The 2Q-Cache implemented the cache using two simple queues:
    1. Short term FIFO as a circular buffer. Items are only inserted
       once into the short term cache. For simplicity, items are not
       explicitly removed, but rather evicted strictly on FIFO ordering
    2. Long term LRU. One requested from the Short Term cache, items are 
       copied into the long term cache. Evicted items are determined based
       on LRU using a deque. When accessed, item indexes are removed from the deque
       and appended to the end. Evictions occur as a popleft
    TBD:  Dynamic sizing of short/long term cache sizes 

    The index is used for quick look up based on the following:
      Keys are the external references (pointers) for look up
      Values will hold the index location in the underlying cache store:
        Positive index --> long term store
        Negative index --> short term short
      No key in the index == Cache Miss
  """

  def __init__(self, max_size=(1000, 1)):
    store_size = list(max_size)
    store_size[0] //= 2        # For now: each store get half total capacity (interim soln)
    store_size = tuple(store_size)
    self.lookup = {}
    self.lrulist = deque()
    self.short = CacheStoreNDArry('testcacheS.npy', store_size)
    self.long = CacheStoreNDArry('testcacheL.npy', store_size)
    self.head_short = 1
    self.capacity_short = store_size[0]
    self.capacity_long = store_size[0]

  def insert(self, index, value):
    """Insert the new value into the short term cache. Assumes the value is 
      new, but checks the cache first and notifies on cache hit. The index
      is the external reference used for access/lookup
    """
    if index in self.lookup.keys():
      logging.warning('[CACHE]  Index value %s already exists in the cache. This is bad. Please check why.', str(index))
      return
    self.lookup[index] = (-1 * self.head_short)
    self.head_short += 1
    if self.head_short == self.capacity_short:
      self.head_short = 1
    self.short.set(self.head_short, value)

  def request(self, index):
    """Checks the cache for the given index.
      HIT in long term:  updates LRU list and returns value
      HIT in short term: inserts value in long term list. Evicts the
        least recently used slot and uses that as the new insertion point
    """
    if index not in self.lookup:
      logging.debug("[CACHE]  Request for index %s.  Cache MISS.", str(index))
      return None

    slot = self.lookup[index]
    if slot == 0:
      logging.debug("[CACHE]  Request for index %s.  Cache HIT at pinned posn", str(index))
      # Item is at position 0 -- special cases (pinned point)
      value = self.long.get(0)
    elif slot > 0:
      # Item is cached in long term store; put slot value at end of LRU
      value = self.long.get(slot)
      self.lrulist.remove(index)
      self.lrulist.append(index)
      logging.debug("[CACHE]  Request for index %s.  Cache HIT in long at slot %d ?", str(index), slot)

    # Item is cached in short term store; insert it into long term
    else:
      value = self.short.get(slot)

      victim = None
      if self.long.size() >= self.capacity_long:
        # L/T Cache is full. Evict a victim
        victim = self.lrulist.popleft()
        newslot = self.lookup.pop(victim)
      else:   
        # Cache is not full, append it
        newslot = self.long.size()
      self.long.set(newslot, value)
      self.lrulist.append(index)    
      self.lookup[index] = newslot
      logging.debug("[CACHE]  Request for index %s.  Cache HIT in short at slot %d posn. Inserted into long at %d", str(index), slot, newslot)
    return value


def testcache():
  data = np.random.random(size=(50, 5, 3))
  logging.debug("Creating Cache.....")
  c = TwoQCache((20, 5, 3))
  for i, d in enumerate(data):
    if i in [10, 30, 49]:
      logging.debug("  S-cache size check (item %d-pre): %d", i, c.short.size())
      logging.debug("  L-cache size check (item %d-pre): %d", i, c.long.size())
    c.insert(i, d)
    if i in [10, 30, 49]:
      logging.debug("  S-cache size check (item %d-pre): %d", i, c.short.size())
      logging.debug("  L-cache size check (item %d-pre): %d", i, c.long.size())
  logging.debug("Inserted all data elms")

  for i in range(50):
    if i in [10, 30, 49]:
      logging.debug("  S-cache size check (item %d-pre): %d", i, c.short.size())
      logging.debug("  L-cache size check (item %d-pre): %d", i, c.long.size())
    s = np.random.randint(50)
    hit = c.request(s)
    if hit is not None:
      logging.debug(" check %d:  HIT!", s)
    else:
      logging.debug(" check %d:  MISS!", s)
      c.insert(s, data[s])
      rehit = c.request(s)
      logging.debug("   re-check:  %s", str(rehit))
    if i in [10, 30, 49]:
      logging.debug("  S-cache size check (item %d-pre): %d", i, c.short.size())
      logging.debug("  L-cache size check (item %d-pre): %d", i, c.long.size())



if __name__ == "__main__":
  testcache()