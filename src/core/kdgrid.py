#!/usr/bin/env python
"""KD Grid Implementation

"""

from collections import deque, OrderedDict
import sys
import json
import logging
import math
import abc
import copy

import numpy as np

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)



class KDGrid(object):
  """
    KD-Grid is a K-dimensional grid with partitions space along
    the coordinate origin. Points are grouped into their respective
    hypercube based on polarity along each dimension

  """
  def __init__(self, K=10, data=None):
    """
    Leafsize is set to 1
    """
    self.dim = K
    self.key = [int(math.pow(2, i)) for i in range(K)]
    self.size = int(math.pow(2, K))
    self.grid = [[] for i in range(self.size)]
    self.toindex = lambda vect: np.sum(np.greater(vect, np.zeros(K)) * self.key)
    self.region = {}
    self.define_region(0, [-1, -1, -1, -1,  0,  0,  0,  0,  0,  0])
    self.define_region(1, [ 1,  0,  0,  0, -1, -1, -1,  0,  0,  0])
    self.define_region(2, [ 0,  1,  0,  0,  1,  0,  0, -1, -1,  0])
    self.define_region(3, [ 0,  0,  1,  0,  0,  1,  0,  1,  0, -1])
    self.define_region(4, [ 0,  0,  0,  1,  0,  0,  1,  0,  1,  1])

  def insert(self, data):
    """
    Inserts a list of points, each pt much be of the same K-dimension vector
    For now: store raw data (vice index)
    """
    for vect in data:
      self.grid[self.toindex(vect)].append(vect)

  def makeregion(self, mask, d=0):
    if d == self.dim:
      return [self.toindex(mask)]
    else:
      if mask[d] == 0:
        m0 = copy.copy(mask)
        m0[d] = -1
        m1 = copy.copy(mask)
        m1[d] = 1
        return self.makeregion(m0, d+1) + self.makeregion(m1, d+1)
      else:
        return self.makeregion(mask, d+1)

  def define_region(self, label, mask):
    self.region[label] = self.makeregion(mask)

  def collect_region(self, label=None):
    if label is None:
      region = {}
      for key in self.region.keys():
        region[key] = []
        for grid in self.region[key]:
          region[key].extend(self.grid[grid])
      return region
    else:
      if label not in region:
        print("Error no region defined for: ", label)
        return
      labellist = [label]
      region = []
      for grid in self.region[label]:
        region.extend(self.grid[grid])

if __name__ == "__main__":
  kdg = KDG.KDGrid(10)
  kdg.define_region(0, [-1, -1, -1, -1,  0,  0,  0,  0,  0,  0])
  kdg.define_region(1, [ 1,  0,  0,  0, -1, -1, -1,  0,  0,  0])
  kdg.define_region(2, [ 0,  1,  0,  0,  1,  0,  0, -1, -1,  0])
  kdg.define_region(3, [ 0,  0,  1,  0,  0,  1,  0,  1,  0, -1])
  kdg.define_region(4, [ 0,  0,  0,  1,  0,  0,  1,  0,  1,  1])