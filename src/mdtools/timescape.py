import mdtraj as md
import numpy as np
import logging
import math
import redis

from core.common import *
 

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"


class TimeScape:

  @classmethod
  def read_log(cls, fname):
    data = [0]
    with open(fname) as src:
      for line in src.readlines():
        if line.startswith('Output'):
          elm = line.split()
          if elm[6][:-1].isdigit():
            data.append(int(elm[6][:-1]))
          else:
            print("BAD Format in log file: ", line)
            break
    return data

class Basin(object):
  def __init__ (self, traj, traj_id, window, mindex):
    self.start, self.end = window
    self.len = self.end - self.start
    self.mindex = mindex
    self.minima = traj.slice(mindex)
    self.id = getUID()
    self.prev = None
    self.next = None
    self.traj = traj_id

  def kv(self):
    d = self.__dict__
    del d['minima']
    return d

    

class TimeScapeTrajectory(object):
  """Wrapper class for the TimeScapes API"""
  """ Placeholder for full integration """

  def __init__(self, pdb, loc, traj_id, dcd=None, traj=None):
    self.pdb = pdb
    if dcd is None:
      self.dcd = os.path.join(loc, traj_id + '.dcd')
    else:
      self.dcd = dcd
    self.out = loc
    self.traj = traj
    self.traj_id = traj
    self.basins = []

  def load_traj(self):
    self.traj = md.load(self.dcd, top=self.pdb)

  def load_basins(self):
    segfile = self.out + '_segmentation.dat'
    basin_list = []
    with open(segfile) as src:
      cur_basin = 0
      last_trans = 0
      index = 0
      for line in src.readlines():
        index += 1
        bnum = int(line.split()[-1])
        if cur_basin == bnum:
          continue
        basin_list.append((last_trans, index))
        last_trans = index
        cur_basin = bnum
      basin_list.append((last_trans, index))

    if self.traj is None:
      self.load_traj()

    minima_list = self.get_minima()
    index = 0
    last = None
    while index < len(minima_list):
      # TODO:  Do we handle First & Last basin in trajectory
      window = basin_list[index+1]
      minima = minima_list[index]
      basin = Basin(self.traj, self.traj_id, window, minima)
      if last is not None:
        basin.prev = last.id
        self.basins[-1].next = basin.id
      self.basins.append(basin)
      last = basin
      index += 1
    return basin_list

  def read_log(self, fname):
    data = [0]
    with open(fname) as src:
      for line in src.readlines():
        if line.startswith('Output'):
          elm = line.split()
          if elm[6][:-1].isdigit():
            data.append(int(elm[6][:-1]))
          else:
            print("BAD Format in log file: ", line)
            break
    return data

  def get_minima(self):
    minima_file = self.out + '_minima.log'
    return self.read_log(minima_file)

