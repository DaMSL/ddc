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
  def __init__ (self, traj_id, window, mindex, traj=None, uid=None):
    self.start, self.end = window
    self.len = self.end - self.start
    self.mindex = mindex
    self.minima = None if traj is None else traj.slice(mindex)
    if uid is None:
      self.id = getUID()
    else:
      self.id = uid
    self.prev = None
    self.next = None
    self.traj = traj_id

  def kv(self):
    d = self.__dict__
    if self.minima is not None:
      del d['minima']
    return d

    

class TimeScapeParser(object):
  """ 
  Class to parse and manage TimeScape Output. Use of this class
  assumes that the TimeScape program has already run. Ergo
  the Trajectory passed in and processed is the FULL underlying
  trajectory -- a frame_rate is provided to the load_basin program
  to convert indexing from a pre-processed file (found in output)
  to a full tranjectory (for follow on manipulation) """

  def __init__(self, pdb, loc, traj_id, dcd=None, traj=None, uniqueid=False):
    """
      pdb - topology for loading f
      loc - output dir location (which should also include src dcd)
      traj_id - identifier for FULL underlying scs
      dcd - pre-proprocessed DCD passed to TimeScape
      traj - FULL trajectory
    """

    self.pdb = pdb
    if dcd is None:
      self.dcd = os.path.join(loc, traj_id + '.dcd')
    else:
      self.dcd = dcd
    self.out = loc
    self.traj = traj
    self.traj_id = traj_id
    self.basins = []
    self.unique_basin_id = uniqueid

  def load_traj(self):
    self.traj = md.load(self.dcd, top=self.pdb)

  def load_basins(self, frame_rate=1, force_load=False):
    """
    frame_rate is measured in #frames / ps
    """
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
        basin_list.append((last_trans*frame_rate, index*frame_rate))
        last_trans = index
        cur_basin = bnum
      basin_list.append((last_trans, index))

    if self.traj is None and force_load:
      self.load_traj()

    minima_list = [i*frame_rate for i in self.get_minima()]
    basin_index = 0
    last = None
    while basin_index < len(minima_list):
      # TODO:  Do we handle First & Last basin in trajectory
      window = basin_list[basin_index]
      minima = minima_list[basin_index]
      
      if self.unique_basin_id:
        basin_id = None 
      else:
        basin_id = str(self.traj_id) + '_' + str(basin_index)

      basin = Basin(self.traj_id, window, minima, traj=self.traj, uid=basin_id)
      if last is not None:
        basin.prev = last.id
        self.basins[-1].next = basin.id
      self.basins.append(basin)
      last = basin
      basin_index += 1
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

