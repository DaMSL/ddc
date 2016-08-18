import mdtraj as md
import numpy as np
import logging
import math
import redis

import itertools as itr

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

  @classmethod
  def windows(cls, fname):
    tran = TimeScape.read_log(fname)
    W = []
    for i in range(1, len(tran)):
      W.append((tran[i-1], tran[i]))
    return W

  @classmethod
  def contact_map(cls, fname, natoms, nframes):
    cmap = np.zeros(shape=(nframes, natoms, natoms))
    # for i in range(natoms):
    #   cmap[0][i][i] = 1
    event_list = []
    with open(fname) as src:
      for line in src.readlines():
        if line.startswith('+++') or line.startswith('---'):
          elm = line.split()
          polar, step,r1,r2 = elm[0],int(elm[5][:-1]),int(elm[7])-1,int(elm[10][:-1])-1
          event_list.append((polar, step, r1, r2))
    cur_frame = 0
    debug = 0
    for ev, step, r1, r2 in event_list:
      if step > cur_frame:
        for fr in range(cur_frame+1, step+1):
          cmap[fr] = cmap[cur_frame]
        cur_frame = step
      val = 1 if ev == '+++' else 0
      cmap[step][r1][r2] = cmap[step][r2][r1] = val
    for fr in range(cur_frame+1, nframes):
      cmap[fr] = cmap[cur_frame]
    return cmap


  @classmethod
  def event_list(cls, fname, e_type='all'):
    if e_type not in ['all', 'init', 'form', 'break']:
      print('Must use e_type of:  form, break, both')
      return
    event_list = []
    with open(fname) as src:
      for line in src.readlines():
        if not(line.startswith('+++') or line.startswith('---')):
          continue
        elm = line.split()
        ev1, ev2, step,r1,r2 = elm[1],elm[2],int(elm[5][:-1]),int(elm[7])-1,int(elm[10][:-1])-1
        if ev1 == 'initial' and e_type in ['all', 'init']:
          event_list.append(('init', step, r1, r2))
        elif ev2 == 'formed' and e_type in ['all', 'form']:
          event_list.append(('form', step, r1, r2))
        elif ev2 == 'broken' and e_type in ['all', 'break']:
          event_list.append(('break', step, r1, r2))
    return event_list


    # def basin_contact_map(cls, prefix, natoms, nframes):
    #   trans = TimeScape.read_log(prefix+'_transitions.log')
    #   events = TimeScape.event_list(prefix + '_events.log')
    #   cmap = np.zeros(shape=(len(trans)-1, natoms, atoms))
    #   W = [(trans[i], trans[i+1]) for i in range(0, len(trans)-1)]
    #   last_tran = trans[-1]
    #   basin_event = [[] for i in range(len(W))]
    #   for etype, step, a, b in events:
    #     val = 

    #   with open(fname) as src:
    #     for line in src.readlines():
    #       if not(line.startswith('+++') or line.startswith('---')):
    #         continue
    #       elm = line.split()
    #       ev, step,r1,r2 = elm[0],int(elm[5][:-1]),int(elm[7])-1,int(elm[10][:-1])-1
    #       if step == 0:
    #         continue
    #       if ev == '+++' and e_type in ['both', 'form']:
    #         event_list.append((step, r1, r2))
    #       if ev == '---' and e_type in ['both', 'break']:
    #         event_list.append((step, r1, r2))


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

    


second_sc_atom = {
  'ALA': 'CA',
  'ARG': 'CZ',
  'ASN': 'CG',
  'ASP': 'CG',
  'CYS': 'CB', 
  'GLN': 'CD', 
  'GLU': 'CD', 
  'GLY': 'CA', 
  'HSD': 'CE1', 
  'HSE': 'CE1', 
  'HSP': 'CE1', 
  'ILE': 'CG1', 
  'LEU': 'CD1', 
  'LYS': 'CE', 
  'MET': 'SD', 
  'PHE': 'CE1', 
  'PRO': 'CD', 
  'SER': 'CB', 
  'THR': 'CB', 
  'TRP': 'CE3',
  'TYR': 'CZ',
  'VAL': 'CB'
}

def side_chain_atoms(traj):
  if isinstance(traj, md.Topology):
    top = traj
  elif isinstance(traj, md.Trajectory):
    top = traj.top
  else:
    print("Only implemented for MDTrajectories and Topologies")
  atom_indices = np.zeros(top.n_residues, dtype=np.int32)
  for i, res in enumerate(top.residues):
    aname = second_sc_atom[res.name]
    idx = top.select('(resid==%d) and (name %s)' % (i, aname))[0]
    print(i, res.name, aname, idx)
    atom_indices[i] = idx
  return atom_indices

def side_chain_pairs(traj):
  atom_idx = side_chain_atoms(traj)
  return list(itr.combinations(atom_idx, 2))  


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

  def load_basins(self, frame_ratio=1, force_load=False):
    """  frame_ratio is the ratio in frame frequence between the 
    FULL sorce tractory (i.e. 1 frame per ps) and the TimeScape
    Analyzed trajectory (i.e. 1 frame every 4 ps)
      # FRAME SRC :  
    """

    #  Parse segmentation file to get each basin's window
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
        basin_list.append((last_trans*frame_ratio, index*frame_ratio))
        last_trans = index
        cur_basin = bnum
      basin_list.append((last_trans, index))

    # Ensure trajectory is loaded into memory
    if self.traj is None and force_load:
      self.load_traj()

    # Derive Correlation Matrix and get local minima frame indices
    # NOTE: minima index is in reference to FULL trajectory
    minima_list = [i*frame_ratio for i in self.get_minima()]
    basin_index = 0
    last = None

    #  Create all basin objects
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
    return self.basins


  def correlation_matrix(self):
    """Parses the event log output from TimeScapes and recreates the trajectory
    correlation matrix. Each frame is a [0,1] matrix indiciating if features
    i and j are correlated based on the TimeScape Event output. Since this is a 
    symmetric matrix, this return a vector of the upper triable (excluding diags)""" 
    if self.traj is None:
      self.load_traj()

    # Initialize data -- Correlation should exclude any solvent
    protein = self.traj.atom_slice(self.traj.top.select('protein'))
    nframes, nresid = protein.n_frames, protein.n_residues
    fname = self.out + '_events.log'
    cmap = np.zeros(shape=(nframes, nresid, nresid))
    event_list = []

    #  Read in all events line by line
    with open(fname) as src:
      for line in src.readlines():
        if line.startswith('+++') or line.startswith('---'):
          elm = line.split()
          polar, step,r1,r2 = elm[0],int(elm[5][:-1]),int(elm[7])-1,int(elm[10][:-1])-1
          event_list.append((polar, step, r1, r2))

    #  For each frame in the trajectory, set (i.j) cell in the matrix for all
    #  events listed. This is set the initial frame at step 0 and iteratively
    #  copy forward previous frames until an event changes a cell
    cur_frame = 0
    debug = 0
    for ev, step, r1, r2 in event_list:
      if step > cur_frame:
        for fr in range(cur_frame+1, step+1):
          cmap[fr] = cmap[cur_frame]
        cur_frame = step
      val = 1 if ev == '+++' else 0
      cmap[step][r1][r2] = cmap[step][r2][r1] = val

    # Last event was entered: copy last matrix to remaining frames in cmap
    for fr in range(cur_frame+1, nframes):
      cmap[fr] = cmap[cur_frame]

    # Only return upper triangle as a (nframes X M) matrix
    upper_tri = np.triu_indices(nresid, 1)
    return cmap[upper_tri]

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

