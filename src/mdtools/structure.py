""" Abstract Class for a Protein Structure
  with implementation for BPTI in solvent
"""
import abc
import itertools

import numpy as np
import mdtraj as md

from core.common import *


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"




class ProteinStructure(object):
  """ Protein Structure consolidates all MD protein specific implementation
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, topology):
    self.top = topology

  def secondary_structure_pairs(self):
    """ 
    defines the sequence pairs for R/L Alpha Helix and Beta Sheet
    """
    self.H1 = [i for i in itertools.combinations(self.alpha1, 2)]
    self.H2 = [i for i in itertools.combinations(self.alpha2, 2)]
    self.B = [i for i in itertools.combinations(self.beta, 2)]

class BPTI(ProteinStructure):

  def __init__(self, topology):
    ProteinStructure.__init__(self, topology)
    self.alpha1 = [i for i in range(2,6)]
    self.alpha2 = [i for i in range(47,56)]
    self.beta = [i for i in range(17,24)] + [i for i in range(28,35)]


class Protein (object):
    """The encapsulates trajectory, filtering and other common data
    for the MD protein. Facilitates standardization for common operations
    (e.g. atom selection strings)"""

    def __init__(self, name, db, load=False):
      settings = systemsettings()
      self.name = name
      self.db = db
      self.pdbfile = db.get('protein:' + self.name)
      traj = md.load(os.path.join(settings.workdir, self.pdbfile))
      traj.atom_slice(traj.top.select('protein'), inplace=True)
      self.pdb = traj
      self.top = self.pdb.top


        # self.loaded = False
        # self.pdb = None
        # self.top = None
        # # if load:
        # self.load_protein()

    # def load_protein(self):
    #   self.pdbfile = db.get('protein:' + self.name)
    #   traj = md.load(self.pdbf)
    #   traj.atom_slice(traj.top.select('protein'), inplace=True)
    #   self.pdb = traj
    #   self.top = self.pdb.top
    #   self.loaded = True

    def pfilt(self):
      """Protein Filter"""
      return self.top.select('protein')

    def afilt(self):
      """Alpha Filter"""
      return self.top.select_atom_indices('alpha')

    def hfilt(self):
      """Heavy Filter"""
      prot = self.pdb.atom_slice(self.pfilt())
      return prot.top.select("name =~ '[C.,N.,O.,S.]'")

    def bfilt(self):
      """Backbone Filter"""
      prot = self.atom_slice(self.pfilt())
      return prot.top.select("backbone")

    def filter(self, string):
      logging.warning('CAUTION: FILTER NOT PRE-DEFINED')
      return self.top.select(string)

