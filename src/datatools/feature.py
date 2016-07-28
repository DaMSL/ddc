"""
Feaure Landscape
""" 
import math

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *
import core.ops as op
from core.kdtree import KDTree
import datatools.datareduce as DR
import datatools.rmsd as rmsd


class feal(object):
  """ The feature landscape (feal) object defines a distribution over multiple
  features
  Current (initial) implementation defines class methods for  atemporal
  """

  def __init__(self, protein, centroids_cart=None, centroids_ds=None, scaleto=10):
    if centroids_cart:
      self.cent_c = centroids_cart
    if centroids_ds:
      #  TODO:  Auto-compute this
      self.cent_ds = centroids_ds
    self.scaleto=scaleto
    self.max_rms_dist = 20.  #11.34
    self.min_rms_dist = 12.




  @classmethod
  def atemporal(cls, rms, scaleto=10):
    """Atemporal (individual frame) featue landscape
    """
    log_reld = op.makeLogisticFunc(scaleto, -3, 0)

    fealand = [0 for i in range(5)]
    fealand[np.argmin(rms)] = scaleto
    tup = []
    # Proximity
    for n, dist in enumerate(rms):
      # tup.append(log_prox(dist))
      maxd = 10.  #11.34
      # tup.append(scaleto*max(maxd-dist, 0)/maxd)
      tup.append(max(maxd-dist, 0))

    # Additional Feature Spaces
    for a in range(4):
      for b in range(a+1, 5):
        rel_dist = rms[a]-rms[b]
        tup.append(log_reld(rel_dist))

    fealand.extend(tup)

    # Additional Feature Spaces Would go here
    return np.array(fealand)   # Tuple or NDArray?


  @classmethod
  def atemporal2(cls, rms, scaleto=10):
    """Atemporal (individual frame) featue landscape
    """
    log_reld = op.makeLogisticFunc(scaleto, -3, 0)
    maxd = 20
    mind = 10

    fealand = [0 for i in range(5)]
    fealand[np.argmin(rms)] = scaleto
    tup = []
    # Proximity
    for dist in rms:
      fealand.append(scaleto*max(maxd-(max(dist, mind)), 0)/(maxd-mind))

    # Additional Feature Spaces
    for a in range(4):
      for b in range(a+1, 5):
        rel_dist = rms[a]-rms[b]
        tup.append(log_reld(rel_dist))

    fealand.extend(tup)

    # Additional Feature Spaces Would go here
    return np.array(fealand)   # Tuple or NDArray?


  def calc_feal_AB(cls, traj):
    """Atemporal (individual frame) featue landscape
    """
    maxd = self.max_rms_dist
    mind = self.min_rms_dist
    ds = DR.distance_space(traj)
    rms = rmsd.calc_rmsd(ds, self.cent_ds)

    # Proximity to State
    for i in range(traj.n_frames):
      fealand = [0 for i in range(5)]
      fealand[np.argmin(rms[i])] = self.scaleto
      # Proximity
      for dist in enumerate(rms):
        fealand.append(scaleto*max(maxd-(max(dist, mind), 0))/(maxd-mind))

      # Additional Feature Spaces
      for a in range(4):
        for b in range(a+1, 5):
          rel_dist = rms[a]-rms[b]
          tup.append(log_reld(rel_dist))

      fealand.extend(tup)

  @classmethod
  def tostring(cls, feal):
      out = 'CountsMax [C]:  %s' % str(feal[:5])
      out += '\nStateDist [S]:  %s' % str(feal[5:10])
      out += '\nRelDist [A-B]:  %s' % str(feal[10:])
      return out

  @classmethod
  def classify(cls, feal_list):
    state_list = []
    tran_list  = []
    for i in feal_list:
      state = [None for i in range(5)]
      state[0] = (i[10] < 5 and i[11] < 5 and i[12] < 5 and i[13] < 5)  #0
      state[1] = (i[14] < 5 and i[15] < 5 and i[16] < 5 and i[10] > 5)  #1
      state[2] = (i[17] < 5 and i[18] < 5 and i[11] > 5 and i[14] > 5)  #2
      state[3] = (i[19] < 5 and i[12] > 5 and i[15] > 5 and i[17] > 5)  #3
      state[4] = (i[13] > 5 and i[16] > 5 and i[18] > 5 and i[19] > 5)  #4
      state_list.append(state)
    return state_list