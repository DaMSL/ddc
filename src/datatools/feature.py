import math

import mdtraj as md
import numpy as np
from numpy import linalg as LA

from core.common import *
import core.ops as op
from core.kdtree import KDTree



class feal(object):

  def __init__(self):
    pass

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
    return np.array(fealand)   # Tuple or NDArray?

  @classmethod
  def tostring(cls, feal):
      out = 'CountsMax [C]:  %s' % str(feal[:5])
      out += '\nStateDist [S]:  %s' % str(feal[5:10])
      out += '\nRelDist [A-B]:  %s' % str(feal[10:])
      return out
