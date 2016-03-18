import mdtraj as md
import numpy as np
import logging

from numpy import linalg as LA

import datatools.datareduce as dr

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

np.set_printoptions(precision=3, suppress=True)
logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


def calc_rmsd(traj, centroid, space='cartesian', title=None, top=None):
  """Calculate the RMSD from each point in traj to each of the centroids
  Input passed can be either a trajectory or an array-list object
  Title, if defined, will be used in the output filename
  """
  # TODO: Check dimenstionality of source points and centroid for given space
  observations = traj.xyz if isinstance(traj, md.Trajectory) else traj
  rmsd = np.zeros(shape=(len(observations), len(centroid)))
  for n, pt in enumerate(observations):
    # TODO:  Check Axis here
    rmsd[n] = np.array([np.sum(LA.norm(pt - C)) for C in centroid])
  return rmsd


def calc_bpti_centroid(traj_list):
  """Given a trajectory list of frames corresponding to the pre-labeled DEShaw
  dataset, calcualte centroids:
    - Groups frames by states and calcuate average (x,y,z) for all atoms
    This wil exclude any near-transition states and does a best fit of 
    using only in-state points (non-transition ones)
  """
  # Assuming distance space (with alpha-filter), hence 1653 dimensions
  sums = np.zeros(shape=(5, 1653))
  cnts = [0 for i in range(5)]
  for n, traj in enumerate(prdist):
    for i in range(0, len(traj), 40):
      try:
        idx = (n*400)  + (i // 1000)
        state = label[idx]
        # Exclude any near transition frames
        if idx < 3 or idx > 4121:
          continue
        if state == label[idx-2] == label[idx-1] == label[idx+1] == label[idx+2]:
          sums[state] += traj[i]
          cnts[state] += 1
      except IndexError as err:
        pass # ignore idx errors due to tail end of DEShaw data
  cent = [sums[i] / cnts[i] for i in range(5)]
  return (np.array(cent))


def check_bpti_rms(traj_list, centroid, skip=40):
  hit = 0
  miss = 0
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      predicted_state = np.argmin(dist)
      if labeled_state == predicted_state:
        hit += 1
      else:
        miss += 1
  print ('Hit rate:  %5.2f  (%d)' % ((hit/(hit+miss)), hit))
  print ('Miss rate: %5.2f  (%d)' % ((miss/(hit+miss)), miss))


def check_bpti_rms_trans(traj_list, centroid, skip=40):
  traj_list = prdist
  centroid = cent_d
  skip=100
  exact_hit = 0
  total_miss = 0
  det_trans = 0
  total = 0
  tdist = 0.
  wdist = 0.
  tlist = []
  theta = 0.15
  for n, traj in enumerate(traj_list[:10]):
    print ('checking traj #', n)
    for i in range(0, len(traj), skip):
      idx = (n*400)  + (i // 1000)
      labeled_state = label[idx]
      dist = [np.sum(LA.norm(traj[i] - C)) for C in centroid]
      prox = np.argsort(dist)
      A = prox[0]
      B = prox[1]
      delta = dist[B] - dist[A]
      if delta > theta:
        B = A
      if labeled_state == A == B:
        exact_hit += 1
        wdist += delta
      elif labeled_state == A or labeled_state == B:
        det_trans += 1
        tdist += delta
        tlist.append((idx, (A, B), labeled_state))
      else:
        total_miss += 1
      total += 1

