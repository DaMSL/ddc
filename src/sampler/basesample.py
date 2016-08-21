"""Basic Sampling Object 
"""
import abc
import os
import time
import logging
import numpy as np

from core.slurm import systemsettings

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


class SamplerBasic(object):
  """Basic Sampler provides an abstract layer for writing/processing 
  a sampling algorithm
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, name, **kwargs):
    self.name = name
    logging.info('Sampler Object created for:  %s', name)

  @abc.abstractmethod
  def execute(self, num):
    """Method to check if service is running on the given host
    """
    pass


class UniformSampler(SamplerBasic):
  """ Uniform sampler takes a list of object and uniformly samples among them
  This is the most simple and naive sampler
  """

  #  TODO:  Ensure catalog and cache run on separate servers
  def __init__(self, choice_list):
    SamplerBasic.__init__(self, "Uniform")
    self.choices = choice_list

  def execute(self, num):
    logging.info('UNIFORM SAMPLER:  sampling called for %d  items', num)
    need_replace = (len(self.choices) < num)
    candidates = np.random.choice(self.choices, size=num, replace=need_replace)
    return candidates  



class CorrelationSampler(SamplerBasic):
  """ Corelation Feature Sampler using correlating features to derive a scoring
  metric. Scoring metric ranks all input sources. It provides a few sampling 
  options for selecting data from which to execute. The input to this class is
  the correlation matrix where:
      1 == features {i,j} correlate in time/space
      0 == features {i,j} are uncorrelated in time/space
      between 0 and 1 == features are spatially correlated in some time

   otional mean and stddev matrices are for noise calculation
  """

  def __init__(self, corr_matrix, mu=None, sigma=None, noise_factor=1.):
    SamplerBasic.__init__(self, "Biased-Correlation")

    if corr_matrix.ndim != 2:
      logging.error('Correlation Sampler requires input to be vectored to TxM (not TxNxN)')
      return

    self.N, self.K = corr_matrix.shape
    self.cm = corr_matrix
    self.mu = mu
    self.sigma = sigma
    self.noise_factor = noise_factor


  def reduce(self):
    # Filter Trivial Features  (all 0's or all 1's)
    allCorr  = [i for i in range(self.K) if (self.cm[:,i]<.001).all()]
    allUncor = [i for i in range(self.K) if (self.cm[:,i]>.999).all()]

    self.selected_features = list(sorted(set(range(self.K)) - set(allCorr) - set(allUncor)))
    self.K_m = len(self.selected_features)

    logging.info('Correlation Sampler reduced total comlexity from %d to %d dimensions', self.K, K_m)
    self.corr_matrix = self.cm[:,self.selected_features]


  def execute(self, num, theta=.15):
    logging.info('BIASED (SKEW) CORRELATIONS SAMPLER:  sampling called for %d items', num)

    # Filter Trivial Features  (all 0's or all 1's)
    allCorr  = [i for i in range(self.K) if (self.cm[:,i]<.001).all()]
    allUncor = [i for i in range(self.K) if (self.cm[:,i]>.999).all()]

    selected_features = list(sorted(set(range(self.K)) - set(allCorr) - set(allUncor)))
    K_m = len(selected_features)

    logging.info('Correlation Sampler reduced total comlexity from %d to %d dimensions', self.K, K_m)
    corr_matrix = self.cm[:,selected_features]

    # For each feature select corr basins and group by feature
    feature_set = [set([i for i in range(self.N) if corr_matrix[i][f] == 1.]) for f in range(K_m)]
    feature_score = [len(v) for v in feature_set]

    # Score each basin for each of its correlating features as a factr of the 
    #  number of basins which also correlate with the same features
    basin_score_vect = corr_matrix*feature_score

    basin_score_scalar = np.zeros(shape=(self.N))
    if self.mu is not None and self.sigma is not None:
        # APPLY NOISE HERE
      variance = self.sigma[:,selected_features]

      # Simple Noise penalty:  n_factor * S * sigma
      noise = self.noise_factor * basin_score_vect * variance
      for i in range(self.N):
          basin_score_scalar[i] = (np.sum(noise[i]) + np.sum(basin_score_vect[i])) / np.sum(corr_matrix[i])

    # Consolidate basin score to a single scalar
    else:
      for i in range(self.N):
        basin_score_scalar[i] = np.sum(basin_score_vect[i]) / np.sum(corr_matrix[i])

    # Rank all basins by correlation score and select top-N
    basin_rank = np.argsort(basin_score_scalar)
    top_N = int(self.N * theta)  # or some rarity threshold

    b = basin_score_scalar
    logging.info("""Basin Scoring Calculated! Some Stats\
      Total Basins:   %8d
      Low Score:      %8f
      High Score:     %8f
      Median Score    %8f
      Mean Score      %8f
      Theta           %8f
      Theta Score     %8f""", self.N, np.min(b), np.max(b),
      np.median(b), np.mean(b), theta, b[basin_rank[top_N]])

    # Apply a skew distribution function (weight extremes)
    skew_dist_func = lambda x: .5 * (x - top_N/2)**2
    skew_dist = [skew_dist_func(i) for i in range(top_N)]
    norm_sum = np.sum(skew_dist)
    skew_pdf = [skew_dist[i]/norm_sum for i in range(top_N)]

    # Select candidates using skew PDF
    choices = basin_rank[:top_N]
    need_replace = (len(choices) < num)
    candidates = np.random.choice(choices, size=num, replace=need_replace, p=skew_pdf)
    return candidates  
