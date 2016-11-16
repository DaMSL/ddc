"""Basic Sampling Object 
"""
import abc
import os
import time
import logging
import numpy as np
import numpy.linalg as LA
import itertools as it
import string

from core.slurm import systemsettings

from datatools.lattice import Lattice, clusterlattice

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"


ascii_greek = ''.join([chr(i) for i in it.chain(range(915,930), range(931, 938), range(945, 969))])
k_domain = label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek
tok   = lambda x: ''.join(sorted([k_domain[i] for i in x]))
toidx = lambda x: [ord(i)-97 for i in x]
fromm = lambda x: ''.join(sorted([k_domain[i] for i,m in enumerate(x) if m == 1]))

 

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


 
class BiasSampler(SamplerBasic):
  """ Biased Sampler sampler takes a list of object and values and performs 
  a biased sampling strategy
  """
  #  TODO:  Ensure catalog and cache run on separate servers
  def __init__(self, distro):
    SamplerBasic.__init__(self, "Biased")
    self.distribution = distro

  def umbrella_pdf(self, vals):
    if max(vals)/sum(vals) > .5 * (sum(vals)):
      max_val = max(vals) * (1/len(vals))
    else:
      max_val = max(vals) + .01 * sum(vals)
    wght = np.array([max(0, (max_val - i)) if i > 0 else 0 for i in vals])
    return wght / np.sum(wght)

  def umbrella2_pdf(self, vals):
    max_val = 1 + sum(vals) * (1/len(vals))
    wght = np.array([max(0, (max_val - i)) if i > 0 else 0 for i in vals])
    return wght / np.sum(wght)


  def execute(self, num):
    n_bins = len(self.distribution)
    logging.info('BIASED SAMPLER (Umbrella):  sampling called for %d  items  on %d bins', num, n_bins)
    pdf = self.umbrella_pdf(self.distribution)
    logging.info('USING Followng Distrbution:\n %s', str(pdf))
    need_replace = (n_bins <= num)
    candidates = np.random.choice(np.arange(n_bins), size=num, \
      replace=need_replace, p=pdf)

    # Check to ensure we don't oversample
    sample_count = np.bincount(candidates, minlength = n_bins)
    adjusted_distro = [i for i in self.distribution]  #copy
    remove_list = []
    for i, num in enumerate(sample_count):
      adjusted_distro[num] -= 1
      if sample_count[i] > self.distribution[i]:
        # Oversampled
        remove_list.append(i)
        pdf = self.umbrella_pdf(adjusted_distro)
        logging.info('RE-SAMPLING.. Oversampled %d. USING Followng Updated Distrbution:\n %s', num, str(pdf))
        candidates.append(np.random.choice(np.arange(n_bins), p=pdf))
        logging.info('Replaced bin %d with bin %s', num, candidates[-1])
    for i in remove_list:
      logging.info('Popping %d', i)
      candidates.pop(i)

    return candidates  





class CorrelationSampler(SamplerBasic):
  """ Corelation Feature Sampler using correlating features to derive a scoring
  metric. Scoring metric ranks all input sources. It provides a few sampling 
  options for selecting data from which to execute. The input to this class is
  the correlation matrix where:
      1 == features {i,j} correlate in time/space
      0 == features {i,j} are uncorrelated in time/space
      between 0 and 1 == features are spatially correlated in some time

   optional mean and stddev matrices are for noise calculation
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



  def umbrella_pdf(self, vals):
    wghts = vals - np.max(vals)
    return wghts / np.sum(wghts)

  def skew_pdf(self, n):
    skew_dist_func = lambda x: .5 * (x - n/2)**2
    skew_dist = [skew_dist_func(i) for i in range(n)]
    norm_sum = np.sum(skew_dist)
    return [skew_dist[i]/norm_sum for i in range(n)]


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

    # Reduce Complexity by only considering relevant features
    selected_features = list(sorted(set(range(self.K)) - set(allCorr) - set(allUncor)))
    K_m = len(selected_features)
    logging.info('Correlation Sampler reduced total complexity from %d to %d dimensions', self.K, K_m)
    corr_matrix = self.cm[:,selected_features]

    # For each feature select corr basins and group by feature
    feature_set = [set([i for i in range(self.N) if corr_matrix[i][f] == 1.]) for f in range(K_m)]
    feature_score = [len(v) for v in feature_set]

    #  Term # 1
    # Score each basin for each of its correlating features as a factr of the 
    #  number of basins which also correlate with the same features and scale result bet 0..1
    basin_score_vect = corr_matrix*feature_score
    basin_score_scalar = np.zeros(shape=(self.N))
    bmin, bmax = np.min(basin_score_scalar), np.max(basin_score_scalar)
    bscale = bmax-bmin
    bscale_func = lambda x: (x-bmin) / bscale
    basin_score_scaled = [bscale_func(i) for i in basin_score_scalar]

    # if self.mu is not None and self.sigma is not None:
    #     # APPLY NOISE HERE
    #   variance = self.sigma[:,selected_features]

    #   # Simple Noise penalty:  n_factor * S * sigma
    #   noise = self.noise_factor * basin_score_vect * variance
    #   for i in range(self.N):
    #       basin_score_scalar[i] = (np.sum(noise[i]) + np.sum(basin_score_vect[i])) / np.sum(corr_matrix[i])

    # # Consolidate basin score to a single scalar
    # else:
    #   for i in range(self.N):
    #     basin_score_scalar[i] = np.sum(basin_score_vect[i]) / np.sum(corr_matrix[i])

    # Perform KMeans on the inverse of distance space & clip values beyond cutoff limit
    #  And collect all the cluster/centroid data
    cut = self.data['sampler:cutoff']
    n_clust = self.data['sampler:cutoff']
    proximity = (cut-ds).clip(0, cut)
    train_set = np.array([proximity[i] for i in range(0, N, 10)])
    km = KMeans(n_clust)
    km.fit(train_set)
    Y = kmprox.predict(proximity)
    centroid = km.cluster_centers_
    cluster = {k: [] for k in range(n_clust)}
    for i, k in enumerate(Y):  
      cluster[k].append(i)

    #  TERM 2:  Cluster Component
    #  2a:  Size = # of points in the cluster
    ksize = [len(cluster[k]) for k in range(n_clust)]

    #  2b:  Spatial size of the cluster (using 2d circular area)
    local_kdist = np.array([LA.norm(proximity[i] - centroid[Y[i]]) for i in range(N)])
    area_eq = lambda r: np.pi * r**2
    karea = [area_eq(np.median(local_kdist[clust])) for k, clust in cluster.items()]

    #  Scale Values between 0..1
    scale_func = lambda v, x: (x-min(v)) / (max(v)-min(v))
    ksize_scaled = [scale_func(ksize, i) for i in ksize]
    karea_scaled = [scale_func(karea, i) for i in karea]

    #  TERM #3:  Noise:  logistic function factored as how well the basin point
    #    clusters with its local K
    knoise = makeLogisticFunc(1, -.5, cut/2)

    #  TERM #4:  Temporal Exploration

    #  TERM #5:  Local Sampling Coverage (how well spread is the sampling in current loop)
    centroid_mask = np.zeros(shape=(N, n_clust))
    for i, k in enumerate(Y):
      centroid_mask[i][k] = 1.


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

    # Select candidates using umbrella PDF
    choices = basin_rank[:top_N]
    pdf = self.umbrella_pdf(basin_score_scalar[choices])
    need_replace = (len(choices) < num)
    candidates = np.random.choice(choices, size=num, replace=need_replace, p=pdf)
    return candidates  


class LatticeSampler(SamplerBasic):
  """ Lattice Sampler uses the derived lattice data to cluster nodes
  and subsequently drive a scoring function for sampling """

  def __init__(self, lattice):
    SamplerBasic.__init__(self, "Lattice")
    self.lattice = lattice
    self.theta = .9
    self.num_cluster = 8
    self.explore = True

  def execute(self, num):
    dlat = self.lattice.dlat
    Ik   = self.lattice.Ik
    CM   = self.lattice._CM()
    D    = self.lattice.E[:, self.lattice.Kr]

    print('DEBUG: ', len(dlat), len(Ik), CM.shape, D.shape)
    clusters, score, elmlist  = clusterlattice(dlat, CM, D, Ik, theta=self.theta, num_k=self.num_cluster)

    samplecount = np.zeros(len(clusters), dtype=np.int16)
    pdf = score / np.sum(score)
    candidates = []
    for i in range(num):
      while True:
        cluster_index = int(np.random.choice(len(pdf), p=pdf))
        if samplecount[cluster_index] < len(elmlist[cluster_index]):
          break
      elm, dist = elmlist[cluster_index][samplecount[cluster_index]]
      samplecount[cluster_index] += 1
      candidates.append(elm)
    return candidates




class LatticeExplorerSampler(SamplerBasic):
  """ Lattice Sampler uses the derived lattice data to cluster nodes
  and subsequently drive a scoring function for sampling """

  def __init__(self, lattice):
    SamplerBasic.__init__(self, "Lattice")
    self.lattice = lattice
    self.theta = .9
    self.num_cluster = 25

  def execute(self, num):
    dlat = self.lattice.dlat
    Ik   = self.lattice.Ik
    CM   = self.lattice._CM()
    D    = self.lattice.E[:, self.lattice.Kr]

    print('DEBUG: ', len(dlat), len(Ik), CM.shape, D.shape)
    cuboids, score, elmlist  = clusterlattice(dlat, CM, D, Ik, theta=self.theta, num_k=self.num_cluster)

    keylist = list(cuboids.keys())
    candlist = [cuboids[k] for k in keylist]

    clu_var = {}
    for k,v in cuboids.items():
      try:
        ew, ev = LA.eigh(np.cov(D[sorted(v)].T))
        clu_var[k] = 10*np.sum(ew)
      except LA.LinAlgError:
        clu_var[k] = 0

    clu_size = [len(v) for v in candlist]
    clu_cent = [np.mean(D[list(v)], axis=0) for v in candlist]
    # clu_elsc = [sorted({i:LA.norm(D[i]-clu_cent[k]) for i in v}.items(), key=lambda x: x[1]) for v in candlist]
    clu_totD = [np.sum(D[list(v)])/len(v) for v in candlist]
    clu_basc = [sorted({i:D[i][toidx(keylist[n])].sum() for i in v}.items(), key=lambda x: x[1], reverse=True) for n, v in enumerate(candlist)]
    # if print_c:
    #   for i, k in enumerate([x[0] for x in sorted(clu_totD.items(), key=lambda p: p[1], reverse=True)]):
    #     expt, explr = clu_basc[k][0][0], clu_elsc[k][-1][0]
    #     print('%3d. %4d  %4.2f  %6.2f - %5d (%s)  vs  %5d (%s)' % (i, clu_size[k], clu_var[k], clu_totD[k], expt, rlab(expt), explr, rlab(explr)))

    samplecount = np.zeros(len(cuboids), dtype=np.int16)
    score = clu_totD
    pdf = score / np.sum(score)
    candidates = []
    for i in range(num):
      while True:
        cluster_index = int(np.random.choice(len(pdf), p=pdf))
        if samplecount[cluster_index] < len(candlist[cluster_index]):
          break
      # elm = candlist[cluster_index][samplecount[cluster_index]]
      elm, _ = clu_basc[cluster_index][samplecount[cluster_index]]
      samplecount[cluster_index] += 1
      candidates.append(elm)
    return candidates



