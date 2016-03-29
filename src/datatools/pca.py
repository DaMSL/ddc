"""Principle Component Analysis tools

  Contains both standalone methods and a PC Analyzer Class
"""
import abc

import datetime as dt

import mdtraj as md
import numpy as np
import logging
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA


logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)


class PCAnalyzer(object):

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.n_components = 0

  @abc.abstractmethod
  def solve(self, X):
    """ Given Training Data X, solve for the prinicple components
    """
    pass

  @abc.abstractmethod
  def project(self, X, numpc):
    """ Given Data X, Project high dimensional points 
    Should return the projected points to numpc components
    """
    pass


class PCALinear(PCAnalyzer):

  def __init__(self, components):
    PCAnalyzer.__init__(self)
    if isinstance(components, int):
      self.n_components = components
    self.pca = PCA(n_components = components)

  def solve(self, X):
    self.dim = np.prod(X.shape[1:])
    self.pca.fit(X.reshape(len(X), dim))
    self.pc = self.pca.components_

  def project(self, X, numpc):
    dimX = np.pod(X.shape[1:])
    if dimX != self.dim:
      logging.error('Projection Error in PCA: Cannot reshape/project %s size data using PC Vects of size, %s', str(X.shape), str(self.dim))
      return None
    projection = np.zeros(shape=(len(X), numpc))
    for i, s in enumerate(X):
      np.copyto(projection[i], np.array([np.dot(s.flatten(),v) for v in self.pc[:numpc]]))
    return projection


def calc_pca(xyz, title=None):
  n_dim = np.prod(xyz.shape[1:])
  pca = PCA(n_components = .99)
  pca.fit(xyz.reshape(len(xyz), n_dim))
  if title is not None:
    np.save('pca_%s_comp' %title, pca.components_)
    np.save('pca_%s_var' %title, pca.explained_variance_ratio_)
    np.save('pca_%s_mean' %title, pca.mean_)
    np.save('pca_%s_applied' %title, pca.transform(xyz.reshape(len(xyz), n_dim)))
  return pca



def calc_kpca(xyz, kerneltype=None, title=None, n_comp=None):
  n_dim = np.prod(xyz.shape[1:])
  result = []
  if kerneltype is None:
    klist = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
  else:
    klist = [kerneltype]
  for ktype in klist:
    kpca = KernelPCA(kernel=ktype, n_components=n_comp)
    st = dt.datetime.now()
    kpca.fit(xyz.reshape(len(xyz), n_dim))
    if title is not None:
      with open('kpca_%s_%s.dat' % (title, ktype), 'wb') as out:
        out.write(pickle.dumps(kpca))
    result.append(kpca)
  if kerneltype is None:
    return result
  else:
    return result[0]


def calc_ipca(r, key, xyz, N, title=None):
  n_dim = np.prod(xyz.shape[1:])
  ipca = IncrementalPCA()
  ipca.fit(xyz.reshape(len(xyz), n_dim))
  return ipca



def project_pca(src, pc, numpc):
  """ Project source pts onto pc's
  """
  projection = np.zeros(shape=(len(src), numpc))
  for i, s in enumerate(src):
    np.copyto(projection[i], np.array([np.dot(s.flatten(),v) for v in pc[:numpc]]))
  return projection


def project_kpca(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)



# ref:  http://sebastianraschka.com/Articles/2014_kernel_pca.html
def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    Returns the k eigenvectors (alphas) that correspond to the k largest
        eigenvalues (lambdas).

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K_norm)

    # Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]

    return alphas, lambdas
