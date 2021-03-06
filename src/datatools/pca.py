"""
  Principle Component Analysis tools
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

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

class PCAnalyzer(object):
  """ 
  Abstract class for wrapping PCA analysis object. Use load 
  method with database and object key to retrieve the analyzer or instantiate
  a child class. 
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.n_components = 0
    self.dim = None
    self.type = None
    self.pca = None
    self.trainsize = 0

  @abc.abstractmethod
  def solve(self, X):
    """ Given Training Data X, solve for the prinicple components
    """
    pass

  @abc.abstractmethod
  def project(self, X):
    """ Given Data X, Project high dimensional points 
    Should return the projected points to numpc components
    """
    pass

  def store(self, db, key):
    logging.info('Storing PCA Analyzer of type: %s', self.type)
    db.hmset(key, {'type': self.type,
        'dim': self.dim,
        'trainsize': self.trainsize,
        'n_comp': self.n_components,
        'data': pickle.dumps(self.pca)})

  @classmethod
  def load(cls, db, key):
    """ Load a PC analyzer from the datastore and cast it to the
    correct child object
    """
    obj = db.hgetall(key)
    if obj == {}:
      return None
    print('Loaded Obj', obj['type'])
    n_comp = obj['n_comp']
    analyzer = None
    if obj['type'] == 'linear':
      analyzer = PCALinear(n_comp)
    elif obj['type'] == 'kernel':
      analyzer = PCAKernel(n_comp)
    elif obj['type'] == 'incremental':
      analyzer = PCAIncremental(n_comp)
    if not analyzer:
      print('Could not load the analyzer')
      return None
    analyzer.pca = pickle.loads(obj['data'])
    analyzer.dim = eval(str(obj['dim']))
    analyzer.trainsize = int(obj['trainsize'])
    return analyzer



class PCALinear(PCAnalyzer):
  """ Simple Linear PCA """
  def __init__(self, components):
    PCAnalyzer.__init__(self)
    if isinstance(components, int):
      self.n_components = components
    self.pca = PCA(n_components = components)
    self.type = 'linear'

  def solve(self, X):
    self.dim = np.prod(X.shape[1:])
    self.pca.fit(X.reshape(len(X), self.dim))
    self.trainsize = len(X)

  def project(self, X):
    if isinstance(X, list):
      X = np.array(X)
    dimX = np.prod(X.shape[1:])
    if dimX != self.dim:
      logging.error('Projection Error in PCA: Cannot reshape/project %s size data using PC Vects of size, %s', str(X.shape), str(self.dim))
      return None
    projection = self.pca.transform(X.reshape(len(X), dimX))
    return projection


class PCAIncremental(PCAnalyzer):
  """ Incremental PCA -- used to batch input over time/space """
  def __init__(self, components):
    PCAnalyzer.__init__(self)
    if isinstance(components, int):
      self.n_components = components
    self.pca = IncrementalPCA(n_components=components, batch_size=500)
    self.num_seen = 0
    self.type = 'incremental'

  def solve(self, X):
    self.dim = np.prod(X.shape[1:])
    self.pca.partial_fit(X.reshape(len(X), self.dim))
    self.trainsize += len(X)

  def project(self, X):
    if isinstance(X, list):
      X = np.array(X)
    dimX = np.prod(X.shape[1:])
    if dimX != self.dim:
      logging.error('Projection Error in PCA: Cannot reshape/project %s size data using PC Vects of size, %s', str(X.shape), str(self.dim))
      return None
    projection = self.pca.transform(X.reshape(len(X), dimX))
    return projection

class PCAKernel(PCAnalyzer):
  """ Non-linear PCA as wrapper over SciKitLearn Kernels """
  def __init__(self, components, ktype='poly'):
    PCAnalyzer.__init__(self)
    if isinstance(components, int):
      self.n_components = components
    self.pca = KernelPCA(kernel=ktype, n_components=components)
    self.type = 'kernel'

  def solve(self, X):
    self.dim = np.prod(X.shape[1:])
    self.pca.fit(X.reshape(len(X), self.dim))
    self.trainsize = len(X)

  def project(self, X):
    if isinstance(X, list):
      X = np.array(X)
    dimX = np.prod(X.shape[1:])
    if dimX != self.dim:
      logging.error('Projection Error in KPCA: Cannot reshape/project %s size data using PC Vects of size, %s', str(X.shape), str(self.dim))
      return None
    projection = self.pca.transform(X.reshape(len(X), dimX))
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



if __name__ == '__main__':
  """ Test Methods """
  X = np.random.random(size=(1000, 12, 3))
  Y = np.random.random(size=(1000, 12, 3))

  print("Linear PCA Check")
  lpca = PCALinear(5)
  lpca.solve(X)
  p1 = lpca.project(X)
  print(p1.shape)

  print("Kernel PCA Check")
  kpca = PCAKernel(5)
  kpca.solve(X)
  p2 = kpca.project(X)
  p3 = kpca.project(Y)
  print(p2.shape)
  print(kpca.pca.get_params())

  import redis
  print("Storage Check")
  r = redis.StrictRedis(decode_responses=True)
  lpca.store(r, 'test_lpca')
  kpca.store(r, 'test_kpca')

  print("Load Check")
  l2 = PCAnalyzer.load(r, 'test_lpca')
  k2 = PCAnalyzer.load(r, 'test_kpca')
  print(kpca.pca.get_params())

  print("Correctness Check")
  y1 = lpca.project(Y)
  y2 = l2.project(Y)
  print(y1 == y2)

  y1 = kpca.project(Y)
  y2 = k2.project(Y)
  print(y1 == y2)
