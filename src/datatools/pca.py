import datetime as dt

import mdtraj as md
import numpy as np
import logging
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA




logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

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
    print('Calculating Kernal for type: ', ktype)
    st = dt.datetime.now()
    kpca.fit(xyz.reshape(len(xyz), n_dim))
    print('  Time=  %5.1f' % ((dt.datetime.now()-st).total_seconds()))
    if title is not None:
      with open('kpca_%s_%s.dat' % (title, ktype), 'wb') as out:
        out.write(pickle.dumps(kpca))
    result.append(kpca)
  if kerneltype is None:
    return result
  else:
    return result[0]



def project_pca(src, pc, numpc):
  """ Project source pts onto pc's
  """
  projection = np.zeros(shape=(len(src), numpc))
  for i, s in enumerate(src):
    np.copyto(projection[i], np.array([np.dot(s.flatten(),v) for v in pc[:numpc]]))
  return projection
