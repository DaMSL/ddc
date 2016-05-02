import mdtraj as md
import numpy as np
import logging
import pickle

from sklearn.mixture import GMM

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)


def calc_gmm(xyz, N, ctype='full', title=None):
  n_dim = np.prod(xyz.shape[1:])
  gmm = GMM(n_components=N, covariance_type=ctype)
  gmm.fit(xyz.reshape(len(xyz), n_dim))
  if title is not None:
    np.save('gmm_%d_%s_mean' % (N, ctype), gmm.means_)
    np.save('gmm_%d_%s_wgt' % (N, ctype), gmm.weights_)
    np.save('gmm_%d_%s_cov' % (N, ctype), gmm.covars_)
    np.save('gmm_fit_%d_%s' % (N, ctype), gmm.predict(xyz.reshape(len(xyz), n_dim)))
  return gmm

  # with open('ipca_pickled.dat', 'wb') as pout:
#    pout.write(ipca_p)

# xyz = DE.loadpts(skip=4, filt=DE.atom_filter['alpha'])
# gmm = calc_gmm(xyz, 5, 'alpha')
# with open('gmm_alpha.dat', 'wb') as pout:
#    pout.write(pickle.dumps(gmm))

cov = np.array(cov)
variance = np.array([np.diag(i) for i in cov])
avg = np.array(avg).reshape(len(avg), 174)

cca = CCA(n_components=3)
cca.fit(avg, variance)
X3,Y3 = cca.transform(avg, variance)

st=dt.datetime.now()
gmm.fit(X3)
print((dt.datetime.now()-st).total_seconds())

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
  for n_components in n_components_range:
    # Fit a mixture of Gaussians with EM
    gmm = GMM(n_components=n_components, covariance_type=cv_type)
    gmm.fit(X3)
    bic.append(gmm.bic(X3))
    if bic[-1] < lowest_bic:
      lowest_bic = bic[-1]
      best_gmm = gmm