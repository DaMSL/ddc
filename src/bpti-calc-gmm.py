
#!/bin/usr/env python
import logging
import datetime as dt

import mdtraj as md
import numpy as np

import datatools.datareduce as DR
import datatools.rmsd as DTrmsd
import mdtools.deshaw as DE

from sklearn.mixture import GMM

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

diff = lambda x: (dt.datetime.now()-x).total_seconds()

def calc_gmm(xyz, N, ctype='full', title=None):
  n_dim = np.prod(xyz.shape[1:])
  gmm = GMM(n_components=N, covariance_type=ctype)
  gmm.fit(xyz.reshape(len(xyz), n_dim))
  if title is not None:
    np.save('gmm_%d_%s_mean' % (N, ctype), gmm.means_)
    np.save('gmm_%d_%s_wgt' % (N, ctype), gmm.weights_)
    np.save('gmm_%d_%s_cov' % (N, ctype), gmm.covars_)
    np.save('gmm_fit_%d_%s' % (N, ctype), gmm.predict(xyz.reshape(len(xyz), n_dim)))

xyz = DE.loadpts(skip=4, filt=DE.atom_filter['alpha'])
gmm = calc_gmm(xyz, 5, title='alpha')
with open('gmm_alpha.dat', 'wb') as pout:
   pout.write(pickle.dumps(gmm))

