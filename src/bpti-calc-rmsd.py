#!/bin/usr/env python
import logging
import datetime as dt

import mdtraj as md
import numpy as np

import datatools.datareduce as DR
import datatools.rmsd as DTrmsd
import mdtools.deshaw as DE


logging.basicConfig(format='%(message)s', level=logging.DEBUG)

diff = lambda x: (dt.datetime.now()-x).total_seconds()

cent = np.load('data/bpti-alpha-dist-centroid.npy')
rmsd = np.zeros(shape = (4125000, 5))
for i in range(41):
  st = dt.datetime.now()
  logging.info('Loading: %d',i)
  tr = md.load(DE.DCD_PROT(i), top=DE.PDB_PROT)
  tr.atom_slice(DE.atom_filter['alpha'], inplace=True)
  logging.info('  Dist: %d  (%5.1f)',i, diff(st))
  st2 = dt.datetime.now()
  ds = DR.distance_space(tr)
  logging.info('  RMSD: %d  (%5.1f)',i, diff(st2))
  rms = DTrmsd.calc_rmsd(ds, cent)
  rmsd[i*100000:i*100000+len(rms)] = rms
  logging.info('  Total for: %d  (%5.1f)',i, diff(st))

np.save('bpti-rmsd-alpha-dspace.npy', rmsd)





dist = np.load('bpti-rmsd-alpha-dspace.npy')
