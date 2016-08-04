"""Following methods were used to time in-situ analysis benefits
"""
import datetime as dt
import numpy as np
import mdtraj as md

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"


def genarr(sf = 1):
  return np.random.random(int(sf * (2**27)))

def iotest(sf=1):
  for loc in ['/tmp', home+'/scratch', '/dev/shm/tmp']:
    arr = genarr(sf)
    data = []
    print('\nDest: ', loc)
    for i in range(5):
      t = timecmd(lambda: np.save(loc+'/arr.npy', arr)) 
      data.append(t)
    print('Avg time [%s]  ' % loc, np.mean(data))

def timecmd(cmd, verbose=True):
  start = dt.datetime.now()
  cmd()
  end = dt.datetime.now()
  diff = (end-start).total_seconds()
  if verbose:
    print ('  Time: ', diff)
  return diff 

def timeld(n):
  start = dt.datetime.now()
  tr = md.load('bpti-all-1%03d.dcd'%n, top=pdb)
  tr.atom_slice(tr.top.select('protein'), inplace=True)
  filtered = tr.slice(idxfilt)
  end = dt.datetime.now()
  print ('Time: ', (end-start).total_seconds())
  return filtered

def timefld(n):
  start = dt.datetime.now()
  tr = md.load_frame('bpti-all-1%03d.dcd'%n, 23, top=pdb)
  tr.atom_slice(tr.top.select('protein'), inplace=True)
  end = dt.datetime.now()
  print ('Time: ', (end-start).total_seconds())
  return tr
