import mdtraj as md
import datetime as dt
import os
import sys
import numpy as np
from numpy import linalg as LA
import math

HOME = os.environ['HOME']
pdb = HOME + '/work/bpti/bpti-prot.pdb'
dcd = lambda i: HOME + '/work/bpti/bpti-prot-100-%d.dcd' % i
out = lambda i: HOME + '/work/bpti/bpti-prot-%d.dcd' % i

if __name__ == '__main__':
  for i in range(10):
    print ('Loading File #', i)
    traj = md.load(dcd(i), top=pdb)
    for j in range(4):
      fnum = i*4 + j
      first = j*100000
      part = traj.slice(list(range(first, first+100000)))
      print('  Saving file #', fnum)
      part.save(out(fnum))
