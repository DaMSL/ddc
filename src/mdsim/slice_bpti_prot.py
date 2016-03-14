#!/usr/bin/env python

import mdtraj as md
import datetime as dt
import os
import sys

NUM_FILES=125

if __name__=='__main__':
  index = int(sys.argv[1])
  first =  int(index * NUM_FILES)
  home = os.environ['HOME']
  dcd = lambda x: home+'/work/bpti/bpti-all-'+('%03d'%x if x <1000 else '%04d'%x)+'.dcd'
  top = home + '/work/bpti/bpti-all.pdb'
  topo = md.load(top)
  # heavyfilter = topo.top.select_atom_indices('heavy')
  protfilter =  topo.top.select('protein')
  start = dt.datetime.now()
  traj = md.load([dcd(i) for i in range(first, first+NUM_FILES)], top=top, atom_indices=protfilter)
  end = dt.datetime.now()
  traj.save(home+'/work/bpti/bpti-prot-%02d.dcd'%index)
  print((end-start).total_seconds())


