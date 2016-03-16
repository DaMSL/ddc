import numpy as np
import mdtraj as md
import os
from collections import namedtuple

home = os.getenv('HOME')

def dcdfile(n):
  return (home + '/work/bpti/bpti-all-' + ('%03d' if n < 1000 else '%04d') + '.dcd') % n


def loadLabels(fn):
  label =namedtuple('window', 'time state')
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

win = loadLabels(os.getenv('HOME') + '/ddc/bpti_labels_ms.txt')


mdload = lambda x: md.load(dcdfile(x), top=home+'/work/bpti/bpti-all.pdb')
mdslice = lambda x: x.atom_slice(x.top.select_atom_indices(selection='heavy'), inplace=True)

avg = np.zeros(shape=(5, 454, 3), dtype=np.float32)
total = [0] * 5
for x in range(4124):
  print ("Calculating %04d" % x)
  st = win[x].state
  total[st] += 1
  traj = mdload(x)
  mdslice(traj)
  avg[st] += traj.slice(500).xyz[0]

np.save('average', avg)


