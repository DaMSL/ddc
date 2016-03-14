import sqlite3
import sys
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

HOME = os.environ['HOME']

def scrap_cw(appl_name):
  ts = None
  logdir=os.path.join(HOME, 'work', 'log', appl_name)
  data = {}
  bench = []
  for a in range(5):
    for b in range(5):
      data[str((a, b))] = []
  ls = sorted([f for f in os.listdir(logdir) if f.startswith('cw')])
  for i, cw in enumerate(ls[1:]):
    with open(logdir + '/' + cw, 'r') as src:
      lines = src.read().split('\n')
      timing = []
      for l in lines:
        if 'TIMESTEP:' in l:
          elms = l.split()
          ts = int(elms[-1])
        elif l.startswith('##CONV'):
          if ts is None:
            print("TimeStep was not found before scrapping data. Check file: %s", cw)
            break
          label = l[11:17]
          vals = l.split()
          conv = vals[7].split('=')[1]
          db.insert_conv()
          data[label].append((i, conv))
        elif l.startswith('##   '):
          vals = l[3:].split()
          timing.append((vals[0].strip(), vals[1].strip()))
      bench.append(timing)


  for k, v in sorted(data.items()):
    print(k, np.mean([float(p[1]) for p in v]))





dbDisabled = False
