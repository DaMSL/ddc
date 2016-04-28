import plot as P
import numpy as np

with open('../ctl.log') as ctllog:
  lines = ctllog.read.strip().split('\n')

raw = [k.split(',') for k in lines]
gkeys = raw[0][2:]
local = {}
for r in raw:
  if r[0] == 'local':
    if r[1] not in local:
      local[r[1]] = {}
    local[r[1]][r[2]] = r[3:]

data = {}
for l in local.keys():
  data[l] = []
  for k in gkeys:
    data[l].append([0 for i in range(len(local[l]['keys']))])


gmap = {k: i for i, k in enumerate(gkeys)}
lmap = {b: {k: i for i, k in enumerate(local[b]['keys'])} for b in local.keys()}

for r in raw:
  if r[0] == 'edge':
    _, b, l, g, n = r
    data[b][gmap[g]][lmap[b][l]] += int(n)


for tbin in data.keys():
  c = np.array(data[tbin])
  gcnt = np.sum(c, axis=1)
  lcnt = np.sum(c, axis=0)
  norm = np.nan_to_num(c / np.linalg.norm(c, axis=-1)[:, np.newaxis])
  P.heatmap(norm.T, gcnt, lcnt, tbin+'_Norm_by_Col')
  d = c.T
  norm = np.nan_to_num(d / np.linalg.norm(d, axis=-1)[:, np.newaxis])
  P.heatmap(norm, gcnt, lcnt, tbin+'_Norm_by_Row')
