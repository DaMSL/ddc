import sqlite3
import sys
import os
import logging
from collections import OrderedDict
import datetime as dt
import subprocess as proc

from core.common import systemsettings

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np




HOME = os.environ['HOME']
default_log_loc = HOME + '/ddc/results/'


dbDisabled = False


class StatCollector:
  def __init__(self, name, uid=None):
    setting = systemsettings()
    self.stat = OrderedDict()
    self.name = name
    self.uid = int('000000' if uid is None else uid)
    # Set up logging
    log = logging.getLogger('stat' + name + str(self.uid))
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(default_log_loc + 'stat_' + name + '.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    if uid is None:
      fmt = logging.Formatter(name + ',%(message)s')
    else:
      fmt = logging.Formatter(uid + ',%(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False
    self.log = log


  def collect(self, key, value):
    self.stat[key] = value

  def show(self):
    num = 0
    for label, data in self.stat.items():
      if isinstance(data, list):
        view = ','.join([str(i) for i in data])
      else:
        view = str(data)
      self.log.info('%d,%s,%s',num,label,data)
      num += 1

  def wipe(self):
    self.stat = OrderedDict()    

  def postdb(self):
    try:
      eid = db.get_expid(self.name)
      num = 0
      for label, data in self.stat.items():
        if isinstance(data, list):
          view = ','.join([str(i) for i in data])
        else:
          view = str(data)
        db.insert('stat_ctl', eid, self.uid, label, view)
        num += 1
      conn.commit()
      conn.close()
    except Exception as inst:
      print("[DB] Failed to insert CTL stats values:" )
      traceback.print_exc()
      conn.close()


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


  # home = os.environ['HOME']
  # dcd = home + '/work/bpti/' + filename
  # pdb = home + '/work/bpti/bpti-all.pdb'


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




