"""Micro-Benchmark class
  Used to gather metrics in code
"""  
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


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

HOME = os.environ['HOME']
default_log_loc = HOME + '/ddc/results/'


dbDisabled = False


class StatCollector:
  """Defines basic capability to collect metrics. To Use:
    1. Create the class
    2. Invoke collect() to key-val pair to collect data (with label)
    3. When done, invoke show() to display
  File logging is automatically includs to the "default_log_loc" 
  """
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
    """ Not Fully Implemented """
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
  """ Legacy for backwards compatibility to collect stats from older files"""
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






