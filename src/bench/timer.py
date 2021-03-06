"""Micro-Benchmark class
  Used to track and time intra-code events
"""  
import logging
import os
import subprocess as proc
import numpy as np
import sys
import datetime as dt

from core.common import *

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

default_log_loc = os.environ['HOME'] + '/ddc/results/'

class microbench:
  """Defines basic capability to set up and
  track microbench marks within places in code. To Use:
    1. Create the class
    2. Start the initial timer, start()
    3. Invoke mark() to mark a time hack relative to the start time
    4. When done, invoke show() to display
  File logging is automatically includs to the "default_log_loc" 
  """
  
  def __init__(self, name, uid=None):
    setting = systemsettings()
    self._begin = None
    self.tick = OrderedDict()
    self.last = None
    self.recent = None
    self.name = name
    self.uid = int('000000' if uid is None else uid)
    # Set up logging
    log = logging.getLogger('bench'+name)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(default_log_loc + 'bench_' + name + '.log')
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

  def start(self):
    self._begin = dt.datetime.now()
    self.tick['START'] = self._begin
    self.recent = self._begin

  def mark(self, label=None):
    if label is None:
      label = 'mark-%02d' % len(self.tick.keys())
    self.tick[label] = dt.datetime.now()
    self.last = self.recent
    self.recent = self.tick[label]

  def delta_last(self):
    if self.last is None:
      return 0.
    return (self.recent-self.last).total_seconds()

  def show(self):
    last = None
    num = 0
    self.log.info('0,%s',self._begin.isoformat())
    for label, ts in self.tick.items():
      t = (ts-self._begin).total_seconds()
      if last is None:
        last = ts
      d = (ts-last).total_seconds()
      last = ts
      str_t = '%f'%t if t < 10 else '%d'%t
      str_d = '%f'%d if d < 10 else '%d'%d
      if num > 0:
        self.log.info('%d,%s,%s,%s',num,str_t,str_d,label)
      num += 1

  def postdb(self):
    try:
      conn = db.getConn()
      cur = conn.cursor()
      qry = "SELECT expid FROM expr WHERE expname='%s';" % self.name
      cur.execute(qry)
      eid = int(cur.fetchone()[0])
      last = None
      num = 0
      for label, ts in self.tick.items():
        r = (ts-self._begin).total_seconds()
        if last is None:
          last = ts
        d = (ts-last).total_seconds()
        last = ts
        db.insert('bench_ctl', eid, self.uid, num, r, d, label)
        num += 1
      conn.commit()
      conn.close()
    except Exception as inst:
      print("[DB] Failed to insert benchmark values:" )
      traceback.print_exc()
      conn.close()