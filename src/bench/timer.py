"""Common definitions and methods
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
  def __init__(self, name, uid=None):
    setting = systemsettings()
    self._begin = None
    self.tick = OrderedDict()
    self.last = None
    self.recent = None
    self.name = name
    # Set up logging
    log = logging.getLogger(name)
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
    for label, ts in self.tick.items():
      t = (ts-self._begin).total_seconds()
      if last is None:
        last = ts
      d = (ts-last).total_seconds()
      last = ts
      str_t = '%f'%t if t < 10 else '%d'%t
      str_d = '%f'%d if d < 10 else '%d'%d
      self.log.info('%d,%s,%s,%s',num,str_t,str_d,label)
      num += 1