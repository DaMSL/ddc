import time
import math
import sys
import os
import subprocess as proc
import argparse
import common
import catalog
from slurm import slurm

# FOR PI DEMO ONLY
import picalc as pi

import logging, logging.handlers
logger = common.setLogger()


class macrothread(object):
  def __init__(self, fname, name):
    self._input = 0
    self._term  = 0
    self._exec  = 0
    self._split = 0
    self.catalog = 0
    self.name = name
    self.fname = fname
    self.upStream = None
    self.downStream = None
  # def __init__(self, inp, term, user, split):
  #   self._input = inp
  #   self._term  = term
  #   self._exec  = user
  #   self._split = split

  def initialize(self, catalog):
    logger.debug("INITIALIZING:")


  def term(self):
    pass

  def split(self):
    print ("passing")

  def execute(self, item):
    pass

  def fetch(self, i):
    pass


  def manager(self, catalog):
    logger.debug("\n==========================\n  %s  -- MANAGER", self.name)

    # TODO: Catalog Service Check here
    if self.term(catalog):
      logger.info('TERMINATION condition for ' + self.name)
      sys.exit(0)
    immed = self.split(catalog)

    # TODO:  For now using incrementing job id counters (det if this is nec'y)
    jobid = int(catalog.load('id_' + self.name))

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = 30
      logger.debug("%s-MANAGER: No Available input data. Delaying %d seconds and rescheduling...." % (self.name, delay))
      slurm.schedule('%04d' % jobid, "python3 %s %s -m" % (self.fname, self.name), delay=delay, name=self.name + '-M')
      jobid += 1

    else:
      for i in immed:
        #TODO: JobID Management
        logger.debug("%s: scheduling worker, input=%s", self.name, i)
        slurm.schedule('%04d' % jobid, "python3 %s %s -w %s" % (self.fname, self.name, i), name=self.name + '-W')
        jobid += 1

      # METHOD 1.  Schedule self after scheduling ALL workers
      slurm.schedule('%04d' % jobid, "python3 %s %s -m" % (self.fname, self.name), name=self.name + '-M')
      jobid += 1

      # METHOD 2.  After.... use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>

    catalog.save('id_' + self.name, jobid)
    logger.debug("==========================")

  def worker(self, catalog, i):
    logger.debug("\n--------------------------\n  %s  -- WORKER", self.name)
    # print ("FETCHING: ", i)
    # jobInput = self.fetch(i)
    jobInput = i      # TODO: Manage job Input w/multiple input items, for now just pass it

    #  catalog.notify(i, "active")
    data = self.execute(catalog, jobInput)

    # Ensure returned results are a list
    if type(data) != list:
      data = [data]

    #  catalog.notify(i, "complete")
    for d in data:
      # catalog.notify(d, 'ready')
      print ("  output: ", d)  
    logger.debug("--------------------------")

