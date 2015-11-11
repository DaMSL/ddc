import time
import math
import sys
import os
import subprocess as proc
import argparse
import abc
import redis
# from retrying import retry

from common import DEFAULT, setLogger, archiveConfig
import catalog
from slurm import slurm

from collections import namedtuple
ddl = namedtuple('key', 'value, type')

# For now: use redis catalog
import redisCatalog

import logging, logging.handlers
logger = setLogger()


class macrothread(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, schema, fname, name):
    self.name = name
    self.fname = fname

    self._input = {}
    self._term  = {}
    self._split = {}
    self._exec  = {}

    self.data = schema

    # TODO: upstream/downstream handling
    # DO: congestion control
    self.upStream = None
    self.downStream = None

    self.catalog = None

    # RUNTIME Parameters
    #  These vary from mt to mt (and among workers), but are passed to SLURM manager
    self.modules = []
    self.slurmParams = {'time':'6:0:0', 'nodes':1, 'cpus-per-task':24, 'job-name':self.name}


  # For now retain 2 copies -- to enable reverting of data
  #   Eventually this may change to a flag based data structure for inp/exec/term, etc...
  def setInput(self, *arg):
    for a in arg:
      self._input[a] = self.data[a]

  def setExec(self, *arg):
    for a in arg:
      self._exec[a] = self.data[a]

  def setTerm(self, *arg):
    for a in arg:
      self._term[a] = self.data[a]

  def setSplit(self, *arg):
    for a in arg:
      self._split[a] = self.data[a]

    # TODO: job ID management  
    self._split['id_' + self.name] = 0


  def setCatalog(self, catalog):
    self.catalog = catalog
    #  TODO: COnnection logic

  def getCatalog(self):
    return self.catalog

  @abc.abstractmethod
  def term(self):
    raise NotImplementedError("This method has not been implemented.")

  @abc.abstractmethod
  def split(self):
    raise NotImplementedError("This method has not been implemented.")

  @abc.abstractmethod
  def execute(self, item):
    raise NotImplementedError("This method has not been implemented.")


  def retry_redisConn(ex):
    return isinstance(ex, redis.ConnectionError)


  def load(self, state):
    """
    Load state from remote catalog to local cache
    """
    # TODO: Check for catalog here (????)
    # pass expeected data types (interim solution)
    self.catalog.load(state)
    for key, value in state.items():
      self.data[key] = value

  def save(self, state):
    """
    Save state to remote catalog
    """
    logger.debug(" --> call for mt.save")    
    # TODO: Check for catalog here (????)
    for key, value in state.items():
      logger.debug("    --> saving: " + key)    
      state[key]      = self.data[key]
    self.catalog.save(state)



  def manager(self, fork=False):

    logger.debug("\n==========================\n  %s  -- MANAGER", self.name)

    # Catalog Service Check here
    self.catalog.conn()

    # Check for termination  
    self.load(self._term)
    if self.term():
      logger.info('TERMINATION condition for ' + self.name)
      sys.exit(0)

    # TODO: what if catalog stops here

    # Split input data set
    self.load(self._split)
    self.load(self._input)

    # # TODO:  JobID mgmt. For now using incrementing job id counters (det if this is nec'y)
    # jobid = int(catalog.load('id_' + self.name))
    # logger.debug("Loaded ID = %d" % jobid)

    #  TODO:  Det if manager should load entire input data set, make this abstract, or
    #     push into UDF portion
    immed = self.split()
    jobid = self.data['id_%s' % self.name]

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = DEFAULT.MANAGER_RERUN_DELAY

      logger.debug("%s-MANAGER: No Available input data. Delaying %d seconds and rescheduling...." % (self.name, delay))
      # slurm.schedule('%04d' % jobid, "python3 %s %s -m" % (self.fname, self.name), delay=delay, name=self.name + '-M')

      self.slurmParams['begin'] = 'now+%d' % delay

    # Dispatch Workers
    else:
      for i in immed:
        logger.debug("%s: scheduling worker, input=%s", self.name, i)


        if self.fork:
          # OPTION B:  Fork worker thread as a separate unsupervised (e.g. daemonized) process
          logger.debug("Forking Worker: " + str(i))
          self.execute(i)        

        else:
          # OPTION A:  Make worker child thread within python worker process
          slurm.sbatch(jobid='%04d' % jobid,
              workdir = os.getcwd(), 
              options = self.slurmParams,
              modules = self.modules,
              cmd = "python3 %s -w %s" % (self.fname, str(i)))


          # OPTION C --> est a "prepare" routine which is invoked to pre-process input
          #   files and returns the sbatch job

        # slurm.sbatch(self.name + '%04d' % jobid, "python3 %s %s -w %s" % (self.fname, self.name, str(i)), name=self.name + '-W')
        jobid += 1

    self.slurmParams['job-name'] += self.name + '-M-' + str(jobid)
    slurm.sbatch(jobid='%04d' % jobid,
              workdir = os.getcwd(), 
              options = self.slurmParams,
              modules = self.modules,
              cmd = "python3 %s" % self.fname)


      # Reschedule Next Manager:

      # METHOD 1.  Schedule self after scheduling ALL workers
            # workdir = '~/bpti/manager', 

        # slurm.sbatch(jobid='%s%04d' % (self.name, jobid),
        #     workdir = os.getcwd(), 
        #     options = self.slurmParams,
        #     modules = self.modules,
        #     cmd = "python3 %s %s -m" % (self.fname, self.name))

      # slurm.schedule('%04d' % jobid, "python3 %s %s -m" % (self.fname, self.name), name=self.name + '-M')
    jobid += 1

      # METHOD 2.  After.... use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>

    self.data['id_%s' % self.name] = jobid

    self.save(self._split)
    self.save(self._exec)
    # catalog.save('id_' + self.name, jobid)
    logger.debug("==========================")

    return 1


  def worker(self, i):
    logger.debug("\n--------------------------\n  %s  -- WORKER", self.name)

    # TODO:  Does the worker need to fetch input data? (ergo: should this be abstracted)
    # jobInput = self.fetch(i)
    jobInput = i      # TODO: Manage job Input w/multiple input items, for now just pass it

    # TODO: Optimization Notifications
    #  catalog.notify(i, "active")

    # Call user-defined execution function

    #  CHECK CATALOG STATUS

    self.load(self._exec)
    self.load(self._term)


    logger.debug("Starting Worker Execution")
    result = self.execute(jobInput)
    logger.debug("Worker Execution Complete")
    # Ensure returned results are a list
    if type(result) != list:
      result = [result]

    #  CHECK CATALOG STATUS
    self.save(self._exec)
    self.save(self._term)

    #  catalog.notify(i, "complete")
    for r in result:

      # TODO:  Notification of downstream data to catalog

      # Option A -> passive model  (for now)
      # Option B -> active. do you want to do something with job???
      #   If data is appended, make it passive
      #   If data should be prioritized, then you need to be active
      # Boils down to scheduling model

      # catalog.notify(r, 'ready')
      # TODO: Add in notification for congestion control
      #   Need feedback to upstreaam to adjust split param
      #     trigger flow bet streams is tuning action (e.g. assume 1:1 to start)

      print ("  output: ", r)  
    logger.debug("--------------------------")


  def addArgs(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workinput')
    parser.add_argument('-i', '--init', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser


  def run(self):
    args = self.addArgs().parse_args()

    if args.debug:
      logging.debug("DEBUGGING: %s", self.name)
      sys.exit(0)

    if args.init:
      initialize(catalog, archive)
      sys.exit(0)

    # TODO:  Abstract The Catalog/Archive to enable mutliple
    #   and dynamic Storage type    
    #   FOR NOW:  Use a Redis Implmentation

    catalog = redisCatalog.dataStore('catalog')
    archive = redisCatalog.dataStore(**archiveConfig)

    self.setCatalog(catalog)

    if args.workinput:
      self.worker(args.workinput)
    else:
      self.manager()