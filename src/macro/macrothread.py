#!/usr/bin/env python
"""Abstract Data Class defining the macro thread

Sub-classes should implement a split and execute method (and optional )
"""

import time
import math
import sys
import os
import subprocess as proc
import argparse
import abc
import copy
import re
from collections import namedtuple
import copy

import redis

from .core.common import * 
from .core.slurm import slurm
from .overlay.redisOverlay import RedisClient

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)


class macrothread(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, fname, name):
    self.name = name
    self.fname = fname

    #  TODO: Better handling of command line args
    self.parser = None
    self.parser = self.addArgs()
    args = self.parser.parse_args()
    self.config = args.config

    #  apply global settings (TODO: Move to system pro-proc)    
    DEFAULT.applyConfig(self.config)

    # Thread State
    self._mut = []
    self._immut = []
    self._append = []

    self._state = {}
    self.data = {}
    self.origin = {}
    self.upstream = None
    self.downstream = None

    # Elasticity
    self.delay = None

    # Data Stre
    self.catalog = None
    self.localcatalogserver = None

    # Runtime Modules to load   -- Assumes Redis as baseline catalog (for now)
    self.modules = set(['redis'])

    # Default Runtime parameters to pass to slurm manager
    #  These vary from mt to mt (and among workers) and can be updated
    #  through the prepare method
    self.slurmParams = {'time':'1:0:0', 
              'nodes':1, 
              'cpus-per-task':1, 
              'partition':DEFAULT.PARTITION, 
              'job-name':self.name,
              'workdir' : os.getcwd()}

    self.slurm_id  = None
    self.job_id    = None
    self.seq_num   = None


  #  Job ID Management  
  #  Sequence ALL JOBS --- good for up to 10,000 managers and 100 concurrently launched workers
  #  TODO:  Can make this larger, if nec'y
  def toMID(self, jobid):
    return '%sm-%04d.00' % (self.name[0], int(jobid))

  def fromMID(self, ):
    if self.job_id is None:
      return None

    if re.match(r'\D[m,w]-\d{4}.{2}', self.job_id):
      job = self.job_id.split('-')[1]
      m, w = job.split('.')
      return int(m)
    else:
      return None

  def toWID(self, jobid, worknum):
    return '%sw-%04d.%02d' % (self.name[0], jobid, worknum)

  def seqNumFromID(self, ):
    """Returns the logical sequenced number for this macrothread
    """
    if self.job_id is None:
      return '000000'

    if re.match(r'\D[m,w]-\d{4}.{2}', self.job_id):
      job = self.job_id.split('-')[1]
      m, w = job.split('.')
      return '%06d' % (100*(int(m)) + int(w))
    else:
      return '000000'


  #  Methods to add data elements to thread state
  def addMut(self, key, value=None):
    self._mut.append(key)
    if value is not None:
      print ("ADDING MUTABLE DATA:  ", key)
      self.data[key] = value

  def addImmut(self, key, value=None):
    self._immut.append(key)

  def addAppend(self, key, value=None):
    self._append.append(key)
    self.origin[key] = None
    if value is not None:
      self.data[key] = value
      self.origin[key] = copy.deepcopy(value)

  # TODO:  Eventually we may want multiple up/down streams
  def setStream(self, upstream, downstream):
    if upstream is None:
      logger.info("Upstream data `%s` not defined", upstream)
    self.upstream = upstream

    if downstream is None:
      logger.warning("Downstream data `%s` not defined in schema", downstream)
    self.downstream = downstream

  # Implementation methods for macrothreads
  @abc.abstractmethod
  def term(self):
    raise NotImplementedError("This method has not been implemented.")

  @abc.abstractmethod
  def split(self):
    raise NotImplementedError("This method has not been implemented.")

  @abc.abstractmethod
  def execute(self, item):
    raise NotImplementedError("This method has not been implemented.")


  def fetch(self, item):
    """
    Retrieve data element associated with given item reference (defaults to returning item ref)
    """
    return item

  def preparejob(self, job):
    """
    Called just prior to submitting a job on the Queue -- allows inherited macrothread
    to affect how a job is scheduled (e.g. dynamically setting Slurm params) in the 
    Manager
    """
    pass


  def configElasPolicy(self):
    """
    Set rescheduling / delay policies for the manager thread 
    """
    self.delay = 60


  #  Catalog access methods
  def setCatalog(self, catalog):
    self.catalog = catalog

  def getCatalog(self):
    return self.catalog

  def load(self, *keylist):
    """
    Load state from remote catalog to local cache
    """
    keys = []
    for k in keylist:
      if isinstance(k, list):
        keys.extend(k)
      elif isinstance(k, str):
        keys.append(k)
      else:
        logging.error("Key Loading error: %s", str(k))
    # pass expeected data types (interim solution)

    data = self.catalog.load(keys)
    logging.debug("Loaded state for %s:", self.name)
    for k,v in data.items():
      self.data[k] = v
      logging.debug("  %10s", k)


    logging.debug("Setting origin for append-only")
    for k in keys:
      if k in self._append:
        self.origin[k] = copy.deepcopy(data[k])
        logging.debug("  Setting Append only origin for: %s", k)


  def save(self, *keylist):
    """
    Save state to remote catalog
    """
    state = {}
    keys = []
    for k in keylist:
      if isinstance(k, list):
        keys.extend(k)
      elif isinstance(k, str):
        keys.append(k)
      else:
        logging.error("Key Loading error: %s", str(k))
    for key in keys:
      if key not in self.data:
        logging.error("KEY ERROR. %s not found in current state", key)
      else:
        state[key] = self.data[key]    
    self.catalog.save(state)


  def append(self, *keylist):
    """
    Save state to remote catalog
    """
    state = {}
    keys = []
    for k in keylist:
      if isinstance(k, list):
        keys.extend(k)
      elif isinstance(k, str):
        keys.append(k)
      else:
        logging.error("Key Loading error: %s", str(k))
    for key in keys:
      if key not in self.data:
        logging.error("KEY ERROR. %s not found in current state", key)
      else:
        if type(self.data[key]) in [int, float, np.ndarray]:
          state[key] = self.data[key] - self.origin[key]
        elif isinstance(self.data, list):
          state[key] = [x for x in self.data[key] if x not in self.origin[key]]
        elif isinstance(self.data, dict):
          state[key] = {k:v for k.v in self.data[key].items if k not in self.origin[key].keys()}
        else:
          logging.error("CANNOT APPEND data `%s` of type of %s", key, str(type(self.data[key])))
    
    return self.catalog.append(state)


  # Manager Algorithm
  def manager(self, fork=False):

    logger.debug("\n==========================\n  MANAGER:  %s", self.name)


    # Check global termination:
    term_flag = self.data['terminate']
    if term_flag and term_flag.lower() in ['halt', 'stop', 'now']:
      logger.info('RECEIVED TERMINATION FLAG. Shutting down')
      sys.exit(0)

    # Load Data from Thread's State and Upstream thread
    if self.upstream:
      logging.debug("Loading upstream data: %s", self.upstream)
      self.load(self.upstream)

    # Check for termination  
    if self.term():
      logger.info('TERMINATION condition for ' + self.name)
      # sys.exit(0)

    # Set Elasticity Policy
    self.configElasPolicy()

    # Note: Manager can become a service daemon. Thus, we allow the manager
    #  to run along with the monitor process and assume the manager overhead
    #  is small enough to not interfere. Eventually, this will be threaded
    #  differently by preventing the local service (within this object's
    #  context) from running while the manager performs its split() function
    #  worker dispatching. The worker (below) starts a local service
    #  for reading, reads in the state, stops it, performs its work, and then
    #  starts it for writing and remain alive to monitor......
    #  Hence, we'll eventually change this next line to False or some other
    #  state value or we'll just let this manager become the monitor and
    #  provide the service which means it will need to immediate re-schedule
    #  itself
    # self.catalogPersistanceState = True
    # if self.localcatalogserver and not self.catalogPersistanceState:
    #   self.catalog.stop()
    #   self.localcatalogserver = None

    #  TODO:  Det if manager should load entire input data set, make this abstract, or
    #     push into UDF portion
    #  Defer can return either a list of items to push back or a "split" value to
    #  perform an in-line data trim on the key-store DB (optimiation)
    immed, defer  = self.split()

    #  Manager oversee's id assignment. 
    idlabel = 'id_%s' % self.name
    self.catalog.incr(idlabel)
    nextid = self.catalog.get(idlabel)

    # first ID check 
    nextid = 0 if nextid is None else int(nextid)
    myid = self.fromMID()
    if myid is None:
      myid = int(nextid - 1)

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = int(self.delay)
      logger.debug("MANAGER %s: No Available input data. Delaying %d seconds and rescheduling...." % (self.name, delay))
      self.slurmParams['begin'] = 'now+%d' % delay

    # Dispatch Workers
    else:
      workernum = 1

      # Set baseline slurm params and modules (to allow for dynamic disatching)
      baseline_param = copy.deepcopy(self.slurmParams)
      baseline_mods  = copy.deepcopy(self.modules)
      for i in immed:
        logger.debug("%s: scheduling worker, input=%s", self.name, i)
        self.preparejob(i)
        self.slurmParams['job-name'] = self.toWID(myid, workernum)
        slurm.sbatch(taskid=self.slurmParams['job-name'],
            options = self.slurmParams,
            modules = self.modules,
            cmd = "python3 %s -c %s -w %s" % (self.fname, self.config, str(i)))
        workernum += 1
        # Reset params and mods
        self.slurmParams = copy.deepcopy(baseline_param)
        self.modules     = copy.deepcopy(baseline_mods)


    # Elas Policy to control manager rescheduling
    delay = self.delay  
    self.slurmParams['begin'] = 'now+%d' % delay
    self.slurmParams['job-name'] = self.toMID(nextid)
    self.slurmParams['cpus-per-task'] = 1
    slurm.sbatch(taskid =self.slurmParams['job-name'],
              options   = self.slurmParams,
              modules   = self.modules,
              cmd = "python3 %s -c %s" % (self.fname, self.config))

    # TODO: Alternate manager rescheduling:  Trigger Based
    #   use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>

    # Consume upstream input data
    if isinstance(defer, list) and len(defer) > 0:
      self.catalog.removeItems(self.upstream, defer)
    elif defer is not None:
      self.catalog.slice(self.upstream, defer)

    # Other interal thread state is saved back to catalog
    self.save(self._mut)

    logger.debug("==========================")
    return 0


  def worker(self, i):
    logger.debug("\n--------------------------\n   WORKER:  %s", self.name)
    # TODO: Optimization Notifications
    #  catalog.notify(i, "active")

    logger.info("WORKER Fetching Input parameters/data for input:  %s", str(i))
    jobInput = self.fetch(i)

    logger.debug("Starting WORKER  Execution  ---->>")
    result = self.execute(jobInput)
    logger.debug("<<----  WORKER Execution Complete")

    # Ensure returned results are a list
    if type(result) != list:
      result = [result]

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

      logging.debug ("  WORKER output:   %s", r)  

    if self.downstream is not None and len(result) > 0:
      self.catalog.append({self.downstream: result})

    logging.debug("Updating all append-only state items")
    self.append(self._append)

    logging.debug("Saving all mutable state items")
    self.save(self._mut)

    logger.debug("--------------------------")


  def addArgs(self):
    if self.parser is None:
      parser = argparse.ArgumentParser()
      parser.add_argument('-w', '--workinput')
      parser.add_argument('-c', '--config', default='default.conf')
      parser.add_argument('-i', '--init', action='store_true')
      parser.add_argument('-d', '--debug', action='store_true')
      self.parser = parser
    return self.parser


  def run(self):
    args = self.parser.parse_args()

    settings = systemsettings()

    logging.info("APPLICATION:    %s", DEFAULT.APPL_LABEL)
    logging.info("WORKDIR:  %s", DEFAULT.WORKDIR)

    # Read in Slurm params  (TODO: Move to abstract slurm call)
    self.job_id   = os.getenv('JOB_NAME')
    self.slurm_id = os.getenv('SLURM_JOB_ID')

    logging.debug('EnVars Follow.q....')

    for i in ['SBATCH_JOBID', 'SBATCH_JOB_NAME', 'SLURM_JOB_ID', 'SLURM_JOBID', 'SLURM_JOB_NAME']:
      logging.debug('    %s : %s', i, os.getenv(i))

    logging.info("JOB NAME :  %s", str(self.job_id))
    logging.info("SLURM JOB:  %s", str(self.slurm_id))

    if args.debug:
      logging.debug("DEBUGGING: %s", self.name)


    if args.init:
      sys.exit(0)

    # Both Worker & Manager need catalog to run; load it here and import schema

    if not self.catalog:
      # self.catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
      self.catalog = RedisClient(DEFAULT.APPL_LABEL)

    if self.catalog.isconnected and self.catalog.ping():
      logging.info('Catalog service is connected')
    else:
      logging.info("Catalog service is not running. Starting the service now")
      service = RedisService(DEFAULT.APPL_LABEL)
      self.localcatalogserver = service.start()
      logging.info("Catalog service started as a background thread.")
      #  TODO: Quit and self-reschedule???

    self.catalog.loadSchema()   # Should this be called from within the catalog module?

    # Load data from Catalog
    logger.info("Loading Thread State for from catalog:")
    self.load(self._mut, self._immut, self._append)

    if args.workinput:
      logger.debug("Running worker.")
      self.worker(args.workinput)
    else:
      self.manager()

    if self.localcatalogserver:
      logger.debug("This thread is running the catalog. Waiting on local service to terminate...")
      self.localcatalogserver.join()
      self.localcatalogserver = None
