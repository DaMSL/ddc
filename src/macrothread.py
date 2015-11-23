import time
import math
import sys
import os
import subprocess as proc
import argparse
import abc
import redis
# from retrying import retry

from common import * 
import catalog
from slurm import slurm

from collections import namedtuple
ddl = namedtuple('key', 'value, type')

# For now: use redis catalog
import redisCatalog

import logging, logging.handlers
logger = setLogger("MT")


class macrothread(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, schema, fname, name):
    self.name = name
    self.fname = fname

    #  TODO: See if this can be Simplied by combining all "state" items into one 
    #     big set of state fields dict

    # self._input = {}
    # self._term  = {}
    # self._split = {}
    # self._exec  = {}

    self._state = {}
    self.data = schema
    self.upstream = None
    self.downstream = None


    self.catalog = None
    self.localCatalogServer = None

    # Catalog support status (bool for now, may need to be state-based)
    self.catalogPersistanceState = False

    # Runtime Modules to load   -- Assumes Redis as baseline catalog (for now)
    self.modules = set(['redis'])

    # Default Runtime parameters to pass to slurm manager
    #  These vary from mt to mt (and among workers) and can be updated
    #  through the prepare method
    self.slurmParams = {'time':'6:0:0', 
              'nodes':1, 
              'cpus-per-task':1, 
              'job-name':self.name,
              'workdir' : os.getcwd()}

  # For now retain 2 copies -- to enable reverting of data
  #   Eventually this may change to a flag based data structure for inp/exec/term, etc...
  # def setInput(self, *arg):
  #   for a in arg:
  #     self._input[a] = self.data[a]

  # def setExec(self, *arg):
  #   for a in arg:
  #     self._exec[a] = self.data[a]

  # def setTerm(self, *arg):
  #   for a in arg:
  #     self._term[a] = self.data[a]

  # def setSplit(self, *arg):
  #   for a in arg:
  #     self._split[a] = self.data[a]

  #   # TODO: job ID management  
  #   self._split['id_' + self.name] = 0


  # TODO:  Eventually we may want multiple up/down streams
  def setStream(self, upstream, downstream):
    if upstream is None or upstream not in self.data.keys():
      logging.error("Upstream data `%s` not defined in schema", upstream)
    else:
      self.upstream = upstream

    if downstream is None or downstream not in self.data.keys():
      logging.error("Downstream data `%s` not defined in schema", downstream)
    else:
      self.downstream = downstream

  def setState(self, *arg):
    for a in arg:
      self._state[a] = self.data[a]
    self._state['id_' + self.name] = 0


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


  def fetch(self, item):
    """
    Retrieve data element associated with given item reference (defaults to returning item ref)
    """
    return item

  def retry_redisConn(ex):
    return isinstance(ex, redis.ConnectionError)


  def load(self, state):
    """
    Load state from remote catalog to local cache
    """
    # pass expeected data types (interim solution)
    self.catalog.load(state)
    for key, value in state.items():
      self.data[key] = value

  def save(self, state):
    """
    Save state to remote catalog
    """
    for key, value in state.items():
      state[key]      = self.data[key]
    self.catalog.save(state)



  def manager(self, fork=False):

    logger.debug("\n==========================\n  MANAGER:  %s", self.name)

    # Catalog Service Check here
    if not self.catalog:
      self.catalog = redisCatalog.dataStore('catalog')
    self.localcatalogserver = self.catalog.conn()

    # Load Data from Thread's State and Upstream thread
    self.load(self._state)
    if self.upstream:
      self.load({self.upstream:[]})

    # Check for termination  
    if self.term():
      logger.info('TERMINATION condition for ' + self.name)
      sys.exit(0)

    # # Split input data set
    # self.load(self._split)
    # self.load(self._input)


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
    self.catalogPersistanceState = True
    if self.localcatalogserver and not self.catalogPersistanceState:
      self.catalog.stop()
      self.localcatalogserver = None

    #  TODO:  Det if manager should load entire input data set, make this abstract, or
    #     push into UDF portion
    immed, defer  = self.split()

    # # TODO:  JobID mgmt. For now using incrementing job id counters (det if this is nec'y)
    # jobid = int(catalog.load('id_' + self.name))
    # logger.debug("Loaded ID = %d" % jobid)
    myid = 'id_%s' % self.name
    jobid  = self.data[myid]

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = DEFAULT.MANAGER_RERUN_DELAY
      logger.debug("MANAGER %s: No Available input data. Delaying %d seconds and rescheduling...." % (self.name, delay))
      self.slurmParams['begin'] = 'now+%d' % delay

    # Dispatch Workers
    else:
      for i in immed:
        logger.debug("%s: scheduling worker, input=%s", self.name, i)
        self.slurmParams['job-name'] = "%sW-%05d" % (self.name, jobid)
        slurm.sbatch(taskid=self.slurmParams['job-name'],
            options = self.slurmParams,
            modules = self.modules,
            cmd = "python3 %s -w %s" % (self.fname, str(i)))
        jobid += 1

    # Reschedule Next Manager:
    # METHOD 1.  Automatic. Schedule self after scheduling ALL workers
    #      FOR NOW, back off delay  (for debug/demo/testing)
    delay = DEFAULT.MANAGER_RERUN_DELAY  
    self.slurmParams['begin'] = 'now+%d' % delay

    self.slurmParams['job-name'] = "%sM-%05d" % (self.name, jobid)
    slurm.sbatch(taskid =self.slurmParams['job-name'],
              options   = self.slurmParams,
              modules   = self.modules,
              cmd = "python3 %s" % self.fname)
    jobid += 1

    # METHOD 2.  Trigger Based
    #   use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>
    self.data[myid] = jobid


    # Ensure the catalog is available. If not, start it for persistance and check
    # if the thread is running before exit
    self.localcatalogserver = self.catalog.conn()
    self.catalogPersistanceState = True


    # Consume upstream input data (should this happen earlier or now?)
    #  TODO: Should we just make this only 1 list allowed for upstream data?
    if isinstance(defer, list) and len(defer) > 0:
      self.catalog.removeItems(self.upstream, defer)
    else:
      self.catalog.slice(self.upstream, defer)

    # Other interal thread state is saved back to catalog
    #  TODO: For now the manager ONLY updates the job ID
    self.save({myid: jobid})
    
    if self.localcatalogserver and self.catalogPersistanceState:
      logger.debug("This Manager is running the catalog. Waiting on local service to terminate...")
      self.localcatalogserver.join()
      self.localcatalogserver = None

    logger.debug("==========================")
    return 1


  def worker(self, i):
    logger.debug("\n--------------------------\n   WORKER:  %s", self.name)

    # TODO:  Does the worker need to fetch input data? (ergo: should this be abstracted)
    # jobInput = self.fetch(i)

    # TODO: Optimization Notifications
    #  catalog.notify(i, "active")

    # Catalog Service Check here. Ensure catalog is available and then retrieve all data up front
    if not self.catalog:
      self.catalog = redisCatalog.dataStore('catalog')
    self.localcatalogserver = self.catalog.conn()

    logger.info("WORKER Loading Thread State for from catalog:")
    self.load(self._state)
    # self.load(self._exec)
    # self.load(self._term)
    logger.info("WORKER Fetching Input parameters/data for input:  %s", str(i))
    jobInput = self.fetch(i)      # TODO: Manage job Input w/multiple input items, for now just pass it

    # In case this worker spun up a service, ensure it is stopped locally
    if self.localcatalogserver and not self.catalogPersistanceState:
      self.catalog.stop()
      self.localcatalogserver = None

    logger.debug("Starting WORKER  Execution  ---->>")
    result = self.execute(jobInput)
    logger.debug("<<----  WORKER Execution Complete")
    # Ensure returned results are a list
    if type(result) != list:
      result = [result]

    #  CHECK CATALOG STATUS for saving data
    self.localcatalogserver = self.catalog.conn()

    logger.info("WORKER Saving Thread State to catalog:")
    self.save(self._state)

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
    
    self.catalog.append(self.downstream, result)

    if self.localcatalogserver and self.catalogPersistanceState:
      logger.debug("This Worker is running the catalog. Waiting on local service to terminate...")
      self.localcatalogserver.join()
      self.localcatalogserver = None

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


    if args.init:
      # TODO:  Abstract The Catalog/Archive to enable mutliple
      #   and dynamic Storage type    
      #   FOR NOW:  Use a Redis Implmentation
      catalog = redisCatalog.dataStore('catalog')
      archive = redisCatalog.dataStore(**archiveConfig)

      initialize(catalog, archive)
      sys.exit(0)

    # self.setCatalog(catalog)

    if args.workinput:
      logger.debug("Running worker.")
      self.worker(args.workinput)
    else:
      self.manager()