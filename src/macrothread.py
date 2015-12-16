import time
import math
import sys
import os
import subprocess as proc
import argparse
import abc
import redis
import copy
# from retrying import retry

from common import * 
import catalog
from slurm import slurm

from collections import namedtuple

# For now: use redis catalog
import redisCatalog

# import logging, logging.handlers
# logger = setLogger("MT")
logger = logging.getLogger(__name__)


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
    DEFAULT.applyConfig(self.config)


    # System-wide meta-data TODO: (part loaded from file, part from catalog)
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
              'partition':DEFAULT.PARTITION, 
              'job-name':self.name,
              'workdir' : os.getcwd()}



  def addMut(self, key, value=None):
    self._mut.append(key)
    if value is not None:
      self.data[key] = value

  def addImmut(self, key, value=None):
    self._immut.append(key)

  def addAppend(self, key, value=None):
    self._append.append(key)
    self.origin[key] = None
    if value is not None:
      self.data[key] = value
      self.origin[key] = copy.deepcopy(value)


  # # TODO:  Eventually we may want multiple up/down streams
  def setStream(self, upstream, downstream):
    if upstream is None:
      logger.info("Upstream data `%s` not defined", upstream)
    self.upstream = upstream

    if downstream is None:
      logger.warning("Downstream data `%s` not defined in schema", downstream)
    self.downstream = downstream

  # def setState(self, *arg):

  #   for a in arg:
  #     self._state[a] = self.data[a]
  #   # self._state['id_' + self.name] = 0

  # def addToState(self, key, value):
  #   self._state[key] = value
  #   self.data[key] = value

  def setCatalog(self, catalog):
    self.catalog = catalog

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

  def configElasPolicy(self):
    """
    Set rescheduling / delay policies for the manager thread 
    """
    self.delay = 60



  def retry_redisConn(ex):
    return isinstance(ex, redis.ConnectionError)


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
      logging.debug("  %10s: %s ", k, str(v))

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
        logging.debug("Current Origin Contents:")
        for k, v in self.origin.items():
          print("   ",k,v)
        logging.debug("------")
        if type(self.data[key]) in [int, float, np.ndarray]:
          state[key] = self.data[key] - self.origin[key]
        elif isinstance(self.data, list):
          state[key] = [x for x in self.data[key] if x not in self.origin[key]]
        elif isinstance(self.data, dict):
          state[key] = {k:v for k.v in self.data[key].items if k not in self.origin[key].keys()}
        else:
          logging.error("CANNOT APPEND data `%s` of type of %s", key, str(type(self.data[key])))
    self.catalog.append(state)




  def manager(self, fork=False):

    logger.debug("\n==========================\n  MANAGER:  %s", self.name)

    # # Catalog Service Check here
    # if not self.catalog:
    #   self.catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
    # self.localcatalogserver = self.catalog.conn()

    # Check global termination:
    term_flag = self.data['terminate']
    if term_flag and term_flag.lower() in ['halt', 'stop', 'now']:
      logger.info('RECEIVED TERMINATION FLAG. Shutting down')
      sys.exit(0)

    # Load Data from Thread's State and Upstream thread
    if self.upstream:
      logging.debug("Loading upstream data: %s", self.upstream)
      self.load(self.upstream)

    for k,v in self.data.items():
      print("  mngrstate:  ", k, v, type(v))


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
    self.catalogPersistanceState = True
    if self.localcatalogserver and not self.catalogPersistanceState:
      self.catalog.stop()
      self.localcatalogserver = None

    #  TODO:  Det if manager should load entire input data set, make this abstract, or
    #     push into UDF portion
    immed, defer  = self.split()

    # # TODO:  JobID mgmt. For now using incrementing job id counters (det if this is nec'y)
    myid = 'id_%s' % self.name
    curid = self.catalog.get(myid)
    if curid is None:
      self.catalog.set(myid, 0)
      jobid = 0
    else:
      jobid = int(curid)
    self.catalog.incr(myid)

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = int(self.delay)
      logger.debug("MANAGER %s: No Available input data. Delaying %d seconds and rescheduling...." % (self.name, delay))
      self.slurmParams['begin'] = 'now+%d' % delay

    # Dispatch Workers
    else:
      workernum = 1
      for i in immed:
        logger.debug("%s: scheduling worker, input=%s", self.name, i)
        self.slurmParams['job-name'] = "%sW%d-%d" % (self.name[0], jobid, workernum)
        slurm.sbatch(taskid=self.slurmParams['job-name'],
            options = self.slurmParams,
            modules = self.modules,
            cmd = "python3 %s -c %s -w %s" % (self.fname, self.config, str(i)))
        workernum += 1

    # Reschedule Next Manager:
    # METHOD 1.  Automatic. Schedule self after scheduling ALL workers
    #      FOR NOW, back off delay  (for debug/demo/testing)
    # TODO: Use Elas Policy to control manager rescheduling
    delay = self.delay  
    self.slurmParams['begin'] = 'now+%d' % delay

    self.slurmParams['job-name'] = "%sM-%04d" % (self.name[0], jobid)
    self.slurmParams['cpus-per-task'] = 1
    slurm.sbatch(taskid =self.slurmParams['job-name'],
              options   = self.slurmParams,
              modules   = self.modules,
              cmd = "python3 %s -c %s" % (self.fname, self.config))

    # METHOD 2.  Trigger Based
    #   use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>


    # Ensure the catalog is available. If not, start it for persistance and check
    # if the thread is running before exit
    self.localcatalogserver = self.catalog.conn()
    self.catalogPersistanceState = True


    # Consume upstream input data (should this happen earlier or now?)
    #  TODO: Should we just make this only 1 list allowed for upstream data?
    if isinstance(defer, list) and len(defer) > 0:
      self.catalog.removeItems(self.upstream, defer)
    elif defer is not None:
      self.catalog.slice(self.upstream, defer)

    # Other interal thread state is saved back to catalog
    #  TODO: For now the manager ONLY updates the job ID

    self.save(self._mut)
    
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

    if self.localcatalogserver and self.catalogPersistanceState:
      logger.debug("This Worker is running the catalog. Waiting on local service to terminate...")
      self.localcatalogserver.join()
      self.localcatalogserver = None

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

    if args.debug:
      logging.debug("DEBUGGING: %s", self.name)


    if args.init:
      # TODO:  Abstract The Catalog/Archive to enable mutliple
      #   and dynamic Storage type    
      #   FOR NOW:  Use a Redis Implmentation
      catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
      archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)

      initialize(catalog, archive)
      sys.exit(0)

    # Both Worker & Manager need catalog to run; load it here and import schema

    if not self.catalog:
      self.catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
    self.localcatalogserver = self.catalog.conn()

    self.catalog.loadSchema()   # Should this be called from within the catalog module?

    # Load data from Catalog
    logger.info("Loading Thread State for from catalog:")
    self.load(self._mut, self._immut, self._append)

    if args.workinput:
      logger.debug("Running worker.")
      self.worker(args.workinput)
    else:
      self.manager()