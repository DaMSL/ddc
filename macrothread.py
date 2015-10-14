import time
import sys
import os
import subprocess as proc
import argparse
import common
import catalog
from ddcslurm import slurm

import logging, logging.handlers
logger = common.setLogger()

# TODO: Move common from class to module
class common:
  catFile = "cat.log"


class macrothread(object):
  def __init__(self, name):
    self._input = 0
    self._term  = 0
    self._exec  = 0
    self._split = 0
    self.catalog = 0
    self.name = name
    self.upStream = None
    self.downStream = None
  # def __init__(self, inp, term, user, split):
  #   self._input = inp
  #   self._term  = term
  #   self._exec  = user
  #   self._split = split

  def term(self):
    pass

  def split(self):
    print ("passing")

  def execute(self, item):
    pass

  def fetch(self, i):
    pass


  def manager(self, catalog):
    logger.debug("MANAGER for %s", self.name)

    # TODO: Catalog Service Check here
    if self.term(catalog):
      sys.exit(0)
    immed = self.split(catalog)
    # TODO: save back state, if nec'y

    #registry = catalog.retrieve(common.catFile)

    for i in immed:
      # TODO: Enum or dev. k-v indexing system
      # catalog.notify(i, 'pending')
    #  job = pyslurm.reservation
    #  job.create(JOB_INFO_DICT_HERE)
      # print ("SCHEDULING:  ", i)
      slurm.schedule("python3 macrothread.py -w -n %s -i %s" % (self.name, i))

  def worker(self, catalog, i):
    print ("FETCHING: ", i)
    jobInput = self.fetch(i)

    print ("EXECUTING: ", jobInput)

    #  catalog.notify(i, "active")
    data = self.execute(catalog, jobInput)
    print ("RESULTS: ", data)

    #  catalog.notify(i, "complete")
    for d in data:
    #  catalog.notify(d, "ready")
          print ("  data: ", d)  

class simThread(macrothread):
  def __init__(self):
    macrothread.__init__(self, 'sim')
    self.downStream = 'anl'

  def initialize(self, catalog, initParams):
    logger.debug("INITIALIZING:")
    catalog.register('JCQueue', initParams)
    catalog.register('JCComplete', 0)
    catalog.register('JCTotal', len(initParams))
    catalog.register('simSplitParam', 2)        
    catalog.register('rawFileList', [])
    logger.debug("Initialization complete\n")

  def term(self, catalog):
    logger.debug("Checking Term")
    jccomplete = catalog.load('JCComplete')
    jctotal    = catalog.load('JCTotal')
    return (jccomplete == jctotal)

  def split(self, catalog):
    print ("splitting....")
    split = int(catalog.load('simSplitParam'))
    jcqueue = catalog.load('JCQueue')
    logger.debug(" JCQ=%s", str(jcqueue))
    immed = jcqueue[:split]
    defer = jcqueue[split:]
    catalog.save('JCQueue', defer)
    return immed

  def execute(self, catalog, i):
#    target = exec(DO_SIM)
    data = int(i)
    target = 'rawout.%d' % data
    proc.call('python3 /ring/ddc/test_sim.py -f %s -i %d' % (target, data), shell=True)
    catalog.append('rawFileList', target)
    catalog.incr('JCComplete')
    return [target]

  def fetch(self, i):
    return i

# TODO: IMplement Analysis Thread
class anlThread(macrothread):
  def __init__(self):
    macrothread.__init__(self, 'anl')
    self.upStream = 'sim'

  def initialize(self, catalog, initParams):
    catalog.register('processed', 0)
    catalog.register('anlSplitParam', 1)        
    catalog.register('outputFile', [])

  def term(self, catalog):
    logger.debug("Checking Term")
    jccomplete = catalog.load('JCComplete')
    anlprocessed = catalog.load('processed')
    return (jccomplete == anlprocessed)

  def split(self, catalog):
    print ("splitting....")
    split = int(catalog.load('anlSplitParam'))
    jcqueue = catalog.load('rawFileList')
    immed = jcqueue[:split]
    defer = jcqueue[split:]
    catalog.save('rawFileList', defer)
    return immed

  def execute(self, catalog, i):
    target = "sample.out"
    catalog.append('rawFileList', 'sample.out')
    catalog.incr('JCComplete')
    return [target]

  def fetch(self, i):
    return i

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--manager', action='store_true')
  parser.add_argument('-w', '--worker', action='store_true')
  parser.add_argument('-n', '--name')
  parser.add_argument('-i', '--input')
  args = parser.parse_args()

  # TODO: common registry for threads
  threads = {'sim': simThread(), 
             'anl': anlThread()}
  mt = threads[args.name]

  # TODO: Catalog check here for now. Det where to put it...
  registry = catalog.serverLess(args.name)
  if not registry.exists():
    params = [1, 2, 3, 4, 5, 6]
    registry.initialize()
    mt.initialize(registry, params)

  if args.manager:
    mt.manager(registry)
  elif args.worker:
    mt.worker(registry, args.input)

  # sim = simThread("sim")
  # sim.manager()
