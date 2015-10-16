import time
import math
import sys
import os
import subprocess as proc
import argparse
import common
import catalog
from ddcslurm import slurm

# FOR PI DEMO ONLY
import picalc as pi

import logging, logging.handlers
logger = common.setLogger()


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

  def initialize(self, catalog):
    logger.debug("INITIALIZING:")
    catalog.register('INIT-' + self.name, True)
    catalog.register('id_' + self.name, 0)


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
      logger.info('TERMINATION condition for ' + self.name)
      sys.exit(0)
    immed = self.split(catalog)

    # TODO:  For now using incrementing job id counters (det if this is nec'y)
    jobid = int(catalog.load('id_' + self.name))

    # No Jobs to run.... Delay and then rerun later
    if len(immed) == 0:
      delay = 15
      logger.debug("%s-MANAGER: No Available input data. Delaying %d seconds and rerunning...." % (self.name, delay))
      slurm.schedule('%04d' % jobid, "python3 macrothread.py -m -n %s" % self.name, delay=delay, name=self.name + '-M')
      jobid += 1

    else:
      for i in immed:
        #TODO: JobID Management
        logger.debug("%s: scheduling worker, input=%s", self.name, i)
        slurm.schedule('%04d' % jobid, "python3 macrothread.py -w -n %s -i %s" % (self.name, i), name=self.name + '-W')
        jobid += 1

      # METHOD 1.  Schedule self after scheduling ALL workers
      slurm.schedule('%04d' % jobid, "python3 macrothread.py -m -n %s" % self.name, name=self.name + '-M')
      jobid += 1

      # METHOD 2.  After.... use after:job_id[:jobid...] w/ #SBATCH --dependency=<dependency_list>


    catalog.save('id_' + self.name, jobid)

  def worker(self, catalog, i):
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

class simThread(macrothread):
  def __init__(self):
    macrothread.__init__(self, 'sim')
    self.downStream = 'anl'

  def initialize(self, catalog, initParams):
    super().initialize(catalog)
    catalog.register('JCQueue', initParams)
    catalog.register('JCComplete', 0)
    catalog.register('JCTotal', len(initParams))
    catalog.register('simSplitParam', 2)        
    catalog.register('rawFileList', [])
    catalog.register('rawFileList', [])
    logger.debug("Initialization complete\n")

  def term(self, catalog):
    logger.debug("Checking Term")
    jccomplete = catalog.load('JCComplete')
    jctotal    = catalog.load('JCTotal')
    logger.debug("Check: %s vs %s", jccomplete, jctotal)
    return (jccomplete == jctotal)

  def split(self, catalog):
    print ("splitting....")
    split = int(catalog.load('simSplitParam'))
    immed = catalog.slice('JCQueue', split)
    # jcqueue = catalog.load('JCQueue')
    # logger.debug(" JCQ=%s", str(jcqueue))
    # immed = jcqueue[:split]
    # defer = jcqueue[split:]
    # catalog.save('JCQueue', defer)
    return immed

  def execute(self, catalog, i):
    # jobnum = str(i[0])
    param  = pi.jc[int(i[0])]

    uid = common.getUID()

    target = 'pi/rawout.%s' % uid
    pi.piSim(target, param)
    # TODO: JobID mgmt
    # slurm.schedule('0002', 'python3 /ring/ddc/test_sim.py -f %s -i %d' % (target, data))

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

  def initialize(self, catalog):
    super().initialize(catalog)
    catalog.register('processed', 0)
    catalog.register('anlSplitParam', 2)        
    # catalog.register('outputFile', [])
    catalog.register('indexPi_in', 0)
    catalog.register('indexPi_tot', 0)

  def term(self, catalog):
    logger.debug("Checking Term")
    jccomplete = catalog.load('JCComplete')
    anlprocessed = catalog.load('processed')
    #    return ((jccomplete == anlprocessed) and jccomplete != 0)
    return False  # For now


  def split(self, catalog):
    print ("splitting....")
    split = int(catalog.load('anlSplitParam'))
    immed = catalog.slice('rawFileList', split)
    # jcqueue = catalog.load('rawFileList')
    # immed = jcqueue[:split]
    # defer = jcqueue[split:]
    # catalog.save('rawFileList', defer)
    return immed

  def execute(self, catalog, i):
    for elm in i:
      inside, num = pi.piAnl(elm)
      ptInside = int(catalog.load('indexPi_in'))
      ptTot = int(catalog.load('indexPi_tot'))
      catalog.save('indexPi_in', ptInside + inside)
      catalog.save('indexPi_tot', ptTot + num)

  def fetch(self, i):
    return i


class ctlThread(macrothread):
  def __init__(self):
    macrothread.__init__(self, 'ctl')
    self.upStream = 'anl'
    self.accuracyGoal = 0.999999999999

  def initialize(self, catalog):
    super().initialize(catalog)
    catalog.register('omega', [0, 0, 0, 0])
    catalog.register('omegaMask', [False, False, False, False])
    catalog.register('piEst', 0.)
    catalog.register('converge', 0.)


            # catalog.register('outputFile', [])
    catalog.register('indexPi_in', 0)
    catalog.register('indexPi_tot', 0)
    catalog.register('indexPi_est', 0.)

  def term(self, catalog):
    logger.debug("Checking Term")
    convergence = catalog.load('converge')
    result = (convergence > self.accuracyGoal)
    return result

  def split(self, catalog):
    print ("splitting....")
    mask = catalog.load('omegaMask')
    omega = catalog.laod('omega')
    catalog.save('omegaMask', [False, False, False, False])
    return mask

  def execute(self, catalog, i):
    ptIn = catalog.load('indexPi_in')
    ptTot = catalog.load('indexPi_tot')

    estimate = pi.piEst(ptIn, ptTot)
    accuracy = 1. - abs((math.pi - estimate)) / math.pi

    logger.debug('  PI esimated at %f.  Accuracy of %f' % (estimate, accuracy))

    catalog.save('piEst', estimate)
    catalog.save('converge', accuracy)

    # Back Projection
    # Grab next of JC's to run (TODO: how many of each?????)
    # For now just re-run 4 demo JC params
    for i in range(3):
      catalog.append('JCQueue', i)

  def fetch(self, i):
    return i


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--manager', action='store_true')
  parser.add_argument('-w', '--worker', action='store_true')
  parser.add_argument('-n', '--name')
  parser.add_argument('-i', '--input', nargs='+')
  args = parser.parse_args()

  # TODO: common registry for threads
  threads = {'sim': simThread(), 
             'anl': anlThread(),
             'ctl': ctlThread()}
  mt = threads[args.name]

  # TODO: Catalog check here for now. Det where to put it...
  # registry = catalog.serverLess()
  registry = catalog.dataStore('redis.lock')

  if not registry.exists():
    logger.debug(" Initializing the registry.....")
    registry.start()
    # registry.initialize()

  init = registry.check('INIT-' + args.name)
  logger.debug("INIT Status for %s = %s" % (args.name, init))
  if not init:
    # Special case to inialize the system
    logger.debug(" Initializing the %s macrothread....." % args.name)
    if args.name == 'sim':
      mt.initialize(registry, [0, 1, 2])
    else:
      mt.initialize(registry)

  if args.manager:
    mt.manager(registry)
  elif args.worker:
    mt.worker(registry, args.input)

  # sim = simThread("sim")
  # sim.manager()
