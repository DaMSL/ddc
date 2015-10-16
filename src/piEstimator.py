import time
import math
import sys
import os
import argparse
import common
import catalog

from macrothread import macrothread

# FOR PI DEMO ONLY
import picalc as pi

import logging, logging.handlers
logger = common.setLogger()

def initialize(catalog):

  catalog.clear()
  # All
  for name in ['sim', 'anl', 'ctl']:
    catalog.register('id_' + name, 0)

  # Simulation Data    
  initParams = [0, 1, 2] * 10
  catalog.register('JCQueue', initParams)
  catalog.register('JCComplete', 0)
  catalog.register('JCTotal', len(initParams))
  catalog.register('simSplitParam', 2)        
  catalog.register('rawFileList', [])

  # Analysis Data
  catalog.register('processed', 0)
  catalog.register('anlSplitParam', 2)        
  catalog.register('indexPi_in', 0)
  catalog.register('indexPi_tot', 0)

  # Control Data
  catalog.register('omega', [0, 0, 0, 0])
  catalog.register('omegaMask', [False, False, False, False])
  catalog.register('piEst', 0.)
  catalog.register('converge', 0.)

  logger.debug("Initialization complete\n")


# Simulation Thread
class simThread(macrothread):
  def __init__(self, fname):
    macrothread.__init__(self, fname, 'sim')

  def term(self, catalog):
    jccomplete = catalog.load('JCComplete')
    jctotal    = catalog.load('JCTotal')
    return (jccomplete == jctotal)

  def split(self, catalog):
    split = int(catalog.load('simSplitParam'))
    immed = catalog.slice('JCQueue', split)
    return immed

  def execute(self, catalog, i):
    # jobnum = str(i[0])
    param  = pi.jc[int(i[0])]

    uid = common.getUID()

    target = 'pi/rawout.%s' % uid
    pi.piSim(target, param)

    catalog.append('rawFileList', target)
    catalog.incr('JCComplete')

    return [target]

  def fetch(self, i):
    return i

# Analysis Thread
class anlThread(macrothread):
  def __init__(self, fname):
    macrothread.__init__(self, fname, 'anl')

  def term(self, catalog):
    jccomplete = catalog.load('JCComplete')
    anlprocessed = catalog.load('processed')
    #    return ((jccomplete == anlprocessed) and jccomplete != 0)
    return False  # For now

  def split(self, catalog):
    split = int(catalog.load('anlSplitParam'))
    immed = catalog.slice('rawFileList', split)
    return immed

  def execute(self, catalog, i):
    result = []
    for elm in i:
      inside, num = pi.piAnl(elm)
      ptInside = int(catalog.load('indexPi_in'))
      ptTot = int(catalog.load('indexPi_tot'))
      catalog.save('indexPi_in', ptInside + inside)
      catalog.save('indexPi_tot', ptTot + num)
      catalog.incr('processed')
      piVal = pi.piEst(ptInside, ptTot)

      # TODO: index this analysis pi est for weighted calculation <--- D.H.
      #    for now, just return it to the worker
      result.append(piVal)
    return result

  def fetch(self, i):
    return i

# Control Thread
class ctlThread(macrothread):
  def __init__(self, fname):
    macrothread.__init__(self, fname, 'ctl')
    self.upStream = 'anl'
    self.accuracyGoal = 0.999999999999

  def term(self, catalog):
    logger.debug("Checking Term")
    convergence = float(catalog.load('converge'))
    result = (convergence > self.accuracyGoal)
    return result

  def split(self, catalog):
    mask = catalog.load('omegaMask')
    omega = catalog.load('omega')
    catalog.save('omegaMask', [False, False, False, False])
    return mask

  def execute(self, catalog, i):
    ptIn = int(catalog.load('indexPi_in'))
    ptTot = int(catalog.load('indexPi_tot'))

    estimate = pi.piEst(ptIn, ptTot)
    accuracy = 1. - abs((math.pi - estimate)) / math.pi

    logger.debug('  PI esimated at %f.  Accuracy of %f' % (estimate, accuracy))

    catalog.save('piEst', estimate)
    catalog.save('converge', accuracy)

    # Back Projection
    # Grab next of JC's to run (TODO: how many of each?????)
    # For now just re-run 3  demo JC params
    for i in range(3):
      catalog.append('JCQueue', i)

    return accuracy

  def fetch(self, i):
    return i


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('name', metavar='N', type=str, nargs='?', help='Name of macrothread [sim, anl, ctl]')
  parser.add_argument('-m', '--manager', action='store_true')
  parser.add_argument('-w', '--workinput', nargs='+')
  parser.add_argument('-i', '--init', action='store_true')
  args = parser.parse_args()

  # TODO: Catalog check here 
  # registry = catalog.serverLess()
  registry = catalog.dataStore('redis.lock')

  if args.init:
    initialize(registry)
    logger.info("Initialization Complete. Exiting")
    sys.exit(0)
  
    # Make DDC app class to hide __main__ details; 
    #  e.g. add macrothread.... ref: front end for gui app
    #  pick registry
    #  add args as needed

  # TODO: common registry for threads
  # Implementation options:  Separate files for each macrothread OR
  #    dispatch macrothread via command line arg
  threads = {'sim': simThread(__file__), 
             'anl': anlThread(__file__),
             'ctl': ctlThread(__file__)}
  mt = threads[args.name]

  if args.manager:
    mt.manager(registry)
  elif args.workinput:
    mt.worker(registry, args.workinput)

