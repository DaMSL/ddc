import time
import math
import sys
import os
import argparse
import common
import redisCatalog

from collections import namedtuple

from macrothread import macrothread



# FOR PI DEMO ONLY
import picalc as pi

import logging, logging.handlers
logger = common.setLogger()


  # simState = dict(
  #     JCQueue = initParams,
  #     JCComplete = 0,
  #     JCTotal = len(initParams),
  #     simSplitParam =  2, 
  #     rawFileList =  [] )

  # # Analysis Data
  # anlState = dict(
  #     processed =  0,
  #     anlSplitParam =  2,
  #     indexPi_in =  0,
  #     indexPi_tot =  0 )

  # # Control Data
  # ctlState = dict(
  #     omega =  [0, 0, 0, 0],
  #     omegaMask = [False, False, False, False],
  #     piEst =  0.,
  #     converge =  0.)



#  TODO:  Move this to abstract and est. 'dispather' method
def initialize(catalog, threadlist, schema):

  logger.debug("Getting the registry...")
  catalog.conn()
  logger.debug(" Registry found on %s" % registry.host)

  catalog.clear()
  # Job ID Management
  ids = {'id_' + name : 0 for name in threadlist.keys()}

  # Simulation Data    
  initParams = [0, 1, 2] * 2

  catalog.save(ids)
  catalog.save(schema)

  logger.debug("Initialization complete\n")


# Simulation Thread
class simThread(macrothread):
  def __init__(self, fname, schema):
    macrothread.__init__(self, schema, fname, 'sim')

    #  Data for Simulation MacroThread -- organized by state
    self.setInput('JCQueue')
    self.setTerm('JCComplete', 'JCTotal')
    self.setExec('rawFileList')
    self.setSplit('simSplitParam')

  def term(self):
    jccomplete = self.data['JCComplete']
    jctotal = self.data['JCTotal']
    return (jccomplete == jctotal)

  def split(self):
    split = int(self.data['simSplitParam'])

    # Note how data is sliced within the data base
    #   User is required to "save back" the deferred input data, if nec'y
    catalog = self.getCatalog()
    immed = catalog.slice('JCQueue', split)
    return immed

  def execute(self, i):
    # jobnum = str(i[0])
    param  = pi.jc[int(i[0])]

    uid = common.getUID()

    target = 'pi/rawout.%s' % uid
    pi.piSim(target, param)

    self.data['rawFileList'].append(target)
    self.data['JCComplete'] = int(self.data['JCComplete']) + 1

    return [target]


# Analysis Thread
class anlThread(macrothread):
  def __init__(self, fname, schema):
    macrothread.__init__(self, schema, fname, 'anl')

    #  Data for Analysis MacroThread -- organized by state
    self.setInput('rawFileList')
    self.setTerm('JCComplete', 'processed')
    self.setExec('indexPi_in', 'indexPi_tot')
    self.setSplit('simSplitParam')


  def term(self):
    return False  # For now

  def split(self):
    split = self.data['anlSplitParam']
    immed = self.catalog.slice('rawFileList', split)
    return immed

  def execute(self, i):
    result = []
    for elm in i:
      inside, num = pi.piAnl(elm)
      self.data['indexPi_in'] += inside
      self.data['indexPi_tot'] += num
      self.data['processed'] += 1
      piVal = pi.piEst(self.data['indexPi_in'], self.data['indexPi_tot'])

      # TODO: index this analysis pi est for weighted calculation <--- D.H.
      #    for now, just return it to the worker
      result.append(piVal)
    return result


# Control Thread
class ctlThread(macrothread):
  def __init__(self, fname, schema):
    macrothread.__init__(self, schema, fname, 'ctl')

    self.setInput('omega')
    self.setTerm('converge')
    self.setExec('indexPi_in', 'indexPi_tot', 'piEst', 'JCQueue')
    self.setSplit('omegaMask')

    self.accuracyGoal = 0.999999999999

  def term(self):
    convergence = self.data['converge']
    return (convergence > self.accuracyGoal)

  def split(self):
    result = []
    omega = self.data['omega']
    for i, mask in enumerate(self.data['omegaMask']):
      if mask:
        result.append(omega[i])

    self.data['omegaMask'] = [False] * len(self.data['omegaMask'])
    return result

  def execute(self, i):
    ptIn  = self.data['indexPi_in']
    ptTot = self.data['indexPi_tot']

    estimate = pi.piEst(ptIn, ptTot)
    accuracy = 1. - abs((math.pi - estimate)) / math.pi

    logger.debug('  PI esimated at %f.  Accuracy of %f' % (estimate, accuracy))

    self.data['piEst'] = estimate
    self.data['converge'] = accuracy

    # Back Projection
    # Grab next of JC's to run (TODO: how many of each?????)
    # For now just re-run 3  demo JC params
    for i in range(3):
      self.data['JCQueue'].append(i)

    return accuracy


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('name', metavar='N', type=str, nargs='?', help='Name of macrothread [sim, anl, ctl]')
  parser.add_argument('-m', '--manager', action='store_true')
  parser.add_argument('-w', '--workinput', nargs='+')
  parser.add_argument('-i', '--init', action='store_true')
  args = parser.parse_args()

  #  USER DEFINED THReAD AND DATA -- DDL/SCHEMA
  initParams = [0,1,2] * 5
  schema = dict(  
        JCQueue = initParams,
        JCComplete = 0,
        JCTotal = len(initParams),
        simSplitParam =  2, 
        rawFileList =  [], 
        processed =  0,
        anlSplitParam =  2,
        indexPi_in =  0,
        indexPi_tot =  0,
        omega =  [0, 0, 0, 0],
        omegaMask = [False, False, False, False],
        piEst =  0.,
        converge =  0.)

  threads = {'sim': simThread(__file__, schema), 
             'anl': anlThread(__file__, schema),
             'ctl': ctlThread(__file__, schema)}

  # Determine type of registry to use
  registry = redisCatalog.dataStore('redis.lock')  

  if args.init:
    initialize(registry, threads, schema)
    logger.info("Initialization Complete. Exiting")
    sys.exit(0)

    # Make DDC app class to hide __main__ details; 
    #  e.g. add macrothread.... ref: front end for gui app
    #  pick registry
    #  add args as needed

  # TODO: common registry for threads
  # Implementation options:  Separate files for each macrothread OR
  #    dispatch macrothread via command line arg
  mt = threads[args.name]
  mt.setCatalog(registry)

  # mt.setCatalog(registry)

  if args.manager:
    mt.manager()
  elif args.workinput:
    mt.worker(args.workinput)

