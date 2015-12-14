import argparse
import os
import sys
import shutil

import redisCatalog
from common import *
from macrothread import macrothread
from slurm import slurm
from kvadt import kv2DArray

# import logging
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)



class simulationJob(macrothread):
  def __init__(self, fname, jobnum = None):
    macrothread.__init__(self, fname, 'sim')

    # State Data for Simulation MacroThread -- organized by state
    self.setStream('JCQueue', 'rawouput')
    self.setState('JCComplete', 'JCTotal', 'simSplitParam', 'simDelay', 'terminate')

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('namd')
    self.modules.add('redis')
    # self.slurmParams['share'] = None

    self.addImmut('sim_conf_template')

  def term(self):
    # jccomplete = self.data['JCComplete']
    # jctotal = self.data['JCTotal']
    if 'terminate' in self.data and self.data['terminate'] is not None:
      return self.data['terminate'].lower() == 'converged'
    else:
      return False

  def split(self):
    split = int(self.data['simSplitParam'])
    immed = self.data['JCQueue'][:split]
    return immed, split

  def configElasPolicy(self):
    self.delay = self.data['simDelay']


  def fetch(self, i):
    # Load parameters from catalog

    key = wrapKey('jc', i)
    inputs = self.catalog.hgetall(key)
    params = {k.decode():v.decode() for k,v in inputs.items()}
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    self.addToState(key, params)

    # Increment launch count
    A, B = eval(params['targetBin'])
    launch = kv2DArray(self.catalog, 'launch')
    launch.incr(A, B)

    return params


  def execute(self, job):

    # Prepare & source to config file
    with open(self.data['sim_conf_template'], 'r') as template:
      source = template.read()

    # Prepare working directory, input/output files
    conFile = os.path.join(self.data[job]['workdir'], unwrapKey(self.data[job]['name']) + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file

    with open(conFile, 'w') as config:
      config.write(source % self.data[job])
      logging.info(" Config written to: " + conFile)

    cmd = 'namd2 %s > %s' % (conFile, logFile)
    logging.debug("Executing Simulation:\n   %s\n", cmd)

    stdout = executecmd(cmd)

    logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
    logging.info(stdout)
    
    self.data[job]['dcd'] = dcdFile

    self.catalog.hset()

    return [job]



if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
