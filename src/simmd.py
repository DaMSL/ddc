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
    self.setStream('jcqueue', 'rawoutput')
    self.addImmut('simSplitParam')
    self.addImmut('simDelay')
    self.addImmut('terminate')
    self.addImmut('sim_conf_template')

    self.addAppend('launch')
    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('namd')
    self.modules.add('redis')
    # self.slurmParams['share'] = None


  def term(self):
    # jccomplete = self.data['JCComplete']
    # jctotal = self.data['JCTotal']
    if 'terminate' in self.data and self.data['terminate'] is not None:
      return self.data['terminate'].lower() == 'converged'
    else:
      return False

  def split(self):

    if len(self.data['jcqueue']) == 0:
      return [], None
    split = int(self.data['simSplitParam'])
    immed = self.data['jcqueue'][:split]
    return immed, split

  def configElasPolicy(self):
    self.delay = self.data['simDelay']


  def fetch(self, i):
    # Load parameters from catalog

    key = wrapKey('jc', i)
    params = self.catalog.hgetall(key)
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    self.addMut(key, params)

    # Increment launch count
    # A, B = eval(params['targetBin'])
    # logging.debug("Increment Launch count for %s", params['targetBin'])
    # self.data['launch'][A][B] += 1

    return params


  def execute(self, job):

    # Prepare & source to config file
    with open(self.data['sim_conf_template'], 'r') as template:
      source = template.read()

    # Prepare working directory, input/output files
    conFile = os.path.join(job['workdir'], job['name'] + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file

    with open(conFile, 'w') as config:
      config.write(source % job)
      logging.info(" Config written to: " + conFile)

    cmd = 'namd2 %s > %s' % (conFile, logFile)
    logging.debug("Executing Simulation:\n   %s\n", cmd)

    stdout = executecmd(cmd)

    logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
    logging.info(stdout)
    
    key = wrapKey('jc', job['name'])
    self.data[key]['dcd'] = dcdFile
    # self.catalog.hset()

    return [job['name']]



if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
