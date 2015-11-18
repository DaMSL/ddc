import argparse
import os
import sys
import shutil

import redisCatalog
from common import *
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


#  TODO:  Move this to abstract and est. 'dispatcher' method




class simulationJob(macrothread):
  def __init__(self, schema, fname, jobnum = None):
    macrothread.__init__(self, schema, fname, 'sim')

    # State Data for Simulation MacroThread -- organized by state
    self.setInput('JCQueue')
    self.setTerm('JCComplete', 'JCTotal')
    self.setExec('dcdFileList')    # 'pendingjobs' <-- todo
    self.setSplit('simSplitParam')

    # Static Data common across all simulationJobs (for now) 
    self.psf = DEFAULT.PSF_FILE
    self.pdb = DEFAULT.PDB_FILE
    # self.forcefield = DEFAULT.FFIELD

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('namd')
    self.modules.add('redis')
    # self.slurmParams['share'] = None



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
    logging.debug("WORKER Input received: " + str(i))

    logging.info("Preparing Simulation: " + i)
    # Prepare 
    with open(DEFAULT.SIM_CONF_TEMPLATE, 'r') as template:
      source = template.read()

    # Load parameters from catalog & source to config file
    logging.debug("Pulling Params from: %s", self.catalog.host)
    inputs = self.catalog.hgetall(wrapKey('jc', i))
    params = {k.decode():v.decode() for k,v in inputs.items()}
    logging.debug(" Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    # Prepare working directory, input/output files
    conFile = os.path.join(params['workdir'], unwrapKey(i) + '.conf')
    logFile = conFile.replace('conf', 'log')      # log in same place as config file
    dcdFile = conFile.replace('conf', 'dcd')      # dcd in same place as config file

    with open(conFile, 'w') as config:
      config.write(source % params)
      logging.info(" Config written to: " + conFile)

    cmd = 'namd2 %s > %s' % (conFile, logFile)
    logging.debug("Executing Simulation:\n   %s\n", cmd)

    stdout = executecmd(cmd)

    logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
    logging.info(stdout)
    
    # Update Local State
    # TODO:  Change this to append to pending jobs and add check in either
    #     this MT or downstream to check for completed pending jobs....
    #     requires key-val match between jcUID and the slurm jobid
    self.data['dcdFileList'].append(dcdFile)
    # self.data['JCComplete'] = int(self.data['JCComplete']) + 1

    return [dcdFile]





if __name__ == '__main__':
  mt = simulationJob(schema, __file__)
  mt.run()
