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
    macrothread.__init__(self, schema, fname, 'simmd')

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
    self.cpu = DEFAULT.CPU_PER_NODE
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.extend(['namd', 'redis'])



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

    # Prepare 
    with open(DEFAULT.SIM_CONF_TEMPLATE, 'r') as template:
      source = template.read()
      logging.info("SOURCE LOADED:")

    # TODO: Better Job ID Mgmt
    # uid = common.getUID()
    jobnum = getJC_UID(i)


    # Load parameters from catalog & source to config file
    inputs = self.catalog.hgetall(getJC_Key(i))
    params = {k.decode():v.decode() for k,v in inputs.items()}
    logging.debug("Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    # Prepare working directory, input/output files
    workdir = os.path.join(DEFAULT.JOB_DIR, str(jobnum))
    conFile = os.path.join(workdir, str(jobnum) + '.conf')
    logFile = os.path.join(workdir, str(jobnum) + '.log')
    dcdFile = os.path.join(workdir, str(jobnum) + '.dcd')

    with open(conFile, 'w') as config:
      config.write(source % params)
      logging.info("Config written to: " + conFile)

    # self.slurmParams['partition'] = 'parallel'

    # Schedule Simulation from within execute function. This will be unsupervised
    stdout = slurm.sbatch(jobid=str(jobnum),
      workdir = workdir, 
      options = self.slurmParams,
      modules = self.modules,
      cmd = 'charmrun +p%d namd2 %s > %s' % (DEFAULT.CPU_PER_NODE, conFile, logFile))

    logging.info("SIMULATION Submitted! STDOUT/ERR Follows:")
    logging.info(stdout)
    
    # Update Local State
    # TODO:  Change this to append to pending jobs and add check in either
    #     this MT or downstream to check for completed pending jobs....
    #     requires key-val match between jcUID and the slurm jobid
    self.data['dcdFileList'].append(dcdFile)
    # self.data['JCComplete'] = int(self.data['JCComplete']) + 1

    return [dcdFile]





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--workinput')
  parser.add_argument('-i', '--init', action='store_true')
  args = parser.parse_args()

  catalog = redisCatalog.dataStore('catalog')
  archive = redisCatalog.dataStore(**archiveConfig)


  if args.init:
    initialize(catalog, archive)
    sys.exit(0)

    # Make DDC app class to hide __main__ details; 
    #  e.g. add macrothread.... ref: front end for gui app
    #  pick registry
    #  add args as needed

  # TODO: common registry for threads
  # Implementation options:  Separate files for each macrothread OR
  #    dispatch macrothread via command line arg
  mt = simulationJob(schema, __file__)
  mt.setCatalog(catalog)

  # mt.setCatalog(registry)

  if args.workinput:
    mt.worker(args.workinput)
  else:
    mt.manager(fork=True)
