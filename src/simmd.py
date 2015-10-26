import argparse
import os
import sys

import redisCatalog
from common import DEFAULT, executecmd
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


#  TODO:  Move this to abstract and est. 'dispather' method
def initialize(catalog, threadlist, schema):

  logging.debug("Getting the registry...")
  catalog.conn()
  logging.debug(" Registry found on %s" % registry.host)

  catalog.clear()
  # Job ID Management
  ids = {'id_' + name : 0 for name in threadlist.keys()}

  catalog.save(ids)
  catalog.save(schema)

  logging.debug("Initialization complete\n")





class simulationJob(macrothread):
  def __init__(self, schema, fname, jobnum = None):
    macrothread.__init__(self, schema, fname, 'simmd')

    # State Data for Simulation MacroThread -- organized by state
    self.setInput('JCQueue')
    self.setTerm('JCComplete', 'JCTotal')
    self.setExec('dcdFileList')
    self.setSplit('simSplitParam')

    # Static Data common across all simulationJobs (for now) 
    self.psf = os.path.join(DEFAULT.WORKDIR, DEFAULT.PSF_FILE)
    self.pdb = os.path.join(DEFAULT.WORKDIR, DEFAULT.PDB_FILE)  # ?? Is this generated from input
    self.forcefield = os.path.join(DEFAULT.WORKDIR, DEFAULT.FFIELD)

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
    with open('src/sim_template.conf', 'r') as template:
      source = template.read()
      logging.info("SOURCE LOADED:")

    # TODO: Better Job ID Mgmt
    # uid = common.getUID()
    jobnum = i

    # Load parameters from catalog & source to config file
    inputs = self.catalog.hgetall(i)
    params = {k.decode():v.decode() for k,v in inputs.items()}
    params['outname'] = i
    logging.debug("Job Candidate Params:")
    for k, v in params.items():
      logging.debug("    %s: %s" % (k, v))

    # Prepare working directory, input/output files
    workdir = os.path.join(DEFAULT.WORKDIR, str(jobnum))
    conFile = os.path.join(workdir, i + '.conf')
    logFile = os.path.join(workdir, i + '.log')
    dcdFile = os.path.join(workdir, i + '.dcd')

    if not os.path.exists(workdir):
      os.mkdir(workdir)

    with open(conFile, 'w') as config:
      config.write(source % params)
      logging.info("Config written to: " + conFile)

    # Run Simulation

    stdout = slurm.sbatch(jobid=str(jobnum),
      workdir = workdir, 
      options = self.slurmParams,
      modules = self.modules,
      cmd = 'charmrun +p%d namd2 %s > %s' % (DEFAULT.CPU_PER_NODE, conFile, logFile))

    logging.info("SIMULATION Submitted! STDOUT/ERR Follows:")
    logging.info(stdout)
    
    # Update Local State
    self.data['dcdFileList'].append(dcdFile)
    # self.data['JCComplete'] = int(self.data['JCComplete']) + 1

    return [dcdFile]





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', metavar='N', type=str, nargs='?', help='Name of macrothread [sim, anl, ctl]')
  parser.add_argument('-m', '--manager', action='store_true')
  parser.add_argument('-w', '--workinput')
  parser.add_argument('-i', '--init', action='store_true')
  args = parser.parse_args()

  #  USER DEFINED THReAD AND DATA -- DDL/SCHEMA

  sampleSimJobCandidate = dict(
    psf     = DEFAULT.PSF_FILE,
    pdb     = DEFAULT.PDB_FILE,
    forcefield = DEFAULT.FFIELD,
    runtime = 2000)

  initParams = {'simmd:0001':sampleSimJobCandidate}

  schema = dict(  
        JCQueue = list(initParams.keys()),
        JCComplete = 0,
        JCTotal = len(initParams),
        simSplitParam =  1, 
        dcdFileList =  [], 
        processed =  0,
        anlSplitParam =  1,
        omega =  [0, 0, 0, 0],
        omegaMask = [False, False, False, False],
        converge =  0.)

  threads = {'simmd': simulationJob(schema, __file__)}

             # 'anl': anlThread(__file__, schema),
             # 'ctl': ctlThread(__file__, schema)}

  # Determine type of registry to use
  registry = redisCatalog.dataStore('redis.lock')  


  if args.init:
    logging.debug("Loading Schema.....")
    initialize(registry, threads, schema)
    logging.debug("Loading initial parameters.....")
    registry.save(initParams)
    logging.info("Initialization Complete. Exiting")
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
    mt.manager(fork=True)
  elif args.workinput:
    mt.worker(args.workinput)