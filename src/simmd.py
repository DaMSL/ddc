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


PARALLELISM = 24

class simulationJob(macrothread):
  """Macrothread to run MD simuation. Each worker runs one simulation
  using the provided input parameters.
    Input -> job candidate key in the data store 
    Execute -> creates the config file and calls namd to run the simulation. 
  """
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
    self.modules.add('namd/2.10')
    self.modules.add('redis')
    # self.slurmParams['share'] = None

    self.slurmParams['cpus-per-task'] = PARALLELISM

  def term(self):
    return False

  def split(self):

    if len(self.data['jcqueue']) == 0:
      return [], None
    split = int(self.data['simSplitParam'])
    immed = self.data['jcqueue'][:split]
    return immed, split

  def configElasPolicy(self):
    self.delay = self.data['simDelay']

  # def preparejob(self, job):
  #   logging.debug('Simlation is preparing job %s', job)
  #   key = wrapKey('jc', i)
  #   params = self.catalog.hgetall(key)
  #   logging.debug(" Job Candidate Params:")
  #   for k, v in params.items():
  #     logging.debug("    %s: %s" % (k, v))
    # if 'parallel' in job:
    #   numnodes = job['parallel']
    #   total_tasks = numnodes * 24       # Total # cpu per node should be detectable
    #   self.modules.add('namd/2.10-mpi')
    #   self.slurmParams['partition'] = 'parallel'
    #   self.slurmParams['ntasks-per-node'] = 24
    #   self.slurmParams['nodes'] = numnodes
    #   del self.slurmParams['cpus-per-task']


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

    ramdisk = '/dev/shm/out/'
    if not os.path.exists(ramdisk):
      os.mkdir(ramdisk)
    job['outputloc'] = ramdisk

    with open(conFile, 'w') as config:
      config.write(source % job)
      logging.info(" Config written to: " + conFile)

    # # Run simulation in parallel
    # if 'parallel' in job:
    #   numnodes = job['parallel']
    #   total_tasks = numnodes * 24
    #   cmd = 'mpiexec -n %d namd2 %s > %s'  % (total_tasks, conFile, logFile)

    # # Run simulation single threaded
    # else:
    #   cmd = 'namd2 %s > %s' % (conFile, logFile)

    cmd = 'namd2 +p%d %s > %s' % (PARALLELISM, conFile, logFile)

    logging.debug("Executing Simulation:\n   %s\n", cmd)

    bench = microbench()
    bench.start()
    stdout = executecmd(cmd)
    logging.info("SIMULATION Complete! STDOUT/ERR Follows:")
    bench.mark('SimExec:%s' % job['name'])
    shm_contents = os.listdir('/dev/shm/out')
    logging.debug('Ramdisk contents (should have files) : %s', str(shm_contents))
    shutil.copy(ramdisk + job['name'] + '.dcd', job['workdir'])
    logging.info("Copy Complete to Lustre.")
    bench.mark('CopyLustre:%s' % job['name'])
    shutil.rmtree(ramdisk)
    shm_contents = os.listdir('/dev/shm')
    logging.debug('Ramdisk contents (should be empty) : %s', str(shm_contents))
    bench.show()

    logging.info("STDOUT/ERR Follows:")
    logging.info(stdout)
    
    key = wrapKey('jc', job['name'])
    self.data[key]['dcd'] = dcdFile
    # self.catalog.hset()

    return [job['name']]



if __name__ == '__main__':
  mt = simulationJob(__file__)
  mt.run()
