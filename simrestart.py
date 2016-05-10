#!/usr/bin/env python

import argparse
import os
import sys
import time
import logging
from datetime import datetime as dt
import subprocess as proc

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

PARALLELISM = 24
RUNTIME = 1000000

def execute(mynum):

  conTemplate = '/home-1/bring4@jhu.edu/work/jc/serial/restart/serial.temp.conf'
  conFile = '/home-1/bring4@jhu.edu/work/jc/serial/restart/serial-%03d.conf' % mynum
  logFile = '/home-1/bring4@jhu.edu/work/jc/serial/restart/serial-%03d.log' % mynum

  # Prepare & source config file
  with open(conTemplate, 'r') as template:
    source = template.read()

  vals = {'last':mynum-1, 'this':mynum, 'runtime':RUNTIME}

  with open(conFile, 'w') as sysconfig:
    sysconfig.write(source % vals)
    logging.info("Config written to: " + conFile)

  cmd = 'namd2 +p%d %s > %s' % (PARALLELISM, conFile, logFile)
  logging.debug("Executing Simulation:\n   %s\n", cmd)

  start = dt.now()
  job = proc.call(cmd, shell=True)
  end   = dt.now()
  logging.info("SIMULATION Complete! STDOUT/ERR Follows:")

  # Internal stats
  sim_length = 1000000
  sim_realtime = (end-start).total_seconds()
  sim_run_ratio =  (sim_realtime/60) / (sim_length/1000000)
  logging.info('##SIM_TIME,%6.3f', sim_realtime)
  logging.info('##SIM_RATIO,%6.3f,(min-per-ns-sim)', sim_run_ratio)

  nextsim = mynum + 1
  logging.info("Scheduling Next Simulation # %d", nextsim)

  jobname = 'serial-%03d'%nextsim

  slurmParams = {'time':'12:00:0', 
              'nodes':1, 
              'cpus-per-task':24, 
              'partition':'shared,parallel', 
              'job-name': jobname,
              'workdir' : '/home-1/bring4@jhu.edu/ddc/'}  
  cmd = './simrestart.py --num=%d' % nextsim      

  inline = '#!/bin/bash -l\n\n#SBATCH\n'

  for k, v in slurmParams.items():
    inline += '#SBATCH --%s=%s\n' % (k, str(v))

  inline += '#SBATCH --output=/home-1/bring4@jhu.edu/work/log/serial/serial-%03d.out\n' % nextsim
  inline += 'module load namd/2.10\n'

  inline += 'echo ================================\n'
  inline += '\n%s\n' % cmd


  logging.debug("Inline SBATCH:------------->>")
  logging.debug(inline)
  logging.debug("<<-------  Batch Complete")

  # Launch job
  job = proc.Popen('sbatch <<EOF\n%sEOF' % inline,
     shell=True, stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
  stdout, stderr = job.communicate()

  logging.info("Slurm Batch Job Submitted. Output follows:")
  logging.info(stdout.decode())

  logging.info("Terminating Simulation Restart # %d", mynum)
      

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num')
  args = parser.parse_args()

  if not args.num:
    logging.info("No Number Provided")

  else:
    execute(int(args.num))
