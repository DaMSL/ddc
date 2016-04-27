#!/usr/bin/env python

import time
import subprocess as proc
from collections import namedtuple
import logging
import argparse
import pytest
import pyslurm

from core.common import systemsettings

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.0.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

slurmJob = namedtuple('slurmJob', 'jobid, partition, name, user, state, time, time_limit, nodes, nodelist')

def chmodX(path):
    """Shell wrapper routine to set POSIX permission mode
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2 
    os.chmod(path, mode)



# USEFUL COMMANDS:

# pyslurm.slurm_get_end_time(uint32_t JobID=0)

#  'eligible_time': 1459546387,
#  'end_time': 1460842402,
#  'job_id': 5640778,
#  'job_state': b'RUNNING',
#  'name': "b'serial'",
#  'run_time': 416671,
#  'run_time_str': '4-19:44:31',
#  'start_time': 1459546388,
#  'submit_time': 1459546387,
#  'time_limit': 21600,  IN MINUTES



class slurm:
  """Slurm interface for peforming simple SLURM operations. Only includes the ability
  to schedule a job and to retrieve job information (for now)
  For simplicity, SLURM interaction is performed via shell-based actions using
  a subprocess Popen object and all current methods are static class methods
  """

  @classmethod
  def info(cls):
    out = proc.call("sinfo")  
    return out

  @classmethod
  def run(cls, cmd):
    out = proc.call('srun ' + cmd, shell=True)
    return out


  @classmethod
  def jobinfo(cls, jobid):
    out = proc.check_output('scontrol show jobid %d' % jobid, shell=True)
    info = {}
    for i in out.decode().split('\n'):
      for j in i.split():
        elm = j.split('=')
        if len(elm) == 2:
          k, v = elm
          info[k] = v
    return info

  @classmethod
  def jobexecinfo(cls, jobid):
    if isinstance(jobid, list):
      job = ','.join(['%d.batch'%i for i in jobid])
    else:
      job = str(jobid)
    print('JOBLIST=', job)
    out = proc.check_output("sacct -n -P --delimiter=',' -j %s --format=jobid,exitcode,MaxVMSizeNode,elapsed,cputime,submit,eligible" \
     % job, shell=True)
    joblist = []
    for line in out.decode('utf-8').strip().split('\n'):
      j, ex, n, t, c, sub, st = line.split(',')
      joblist.append(dict(jobid=j[:-6], exitcode=ex, node=n, time=t, cpu=c, submit=sub, start=st))
    return joblist



  @classmethod
  def getJob(cls, jobid):
    job = pyslurm.job().get()[jobid]
    p = {}
    p['time']     = job['time_limit_str']
    p['nodes']    = job['num_nodes']
    p['cpus-per-task'] = job['cpus_per_task']
    p['partition']= job['partition'].decode()
    p['job-name'] = job['name'].decode()
    p['workdir']  = job['work_dir'].decode()
    return p

  @classmethod
  def time_left(cls, jobid):
    """ Return Time left IN SECONDS
    """
    job = pyslurm.job().get()[jobid]
    ts = dt.now().timestamp()
    return (job['end_time'] - ts)


  @classmethod
  def getJobs(**kwargs):
    cmd = ['squeue','-h', '-l']
    if 'state' in kwargs: 
      cmd.append('--state=' + kwargs.get('state'))
    joblist = filter(None, proc.check_output(cmd).decode().split('\n'))
    return [slurmJob(*(job.split())) for job in joblist]

  @classmethod
  def sbatch(cls, taskid, options, modules, cmd, environ={}):

    logging.info("Slurm sbatch Job submitted for " + str(taskid))

    config = systemsettings()
    inline = '#!/bin/bash -l\n\n#SBATCH\n'

    for k, v in options.items():
      if v == None:
        inline += '#SBATCH --%s\n' % k
      else:  
        inline += '#SBATCH --%s=%s\n' % (k, str(v))

    joboutput = "%s/%s.out" % (config.LOGDIR, str(taskid))
    inline += '#SBATCH --output=%s\n' % joboutput

    for mod in modules:
      inline += 'module load %s\n' % mod

    environ['JOB_NAME'] = options['job-name']
    for k, v in environ.items():
      inline += 'export %s=%s\n' % (k, v) 
    inline += 'echo ================================\n'
    inline += 'echo JOB NAME:  %s\n' % options['job-name']
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
    return stdout.decode()

  @classmethod
  def schedule(cls, jobid, cmd, **kwargs):

    name = kwargs.get('name', '')

    batchfile = 'sh/' + name + jobid + '.sh'
    with open(batchfile, 'w') as script:
      script.write('#!/bin/bash\n\n')

      if 'delay' in kwargs:
        script.write("#SBATCH --begin=now+%d\n" % kwargs.get('delay'))

      if 'afterstart' in kwargs:
        script.write("#SBATCH --dependency=after:%s\n" % str(kwargs.get('afterstart')))

      if 'singleton' in kwargs and kwargs.get('singleton'):
        script.write("#SBATCH --dependency=singleton\n")


      script.write("#SBATCH --job-name=%s-%s\n" % (name, jobid))
      script.write("#SBATCH --output=out/%s-%%j.out\n\n" % name)
     
      # TODO: Other batch scripting goes here

      script.write('srun ' + cmd + '\n\n')
    chmodX(batchfile)

    out = proc.call(['sbatch', batchfile])
    return out

  @classmethod
  def parallel(cls, jobid, cmd, numnodes, **kwargs):
    name = kwargs.get('name', '')

    batchfile = 'sh/' + name + jobid + '.sh'
    with open(batchfile, 'w') as script:
      script.write('#!/bin/bash\n\n')

      if 'delay' in kwargs:
        script.write("#SBATCH --begin=now+%d\n" % kwargs.get('delay'))

      script.write("#SBATCH --job-name=%s-%s\n" % (name, jobid))
      script.write("#SBATCH --output=out/%s-%%j.out\n" % name)
      script.write("#SBATCH --partition=parallel\n")
      script.write("#SBATCH --nodes=%d\n" % numnodes)
      script.write("#SBATCH --ntasks-per-node=1\n\n")
     

      script.write('srun ' + cmd + '\n\n')
    chmodX(batchfile)

    out = proc.check_output(['sbatch', batchfile]).decode()
    return out


def test_slurm():
  print ("Testing slurm routines\n")
  result = slurm.info()
  assert True == True, 'Testing'
  print (result)
  print ()
  # result = slurm.schedule('0001', "/ring/ddc/hello.sh")
  # print (result)

  # print ()
  # result = slurm.parallel('0002', "memcached -vvv -u root", 2)
  # print (result)

  print ()
  print ("Currently running jobs:")
  for job in slurm.getJobs():
    print (job.name, job.state, job.nodes, ': ', job.nodelist)
  print (result)
