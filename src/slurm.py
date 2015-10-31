import time
import subprocess as proc
from common import *

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


from collections import namedtuple
slurmJob = namedtuple('slurmJob', 'jobid, partition, name, user, state, time, time_limit, nodes, nodelist')


class slurm:

  @classmethod
  def info(cls):
    out = proc.call("sinfo")  
    return out

  @classmethod
  def run(cls, cmd):
    out = proc.call('srun ' + cmd, shell=True)
    return out

  @classmethod
  def getJobs(cls, **kwargs):
    cmd = ['squeue','-h', '-l']
    if 'state' in kwargs: 
      cmd.append('--state=' + kwargs.get('state'))
    joblist = filter(None, proc.check_output(cmd).decode().split('\n'))
    return [slurmJob(*(job.split())) for job in joblist]

  @classmethod
  def sbatch(cls, jobid, workdir, options, modules, cmd):

    logging.info("Sbatch Job submitted for " + str(jobid))

    inline = '#!/bin/bash -l\n\n#SBATCH\n'
    for k, v in options.items():
      inline += '#SBATCH --%s=%s\n' % (k, str(v))

    inline += '#SBATCH --output=%s.out\n' % str(jobid)
    inline += '#SBATCH --workdir=%s\n' % str(workdir)

    for mod in modules:
      inline += 'module load %s\n' % mod

    inline += '\n%s\n' % cmd


    logging.info("Inline SBATCH: ")
    logging.info(inline)

    # Launch job
    job = proc.Popen('sbatch <<EOF\n%sEOF' % inline,
      shell=True, stdin=None, stdout=proc.PIPE, stderr=proc.STDOUT)
    stdout, stderr = job.communicate()

    logging.info("SBAtch Submitted. Output follows:")
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



if __name__ == '__main__':
  print ("Testing slurm routines\n")
  result = slurm.info()
  print (result)
  print ()
  result = slurm.schedule('0001', "/ring/ddc/hello.sh")
  print (result)

  print ()
  result = slurm.parallel('0002', "memcached -vvv -u root", 2)
  print (result)

  print ()
  print ("Currently running jobs:")
  for job in slurm.getJobs():
    print (job.name, job.state, job.nodes, ': ', job.nodelist)
  print (result)
