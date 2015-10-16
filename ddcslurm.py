import time
import subprocess as proc
from common import *

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
  def schedule(cls, jobid, cmd, **kwargs):

    name = kwargs.get('name', '')

    batchfile = 'job' + jobid + '.sh'
    with open(batchfile, 'w') as script:
      script.write('#!/bin/bash\n\n')

      if 'delay' in kwargs:
        script.write("#SBATCH --begin=now+%di\n" % kwargs.get('delay'))

      script.write("#SBATCH --output=out/%s-%%j.out\n\n" % name)

      
      # TODO: Other batch scripting goes here
      script.write('srun ' + cmd + '\n\n')
    chmodX(batchfile)

    out = proc.call(['sbatch', batchfile])
    return out


if __name__ == '__main__':
  print ("Testing slurm routines\n")
  result = slurm.info()
  print (result)
  print ()
  result = slurm.schedule('0001', "/ring/ddc/hello.sh")
  print (result)


