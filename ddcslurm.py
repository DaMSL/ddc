import time
import subprocess as proc

class slurm:

  @classmethod
  def info(cls):
    out = proc.call("sinfo")  
    return out

  @classmethod
  def schedule(cls, cmd):
    out = proc.call('srun ' + cmd, shell=True)
    return out

if __name__ == '__main__':
  print ("Testing slurm routines\n")
  result = slurm.info()
  print (result)
  print ()
  result = slurm.schedule("/ring/ddc/hello.sh")
  print (result)