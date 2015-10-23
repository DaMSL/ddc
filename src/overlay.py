from slurm import slurm
import subprocess as proc
import os
import abc

from functools import reduce

from common import *
import logging
logger = setLogger()


PORT = '7000'

class service(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, name):
    self.name = name
    self.activenodes = []

    #  TODO:  Pending TTL
    #  TODO:  Probability of launching new job

  @abc.abstractmethod
  def start(self):
    pass






class redisService(service):

  def __init__(self, name):
    service.__init__(self, name)

  def sbatch(self, numnodes):

    jobid = 'test'
    name = 'clu'
    
    batchfile = 'sh/' + self.name + jobid + '.sh'
    with open(batchfile, 'w') as script:
      script.write('#!/bin/bash\n\n')

      script.write("#SBATCH --job-name=%s-%s\n" % (name, jobid))
      script.write("#SBATCH --output=out/%s-%%j.out\n" % name)
      script.write("#SBATCH --partition=parallel\n")
      script.write("#SBATCH --nodes=%d\n" % numnodes)
      script.write("#SBATCH --ntasks-per-node=1\n\n")

      # script.write('cd /ring/ddc/cluster-test')
      # script.write('mkdir -p $(hostname)')
      # script.write('cd $(hostname)')
      # script.write('redis-server ../rediscl.conf')

      script.write('srun redis-server /ring/ddc/cluster-test/redis_cluster.conf\n\n')
    chmodX(batchfile)

    out = proc.check_output(['sbatch', batchfile]).decode()
    logger.debug('PROC Output: ' + out)
    return out



  def startNodes(self, numnodes):

    # 1. launch redis nodes to start separtely


    # job = slurm.parallel('0001', , numnodes)
    job = self.sbatch(2)

    jobid = job.split()[-1]

    logger.debug("Submitted Cluster Job. Job ID = " + jobid)

    return jobid

    # slurm.schedule('cluster')

    # nodelist = ....

    # for node in nodelist:
    #   # cd or append cluster redis parent dir
    #   os.mkdir(node)
    #   with open(os.path.join(node, 'redis.conf')) as conf:
    #     conf.write('')
    #     conf.write('port ' + PORT)
    #     conf.write('cluster-enabled yes')
    #     conf.write('cluster-config-file nodes.conf')
    #     conf.write('cluster-node-timeout 5000')
    #     conf.write('appendonly yes')



  def joinCluster(self, jobid):

    joblist = slurm.getJobs(state='all')

    redisjob = list(filter(lambda x: x.jobid==jobid, joblist))[0]

    hosts = hostlist.expand_hostlist(redisjob.nodelist)

    me = socket.gethostname()

    online = [False] * len(hosts)
    if me == hosts[0]:
      cmd = '/ring/ddc/cluster-test/redis-trib.rb create --replicas 0 '
      for h in hosts:
        cmd += h + ':' + PORT

      # Check online status of all nodes
      while not reduce(lambda x, y: x and y, online):
        for i, h in enumerate(online):
          if not h:
            try:
              cli = redis.StrictRedis(host=hosts[i], port=PORT, db=0)
              online[i] = cli.ping()
            except redis.ConnectionError as ex:
              online[i] = False

      # Join cluster
      result = proc.check


    # TODO:  JOIN Cluster Here


    # Insert hosts into catalog
    slurm.parallel('cluster_' + jobid, cmd)


if __name__ == '__main__':
  logger.debug('Starting redis cluster test')
  cluster = redisService('clu')
  job = cluster.startNodes(2)

  cluster.joinCluster