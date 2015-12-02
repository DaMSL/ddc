import shutil

from common import *
from macrothread import macrothread
import datetime as dt


import logging
logging.basicConfig(format='[%(module)s]: %(message)s', level=logging.DEBUG)

class garbageCollect(macrothread):
  def __init__(self, schema, fname):
    macrothread.__init__(self, schema, fname, 'gc')

    # State Data for Simulation MacroThread -- organized by state
    self.setStream(None, None)
    self.setState('gcDelay')

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('redis')
    # self.slurmParams['share'] = None


  def term(self):
    return False

  def split(self):
    return [1], None

  def configElasPolicy(self):
    self.delay = self.data['gcDelay']

  def execute(self, params):
    jclist = [key.decode() for key in self.catalog.keys('jc_*')]
    dead = 0
    zombie = 0
    alive = 0

    start = dt.datetime.now()
    for job in jclist:
      config = {k.decode(): v.decode() for k, v in self.catalog.hgetall(job).items()}
      # for k, v in config.items():
      #   logging.debug('%s, %s', k, str(v))

      if 'gc' not in config:
        logging.warning("Garbage Collection not set for:  %s", job)
        zombie += 1
        continue

      if int(config['gc']) == 0:
        if 'workdir' not in config:
          logging.warning("Cannot Garbage collect: %s", job)          
          zombie += 1
        else:
          if os.path.exists(config['workdir']):
            shutil.rmtree(config['workdir'])
          logging.warning("GC: cleaned `%s`, and removed dir: %s", job, config['workdir'])
          self.catalog.delete(job)
          dead += 1
      else:
        alive += 1

      # logging.debug("%s :  %d", job, config['gc'])


    end = dt.datetime.now()
    timediff = lambda x, y: (y-x).seconds + (.001)*((y-x).microseconds//1000)


    logging.info("GC Stats:")
    logging.info("  JC Cleaned:     %d", dead)
    logging.info("  JC Alive  :     %d", alive)
    logging.info("  JC Zombied:     %d", zombie)
    logging.info("Total time in GC: %f", timediff(start, end))
    logging.info("GC Complete")
    return []


if __name__ == '__main__':
  mt = garbageCollect(schema, __file__)
  mt.run()


# import os
# import redis
# r = redis.StrictRedis(host='login-node02')
# jcq = r.lrange('JCQeueu', 0, -1)
# jkeys = r.keys('jc_*')
# count = 0
# for j in jkeys:
#   jc = j.decode()
#   job = {k.decode(): v.decode() for k, v in r.hgetall(jc).items()}
#   if 'gc' in job and int(job['gc']) == 1:
#     print('GC Already =1 for %s', jc)
#     count += 1
#   elif 'workdir' in job and len(os.listdir(job['workdir'])) > 2:
#     print('Saving: %s' % jc)
#     r.hset(jc, 'gc', 1)
#     count += 1
#   elif 'name' in job and job['name'] in jcq:
#     print('Saving: %s' % jc)
#     r.hset(jc, 'gc', 1)
#     count += 1