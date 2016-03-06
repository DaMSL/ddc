import shutil

from common import *
from macrothread import macrothread
import datetime as dt

logger = logging.getLogger(__name__)

class garbageCollect(macrothread):
  def __init__(self, fname):
    macrothread.__init__(self, fname, 'gc')

    # State Data for Simulation MacroThread -- organized by state
    self.setStream(None, None)
    self.addImmut('gcDelay')

    # Local Data to this running instance
    self.cpu = 1
    self.numnodes = 1

    #  Update Runtime Parameters
    self.modules.add('redis')


  def term(self):
    return False

  def split(self):
    return [1], None

  def configElasPolicy(self):
    self.delay = self.data['gcDelay']

  def execute(self, params):

    # Get all job candidate keys
    jclist = self.catalog.keys('jc_*')
    dead = 0
    zombie = 0
    alive = 0

    start = dt.datetime.now()

    # iterate through each job
    for job in jclist:
      # Load job params
      config = self.catalog.hgetall(job)

      # Erroneous cases were GC was never set
      if 'gc' not in config:
        logging.warning("Garbage Collection not set for:  %s", job)
        zombie += 1
        continue

      # Remove job from disk and K-V store if GC counter is 0
      if int(config['gc']) == 0:
        # Erroneous cases with no work dir
        if 'workdir' not in config:
          logging.warning("Cannot Garbage collect: %s", job)          
          zombie += 1
        else:

          # Clean up
          if os.path.exists(config['workdir']):
            shutil.rmtree(config['workdir'])
          logging.warning("GC: cleaned `%s`, and removed dir: %s", job, config['workdir'])
          self.catalog.delete(job)
          dead += 1
      else:
        alive += 1

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
  mt = garbageCollect(__file__)
  mt.run()