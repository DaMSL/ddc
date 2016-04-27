
import redis, time
from overlay.redisOverlay import RedisClient
from overlay.overlayException import *
from core.common import *
from datetime import datetime as dt
import argparse
import logging






def check_except(name):
  log = logging.getLogger(name)
  total_uptime = 0.
  total_downtime = 0.
  run_time = 0.
  counter = 0
  up = False
  uptxt = 'DOWN'
  ts = dt.now()
  log.info('START,%f,%f,%f', total_uptime, total_downtime, run_time)
  while True:
    ts = dt.now()
    time.sleep(.5)
    try:
      client = RedisClient(name)
      status = client.ping()
      if counter == 0:
        client.set('testcounter', 1)
      else:
        client.incr('testcounter')
      counter += 1
      testcounter = int(client.get('testcounter'))
      assert (counter == testcounter)
    except redis.RedisError as e:
      print(' REDIS ERROR ===>   ', e.__name__)
      status = False
    except OverlayNotAvailable as e:
      print(' OVERLAY not available')
      status = False
    delta = (dt.now()-ts).total_seconds()
    if status == up:
      run_time += delta
    else:
      print('STATUS Change from %s' % uptxt)
      log.info('%s,%f,%f,%f', uptxt, total_uptime, total_downtime, run_time)
      run_time = 0.
    if status:
      uptxt = 'UP'
      total_uptime += delta
    else:
      uptxt = 'DOWN'
      total_downtime += delta
    print('%s,%f' % (uptxt, run_time))
    up = status

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  args = parser.parse_args()

  name = args.name

  # External Logger to track metrics:
  logfile = os.path.join(os.environ['HOME'], 'ddc', 'results', 'osvc_watch_%s.log'%name)
  monlog = logging.getLogger(name)
  monlog.setLevel(logging.INFO)
  fh = logging.FileHandler(logfile)
  fh.setLevel(logging.INFO)
  fmt = logging.Formatter('%(asctime)s,%(message)s')
  fh.setFormatter(fmt)
  monlog.addHandler(fh)
  monlog.propagate = False

  conf = systemsettings()
  conf.applyConfig(name + '.json')
  check_except(name)
