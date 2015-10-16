import redis
import os

if __name__ == '__main__':
  for h in ['mddb2', 'qp-hd14', 'qp-hd15', 'qp-hd16']:
    r = redis.StrictRedis(host=h)
    try:
      r.shutdown()
      print('Shutdown host on ' + h)
    except redis.ConnectionError:
      print('Host %s not alive' % h)

  lf = 'redis.lock'
  if os.path.exists(lf):
    os.remove(lf)

