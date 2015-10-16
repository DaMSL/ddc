import redis
import os

if __name__ == '__main__':
  lf = 'redis.lock'
  if os.path.exists(lf):
    with open(lf) as f:
      h,p,d = f.read().split(',')
    r = redis.StrictRedis(host=h, port=p, db=d)
    r.flushdb()
    r.shutdown()
    os.remove(lf)

