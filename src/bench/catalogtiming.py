"""Simple test routine to check Redis Exceptions
""" 
import traceback
import redis
import time
from datetime import datetime as dt

r = redis.StrictRedis(decode_responses=True)

iowait = 0.
while True:
  print ('TRYING..... Wait = ', iowait)
  try:
    time.sleep(1)
    st = dt.now()
    r.ping()
    iowait += (dt.now()-st).total_seconds()
  except KeyboardInterrupt as exp:
    traceback.print_exc()
    break
  except redis.BusyLoadingError as exp:
    print('Service is loading:  ', exp.__class__)
  except redis.ConnectionError as exp:
    print('Service Not Available.  ', exp.__class__)
  except Exception as exp:
    print(exp.__class__)
    print(exp)
