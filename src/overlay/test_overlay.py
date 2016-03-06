import pytest

from overlay.redisOverlay import RedisClient

def test_overlay():
  assert True == True
  print ('Is True')

def test_redisclient():
  name = 'test_overlay'
  client = RedisClient(name)
  if not client.isconnected:
    print('I am not connected. Service is not running')
    return
  client.ping()
  print(client.get('foo'))
  print('Running a long pipeline...')
  pipe = client.pipeline()
  for i in range(40000):
    allkeys = pipe.keys('*')
    pipe.set('allkeys', str(allkeys))
  result = pipe.execute()
  print('Pipeline complete.')
  # print('Promote. local slave')
  # time.sleep(5)
  # print('demote master')
  # time.sleep(5)
  # print('Waiting...')
  print(client.incr('foo'))
  print(client.get('foo'))


