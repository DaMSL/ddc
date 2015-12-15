import abc
import redis
import numpy as np

class kvadt(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, database, name):
    self._db   = database
    self._key = name
    self._value = None

  def __get__(self):
    return self._value

  def __set__(self, value):
    self._value = value

  @abc.abstractmethod
  def get(self, index):
    """
    Retrieve value in the adt at position, index
    """
    raise NotImplemented

  @abc.abstractmethod
  def set(self, index):
    """
    Set value in the adt at key, index
    """
    raise NotImplemented

  @abc.abstractmethod
  def merge(self, index):
    """
    Returns redis key, given the adt index
    """
    raise NotImplemented






def fromByte(dataType):
  if dataType == int:
    return lambda x: int(x.decode())
  if dataType == float:
    return lambda x: float(x.decode())
  return lambda x: x.decode()



class kv2DArray(kvadt):

  """
  Wrapper for a 2-D array in K-V. For now, only allow values (int, float, str)
  Each element is a separate key
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, db, name, mag=5, init=0):
    kvadt.__init__(self, db, name)

    self.db = db
    self.name = name
    self.mag = 0

    # Check if the array already exists
    stored_mag = self.db.get(self.name + '_magnitude')
    if stored_mag:
      self.mag = int(stored_mag.decode())
      self._value = self.get()
    else:
      # Initialize the array
      self.mag = mag
      self.set(init)


  def __get__(self):
    return self._value

  def __set__(self, value):
    self._value = value


  def _elmkey (self, x, y):
    return self.name + '_%d_%d' % (x, y)

  def get (self, pipeline=None):
    arr = np.zeros(shape=(self.mag,self.mag))
    pipe = self.db.pipeline() if pipeline is None else pipeline
    for x in range(self.mag):
      for y in range(self.mag):
        pipe.get(self._elmkey(x, y))
    if pipeline is not None:
      return pipe

    vals = pipe.execute()
    v = 0
    for x in range(self.mag):
      for y in range(self.mag):
        arr[x][y] = float(vals[v])
        v += 1
    return arr


  def set(self, value):
    if isinstance(value, np.ndarray):
      data = value
    elif isinstance(value, int) or isinstance(value, float):
      data = np.full((self.mag, self.mag), value)
    pipe = self.db.pipeline()
    pipe.set(self.name + '_magnitude', self.mag)
    for x in range(self.mag):
      for y in range(self.mag):
        pipe.set(self.name + '_%d_%d' % (x, y), data[x][y])
    pipe.execute()


  def merge (self, arr):
    pipe = self.db.pipeline()
    for x in range(self.mag):
      for y in range(self.mag):
        pipe.incrbyfloat(self._elmkey(x, y), arr[x][y])
    pipe.execute()


  # Single Element operations
  def setelm (self, x, y, elm):
    self.db.set(self._elmkey(x, y), elm)

  def getelm (self, x, y):
    return float(self.db.get(self._elmkey(x, y)))

  def incr (self, x, y, amt=1):
    self.db.incrbyfloat(self._elmkey(x, y), amt)
    # else:
    #   logging.error("ERROR!  Trying to increment a non-number")

  def display(self):
    mat = self.getAll()
    for row in mat:
      logging.info('   ' + " ".join([fmt%x for x in row]))



if __name__ == '__main__':

  try:

    r = redis.StrictRedis()
    test2d_Stor = kv2DArray(r, 'test2d', 3)
    test2d = test2d_Stor.get()
    print(test2d)

    test2d_Stor.incr(1,1,12)

    delta = np.zeros(shape=(3,3))
    delta[1][2] = 5

    test2d_Stor.merge(delta)
    test2d = test2d_Stor.get()
    print(test2d)

    t2 = test2d_Stor
    print(t2)

  except redis.ConnectionError as ex:
    print("Make sure Redis is up")