#!/usr/bin/env python

import abc

import redis
import logging
import numpy as np

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.0.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

# NOTE:  Possible bug in Enum class prevents proper init of __members__
#  Following is a shell of a wrapper type-class for all data types in the system
#  Will need to re-design another time (found a bug in the enum module)
#  Attempt to bypass bug:
# class DType:
#   int = int
#   float = float
#   num = float
#   list = list
#   dict = dict
#   str = str
#   ndarray = np.ndarray
#   matrix = kv2DArray
#   @classmethod
#   def cmp(cls, this, other):
#     return this.__name__ == other.__name__



def infervalue(value):
  """For use with Dynamic dispatching
  """
  try:
    castedval = None
    if value.isdigit():
      castedval = int(value)
    else:
      castedval = float(value)
  except ValueError as ex:
    castedval = value
  return castedval

def decodevalue(value):
  """For use with K-V Stores to dynamically dispath string based data types
  """
  data = None
  if isinstance(value, list):
    try:
      if len(value) == 0:
        data = []
      elif value[0].isdigit():
        data = [int(val) for val in value]
      else:
        data = [float(val) for val in value]
    except ValueError as ex:
      data = value
  elif isinstance(value, dict):
    # logging.debug("Hash Loader")
    data = {}
    for k,v in value.items():
      data[k] = infervalue(v)
  elif value is None:
    data = None
  else:
    data = infervalue(value)
  return data



#  K-V ADT Nucleaus of a type-wrapper class
class kvadt(object):
  """The Key-Value Abstract Data Type (kvadt) is employed with key-value
  stores to faciltate dynamic disatching of serialized, string-stored data
  into non-primitve data types (anything other than str, int, float, list, dict)
  """

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
  def set(self, key, index):
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
      self.mag = int(stored_mag)
      self._value = self.get()
    else:
      # Initialize the array
      self.mag = mag
      self.set(init)

  @classmethod
  def key(cls, name, x, y):
    return name + '_%d_%d' % (x, y)

  def __get__(self):
    return self._value

  def __set__(self, value):
    self._value = value

  def _elmkey (self, x, y):
    return self.key(self.name, x, y)

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



class kvMapList(kvadt):

  """
  Wrapper for a map of key->list  
  Each element in the map is a list; Ensures every list is the same size
  and pads new elements with the default value
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, db, name, default=0):
    kvadt.__init__(self, db, name)

    self.db = db
    self.name = name
    self.default = default
    self.length = 0
    self.labellist = []
    self.type = None

    # Check if the array already exists
    length = self.db.exists(self.name + ':total')
    if stored:
      self.length = int(length)
      self.labellist = self.db.lrange(self.name + ':labels', 0, -1)

  def key(self, name, x):
    """X is either a number or a pair of numbers
    """
    if isinstance(x, str):
      return self.name + x
    elif isinstance(x, tuple):
      return self.name +':%d_%d' % x
    else:
      return self.name + str(x)

  def __get__(self):
    return self._value

  def __set__(self, value):
    self._value = value

  def _elmkey (self, x, y):
    return self.key(self.name, x, y)

  def get (self):
    mapping = {}
    if self.labellist is None:
      logging.warning('Cannot get lists for %s. No data exists')
      return {}

    lcast = lambda x: int(x) if isinstance(x, int) else tuple(x)

    pipe = self.db.pipeline()
    for label in self.labellist:
      pipe.lrange(self.name + ':' + label)
    results = pipe.execute()

    for i, vals in enumerate(results):
      key = lcast(self.labellist[i])
      mapping[key] = vals

    self._value = mapping
    return mapping


  def merge(self, mapping):
    pipe = self.db.pipeline()
    for k in mapping.keys():
      label = self.key(k)
      pipe.rpush(self.name + ':' + label)
      if label not in self.labellist:
        self.labellist.append(label)
        pipe.rpush(self.name + ':labels', label)
    pipe.incr(self.name + ':total')
    pipe.execute()


  def set (self, key, index):
    pass
    

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



def runtest():
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

if __name__ == '__main__':
  runtest()
