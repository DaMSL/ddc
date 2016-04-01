import pickle
import numpy as np
import logging
import math

from core.common import *

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)


class ReservoirSample(object):
  """ Retains management over set of samples for common data types. 
  For now: support redis as storage and hold numpy arrays
  All data items in the sample are consistent (same type/shape)
  """
  def __init__(self, name, datastore, maxsize=10000, ):
    #  For now: shape is required with NP array. Eventually this should
    #  be able to support different data types
    self.name = name
    self.redis = datastore  # Assume redis for now
    self.maxsize = maxsize
    self.dtype = None
    self.shape = None
    if datastore.exists(self.getkey('_dtype')):
      self.dtype = datastore.get(self.getkey('_dtype'))
      self.shape = eval(datastore.get(self.getkey('_shape')))

  def getkey(self, label):
    """ Returns the key label for the data store
    """
    return 'rsamp:%s:%s' % (self.name, str(label))

  def getsize(self, label):
    """ returns the current size of the reservoir
    """
    key = self.getkey(str(label))
    return self.redis.llen(key)

  def getN(self, label):
    """ N is the total of data elements for which the reservoir represents
    """
    key = self.getkey(str(label) + ':full')
    val = self.redis.get(key)
    return 0 if val is None else int(val)

  def getpercent(self, label):
    rsize = self.getsize(key)
    N = self.getN(key)
    if N is None or N == 0:
      return 0
    else:
      return rsize / N

  def insert(self, label, data):
    """ Processes the batch of data one element at a time and applies the 
    reservoir function to decide on whether or not to insert
    """
    if isinstance(data, list):
      data = np.array(data)
    logging.info("Reservoir INSERT request for %d points into %s", len(data), label)
    key = self.getkey(label)
    rsize = self.redis.llen(key)
    N = self.getN(label)
    num_inserted = 0

    # Process everything as one pipeline
    pipe = self.redis.pipeline()
    if self.dtype is None:
      pipe.set(self.getkey('_dtype'), np.lib.format.dtype_to_descr(np.dtype(data.dtype)))
      pipe.set(self.getkey('_shape'), data.shape[1:])

    spill = 0
    for i, elm in enumerate(data):
      # Caclulate probabilty to insert
      #  When N < maxsize, rsize == N and data is inserted with P = 1.
      #  Otherwise probability decreased as N grows
      P = 1 if N == 0 else rsize / N
      N += 1
      do_insert = np.random.random() <= P

      # Not full: Fill sequentially
      if do_insert and rsize < self.maxsize:
        # logging.debug('Available Space in Reservoir %s: %d', str(label), self.maxsize-rsize)
        pipe.rpush(key, pickle.dumps(elm))
        rsize += 1
        num_inserted += 1

      # Full Reservoir: evict an element
      elif do_insert:
        # logging.debug('Full Reservoir %s: %d', str(label), rsize)
        evict = np.random.randint(rsize)
        pipe.lset(key, evict, pickle.dumps(elm))
        spill += 1
        num_inserted += 1

    # Keep track of total # of insertions & evictions
    logging.info("##RSAMP Inserted= %d Evict= %d", num_inserted, spill)
    pipe.incr(key + ':full', len(data))
    pipe.incr(key + ':spill', spill)
    pipe.execute()

    # Return # of inserted points
    return num_inserted

  def get(self, label):
    """ Retrieve the entire reservoir
    """
    key = self.getkey(str(label))
    if self.dtype is None:
      logging.error('Reservoir Sample for %s is not defined in the datastore.', key)
      return []
    data_raw = self.redis.lrange(key, 0, -1)
    R = len(data_raw)
    N = self.getN(label)
    spillamt = self.redis.get(key + ':spill')
    if N is None:
      N = 0
    if spillamt is None:
      spillamt = 0
    logging.info('##RSAMP SIZE=%d  FULL=%d  SPILL=%d ', R, int(N), int(spillamt))
    arr = np.zeros(shape = (R,) + self.shape)
    for i in range(R):
      raw = pickle.loads(data_raw[i])
      arr[i] = np.fromstring(raw, dtype=self.dtype).reshape(self.shape)
    return arr
    



#  Original Implementation using files as the sample storage
def reservoirSampling(dataStore, hiDimData, subspaceIndex, subspaceHash, resizeFunc, label, labelNameFunc):

  # hiDimData --> Delta X = indexed from 0...N (actual points in hi-dim space)
  # subspaceIndex -> Global index for Delta S_m  (index for the projected points from Delta X)
  # subspaceHashDelta --> Hash table for variables discovered (or tiled, labeled, etc..)
  #      label/bin/hcube --> list of indecies into DeltaX / DeltaS (0...N)

  for key in subspaceHash.keys():
    storeKey  = 'rsamp:%s:%s' % (label, labelNameFunc(key))
  
    while True:
      rsize     = dataStore.llen(storeKey)
          
      # Newly discovered Label
      if rsize == 0:
        logging.debug('New Data Label --> new reservoir Sample')

        reservoirSamp = np.zeros(shape=resizeFunc(len(subspaceHash[key])))

        # Assume subspaceHash[l] < MAX_SIZE (otherwise, use random.choice to select MAX_SIZE)
        #  TODO: Pipeline & optimize (here and below)
        pipe = dataStore.pipeline()
        for i, si in enumerate(subspaceHash[key]):
          pipe.rpush(storeKey, subspaceIndex[si])

      # Reservoir Sample already exists
      else:
        logging.debug('Old Data Label. Retreiving sample from : %s', rsampfile + '.npy')

        try:
          # Check to ensure lock is not already acquired
          lock = os.open(rsampfile, os.O_RDWR)
          fcntl.lockf(lock, fcntl.LOCK_EX)

        except FileNotFoundError as ex:
          logging.error("Reservoir Sample File not found for `%s`: %s" % (label, labelNameFunc(key)))

        reservoirSamp = np.load(rsampfile + '.npy')
        
        # New points can fit inside reservoir
        if rsize + len(subspaceHash[key]) < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Undersized Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(rsize + len(subspaceHash[key])), refcheck=False)
          for i, si in enumerate(subspaceHash[key]):
            np.copyto(reservoirSamp[rsize+i], hiDimData[si])
            dataStore.rpush(storeKey, subspaceIndex[si])

        # Some new points can fit inside reservoir (sample what can fit)
        elif rsize < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Nearly Full Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(DEFAULT.MAX_RESERVOIR_SIZE), refcheck=False)
          sample = np.random.choice(subspaceHash[key], DEFAULT.MAX_RESERVOIR_SIZE - rsize)
          for i, si in enumerate(sample):
            np.copyto(reservoirSamp[key][rsize+i], hiDimData[sample])
            dataStore.rpush(storeKey, subspaceIndex[sample])

        # Implement Eviction policy & replace with new points
        else:
          logging.debug('Full Reservoir: %d', rsize)
          evictNum = min(len(subspaceHash[l]), DEFAULT.MAX_RESERVOIR_SIZE // 20)         #  5% of reservoir -- for now
          evict = np.random.choice(DEFAULT.MAX_RESERVOIR_SIZE, evictNum)
          sample = np.random.choice(subspaceHash[key], evictNum)
          for i in range(evictNum):
            np.copyto(reservoirSamp[key][evict[i]], hiDimData[sample])
            dataStore.rset(storeKey, evict[i], subspaceIndex[sample])

      logging.debug("Saving Reservoir Sample File: %s", os.path.basename(rsampfile))
      np.save(rsampfile, reservoirSamp)
      os.close(lock)
      break



def reservoirSampling(dataStore, hiDimData, subspaceIndex, subspaceHash, resizeFunc, label, labelNameFunc):

  # hiDimData --> Delta X = indexed from 0...N (actual points in hi-dim space)
  # subspaceIndex -> Global index for Delta S_m  (index for the projected points from Delta X)
  # subspaceHashDelta --> Hash table for variables discovered (or tiled, labeled, etc..)
  #      label/bin/hcube --> list of indecies into DeltaX / DeltaS (0...N)

  for key in subspaceHash.keys():
    storeKey  = 'rsamp:%s:%s' % (label, labelNameFunc(key))
    rsampfile = os.path.join(DEFAULT.DATADIR, 'rSamp_%s' % (labelNameFunc(key)))
  
    while True:
      rsize     = dataStore.llen(storeKey)
          
      # Newly discovered Label
      if rsize == 0:
        logging.debug('New Data Label --> new reservoir Sample (acquiring lock...)')
        try:
          # Check to ensure lock is not already acquired
          lock = os.open(rsampfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as ex:
          logging.debug("Sample File exists (someone else has acquired it). Backing off rand 1..5 seconds and re-checking")
          time.sleep(np.random.randint(4)+1)
          continue

        reservoirSamp = np.zeros(shape=resizeFunc(len(subspaceHash[key])))

        # Assume subspaceHash[l] < MAX_SIZE (otherwise, use random.choice to select MAX_SIZE)
        #  TODO: Pipeline & optimize (here and below)
        for i, si in enumerate(subspaceHash[key]):
          np.copyto(reservoirSamp[i], hiDimData[si])
          dataStore.rpush(storeKey, subspaceIndex[si])

      # Reservoir Sample already exists
      else:
        logging.debug('Old Data Label. Retreiving sample from : %s', rsampfile + '.npy')

        try:
          # Check to ensure lock is not already acquired
          lock = os.open(rsampfile, os.O_RDWR)
          fcntl.lockf(lock, fcntl.LOCK_EX)

        except FileNotFoundError as ex:
          logging.error("Reservoir Sample File not found for `%s`: %s" % (label, labelNameFunc(key)))

        reservoirSamp = np.load(rsampfile + '.npy')
        
        # New points can fit inside reservoir
        if rsize + len(subspaceHash[key]) < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Undersized Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(rsize + len(subspaceHash[key])), refcheck=False)
          for i, si in enumerate(subspaceHash[key]):
            np.copyto(reservoirSamp[rsize+i], hiDimData[si])
            dataStore.rpush(storeKey, subspaceIndex[si])

        # Some new points can fit inside reservoir (sample what can fit)
        elif rsize < DEFAULT.MAX_RESERVOIR_SIZE:
          logging.debug('Nearly Full Reservoir: %d', rsize)
          reservoirSamp.resize(resizeFunc(DEFAULT.MAX_RESERVOIR_SIZE), refcheck=False)
          sample = np.random.choice(subspaceHash[key], DEFAULT.MAX_RESERVOIR_SIZE - rsize)
          for i, si in enumerate(sample):
            np.copyto(reservoirSamp[key][rsize+i], hiDimData[sample])
            dataStore.rpush(storeKey, subspaceIndex[sample])

        # Implement Eviction policy & replace with new points
        else:
          logging.debug('Full Reservoir: %d', rsize)
          evictNum = min(len(subspaceHash[l]), DEFAULT.MAX_RESERVOIR_SIZE // 20)         #  5% of reservoir -- for now
          evict = np.random.choice(DEFAULT.MAX_RESERVOIR_SIZE, evictNum)
          sample = np.random.choice(subspaceHash[key], evictNum)
          for i in range(evictNum):
            np.copyto(reservoirSamp[key][evict[i]], hiDimData[sample])
            dataStore.rset(storeKey, evict[i], subspaceIndex[sample])

      logging.debug("Saving Reservoir Sample File: %s", os.path.basename(rsampfile))
      np.save(rsampfile, reservoirSamp)
      os.close(lock)
      break