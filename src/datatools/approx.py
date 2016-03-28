import pickle
import numpy as np
import logging
import math

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
    return 'rsamp:%s:%s' % (self.name, str(label))

  def getsize(self, label):
    key = self.getkey(label)
    return self.redis.llen(key)

  def insert(self, label, data):
    logging.info("Reservoir inserting request for %d points into %s", len(data), label)
    key = self.getkey(label)
    rsize = self.redis.llen(key)
    num_inserted = 0
    pipe = self.redis.pipeline()
    if self.dtype is None:
      pipe.set(self.getkey('_dtype'), data.dtype.__name__)
      pipe.set(self.getkey('_shape'), data.shape[1:])

    # ALl new points fit inside the reservoir
    if rsize + len(data) <= self.maxsize:
      logging.debug('Available Space in Reservoir %s: %d', str(label), self.maxsize-rsize)
      for si in data:
        num_inserted += 1
        pipe.rpush(key, pickle.dumps(si))
    # Implement Eviction policy & replace with new points
    else:
      # Freshness Value (retaining more of newer data --> more "fresh" data)
      logging.debug('Full Reservoir %s: %d', str(label), rsize)
      PERCENT_OF_NEW_DATA = .5    
      evictNum = math.round(len(data) * PERCENT_OF_NEW_DATA)
      evict = np.random.choice(DEFAULT.MAX_RESERVOIR_SIZE, evictNum)
      store_sample = np.random.choice(data, evictNum)
      for i in range(evictNum):
        num_inserted += 1
        pipe.rset(key, evict[i], pickle.dumps(store_sample[i]))
    pipe.execute()
    return num_inserted

  def get(self, label):
    key = self.getkey(label)
    if self.dtype is None:
      logging.error('Reservoir Sample for %s is not defined in the datastore.', key)
      return []
    data_raw = self.redis.lrange(key, 0, -1)
    N = len(data_raw)
    arr = np.zeros(shape = (N,) + self.shape)
    for i in range(N):
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