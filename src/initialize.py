from simmd import *
from anlmd import *
from ctlmd import *
from deshaw import *
from indexing import *
import math
import json
import bisect

DO_COV_MAT = False


def init_catalog(catalog):

  #  Create a "seed" job
  logging.debug("System schema:")

  # Loading schema and insert the start job into the queue
  for k in sorted(DEFAULT.schema.keys()):
    logging.info("  %15s:   %s", k, DEFAULT.schema[k])

  # executecmd("shopt -s extglob | rm !(SEED.p*)")

  # TODO:  Job ID Management
  ids = {'id_' + name : 0 for name in ['sim', 'anl', 'ctl', 'gc']}

  logging.debug("Catalog found on `%s`. Clearing it.", catalog.host)
  catalog.clear()

  logging.debug("Loading schema into catalog.")
  catalog.save(ids)
  catalog.save(DEFAULT.schema)

  # TODO:  LOAD DEFAULT SETTINGS

  # Load DEShaw meta-data
  # histo = np.load('histogram.npy')


# TODO: For use when migrating schema into catalog
def create_schema(catalog, schema):
  dtype_map = {}
  for k, v in schema.items():
    dtype_map[k] = type(v).__name__
  catalog.hmset("META_schema", dtype_map)


def init_archive(archive, hashsize=8):

  logging.debug("Archive found on `%s`. Stopping it.", archive.host)

  archive.clear()

  # Create redis storage adapter
  redis_storage = RedisStorage(archive)

  # Create Hash
  lshash = RandomBinaryProjections(DEFAULT.HASH_NAME, hashsize)
  lshash.reset(hashsize)

  config = lshash.get_config()
  for k,v in config.items():
    logging.info("%s: %s", str(k), str(v))
  # Assume vects is 
  # pcahash = PCABinaryProjections('pcahash', 10, [v[0] for v in vects])
  # redis_storage.store_hash_configuration(pcahash)
  # eng2 = Engine(454, lshashes=[pcahash], storage=redis_storage)
  # for v in vects:
  #   eng2.store_vector(v[0], v[1])

  # Store hash configuration in redis for later use
  logging.debug('Storing Hash in Archive')
  redis_storage.store_hash_configuration(lshash)

    # TODO:  Automate Historical Archive (Re)Loading


  archive.stop()
  if os.path.exists('archive.lock'):
    os.remove('archive.lock')


  logging.debug("Initialization complete\n")



def index_DEShaw(catalog, archive, start, num, winsize, slide):
    # winperfile = 1000 // slide
    winperfile = 1000
    totalidx = winperfile * num
    # end = 1 + start + num + math.ceil(slide // 1000)
    end = start + num
    logging.info("LOADING D.E.Shaw index as follows:")
    logging.info('  Start        %d', start)
    logging.info('  End          %d', end)
    logging.info('  WinSize      %d', winsize)
    logging.info('  Slide        %d', slide)
    logging.info('  Idx / File   %d', winperfile)
    logging.info('  Total Idx    %d', totalidx)
    logging.info('  Calculation: %s', 'COVARIANCE' if DO_COV_MAT else 'DIST_MATRIX')

    # Load all trajectories up front
    trajectory = loadDEShawTraj(start, end)
    n_var = trajectory.xyz.shape[1]
    if DO_COV_MAT:
      n_var *= 3
    # egm = np.zeros(shape=(num * 1000//slide, n_var), dtype=np.float32)
    # evm = np.zeros(shape=(num * 1000//slide, n_var, n_var), dtype=np.float32)
    egm = np.zeros(shape=(num * 1000, n_var), dtype=np.float32)
    evm = np.zeros(shape=(num * 1000, n_var, n_var), dtype=np.float32)
    logging.info("Traj Shapes: " + str(egm.shape) + " " + str(evm.shape))

    # Slide window & calc eigens
    # for k, w in enumerate(range(0, len(trajectory.xyz) - winsize+1, slide)):
    for k, frame in enumerate(trajectory.xyz):
      # if k == totalidx or w + winsize > len(trajectory.xyz):
      #   break
      if k % 10 == 0:
        logging.info("Trajectory  %d:   %d", start+(k//1000), k%1000)
      # sys.exit(0)
      distmat =  np.zeros(shape=(n_var,n_var))
      for A in range(n_var):
        for B in range(A, n_var):
          delta = LA.norm(frame[A] - frame[B])
          distmat[A][B] = delta
          distmat[B][A] = delta
      eg, ev = LA.eigh(distmat)
      # if DO_COV_MAT:
      #   eg, ev = LA.eigh(covmatrix(trajectory.xyz[w:w+winsize]))
      # else:
      #   eg, ev = LA.eigh(distmatrix(trajectory.xyz[w:w+winsize]))
      np.copyto(egm[k], eg)
      np.copyto(evm[k], ev)

    # Check Index Size
    indexSize = archive.get('indexSize')
    if indexSize is None:
      indexSize = n_var * DEFAULT.NUM_PCOMP
      logging.debug("Index Size not set. Setting to: %d", indexSize)
      catalog.set('indexSize', indexSize)
      archive.set('indexSize', indexSize)
      archive.set('num_var', n_var)
      archive.set('num_pc', DEFAULT.NUM_PCOMP)
      archive.set('num_atoms', trajectory.n_atoms)
      logging.debug("STORING META-DATA:")
      logging.debug('  indexSize   %d', indexSize)
      logging.debug('  num_var     %d', n_var)
      logging.debug('  num_pc      %d', DEFAULT.NUM_PCOMP)
      logging.debug('  num_atoms   %d', trajectory.n_atoms)
    else:
      indexSize = int(indexSize.decode())
      logging.debug("Meta Data already configured:  Indexsize = %d", indexSize)
      if indexSize != (n_var * DEFAULT.NUM_PCOMP):
        logging.error("Inconsistent Index Size:  Setting %d x %d  but had %d stored", n_var, DEFAULT.NUM_PCOMP, indexSize)

    engine = getNearpyEngine(archive, indexSize)

    # win = loadLabels()
    for i in range(len(egm)):
      index = makeIndex(egm[i], evm[i])
      seqnum = start + i//winperfile
      frame  = i % winperfile
      # state = win[seqnum].state
      # datalabel = '%d %04d-%03d' % (state, seqnum, slide*(i%winperfile))
      datalabel = '%04d-%03d' % (seqnum, frame)
      logging.debug("Storing Index:  %s", datalabel)
      engine.store_vector(index, datalabel)


    logging.debug("Index Load Complete")

    # TO JUST SAVE Eigens:
    # if args.saveeigen:
    #   evfile = home+'/work/eigen/evC%d_%04d' % (winsize, start)
    #   egfile = home+'/work/eigen/egC%d_%04d' % (winsize, start) 
    #   logging.info("Saving:  %s, %s", evfile, egfile) 
    #   np.save(evfile, evm)
    #   np.save(egfile, egm)

def init_control(catalog):
  
  # Initialize Candidate Pools
  numLabels  = int(catalog.get('numLabels').decode())
  pools      = [[[] for i in range(numLabels)] for j in range(numLabels)]
  candidates = [[[] for i in range(numLabels)] for j in range(numLabels)]

  # # init first and last
  # pools[labels[0].state].append(0)
  # pools[labels[-1].state].append(len(labels))

  # logging.info("Scanning labellist for potential candidates")
  # for i in range(1, len(labels)-1):
  #   if labels[i-1].state == labels[i].state == labels[i+1].state:
  #     pools[labels[i].state][labels[i].state].append(i)
  #   elif labels[i-1].state == labels[i].state:
  #     pools[labels[i].state][labels[i+1].state].append(i)
  #   else:
  #     pools[labels[i-1].state][labels[i].state].append(i)

  # logging.info("Creating candidate pools")
  # for i in range(numLabels):
  #   for j in range(numLabels):
  #     if len(pools[i][j]) <= DEFAULT.CANDIDATE_POOL_SIZE:
  #       candidates[i][j] = pools[i][j]
  #     else:    
  #       candidates[i][j] = np.random.choice(pools[i][j], DEFAULT.CANDIDATE_POOL_SIZE, replace=False)
    
  #  Create Candidate Pools from RMSD for all source DEShaw data
  pipe = catalog.pipeline()  
  logging.info("Deleting existing candidates and controids")
  pipe.delete('centroid')
  for i in range(numLabels):
    for j in range(numLabels):
      pipe.delete(candidPoolKey(i, j))

  logging.info("Loading DEShaw RMS Values")
  rms = np.load('rmsd.npy')
  nn = []
  cpools = {(i,j):[] for i in range(numLabels) for j in range(numLabels)}
  logging.info("Creating candidate pools")
  wpools = [[] for i in range(numLabels)]
  for i, traj in enumerate(rms):
    for f, conform in enumerate(traj):
      ordc = np.argsort(conform)
      A = ordc[0]
      relproximity = conform[A] / (conform[A] + conform[ordc[1]])
      nn.append((i, f, A, ordc[1], relproximity))
      if relproximity > .49:
        cpools[(A, ordc[1])].append((.5 - relproximity, i,f))
      elif len(wpools[A]) == 0 or relproximity > wpools[A][0][0]:
        bisect.insort(wpools[A], (relproximity, i, f))
        if len(wpools[A]) == 1000:
          del wpools[A][0]

  logging.info("All Candidates Found! Breakdown by bin:")
  candidatepools = {k:sorted(v, key=lambda x : x[2]) for k,v in cpools.items()}

  for i in range(numLabels):
    for j in range(numLabels):
      if i == j:
        candidates[i][j] = ['%03d:%03d'% (c[1], c[2]) for c in wpools[i]]
      else:
        candidates[i][j] = ['%03d:%03d'% (c[1], c[2]) for c in cpools[(i,j)]]
      logging.info("  (%d,%d)  %d", i, j, len(candidates[i][j]))

      # if len(pools[i][j]) <= DEFAULT.CANDIDATE_POOL_SIZE:
      #   candidates[i][j] = pools[i][j]
      # else:    
      #   candidates[i][j] = np.random.choice(pools[i][j], DEFAULT.CANDIDATE_POOL_SIZE, replace=False)

  logging.info("Updating Catalog")
  for i in range(numLabels):
    for j in range(numLabels):
      for c in candidates[i][j]:
        pipe.rpush(candidPoolKey(i, j), c)
  pipe.execute()


  centroid = np.load('centroid_heavy.npy')
  catalog.storeNPArray(centroid, 'centroid')

  # Initialize observations matrix
  obsMat_Store = kv2DArray(catalog, 'observations')
  selMat_Store = kv2DArray(catalog, 'selection')
  launchMat_Store = kv2DArray(catalog, 'launch')

def reindex(archive, size=10):
  indexsize = 1362 #int(archive.get('indexSize').decode())
  indexlist = getindexlist(archive, DEFAULT.HASH_NAME)
    # engine = getNearpyEngine(archive, indexsize)
  eucl = EuclideanDistance()
    # lshash = UniBucket(DEFAULT.HASH_NAME)
  lshash = RandomBinaryProjections(DEFAULT.HASH_NAME, size)
    # lshash = RandomBinaryProjections(None, None)
    # lshash = PCABinaryProjections(None, None, None)

  redis_storage = RedisStorage(archive)
  engine = nearpy.Engine(indexsize, distance=eucl, lshashes=[lshash], storage=redis_storage)
  engine.clean_all_buckets()
  logging.debug("Cleared Buckets. Storing.....")

  count = 0
  for idx in indexlist:
    if len(idx[1]) == 10:
      engine.store_vector(idx[0].astype(np.float32), idx[1])
      count += 1

  logging.debug("%d Vectors Stored. Hash Config follows:", count)

  config = lshash.get_config()
  for k,v in config.items():
    logging.info("%s: %s", str(k), str(v))
    # Assume vects is 
    # pcahash = PCABinaryProjections('pcahash', 10, [v[0] for v in vects])
    # redis_storage.store_hash_configuration(pcahash)
    # eng2 = Engine(454, lshashes=[pcahash], storage=redis_storage)
    # for v in vects:
    #   eng2.store_vector(v[0], v[1])

    # Store hash configuration in redis for later use
    logging.debug('Storing Hash in Archive')
  redis_storage.store_hash_configuration(lshash)

def findstartpts():

  startfiles = {}

  midpt = []
  total = 0
  start = 1
  last = win[0].state
  for i, w in enumerate(win):
    if w.state == last:
      total += 1
    else:
      midpt.append((last, (i-start), start + (i - start)//2))
      start = i
      last = w.state

  for i in  sorted(midpt, key=lambda x: x[1], reverse=True):
    b = (i[0], i[0])
    if b in startfiles:
      continue
    startfiles[b] = i[3]

  trans = []
  last = win[0].state
  for i, w in enumerate(win[1:]):
    if w.state != last:
      trans.append(((last, w.state), i))
      last = w.state


  for i in  sorted(midpt, key=lambda x: x[1], reverse=True):
    b = (i[0], i[0])
    if b in startfiles:
      continue
    startfiles[b] = i[3]

def seedData(catalog):
  with open('seeddata.json') as src:
    seedlist = json.loads(src.read())

  for s in seedlist:
    key =  tuple(s.keys())[0]
    catalog.hmset(key, s[key])
    catalog.hdel(key, 'indexList')
    catalog.hdel(key, 'actualBin')


def seedJob(catalog, num=1):
  """
  Seeds jobs into the JCQueue -- pulled from DEShaw
  """
  numLabels  = int(catalog.get('numLabels').decode())


  #  1 Loop to create pipeline
  pipe = catalog.pipeline()  
  logging.info("Getting candidate pools")
  for i in range(numLabels):
    for j in range(numLabels):
      pipe.lrange(candidPoolKey(i, j), 0, -1)
  pools = pipe.execute()

  idx = 0
  for i in range(numLabels):
    for j in range(numLabels):
      for k in range(num):
        src, frame = choice(pools[idx]).decode().split(':')
        logging.debug("(%d,%d) Selected: BPTI %s, frame: %s", i, j, src, frame)
        pdbfile, archiveFile = getHistoricalTrajectory(int(src))

        # Generate new set of params/coords
        jcID, params = generateNewJC(archiveFile, pdbfile, DEFAULT.TOPO, DEFAULT.PARM, frame=int(frame))

        # Update Additional JC Params and Decision History, as needed
        config = dict(params,
            name    = jcID,
            runtime = DEFAULT.RUNTIME,
            temp    = 310,
            state   = i,
            weight  = 0.,
            timestep = 0,
            interval = 500,
            gc      = 1,
            application   = DEFAULT.APPL_LABEL,
            converge = 0.,
            targetBin  = str((i,j)))
        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in config.items():
          logging.debug("   %s:  %s", k, str(v))
        catalog.rpush('JCQueue', jcID)
        catalog.hmset(wrapKey('jc', jcID), config)
        idx += 1      


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--conf', default='default.conf')
  parser.add_argument('--initarchive', action='store_true')
  parser.add_argument('--initcatalog', action='store_true')
  parser.add_argument('--initcontrol', action='store_true')
  parser.add_argument('--seedjob', action='store_true')
  parser.add_argument('--seeddata', action='store_true')
  parser.add_argument('--reindex', nargs='?', const=10, type=int)
  parser.add_argument('--loadindex', type=int)
  parser.add_argument('--num', type=int, default=50)
  parser.add_argument('--winsize', type=int, default=200)
  parser.add_argument('--slide', type=int, default=100)
  args = parser.parse_args()


  confile = args.conf
  DEFAULT = systemsettings()
  DEFAULT.applyConfig(confile)
  catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)

  if args.initcatalog:
    DEFAULT.envSetup()
    init_catalog(catalog)

  if args.initarchive:
    archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    logging.warning("DON'T DO THIS!")
    # init_archive(archive)
    sys.exit(0)

  if args.loadindex is not None:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    index_DEShaw(catalog, archive, args.loadindex, args.num, args.winsize, args.slide)
    sys.exit(0)

  if args.reindex:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    reindex(archive, args.reindex)


  if args.initcontrol:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    init_control(catalog)

  if args.seedjob:
    seedJob(catalog)


  if args.seeddata:
    seedData(catalog)

  logging.debug("Stopping the catalog.")
  catalog.stop()
  if os.path.exists('catalog.lock'):
    os.remove('catalog.lock')


