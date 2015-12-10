from initialize import *
from anlmd import *
from ctlmd import *
from deshaw import *
from indexer import *

import datetime as dt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


timediff = lambda x, y: (y-x).seconds + (.001)*((y-x).microseconds//1000)
ddump = lambda x: print('\n'.join([' %-10s: %s' % (k,v) for k,v in sorted(x.items())]))

conform =namedtuple('conform', 'state pca X')

def savePCAPoints(archive, start):
  indexlist = getindexlist(archive, DEFAULT.HASH_NAME)
  index = [[None for j in range(10)] for i in range(4125)]
  for idx in indexlist:
      state = int(idx[1][0])
      traj, seqnum = idx[1][2:].split('-')
      pca = idx[0].reshape(3, 454)
      index[int(traj)][int(seqnum)//100] = conform(state, pca, np.zeros(3))

  logging.debug('Loading DEShaw Trajectories')
  traj = loadDEShawTraj(start, start+100)
  with open('points/pca.%d.data'%start, 'w') as ptfile:
    for dcd in range(100):
      logging.debug("Calulating points for file %d", (start + dcd))
      for win in range(10):
        dmat = distmatrix(traj.xyz[dcd*1000+win*100:dcd*1000+win*100+200])
        x = np.mean(dmat.dot(index[1448][4].pca[0]), axis=0)
        y = np.mean(dmat.dot(index[3826][4].pca[1]), axis=0)
        z = np.mean(dmat.dot(index[1135][4].pca[2]), axis=0)
        ptfile.write('%d,%d,%d,%f,%f,%f\n' % ((start+dcd), win*100, index[dcd][win].state, x, y, z))

# index[1448][400].pca[0]
# index[3826][400].pca[1]
# index[1135][400].pca[2]


def makePlot():
  labels = loadLabels()
  pts = []
  for i in range(0, 4100, 100):
    with open('points/pca.%d.data'%i) as src:
      pts.extend(src.read().strip().split('\n'))

  X = np.zeros(shape=(len(pts), 3))
  state = []
  for i in range(len(pts)):
    if len(pts[i]) == 0:
      print('BAD')
      continue
    p = pts[i].split(',')
    print (p, labels[int(p[0])].state)
    state.append(labels[int(p[0])].state)
    X[i][0] = float(p[3])
    X[i][1] = float(p[4])
    X[i][2] = float(p[5])

  color = ['red', 'green', 'blue', 'cyan', 'black']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i, x in enumerate(X):
      ax.scatter(x[0], x[1], x[2], c=color[state[i]])

  ax.set_xlabel('PC0')
  ax.set_ylabel('PC1')
  ax.set_zlabel('PC2')
  plt.savefig('plot.png')
  plt.close(fig)





def getDEShawIndex(archive, num, frame=400, winsize=200):

  start = dt.datetime.now()
  indexSize = int(archive.get('indexSize').decode())
  numpc = int(archive.get('num_pc').decode())
  ts_index = dt.datetime.now()
  traj = loadDEShawTraj(num)
  ts_load = dt.datetime.now()
  logging.debug('Trajectory Loaded %s', str(traj))
  eg, ev = LA.eigh(distmatrix(traj.xyz[frame:frame+winsize]))
  ts_eigen = dt.datetime.now()

  logging.debug('%s', eg[:10])

  index = makeIndex(eg, ev, num_pc=numpc)
  logging.debug('Index created:  Size=%d', indexSize)
  engine = getNearpyEngine(archive, indexSize)

  logging.debug("Probing:")
  ts_nearpy = dt.datetime.now()
  neigh = engine.neighbours(index)
  ts_probe = dt.datetime.now()
  if len(neigh) == 0:
    logging.info ("Found no near neighbors for %s", str(num))
  else:
    logging.debug("Found %d neighbors:", len(neigh))
    logging.debug(" SELF PROBE CHECK:   %04d:%d vs %s", num, frame, neigh[0][1])
    #  Track the weighted count (a.k.a. cluster) for this index's nearest neighbors
    clust = np.zeros(5)
    count = np.zeros(5)
    for n in neigh[1:]:
      nnkey = n[1]
      distance = n[2]
      nn_state = int(nnkey[0])
      logging.info(  '%s, %f', nnkey, distance)
      count[nn_state] += 1
      clust[nn_state] += abs(1/distance)
    # Classify this index with a label based on the highest weighted observed label among neighbors
    state = np.argmax(clust)
    win = loadLabels()
    logging.info("STATE LABEL:   %d", win[num].state)
    logging.info("PROBE GOT:     %d", state)
    logging.info("Neigh count:  %s", str(count))
    logging.info("Neigh clust:  %s", str(clust))
    logging.info("Benchmarking:")
    logging.info("  Redis :     %0.3f", timediff(start, ts_index))
    logging.info("  MDLoad:     %0.3f", timediff(ts_index, ts_load))
    logging.info("  Eigen :     %0.3f", timediff(ts_load, ts_eigen))
    logging.info("  Probe :     %0.3f", timediff(ts_nearpy, ts_probe))



def benchmarkDEShaw(archive, theta=.6, winsize=200):

  logging.basicConfig(format='%(message)s', level=logging.DEBUG)


  indexSize = int(archive.get('indexSize').decode())
  numpc = int(archive.get('num_pc').decode())
  engine = getNearpyEngine(archive, indexSize)
  win = loadLabels()

  frame = 400

  skip = 5

  loadtime = 0.
  eigentime = 0.
  probetime = 0.

  bucket = {(i,j): 0 for i in range (5) for j in range(5)}
  total = 0
  hit = 0
  nearmiss = 0
  miss = 0
  actualIndex = {}

  for num in range(0, 4100):

    try:
      if num % skip == 0:
        total += 1
        start = dt.datetime.now()
        traj = loadDEShawTraj(num)
        ts_load = dt.datetime.now()
        eg, ev = LA.eigh(distmatrix(traj.xyz[frame:frame+winsize]))
        index = makeIndex(eg, ev, num_pc=numpc)
        ts_eigen = dt.datetime.now()
        neigh = engine.neighbours(index)
        ts_probe = dt.datetime.now()
        if len(neigh) == 0:
          miss += 1
          continue
        else:
          clust_i = np.zeros(5)
          count_i = np.zeros(5)
          for n in neigh[1:]:
            nnkey = n[1]
            distance = n[2]
            nn_state = int(nnkey[0])
            count_i[nn_state] += 1
            clust_i[nn_state] += abs(1/distance)

          clust_i = clust_i/np.sum(clust_i)
          order = np.argsort(clust_i)[::-1]
          A = order[0]
          B = A if clust_i[A] > theta else order[1]
          bucket[(A, B)] += 1

          if win[num].state == A:
            hit += 1
          elif win[num].state == B:
            nearmiss += 1
          else:
            miss += 1

          loadtime  += timediff(start, ts_load)
          eigentime += timediff(ts_load, ts_eigen)
          probetime += timediff(ts_eigen, ts_probe)
    except KeyboardInterrupt as ex:
      print ("Halting Benchmark...\n")
      break

  logging.info("\nBenchmark Complete")
  logging.info("Results:")
  logging.info("  Total                   :    %4d", total)
  logging.info("  Hit (labeled state)     :    %4d  (%2.1f%%)", hit, 100*hit/total)
  logging.info("  NearMiss (detect x-tion):    %4d  (%2.1f%%)", nearmiss, 100*nearmiss/total)
  logging.info("  Miss                    :    %4d  (%2.1f%%)", miss, 100*miss/total)
  logging.info("")
  logging.info("Timing Benchmarking:")
  logging.info("  MDLoad:     %3.2f sec,  (%0.2f idx/sec)", loadtime, loadtime/total)
  logging.info("  Eigen :     %3.2f sec,  (%0.2f idx/sec)", eigentime, eigentime/total)
  logging.info("  Probe :     %3.2f sec,  (%0.2f idx/sec)", probetime, probetime/total)
  logging.info("")
  for k, v in sorted(bucket.items()):
    logging.info(" %s  %4d", str(k), v)



def checkarchive(archive, state=-1):
  theta = .6
  indexsize = int(archive.get('indexSize').decode())
  redis_storage = RedisStorage(archive)
  hashkeys = [i.decode().split('_')[-1] for i in archive.keys('nearpy_'+DEFAULT.HASH_NAME+'_*')]
  indexlist = []
  logging.debug("Pulling  keys")
  for h in hashkeys:
    bucket = redis_storage.get_bucket(DEFAULT.HASH_NAME, h)
    for b in bucket:
      indexlist.append(b)
  logging.debug("Pulled %d keys", len(indexlist))
  count = np.zeros(5)
  deshaw = 0
  nondeshaw = 0
  for idx in indexlist:
    count[int(idx[1][0])] += 1
    if len(idx[1]) == 10:
      deshaw += 1
    else:
      nondeshaw += 1
  logging.debug("DEshaw Indices:  %d     Others:  %d", deshaw, nondeshaw)
  logging.debug("Index Count per state: %s", str(count))
  engine = getNearpyEngine(archive, indexsize)
  hit_cl =0
  miss_cl = 0
  trans_cl = 0
  hit_nn =0
  miss_nn = 0
  trans_nn = 0
  total = 0
  badindex = 0
  badindexcnt = np.zeros(5)
  logging.debug("Checking state: %d", state)
  newindex = []
  statecount = np.zeros(5)
  bincount = np.zeros(shape=(5,5))
  logging.debug('idx:  SelfChk: NN CL CNT  LabNN  LabCL    Counts')
  maxcounts = np.zeros(5)

  for x in range(6, len(indexlist), ):
    if total == 500:
      break
    idx = indexlist[x]
    idx_state = int(idx[1][0])
    maxcounts[idx_state] += 1
    if (state > 0 and idx_state != state) or (state == -1 and maxcounts[idx_state] > 100):
      # print(idx[1][0])
      continue
    total += 1
    neigh = engine.neighbours(idx[0])
    if len(neigh) == 0:
      miss_cl += 1
      miss_nn += 1
      continue
    else:
      clust_i = np.zeros(5)
      count_i = np.zeros(5)
      for n in neigh[1:]:
        nnkey = n[1]
        distance = n[2]
        nn_state = int(nnkey[0])
        count_i[nn_state] += 1
        clust_i[nn_state] += abs(1/distance)

      # Cluster Heuristic
      clust_i = clust_i/np.sum(clust_i)
      order = np.argsort(clust_i)[::-1]
      mostclustered = order[0]
      nextclustered = mostclustered if clust_i[mostclustered] > theta else order[1]
      if idx_state == mostclustered:
        hit_cl += 1
      elif idx_state == nextclustered:
        trans_cl += 1
      else:
        miss_cl += 1

      ord_cnt = np.argsort(count_i)[::-1]
      if count_i[ord_cnt[0]] > (len(neigh))//2:
        highcnt = ord_cnt[0]
      elif count_i[ord_cnt[0]] == count_i[idx_state]:
        highcnt = idx_state
      else:
        highcnt = ord_cnt[0]

      # NN Heuristic
      nn_A = int(neigh[0][1][0])
      nn_B = int(neigh[1][1][0])
      if idx_state == nn_B:
        hit_nn += 1
      elif count_i[idx_state] > 0:
        trans_nn += 1
      else:
        miss_nn += 1

      asterisk = ''
      labelA = idx_state
      if idx_state in [nn_B, highcnt, mostclustered]:
        labelB = idx_state
      else:
        badindexcnt[idx_state] += 1
        asterisk = '*'
        if nn_B == highcnt == mostclustered:
          labelB = nn_B
        elif nn_B == highcnt or nn_B == mostclustered:
          labelB = nn_B
        elif highcnt == mostclustered:
          labelB = highcnt
        else:
          labelB = nn_B
          asterisk = '****'
          badindex += 1



      refinedLabel = (labelA, labelB)
      statecount[idx_state] += 1
      bincount[labelA][labelB] += 1       
      logging.debug('%5d   %s-%s     %s  %d  %d  %s   %s %s', 
              x, idx[1][0], nn_A, nn_B, mostclustered, highcnt, str(refinedLabel), str(count_i), asterisk)
  logging.info("\nBenchmark Complete")
  logging.info("Results (Cluster Heuristic:")
  logging.info("  Total               :    %4d", total)
  logging.info("  Hit (labeled state) :    %4d  (%2.1f%%)", hit_cl, 100*hit_cl/total)
  logging.info("  Trans (detected)    :    %4d  (%2.1f%%)", trans_cl, 100*trans_cl/total)
  logging.info("  Miss                :    %4d  (%2.1f%%)", miss_cl, 100*miss_cl/total)
  logging.info("Results (NN Heuristic:")
  logging.info("  Total               :    %4d", total)
  logging.info("  In State Well       :    %4d  (%2.1f%%)", hit_nn, 100*hit_nn/total)
  logging.info("  Transition          :    %4d  (%2.1f%%)", trans_nn, 100*trans_nn/total)
  logging.info("  Miss                :    %4d  (%2.1f%%)", miss_nn, 100*miss_nn/total)
  logging.info("Very Bad Indices (divergent)      :    %4d", badindex)
  logging.info("MisLabeledDEShaw Indices  : %s  ", str(badindexcnt))
  logging.info("\nState Counts:%s", str(statecount))
  logging.info("\nBin Counts:\n%s", str(bincount))



def showMatrices():
  logging.debug("============================  <CONVEGENCE>  =============================")

  # Load Transition Matrix (& TODO: Historical index state labels)
  tmat = loadNPArray(self.catalog, 'transitionmatrix')
  if tmat is None:
    tmat = np.zeros(shape=(5,5))    # TODO: Move to init
  tmat_before = np.zeros(shape=tmat.shape)
  np.copyto(tmat_before, tmat)
  logging.debug("TMAT BEFORE\n" + str(tmat_before))

  # Load Selection Matrix
  smat = loadNPArray(self.catalog, 'selectionmatrix')
  if smat is None:
    smat = np.full((5,5), 1.)    # SEED Selection matrix (it cannot be 0) TODO: Move to init
  logging.debug("SMAT:\n" + str(smat))

  # Load Convergence Matrix
  cmat = loadNPArray(self.catalog, 'convergencematrix')
  if cmat is None:
    cmat = np.full((5,5), 0.04)    # TODO: Move to init
  logging.debug("CMAT:\n" + str(cmat))

  #  1. Load current fatigue values
  fatigue = loadNPArray(self.catalog, 'fatigue')   # TODO: Move to self.data(??) and/or abstract the NP load/save
  if fatigue is None:
    fatigue = np.full((5,5), 0.04)    # TODO: Move to init



  logging.debug("TMAT:\n" + str(tmat))
  logging.debug("Fatigue:\n" + str(fatigue))

  # Update global selection matrix
  smat += selectionTally
  logging.debug("SMAT:\n" + str(smat))

  # Remove bias from selection and add to observations 
  #  which gives the unbiased, uniform selection 
  inverseSelectFunc = lambda x: (np.max(x)) - x

  #  For now, Assume a 4:1 ratio (input:output) and do not factor in output prob distro
  unbias = 4 * inverseSelectFunc(smat) + tmat
  logging.debug("UNBIAS:\n" + str(unbias))

  # Calculcate "convergence" matrix as unbias(i,j) / sel(i.j)
  updated_cmat = unbias / np.sum(unbias)
  logging.debug("CMAT_0:\n" + str(cmat))
  logging.debug("CMAT_1:\n" + str(updated_cmat))


  # Calculate global convergence as summed difference:  cmat_t1 - cmat_t0
  convergence = 1 - (np.sum(abs(updated_cmat - cmat)))

  logging.info('\nLAST CONVERGENCE:  %0.6f', self.data['converge'])
  logging.info("NEW CONVERGENCE:  %0.6f\n", convergence)
  self.data['converge'] = convergence

  # Control Thread requires the catalog to be accessible. Hence it starts it:
  #  TODO: use addState HERE & UPDATE self.data['JCQueue']
  self.catalogPersistanceState = True
  self.localcatalogserver = self.catalog.conn()





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--testindex', type=int)
  parser.add_argument('--testarchive', nargs='?', const=-1)
  parser.add_argument('--benchmark', action='store_true')
  parser.add_argument('--makeplot', action='store_true')
  parser.add_argument('--num', type=int, default=10)
  parser.add_argument('--pcaplot', type=int)
  args = parser.parse_args()

  DEFAULT = systemsettings()
  DEFAULT.applyConfig('debug.conf')

  archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)

  if args.testindex is not None:
    getDEShawIndex(archive, args.testindex)
    sys.exit(0)

  if args.benchmark:
    benchmarkDEShaw(archive)
    sys.exit(0)

  if args.pcaplot is not None:
    savePCAPoints(archive, args.pcaplot)

  if args.makeplot:
    makePlot()

  if args.testarchive:
    checkarchive(archive, int(args.testarchive))
    sys.exit(0)