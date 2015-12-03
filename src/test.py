from initialize import *
import datetime as dt

timediff = lambda x, y: (y-x).seconds + (.001)*((y-x).microseconds//1000)
ddump = lambda x: print('\n'.join([' %-10s: %s' % (k,v) for k,v in sorted(x.items())]))



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

  skip = 10

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


  logging.info("Results:")
  logging.info("  Total   :    %4d", total)
  logging.info("  Hit     :    %4d  (%2.1f%%)", hit, 100*hit/total)
  logging.info("  NearMiss:    %4d  (%2.1f%%)", nearmiss, 100*nearmiss/total)
  logging.info("  Miss    :    %4d  (%2.1f%%)", miss, 100*miss/total)
  logging.info("")
  logging.info("Benchmarking:")
  logging.info("  MDLoad:     %3.2f sec,  (%0.2f idx/sec)", loadtime, loadtime/total)
  logging.info("  Eigen :     %3.2f sec,  (%0.2f idx/sec)", eigentime, eigentime/total)
  logging.info("  Probe :     %3.2f sec,  (%0.2f idx/sec)", probetime, probetime/total)
  logging.info("")
  for k, v in sorted(bucket.items()):
    logging.info(" %s  %4d", str(k), v)




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
  parser.add_argument('--benchmark', action='store_true')
  parser.add_argument('--num', type=int, default=10)
  args = parser.parse_args()

  DEFAULT = systemsettings()
  DEFAULT.applyConfig('default.conf')

  archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)

  if args.testindex is not None:
    getDEShawIndex(archive, args.testindex)
    sys.exit(0)

  if args.benchmark:
    benchmarkDEShaw(archive)
    sys.exit(0)