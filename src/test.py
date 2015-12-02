from initialize import *


def getDEShawIndex(archive, num, frame=400, winsize=200):

  indexSize = int(archive.get('indexSize').decode())
  numpc = int(archive.get('num_pc').decode())
  traj = loadDEShawTraj(num)
  logging.debug('Trajectory Loaded %s', str(traj))
  eg, ev = LA.eigh(distmatrix(traj.xyz[frame:frame+winsize]))

  logging.debug('%s', eg[:10])

  index = makeIndex(eg, ev, num_pc=numpc)
  logging.debug('Index created:  Size=%d', indexSize)
  engine = getNearpyEngine(archive, indexSize)

  logging.debug("Probing:")
  neigh = engine.neighbours(index)
  if len(neigh) == 0:
    logging.info ("Found no near neighbors for %s", key)
  else:
    logging.debug("Found %d neighbors:", len(neigh))
    logging.debug(" SELF PROBE CHECK:   %04d:%d vs %s", num, frame, neigh[0][1])
    #  Track the weighted count (a.k.a. cluster) for this index's nearest neighbors
    clust = np.zeros(5)
    count = np.zeros(5)
    for n in neigh[1:]:
      nnkey = n[1]
      distance = n[2]
      logging.info(  '%s, %f', nnkey, distance)
      # trajectory, seqNum = nnkey.split(':')
      # nn_state = labels[int(trajectory)].state
      # logging.info ("    NN:  %s   dist = %f    state=%d", nnkey, distance, nn_state)
      # count[nn_state] += 1
      # clust[nn_state] += abs(1/distance)
    # Classify this index with a label based on the highest weighted observed label among neighbors
    state = np.argmax(clust)
    win = loadLabels()
    logging.info("STATE LABEL:   %d", win[num].state)
    logging.info("PROBE GOT:     %d", state)
    logging.info("Neigh count:  %s", str(count))
    logging.info("Neigh clust:  %s", str(clust))





def showMatrices():

    # CONVERGENCE CALCUALTION  -------------------------------------
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
  parser.add_argument('--num', type=int, default=10)
  args = parser.parse_args()


  if args.testindex is not None:
    archive = redisCatalog.dataStore(**archiveConfig)
    getDEShawIndex(archive, args.testindex)
    sys.exit(0)