import argparse
import sys
import os
import sys
import json

import mdtraj as md
import numpy as np
np.set_printoptions(precision=5, suppress=True)

from numpy import linalg as LA
import nearpy
from nearpy.storage.storage_redis import RedisStorage
from nearpy.hashes import RandomBinaryProjections, PCABinaryProjections

from collections import namedtuple, deque

import redisCatalog
from common import *
from macrothread import macrothread
from slurm import slurm
from random import choice, randint

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)



def psfgen(params):
  return '''psfgen << ENDMOL

# 1. Load Topology File
topology %(topo)s

# 2. Load Protein
segment BPTI {pdb %(coord)s}

# 3. Patch protein segment
patch DISU BPTI:5 BPTI:55
patch DISU BPTI:14 BPTI:38
patch DISU BPTI:30 BPTI:51

# 4. Define aliases
pdbalias atom ILE CD1 CD ;
pdbalias atom ALA H HN ;
#pdbalias atom ALA OXT O ;
pdbalias atom ARG H HN ;
#pdbalias atom ARG H2 HN;
#pdbalias atom ARG H3 HN;
pdbalias atom ARG HB3 HB1 ;
pdbalias atom ARG HD3 HD1 ;
pdbalias atom ARG HG3 HG1 ;
pdbalias atom ASN H HN ;
pdbalias atom ASN HB3 HB1 ;
pdbalias atom ASP H HN ;
pdbalias atom ASP HB3 HB1 ;
pdbalias atom CYS H HN ;
pdbalias atom CYS HB3 HB1 ;
pdbalias atom GLN H HN ;
pdbalias atom GLN HB3 HB1 ;
pdbalias atom GLN HG3 HG1 ;
pdbalias atom GLU H HN ;
pdbalias atom GLU HB3 HB1 ;
pdbalias atom GLU HG3 HG1 ;
pdbalias atom GLY H HN ;
pdbalias atom GLY HA3 HA1 ;
pdbalias atom ILE H HN ;
pdbalias atom ILE HD11 HD1 ;
pdbalias atom ILE HD12 HD2 ;
pdbalias atom ILE HD13 HD3 ;
pdbalias atom ILE HG13 HG11 ;
pdbalias atom LEU H HN ;
pdbalias atom LEU HB3 HB1 ;
pdbalias atom LYS H HN ;
pdbalias atom LYS HB3 HB1 ;
pdbalias atom LYS HD3 HD1 ;
pdbalias atom LYS HE3 HE1 ;
pdbalias atom LYS HG3 HG1 ;
pdbalias atom MET H HN ;
pdbalias atom MET HB3 HB1 ;
pdbalias atom MET HG3 HG1 ;
pdbalias atom PHE H HN ;
pdbalias atom PHE HB3 HB1 ;
pdbalias atom PRO HB3 HB1 ;
pdbalias atom PRO HD3 HD1 ;
pdbalias atom PRO HG3 HG1 ;
pdbalias atom SER H HN ;
pdbalias atom SER HB3 HB1 ;
pdbalias atom SER HG HG1 ;
pdbalias atom THR H HN ;
pdbalias atom TYR H HN ;
pdbalias atom TYR HB3 HB1 ;
pdbalias atom VAL H HN ;

# 5. Read protein coordinates from PDB file & set coords
coordpdb %(coord)s BPTI
guesscoord

# 6. Output psf/pdb files
writepsf %(psf)s
writepdb %(pdb)s

ENDMOL''' % params


candidPoolKey = lambda x, y: 'candidatePool_%d_%d' % (x, y)



def storeNPArray(store, arr, key):
  #  Force numpy version 1.0 formatting
  header = {'shape': arr.shape,
            'fortran_order': arr.flags['F_CONTIGUOUS'],
            'dtype': np.lib.format.dtype_to_descr(np.dtype(arr.dtype))}
  store.hmset(key, {'header': json.dumps(header), 'data': bytes(arr)})

def loadNPArray(store, key):
  elm = {k.decode(): v for k, v in store.hgetall(key).items()}
  if elm == {}:
    return None
  header = json.loads(elm['header'].decode())
  arr = np.fromstring(elm['data'], dtype=header['dtype'])
  return arr.reshape(header['shape'])


def get2DKeys(key, X, Y):
  return ['key_%d_%d' % (x, y) for x in range(X) for y in range(Y)]

def generateNewJC(rawfile, pdbfile):

    logging.debug("Generating new simulation coordinates from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()
    # jcuid = 'DEBUG'

    # Write out coords (TODO: should this go to catalog or to file?)
    # tmpCoord = os.path.join(DEFAULT.COORD_FILE_DIR, '%s_tmp.pdb' % jcuid)
    jobdir = os.path.join(DEFAULT.JOB_DIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)

    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    # Retrieve referenced file from storage
    #   TODO: Set up historical archive for retrieval (this may need to be imported)
    #   TODO: Use load_frame instead of loading entire trajectory

    #  Load in Historical Referenced trajectory file, filter out proteins & slice
    traj  = md.load(rawfile, top=pdbfile)
    traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)
    
    #  For now, pick a random frame from this trajectory
    #  TODO:  ID specific window reference point
    frame = randint(0, traj.n_frames)
    coord = traj.slice(frame)

    logging.debug("  Source trajectory: %s   (frame # %d)", str(coord), frame)

    # Save this as a temp file to set up simulation input file
    coord.save_pdb(coordFile)

    newsimJob = dict(workdir=jobdir,
        coord = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = DEFAULT.TOPO,
        parm    = DEFAULT.PARM)

    logging.info("  Running PSFGen to set up simulation pdf/pdb files.")
    stdout = executecmd(psfgen(newsimJob))
    logging.debug("  PSFGen COMPLETE!!\n")

    os.remove(coordFile)
    del newsimJob['coord']

    return jcuid, newsimJob




class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('LDIndexList', 'JCQueue')
      self.setState('indexSize', 'converge', 'ctlSplitParam', *tuple([candidPoolKey(i,j) for i in range(5) for j in range(5)]))
      self.modules.add('namd')


    def term(self):
      # For now
      return False

    def split(self):

      catalog = self.getCatalog()

      # TODO:  Provide better organization/sorting of the input queue based on weights
      # For now: just take the top N
      split = self.data['ctlSplitParam']
      immed = self.data['LDIndexList'][:split]
      return immed,split

    def fetch(self, i):
      return {k.decode():np.fromstring(v, dtype=np.float64) for k, v in self.catalog.hgetall(wrapKey('idx', i)).items()}
      

    def execute(self, ld_index):
      logging.debug('CTL MT')

      logging.debug("============================  <PRE-PROCESS>  =============================")
      # TODO:  Treat Archive as an overlay service. For now, wrap inside here and connect to it
      archive = redisCatalog.dataStore(**archiveConfig)
      redis_storage = RedisStorage(archive)
      config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
      if not config:
        logging.error("LSHash not configured")
        #TODO: Gracefully exit
        return []

      # Create empty lshash and load stored hash
      lshash = DEFAULT.getEmptyHash()
      lshash.apply_config(config)
      indexSize = DEFAULT.NUM_VAR * DEFAULT.NUM_PCOMP
      logging.debug("INDEX SIZE = %d:  ", indexSize)
      engine = nearpy.Engine(indexSize, 
            lshashes=[lshash], 
            storage=redis_storage)

      # Load current set of known states
      #  TODO:  Injection Point for clustering. If clustering were applied
      #    this would be much more dynamic (static fileload for now)
      labels = loadLabels(DEFAULT.DATA_LABEL_FILE)
      labelNames = getLabelList(labels)
      numLabels = len(labelNames)

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

 
      # PROBE   ------------------------
      logging.debug("============================  <PROBE>  =============================")

      # Set initial params for index calculations
      prevState = -1    # To track each state transition
      prevTrajectory = None   # To check for unique trajectories
      decisionHistory = {}    # Holds Decision History data from source JC used to create the data
      observationDistribution = {}   #  distribution of observed states (for each input trajectory)
      observationCount = {}
      # observationSet = set()  # To track the # of unique observations (for each input trajectory)

      # NOTE: ld_index is a list of indexed trajectories. They may or may NOT be from
      #   the same simulation (allows for grouping of multiple downstream data into 
      #   one control task). Thus, this algorithm tracks subsequent windows within one
      #   trajectories IOT track state changes. 

      for key in sorted(ld_index.keys()):

        index = np.array(ld_index[key])   # Get actual Index for this window
        sourceJC, frame = key.split(':')  # Assume colon separator
        # logging.info(' Index Loaded from %s:   Shape=%s,  Type=%s', sourceJC, str(index.shape), str(index.dtype))

        # Get Decision History for the index IF its a new index not previously processed
        #  and initialize observation distribution to zeros
        if sourceJC != prevTrajectory:
          decisionHistory[sourceJC] = {}
          observationDistribution[sourceJC] = np.zeros(5)
          observationCount[sourceJC] = np.zeros(5)

          #  TODO: Load indices up front with source history
          self.catalog.load({wrapKey('jc', sourceJC): decisionHistory[sourceJC]})  #TODO: Load these up front
          if 'state' not in decisionHistory[sourceJC]:
            prevState = None
            logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
          else:
            prevState = int(decisionHistory[sourceJC]['state'])
            logging.debug("New Index to analyze, %s: Source JC was supposed to start in state %d", sourceJC, prevState)
          prevTrajectory = sourceJC
          self.addToState(wrapKey('jc', sourceJC), decisionHistory[sourceJC])

          # Note:  Other Decision History is loaded here

        logging.info("Probing `%s` window at frame # %s  (state %s)", sourceJC, frame, str(prevState))

        # Probe historical index  -- for now only probing DEShaw index
        neigh = engine.neighbours(index)
        if len(neigh) == 0:
          logging.info ("Found no near neighbors for %s", key)
        else:
          logging.debug ("  Found %d neighbours:", len(neigh))

          #  Track the weighted count (a.k.a. cluster) for this index's nearest neighbors
          clust = np.zeros(5)
          count = np.zeros(5)
          for n in neigh:
            nnkey = n[1]
            distance = n[2]
            trajectory, seqNum = nnkey[2:].split('-')
            nn_state = labels[int(trajectory)].state
            # logging.info ("    NN:  %s   dist = %f    state=%d", nnkey, distance, nn_state)
            count[nn_state] += 1
            clust[nn_state] += abs(1/distance)

            # Add total to the original input trajectory (for historical and feedback decisions)
            observationCount[sourceJC][nn_state] += 1
            observationDistribution[sourceJC][nn_state] += abs(1/distance)

          # Classify this index with a label based on the highest weighted observed label among neighbors
          state = np.argmax(clust)

          # Increment the transition matrix
          if prevState == None:
            prevState = state
          logging.debug("  Transition %d  --->  %d    Incrementing transition counter (%d, %d)", prevState, state, prevState, state)
          logging.debug("  Clustered at: %s", str(clust))
          logging.debug("  Counts:       %s", str(count))

          # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
          tmat[prevState][state] += 1
          prevState = state

      # Build Decision History Data for the Source JC's from the indices
      # transitionBins = kv2DArray(archive, 'transitionBins', mag=5, dtype=str, init=[])      # Should this load here (from archive) or with other state data from catalog?


      logging.debug("Update TMAT:\n" + str(tmat))

      #  Theta is calculated as the probability of staying in 1 state (from observed data)
      theta = .6  #self.data['observation_counts'][1] / sum(self.data['observation_counts'])
      logging.debug("  THETA  = %0.3f", theta)


      #  DECISION HISTORY  --------------------------
      logging.debug("============================  <DECISION HIST>  =============================")

      #  Process output data for each unque input trajectory (as produced by a single simulation)
      for sourceJC, cluster in observationDistribution.items():
        logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point here for better classification)", sourceJC)
        if sum(observationDistribution[sourceJC]) == 0:
          logging.debug(" No observed data for, %s", sourceJC)
          continue

        #  TODO: Another injection point for better classification. Here, the classification is for the input trajectory
        #     as a whole for future job candidate selection. 

        logging.debug("Observation Counts  for input index, %s\n  %s", sourceJC, str(observationCount[sourceJC]))
        logging.debug("Observation weights for input index, %s\n  %s", sourceJC, str(cluster))
        index_distro = cluster / sum(cluster)
        # logging.debug("Observations for input index, %s\n  %s", sourceJC, str(distro))
        sortedLabels = np.argsort(index_distro)[::-1]    # Should this be normaized to check theta????? or is theta a global calc?
        stateA = sortedLabels[0]

        # Source Trajectory spent most of its time in 1 state
        if max(index_distro) > theta: 
          logging.debug(" Trajectory `%s`  classified as staying in state :  %d", sourceJC, stateA)
          stateB = stateA

        # Observation showed some transition 
        else:
          stateB = sortedLabels[1]
          logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", sourceJC, stateA, stateB)

        #  Add this candidate to list of potential pools:
        #  TODO: Cap candidate pool size
        outputBin = str((stateA, stateB))
        # self.data[candidPoolKey(stateA, stateB)].append(sourceJC)  
        if 'targetBin' in self.data[wrapKey('jc', sourceJC)]:
          inputBin  = self.data[wrapKey('jc', sourceJC)]['targetBin']
          logging.info("  Predicted Taget Bin was       :  %s", inputBin) 
          logging.info("  Actual Trajectory classified  :  %s", outputBin) 
          logging.debug("    TODO: ID difference, update weights, etc..... (if desired)")

        self.data[wrapKey('jc', sourceJC)]['actualBin'] = outputBin

      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's

      #  WEIGHT CALCULATION ---------------------------
      logging.debug("============================  <WEIGHT CALC>  =============================")

      # Weight Calculation 

      bins = [(x, y) for x in range(numLabels) for y in range(numLabels)]
      # TODO:   Load New Transition Matrix  if consistency is necessary, otherwise use locally updated tmat
 
      #  2. Calculate new "preference"  targetbin portion of the weight
      totalObservations = np.sum(tmat)
      tmat_distro = {(i, j): tmat[i][j]/totalObservations for i in range(numLabels) for j in range(numLabels)}
      quota = max(totalObservations / len(tmat_distro.keys()), len(tmat_distro.keys()))
      
      pref = {(i, j): max((quota-tmat[i][j])/quota, 1/quota) for i in range(numLabels) for j in range(numLabels)}

      logging.debug("CURRENT (BIASED) OUTPUT DISTRIBUTION:")
      logging.debug("  Total observations = %d", totalObservations)
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% tmat_distro[(i,k)] for k in range(5)]))


      logging.debug("PREFERENCE WEIGHTS:")
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% pref[(i,k)] for k in range(5)]))

      #  3. Apply constancs. This can be user influenced
      alpha = self.data['weight_alpha']
      beta = self.data['weight_beta']

      #  4. Set new weight and order from high to low
      weight = {}
      for k in bins:
        weight[k] =  alpha * pref[k] + beta * (1 - fatigue[k])

      logging.debug("UPDATED WEIGHTS:")
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% weight[(i,k)] for k in range(5)]))

      updatedWeights = sorted(weight.items(), key=lambda x: x[1], reverse=True)


      #  5. Load JC Queue and all items within to get respective weights and projected target bins
      #   TODO:  Pipeline this or load all up front!
      curqueue = []
      for i in self.data['JCQueue']:
        key = wrapKey('jc', i)
        curqueue.append(self.catalog.load())

      #  5a. (PreProcess current queue) in case weights were never set
      for jc in range(len(curqueue)):
        if 'weight' not in curqueue[jc]:
          curqueue[jc]['weight'] = 1.0

      #  6. Sort current queue
      existingQueue = deque(sorted(curqueue, key=lambda x: x['weight'], reverse=True))

      #  7. Det. potential set of  new jobs  (base on launch policy)
      #     TODO: Set up as multiple jobs per bin, cap on a per-control task basis, or just 1 per bin
      potentialJobs = deque(updatedWeights)

      #  7. Prepare a new queue (for output)
      jcqueue = deque()

      targetBin = potentialJobs.popleft()
      oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()
      selectionTally = np.zeros(shape=(numLabels, numLabels))
      newJobCandidate = {}

      #  SCHEDULING   -----------------------------------------
      logging.debug("============================  <SCHEDULING>  =============================")

      while len(jcqueue) < DEFAULT.MAX_JOBS_IN_QUEUE:

        if oldjob == None and targetBin == None:
          logging.info("No more jobs to queue.")
          break

        if (targetBin == None) or (oldjob and oldjob['weight'] > targetBin[1]):
          jcqueue.append(oldjob)
          oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()

        else:
          A, B = targetBin[0]
          logging.debug("\n\nCONTROL: Target transition bin:  %s", str((A, B)))

          # Identify a pool of starting points
          candidatePool = self.data[candidPoolKey(A, B)]


          #  TODO:  Bin J.C. Selection into a set of potential new J.C. params/coords
          #    can either be based on historical or a sub-state selection
          #    This is where the infinite # of J.C. is bounded to something manageable
          #     FOR NOW:  Pick a completely random DEShaw Window of the same state
          #       and go withit
          # Ensure there are candidates to pick from
          if len(candidatePool) == 0:
            logging.info("No DEShaw reference for transition, (%d, %d)  -- checking reverse direction", A, B)

            # Flip direction of transition (only good if we're assuming transitions are non-symetric)
            candidatePool = self.data[candidPoolKey(B, A)]

            if len(candidatePool) == 0:
              logging.info("No DEShaw reference for transition, (%d, %d)  -- checking all bins starting from state %d", B, A, A)
              candidatePool = []

              # Finally, pick any start point from the initial state (assume that state had a candidate)
              for z in range(5):
                candidatePool.extend(self.data[candidPoolKey(A, z)])

          logging.debug('Prime CANDIDPOOLKEY = %s', candidPoolKey(A, B))
          logging.debug('final CANDIDPOOL    = %s', str(candidatePool))

          # selectedBins.append((A, B))

          # Pick a random trajectory from the bin
          sourceTraj = choice(candidatePool)
          logging.debug("Selected DEShaw Trajectory # %s based on state %d", sourceTraj, A)


          # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
          #   or we have a pre-fetch algorithm to get the data
          # Back-project  <--- Move to separate Function tied to decision history
          # For now:
          if isinstance(sourceTraj, int) or sourceTraj.isdigit():      # It's a DEShaw file
            fname = 'bpti-all-%03d.dcd' if int(sourceTraj) < 1000 else 'bpti-all-%04d.dcd'
            archiveFile = os.path.join(DEFAULT.RAW_ARCHIVE, fname % int(sourceTraj))
            pdbfile = DEFAULT.PDB_FILE
          else:
            archiveFile = os.path.join(DEFAULT.JOB_DIR, sourceTraj, '%s.dcd' % sourceTraj)
            pdbfile     = os.path.join(DEFAULT.JOB_DIR, sourceTraj, '%s.pdb' % sourceTraj)

          # Generate new set of params/coords
          jcID, params = generateNewJC(archiveFile, pdbfile)

          # Update Additional JC Params and Decision History, as needed
          jcConfig = dict(params,
              name    = jcID,
              runtime = 51000,
              temp    = 310,
              state   = A,
              weight  = targetBin[1],
              targetBin  = str((A, B)))

          logging.info("New Simulation Job Created: %s", jcID)
          for k, v in jcConfig.items():
            logging.debug("   %s:  %s", k, str(v))

          #  Add to the output queue & save config info
          jcqueue.append(jcID)
          newJobCandidate[jcID] = jcConfig
          selectionTally[A][B] += 1
          logging.info("New Job Candidate Complete:  %s" % jcID)
          
          targetBin = None if len(potentialJobs) == 0 else potentialJobs.popleft()

      self.data['JCQueue'] = list(jcqueue)
      for jcid, config in newJobCandidate.items():
        jckey = wrapKey('jc', jcid)
        self.catalog.save({jckey: config})


      # Updated the "fatigue" values for all selected jobs
      #    -- FOR NOW do it all at once after selection process
      #    TODO: This can be weighted based on # of selections
      for i in range(numLabels):
        for j in range(numLabels):
          if selectionTally[i][j] > 0:
            fatigue[i][j] += (1-fatigue[i][j])/25
            logging.debug("Increasing fatigue value for %d, %d to %f", i, j, fatigue[i][j])
          else:
            fatigue[i][j] -= (fatigue[i][j]/25)**2

      # CONVERGENCE CALCUALTION  -------------------------------------
      logging.debug("============================  <CONVEGENCE>  =============================")

      logging.debug("TMAT:\n" + str(tmat))
      logging.debug("Fatigue:\n" + str(fatigue))

      # Update global selection matrix
      smat += selectionTally
      logging.debug("SMAT:\n" + str(smat))

      # Remove bias from selection and add to observations 
      #  which gives the unbiased, uniform selection 
      inverseSelectFunc = lambda x: (np.max(x)) - x

      #  For now, Assume a 4:1 ratio (input:output) and do not factor in output prob distro
      unbias = 3 * inverseSelectFunc(smat) + tmat
      logging.debug("UNBIAS:\n" + str(unbias))

      # Calculcate "convergence" matrix as unbias(i,j) / sel(i.j)
      updated_cmat = unbias / np.sum(unbias)
      logging.debug("CMAT_0:\n" + str(cmat))
      logging.debug("CMAT_1:\n" + str(updated_cmat))


      # Calculate global convergence as summed difference:  cmat_t1 - cmat_t0
      convergence = np.sum(abs(updated_cmat - cmat))

      logging.info('\nLAST CONVERGENCE:  %0.6f', self.data['converge'])
      logging.info("NEW CONVERGENCE:  %0.6f\n", convergence)
      self.data['converge'] = convergence

      # Control Thread requires the catalog to be accessible. Hence it starts it:
      #  TODO: use addState HERE & UPDATE self.data['JCQueue']
      self.catalogPersistanceState = True
      self.localcatalogserver = self.catalog.conn()



      # Save all  packed matrices to catalog
      logging.info("Saving global matrices to catalog")
      storeNPArray(self.catalog, tmat, 'transitionmatrix')
      storeNPArray(self.catalog, smat, 'selectionmatrix')
      storeNPArray(self.catalog, updated_cmat, 'convergencematrix')
      storeNPArray(self.catalog, fatigue, 'fatigue')

  
      return list(newJobCandidate.keys())


    def addArgs(self):
      parser = macrothread.addArgs(self)
      parser.add_argument('--gendata')
      return parser



if __name__ == '__main__':
  mt = controlJob(schema, __file__)

  #  For generating a seed JC
  args = mt.addArgs().parse_args()

  if args.gendata:
    jcID, params = generateNewJC(args.gendata, 500)
    jcConfig = dict(params,
        name    = jcID,
        runtime = 51000,
        temp    = 310,
        state   = 1)
    for k, v in jcConfig.items():
      logging.info('%s: %s', k, str(v))

    catalog = redisCatalog.dataStore('catalog')
    catalog.save({'jc_'+jcID: jcConfig})
    sys.exit(0)


  mt.run()
