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
logging.basicConfig(format='%(message)s', level=logging.DEBUG)



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

def generateNewJC(rawfile, pdbfile, topo, parm):

    logging.debug("Generating new simulation coordinates from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()
    # jcuid = 'DEBUG'

    # Write out coords (TODO: should this go to catalog or to file?)
    # tmpCoord = os.path.join(DEFAULT.COORD_FILE_DIR, '%s_tmp.pdb' % jcuid)
    jobdir = os.path.join(DEFAULT.JOBDIR,  jcuid)
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
    def __init__(self, fname):
      macrothread.__init__(self, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('LDIndexList', None)
      self.setState('JCQueue', 'indexSize', 'timestep', 
                    'converge', 'ctlSplitParam', 'ctlDelay',
                    *tuple([candidPoolKey(i,j) for i in range(5) for j in range(5)]))
      self.modules.add('namd')

      self.addImmut('num_var')
      self.addImmut('num_pc')


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
      

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']


    def execute(self, ld_index):
      logging.debug('CTL MT')

      logging.debug("============================  <PRE-PROCESS>  =============================")
      # TODO:  Treat Archive as an overlay service. For now, wrap inside here and connect to it

      # Increment the timestep
      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

      archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
      redis_storage = RedisStorage(archive)
      config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
      if not config:
        logging.error("LSHash not configured")
        #TODO: Gracefully exit
        return []

      # Create empty lshash and load stored hash
      lshash = DEFAULT.getEmptyHash()
      lshash.apply_config(config)
      indexSize = self.data['num_var'] * self.data['num_pc']
      logging.debug("INDEX SIZE = %d:  ", indexSize)
      engine = nearpy.Engine(indexSize, 
            lshashes=[lshash], 
            storage=redis_storage)

      # Load current set of known states
      #  TODO:  Injection Point for clustering. If clustering were applied
      #    this would be much more dynamic (static fileload for now)
      labels = loadLabels()
      labelNames = getLabelList(labels)
      numLabels = len(labelNames)

 
      # PROBE   ------------------------
      logging.debug("============================  <PROBE>  =============================")

      # Set initial params for index calculations
      prevState = -1    # To track each state transition
      prevTrajectory = None   # To check for unique trajectories
      decisionHistory = {}    # Holds Decision History data from source JC used to create the data
      observationDistribution = {}   #  distribution of observed states (for each input trajectory)
      observationCount = {}
      statdata = {}
      delta_tmat = np.zeros(shape=(numLabels, numLabels))
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
          observationDistribution[sourceJC] = np.zeros(5)
          observationCount[sourceJC] = np.zeros(5)

          #  TODO: Load indices up front with source history
          config = {}
          self.catalog.load({wrapKey('jc', sourceJC): config})  #TODO: Load these up front
          for k, v in config.items():
            logging.debug("  %s:  %s", k, str(v))
          decisionHistory[sourceJC] = config
          if 'state' not in config:
            prevState = None
            logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
          else:
            prevState = int(config['state'])
            logging.debug("New Index to analyze, %s: Source JC was previously in state %d", sourceJC, prevState)

          sourceJCKey = wrapKey('jc', sourceJC)
          self.addToState(sourceJCKey, config)
          prevTrajectory = sourceJC
          statdata[sourceJC] = {}

          # Note:  Other Decision History is loaded here

        logging.info("  Probing `%s` window at frame # %s  (state %s)", sourceJC, frame, str(prevState))
        statdata[sourceJC][frame] = {}

        # Probe historical index  -- for now only probing DEShaw index
        neigh = engine.neighbours(index)
        if len(neigh) == 0:
          logging.info ("    Found no near neighbors for %s", key)
        else:
          logging.debug ("    Found %d neighbours:", len(neigh))

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
          state = int(np.argmax(clust))
          statdata[sourceJC][frame]['state'] = state
          statdata[sourceJC][frame]['count'] = count.tolist()
          statdata[sourceJC][frame]['clust'] = clust.tolist()

          # Increment the transition matrix
          if prevState == None:
            prevState = state
          logging.debug("    Transition (%d, %d)", prevState, state)
          logging.debug("      Clustered at: %s", str(clust/np.sum(clust)))
          logging.debug("      Counts:       %s", str(count))

          # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
          delta_tmat[prevState][state] += 1
          prevState = state

      # Build Decision History Data for the Source JC's from the indices
      # transitionBins = kv2DArray(archive, 'transitionBins', mag=5, dtype=str, init=[])      # Should this load here (from archive) or with other state data from catalog?


      logging.debug("Delta Observation Matrix:\n" + str(delta_tmat))

      #  Theta is calculated as the probability of staying in 1 state (from observed data)
      theta = .6  #self.data['observation_counts'][1] / sum(self.data['observation_counts'])
      logging.debug("  THETA  = %0.3f   (static for now)", theta)


      #  DECISION HISTORY  --------------------------
      logging.debug("============================  <DECISION HIST>  =============================")


      #  Process output data for each unque input trajectory (as produced by a single simulation)
      for sourceJC, cluster in observationDistribution.items():
        logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point here for better classification and/or move to analysis)", sourceJC)
        if sum(observationDistribution[sourceJC]) == 0:
          logging.debug(" No observed data for, %s", sourceJC)
          continue

        #  TODO: Another injection point for better classification. Here, the classification is for the input trajectory
        #     as a whole for future job candidate selection. 

        logging.debug("  Observations:      %s", str(observationCount[sourceJC]))
        logging.debug("  Cluster weights:   %s", str(cluster/np.sum(cluster)))
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
          logging.debug("    TODO: ID difference, update ML weights, provonance, etc..... (if desired)")

        # Update other History Data
        key = wrapKey('jc', sourceJC)
        self.data[key]['actualBin'] = outputBin
        self.data[key]['epoch'] = DEFAULT.EPOCH_LABEL
        logging.debug("%s", str(statdata[sourceJC]))
        self.data[key]['indexList'] = json.dumps(statdata[sourceJC])

      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's

      #  WEIGHT CALCULATION ---------------------------
      logging.debug("============================  <WEIGHT CALC>  =============================")

 
      #  TODO:  Consistency
      # Load Transition Matrix (& TODO: Historical index state labels)
      tmat = loadNPArray(self.catalog, 'transitionmatrix')
      if tmat is None:
        tmat = np.zeros(shape=(5,5))    # TODO: Move to init

      # Merge in delta
      tmat += delta_tmat

      # Write back out  (limit inconsitency for now)
      storeNPArray(self.catalog, tmat, 'transitionmatrix')



      # Weight Calculation 

      #  Load current fatigue values
      fatigue = loadNPArray(self.catalog, 'fatigue')   # TODO: Move to self.data(??) and/or abstract the NP load/save
      if fatigue is None:
        fatigue = np.full((5,5), 0.04)    # TODO: Move to init


      bins = [(x, y) for x in range(numLabels) for y in range(numLabels)]
      # TODO:   Load New Transition Matrix  if consistency is necessary, otherwise use locally updated tmat
 
      #  2. Calculate new "preference"  targetbin portion of the weight
      totalObservations = np.sum(tmat)
      tmat_distro = {(i, j): tmat[i][j]/totalObservations for i in range(numLabels) for j in range(numLabels)}
      quota = max(totalObservations / len(tmat_distro.keys()), len(tmat_distro.keys()))
      
      pref = {(i, j): max((quota-tmat[i][j])/quota, 1/quota) for i in range(numLabels) for j in range(numLabels)}

      logging.debug("CURRENT (BIASED) OBSERVATION DISTRIBUTION OF OUTPUT:")
      logging.debug("  Total observations = %d", totalObservations)
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% tmat_distro[(i,k)] for k in range(5)]))


      logging.debug("PREFERENCE WEIGHTS:")
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% pref[(i,k)] for k in range(5)]))

      #  3. Apply constants. This can be user influenced
      alpha = self.data['weight_alpha']
      beta = self.data['weight_beta']

      #  4. Set new weight and order from high to low
      weight = {}
      for k in bins:
        weight[k] =  alpha * pref[k] + beta * (1 - fatigue[k])




      logging.debug("UPDATED WEIGHTS (INCL FATIGUE):")
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% weight[(i,k)] for k in range(5)]))

      updatedWeights = sorted(weight.items(), key=lambda x: x[1], reverse=True)


      #  SCHEDULING   -----------------------------------------
      logging.debug("============================  <SCHEDULING>  =============================")

      #  5. Load JC Queue and all items within to get respective weights and projected target bins
      #   TODO:  Pipeline this or load all up front!
      curqueue = []
      logging.debug("Loading Current Queue of %d items", len(self.data['JCQueue']))
      debug = True
      for i in self.data['JCQueue']:
        key = wrapKey('jc', i)
        config = {}
        self.catalog.load({key: config})
        # if debug:
        #   logging.debug('Sample Job output (1st on queue: idx `%s`)', i)
        #   for k, v in config.items():
        #     logging.debug('  %s:  %s',  k, str(v))  
        #   debug = False        

        # Dampening Factor: proportional it currency (if no ts, set it to 1)
        jc_ts = config['timestep'] if 'timestep' in config else 1
        

        #   May want to consider incl convergence of sys at time job was created
        w_before    = config['weight']
        config['weight'] = config['weight'] * (jc_ts / self.data['timestep'])
        logging.debug("Dampening Factor Applied (jc_ts = %d):   %0.5f  to  %05f", jc_ts, w_before, config['weight'])
        curqueue.append(config)


      #  5a. (PreProcess current queue) for legacy JC's
      logging.debug("Loaded %d items", len(curqueue))
      for jc in range(len(curqueue)):
        if 'weight' not in curqueue[jc]:
          curqueue[jc]['weight'] = 1.0

        if 'gc' not in curqueue[jc]:
          curqueue[jc]['gc'] = 1


      #  6. Sort current queue
      if len(curqueue) > 0:
        existingQueue = deque(sorted(curqueue, key=lambda x: x['weight'], reverse=True))
        logging.debug("Existing Queue has %d items between weights: %0.5f - %0.5f", len(existingQueue), existingQueue[0]['weight'], existingQueue[-1]['weight'])
      else:
        existingQueue = deque()
        logging.debug("Existing Queue is empty.")

      #  7. Det. potential set of  new jobs  (base on launch policy)
      #     TODO: Set up as multiple jobs per bin, cap on a per-control task basis, or just 1 per bin
      potentialJobs = deque(updatedWeights)
      logging.debug("Potential Job Queue has %d items between weights: %0.5f - %0.5f", len(potentialJobs), potentialJobs[0][1], potentialJobs[-1][1])

      #  7. Prepare a new queue (for output)
      jcqueue = deque()

      targetBin = potentialJobs.popleft()
      oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()
      selectionTally = np.zeros(shape=(numLabels, numLabels))
      newJobCandidate = {}
   
      while len(jcqueue) < DEFAULT.MAX_JOBS_IN_QUEUE:

        if oldjob == None and targetBin == None:
          logging.info("No more jobs to queue.")
          break

        if (targetBin == None) or (oldjob and oldjob['weight'] > targetBin[1]):
          logging.debug("Re-Queuing OLD JOB `%s`   weight= %0.5f", oldjob['name'], oldjob['weight'])
          jcqueue.append(oldjob['name'])
          oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()

        else:
          A, B = targetBin[0]
          logging.debug("\n\nCONTROL: Target transition bin  %s  (new job #%d,  weight=%0.5f)", str((A, B)), len(newJobCandidate), targetBin[1])

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

          logging.debug('Final Candidate Pool has  %d  candidates', len(candidatePool))

          # selectedBins.append((A, B))

          # Pick a random trajectory from the bin
          sourceTraj = choice(candidatePool)
          logging.debug("Selected DEShaw Trajectory # %s based from candidate pool.", sourceTraj)


          # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
          #   or we have a pre-fetch algorithm to get the data
          # Back-project  <--- Move to separate Function tied to decision history
          # For now:
          if isinstance(sourceTraj, int) or sourceTraj.isdigit():      # It's a DEShaw file
            fname = 'bpti-all-%03d.dcd' if int(sourceTraj) < 1000 else 'bpti-all-%04d.dcd'
            archiveFile = os.path.join(DEFAULT.RAW_ARCHIVE, fname % int(sourceTraj))
            pdbfile = DEFAULT.PDB_FILE
          else:
            archiveFile = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.dcd' % sourceTraj)
            pdbfile     = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.pdb' % sourceTraj)

          # Generate new set of params/coords
          jcID, params = generateNewJC(archiveFile, pdbfile, DEFAULT.TOPO, DEFAULT.PARM)

          # Update Additional JC Params and Decision History, as needed
          jcConfig = dict(params,
              name    = jcID,
              runtime = DEFAULT.RUNTIME,
              temp    = 310,
              state   = A,
              weight  = targetBin[1],
              timestep = self.data['timestep'],
              gc      = 1,
              epoch   = DEFAULT.EPOCH_LABEL,
              targetBin  = str((A, B)))

          logging.info("New Simulation Job Created: %s", jcID)
          for k, v in jcConfig.items():
            logging.debug("   %s:  %s", k, str(v))

          #  Add to the output queue & save config info
          jcqueue.append(jcID)
          newJobCandidate[jcID] = jcConfig
          selectionTally[A][B] += 1
          logging.info("New Job Candidate Completed:  %s   #%d on the Queue", jcID, len(jcqueue))
          
          if len(potentialJobs) == 0 or len(newJobCandidate) == DEFAULT.MAX_NUM_NEW_JC:
            targetBin = None
          else:
            targetBin = potentialJobs.popleft()


      # Mark obsolete jobs for garbage collection:
      logging.info("Marking %d obsolete jobs for garbage collection", len(existingQueue))
      while len(existingQueue) > 0:
        config = existingQueue.popleft()
        config['gc'] = 0

        # Ensure to add it to the state to write back to catalog
        self.addToState(wrapKey('jc', config['name']), config)



      # Updated the "fatigue" values for all selected jobs
      #    -- FOR NOW do it all at once after selection process
      #    TODO: This can be weighted based on # of selections
      #  TODO:  RESOLVE CONSITENCY
      for i in range(numLabels):
        for j in range(numLabels):
          if selectionTally[i][j] > 0:
            fatigue[i][j] += (1-fatigue[i][j])/25
            logging.debug("Increasing fatigue value for %d, %d to %f", i, j, fatigue[i][j])
          else:
            fatigue[i][j] -= (fatigue[i][j]/25)**2


      # CONVERGENCE CALCUALTION  -------------------------------------
      logging.debug("============================  <CONVEGENCE>  =============================")

      logging.debug("Observations MAT:\n" + str(tmat))
      logging.debug("Fatigue:\n" + str(fatigue))

      # Load Selection Matrix
      smat = loadNPArray(self.catalog, 'selectionmatrix')
      if smat is None:
        smat = np.full((5,5), 1.)    # SEED Selection matrix (it cannot be 0) TODO: Move to init

      # Merge Update global selection matrix
      smat += selectionTally
      logging.debug("SMAT:\n" + str(smat))


      # Load Convergence Matrix
      cmat = loadNPArray(self.catalog, 'convergencematrix')
      if cmat is None:
        cmat = np.full((5,5), 0.04)    # TODO: Move to init
      logging.debug("CMAT:\n" + str(cmat))



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
      # self.catalogPersistanceState = True
      # self.localcatalogserver = self.catalog.conn()

      self.data['JCQueue'] = list(jcqueue)
      # Update Each new job with latest convergence score and save to catalog(TODO: save may not be nec'y)
      logging.debug("Updated Job Queue length:  %d", len(self.data['JCQueue']))
      for jcid, config in newJobCandidate.items():
        config['converge'] = self.data['converge']
        jckey = wrapKey('jc', jcid)
        self.catalog.save({jckey: config})

      # Save all  packed matrices to catalog
      logging.info("Saving global matrices to catalog")
      storeNPArray(self.catalog, smat, 'selectionmatrix')
      storeNPArray(self.catalog, updated_cmat, 'convergencematrix')
      storeNPArray(self.catalog, fatigue, 'fatigue')

  
      return list(newJobCandidate.keys())



if __name__ == '__main__':
  mt = controlJob(__file__)

  #  For generating a seed JC
  mt.parser.add_argument('--gendata')
  args = mt.addArgs().parse_args()

  if args.gendata:
    win = args.gendata
    labels = loadLabels()
    jcID, params = generateNewJC(win, 500)

    jcConfig = dict(params,
        name    = jcID,
        runtime = DEFAULT.RUNTIME,
        temp    = 310,
        state   = labels[win].state,
        weight  = 0.,
        timestep = 0,
        gc      = 1,
        epoch   = DEFAULT.EPOCH_LABEL,
        converge = 0.,
        targetBin  = str((labels[win].state, labels[win].state)))
    for k, v in jcConfig.items():
      logging.info('%s: %s', k, str(v))

    catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
    catalog.save({'jc_'+jcID: jcConfig})
    sys.exit(0)


  mt.run()
