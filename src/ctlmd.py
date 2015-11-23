import argparse
import sys
import os
import sys

import mdtraj as md
import numpy as np
from numpy import linalg as LA
import nearpy
from nearpy.storage.storage_redis import RedisStorage
from nearpy.hashes import RandomBinaryProjections, PCABinaryProjections

from collections import namedtuple

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


def loadLabels(fn):
  label =namedtuple('window', 'time state')
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

def getLabelList(labels):
  labelset = set()
  for lab in labels:
    labelset.add(lab.state)
  return sorted(list(labelset))

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

    #  TODO:  Use load_frame instead of loading entire trajectory

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



def fromByte(dataType, valType):
  logging.debug("DT= %s,  VT=%s", str(dataType), str(valType))
  if dataType == int:
    return lambda x: int(x.decode())
  if dataType == float:
    return lambda x: float(x.decode())
  if dataType == list:
    if valType == int:
      return lambda x: [int(y.decode()) for y in x]
    if valType == float:
      return lambda x: [float(y.decode()) for y in x]
    return lambda x: [y.decode() for y in x]
  return lambda x: x.decode()

class kv1DArray:
  def __init__(self, redis_db, name, mag=5, dtype=int, init=0):
    self.db = redis_db
    self.name = name
    self.mag = 0
    self.datatype = dtype
    self.conv = fromByte(type(init), dtype)
    self.init = init
    # Check if the array already exists
    stored_mag = self.db.get(self.name + '_magnitude')
    if stored_mag:
      self.mag = int(stored_mag)
    else:
      # Initialize the array
      self.mag = mag
      self.db.set(self.name + '_magnitude', mag)
      if self.datatype != list: 
        for x in range(mag):
          for y in range(mag):
            self.db.set(self.name + '_%d_%d' % (x, y), init)
  def key (self, x, y):
    return self.name + '_%d_%d' % (x, y)
  # Only for scalar values
  def incr (self, x, y, amt=1):
    if self.datatype == int:
      self.db.incr(self.key(x, y), amt)
    elif self.datatype == float:
      self.db.incrbyfloat(self.key(x, y), amt)
    else:
      logging.error("ERROR!  Trying to set scalar to 2D K-V non-scalar Array")
  def set (self, x, y, elm):
    self.db.set(self.key(x, y), elm)
  def add (self, x, y, elm):
    if isinstance(self.datatype, list):
      self.db.rpush(self.key(x, y), elm)
    else:
      logging.error("ERROR!  Trying to insert element to scalar 2D K-V list Array")
  def get (self, x, y):
    if isinstance(self.datatype, list):
      return self.conv(self.db.lrange(self.key(x, y), 0, -1))
    return self.conv(self.db.get(self.key(x, y)))


class kv2DArray:
  def __init__(self, redis_db, name, mag=5, dtype=int, init=0):
    self.db = redis_db
    self.name = name
    self.mag = 0
    self.datatype = list if isinstance (init, list) else dtype
    self.conv = fromByte(type(init), dtype)
    self.init = init
    # Check if the array already exists
    stored_mag = self.db.get(self.name + '_magnitude')
    if stored_mag:
      self.mag = int(stored_mag)
    else:
      # Initialize the array
      self.mag = mag
      self.db.set(self.name + '_magnitude', mag)
      if self.datatype != list: 
        for x in range(mag):
          for y in range(mag):
            self.db.set(self.name + '_%d_%d' % (x, y), init)
  def key (self, x, y):
    return self.name + '_%d_%d' % (x, y)
  # Only for scalar values
  def incr (self, x, y, amt=1):
    if self.datatype == int:
      self.db.incr(self.key(x, y), amt)
    elif self.datatype == float:
      self.db.incrbyfloat(self.key(x, y), amt)
    else:
      logging.error("ERROR!  Trying to set scalar to 2D K-V non-scalar Array")
  def set (self, x, y, elm):
    self.db.set(self.key(x, y), elm)
  def add (self, x, y, elm):
    if self.datatype == list:
      self.db.rpush(self.key(x, y), elm)
    else:
      logging.error("ERROR!  Trying to insert element to scalar 2D K-V list Array")
  def get (self, x, y):
    if self.datatype == list:
      return self.conv(self.db.lrange(self.key(x, y), 0, -1))
    return self.conv(self.db.get(self.key(x, y)))
  def getAll(self):
    arr = [[None for x in range(self.mag)] for y in range(self.mag)]
    for x in range(self.mag):
      for y in range(self.mag):
        arr[x][y] = self.get(x, y)
        if arr[x][y] == None:
           arr[x][y] = self.init
    return arr
  def display(self):
    if self.datatype == list:
      logging.warning("Will not displace 2D list Array")
      return
    mat = self.getAll()
    fmt = '%8d' if self.datatype == int else '%8.2f'
    for row in mat:
      logging.info('   ' + " ".join([fmt%x for x in row]))
      


class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('LDIndexList', 'JCQueue')
      self.setState('indexSize', 'ctlSplitParam')

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

      # TODO:  Treat Archive as an overlay service. For now, wrap inside here and connect to it
      archive = redisCatalog.dataStore(**archiveConfig)
      redis_storage = RedisStorage(archive)
      config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
      if not config:
        logging.error("LSHash not configured")
        #TODO: Gracefully exit
      # Create empty lshash and load stored hash
      lshash = PCABinaryProjections(None, None, None)
      lshash.apply_config(config)


      logging.debug("INDEX SIZE = %d:  ", self.data['indexSize'])
      engine = nearpy.Engine(454*DEFAULT.NUM_PCOMP, 
            lshashes=[lshash], 
            storage=redis_storage)

      # Load current set of known states
      #  TODO:  Injection Point for clustering. If clustering were applied
      #    this would be much more dynamic and would not change (static for now)
      labels = loadLabels(DEFAULT.DATA_LABEL_FILE)
      labelNames = getLabelList(labels)
      numLabels = len(labelNames)

      # Load Transition Matrix & Historical index state labels
      tmat = kv2DArray(archive, 'tmat', mag=5, dtype=int, init=0)
      tmat_before = tmat.getAll()
      tmat.display()


      # Set initial params for index calculations
      prevState = -1    # To track each state transition
      prevTrajectory = None   # To check for unique trajectories
      sourceHistory = None    # Holds Decision History data from source JC used to create the data
      observationDistribution = {}   #  distribution of observed states (for each input trajectory)
      observationSet = set()  # To track the # of unique observations (for each input trajectory)

      sourceTrajectories = set()


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
          sourceHistory = {}
          observationDistribution[sourceJC] = np.zeros(5)
          self.catalog.load({wrapKey('jc', sourceJC): sourceHistory})
          if 'state' not in sourceHistory:
            prevState = None
            logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
          else:
            prevState = int(sourceHistory['state'])
            logging.debug("New Index to analyze, %s: Source JC was supposed to start in state %d", sourceJC, prevState)
          prevTrajectory = sourceJC

          # Note:  Other Decision History is loaded here

        logging.info("\nProbing `%s` window at frame # %s  (state %s)", sourceJC, frame, str(prevState))

        # Probe historical index  -- for now only probing DEShaw index
        #   TODO: Take NN distance into account when calculating stateCounts
        observationSet.clear()
        neigh = engine.neighbours(index)
        if len(neigh) == 0:
          logging.info ("Found no near neighbors for %s", key)
        else:
          logging.info ("Found %d neighbours:", len(neigh))
          for n in neigh:
            nnkey = n[1]
            distance = n[2]
            trajectory, seqNum = nnkey.split(':')
            nn_state = labels[int(trajectory)].state
            logging.info ("    NN:  %s   dist = %f    state=%d", nnkey, distance, nn_state)
            
            observationDistribution[sourceJC][nn_state] += 1
            observationSet.add(nn_state)

            # TODO: Factor in distance when calculating the observationDistribution
            # TODO: Index observationDistribution and factor into future decisions

          # For now, just grab the top NN & closest relative state for this index from DEShaw Data
          #  Possible future implementations can insert a auto-encoder and/or better ML algorithm
          #  to classify the index and get a "state"
          nnkey = neigh[0][1]
          trajectory, seqNum = nnkey.split(':')
          state = labels[int(trajectory)].state

          # Increment the transition matrix
          if prevState == None:
            prevState = state
          logging.debug("  Transition %d  --->  %d    Incrementing transition counter (%d, %d)", prevState, state, prevState, state)

          # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
          tmat.incr(prevState, state, 1)
          # TODO:  Update observation counts
          # self.data['observation_counts'][len(observationSet)] += 1
          prevState = state

      # Build Decision History Data for the Source JC's from the indices
      transitionBins = kv2DArray(archive, 'transitionBins', mag=5, dtype=str, init=[])      # Should this load here (from archive) or with other state data from catalog?

      logging.debug("Updated Transition Matrix is below.")
      tmat.display()

      #  Theta is calculated as the probability of staying in 1 state (from observed data)
      theta = .6  #self.data['observation_counts'][1] / sum(self.data['observation_counts'])
      logging.debug("  THETA  = %0.3f", theta)


      #  Process output data for each unque input trajectory (as produced by a single simulation)
      for srckey, resultStates in observationDistribution.items():
        logging.debug("\nFinal processing for Source Trajectory: %s : %s   (note: injection point here for better classification)", srckey, str(resultStates))
        if sum(resultStates) == 0:
          logging.debug(" No observed data for, %s", srckey)
          continue

        #  TODO: Another injection point for better classification. Here, the classification is for the input trajectory
        #     as a whole for future job candidate selection. 
        dist = resultStates / sum(resultStates)
        stateA = np.argmax(dist)

        # Source Trajectory spent most of its time in 1 state
        if max(dist) > theta: 
          logging.debug(" Trajectory `%s`  classified as staying in state :  %d", srckey, stateA)
          transitionBins.add(stateA, stateA, srckey)
          stateB = stateA

        # Observation showed some trandition 
        else:
          dist[stateA] = 0        
          stateB = np.argmax(dist)
          logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", srckey, stateA, stateB)
          transitionBins.add(stateA, stateB, srckey)

        # if 'targetBin' in sourceHistory[srckey]:
        #   outputBin = str(stateA, stateB)
        #   inputBin  = sourceHistory[srcKey]['targetBin']
        #   logging.info("  Predicted Taget Bin was       :  %s", inputBin) 
        #   logging.info("  Actual Trajectory classified  :  %s", outputBin) 
        #   logging.debug("    TODO: ID difference, update weights, etc..... (if desired)")

      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's

      # Weight Calculation 

      # 1. Load New Transition Matrix
      #   TODO:  Compare tmat_after with tmat_before to get change in observations and calculate 
      #     convergence 
      tmat_after = tmat.getAll()

      # 2. Calculate new transition probability based on observed transitions
      total_t = np.sum([[tmat_after[x][y] for x in range(numLabels)] for y in range(numLabels)])
      # weights = sorted([[((x, y), tmat_after[x][y]/total) for x in range(5)] for y in range(5)], key=lambda x: x[1])
      probability = [[tmat_after[x][y]/total_t for x in range(numLabels)] for y in range(numLabels)]

      # 3. Load current "fatigue" values
      fmat = kv2DArray(archive, 'fatigue', mag=numLabels, dtype=float, init=0.)
      fatigue = fmat.getAll() 

      # 4. Appply weighted constant (TODO: Pull in from external or dymically set)
      alpha = 0.5   
      beta  = 0.5 

      # 5. Set weights based on probability and fatigue values for job candidate bins
      wght_mat = np.zeros(shape=(numLabels, numLabels))
      for x in range(numLabels):
        for y in range(numLabels):
          wght_mat[x][y] = alpha * probability[x][y] + beta * fatigue[x][y]
          logging.debug("Setting Weight (%d, %d):  %0.3f     Prob= %0.3f,  Fatig= %0.3f", x, y, wght_mat[x][y], probability[x][y], fatigue[x][y])

      # 6. SORT WEIGHTS  -- MIN is PREFERABLE
      #   TODO:  Selection Process for deterining the next job candidates
      #    INitial approach:  sort weights & choose the top N-bins
      #       Question: Should we select 1 per bin or weighted # per bin??)
      #    For exploration:  May want to examine more detailed stats to ID new regions of interest 
      weights = sorted([((x, y), wght_mat[x][y]) for x in range(numLabels) for y in range(numLabels)], key=lambda x: x[1])

      tbin = transitionBins.getAll()

      newJobCandidate = {}
      selectedBins = []
      for w in weights[:DEFAULT.MAX_NUM_NEW_JC]:

        #  TODO:  Bin J.C. Selection into a set of potential new J.C. params/coords
        #    can either be based on historical or a sub-state selection
        #    This is where the infinite # of J.C. is bounded to something manageable
        #     FOR NOW:  Pick a completely random DEShaw Window of the same state
        #       and go withit
        A = w[0][0]
        B = w[0][1] 
        logging.debug("\n\nCONTROL: Target transition bin:  %s    weight=%f", str(w[0]), w[1])

        # Pick a "target bin"
        targetBin = tbin[A][B]

        # Ensure there are candidates to pick from
        if len(targetBin) == 0:
          logging.info("No DEShaw reference for transition, (%d, %d)  -- checking reverse direction", A, B)

          # Flip direction of transition (only good if we're assuming transitions are non-symetric)
          targetBin = tbin[B][A]

          if len(targetBin) == 0:
            logging.info("No DEShaw reference for transition, (%d, %d)  -- checking all bins starting from state %d", B, A, A)
            targetBin = []

            # Finally, pick any start point from the initial state (assume that state had a candidate)
            for z in range(5):
              targetBin.extend(tbin[A][z])

        selectedBins.append((A, B))

        # Pick a random trajectory from the bin
        sourceTraj = choice(targetBin)
        logging.debug("Selected DEShaw Trajectory # %s based on state %d", sourceTraj, A)


        # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
        #   or we have a pre-fetch algorithm to get the data
        # Back-project  <--- Move to separate Function tied to decision history
        # For now:
        if sourceTraj.isdigit():      # It's a DEShaw file
          fname = 'bpti-all-%03d.dcd' if int(sourceTraj) < 1000 else 'bpti-all-%04d.dcd'
          archiveFile = os.path.join(DEFAULT.RAW_ARCHIVE, fname % int(sourceTraj))
          pdbfile = DEFAULT.PDB_FILE
        else:
          archiveFile = os.path.join(DEFAULT.JOB_DIR, sourceTraj, '%s.dcd' % sourceTraj)
          pdbfile     = os.path.join(DEFAULT.JOB_DIR, sourceTraj, '%s.pdb' % sourceTraj)

        jcID, params = generateNewJC(archiveFile, pdbfile)


        # NOTE: Update Additional JC Params and Historical Data, as needed
        jcConfig = dict(params,
            name    = jcID,
            runtime = 51000,
            temp    = 310,
            state   = A,
            targetBin  = str((A, B)))


        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in jcConfig.items():
          logging.debug("   %s:  %s", k, str(v))

        newJobCandidate[jcID] = jcConfig

        logging.info("New Job Candidate Complete:  %s" % jcID)
          

      # Updated the "fatigue" values  -- FOR NOW do it all at once after selection process
      #   The associated bin of choice is increased by a ratio of 1/N where N is total # of bins
      #   All other bins are "replenished" by a fractional amount
      #   
      #  TODO:  Flush Out these values
      # expendedEnergy =   # 1/N of 
      # replenishEnergy = expendedEnergy / (24.)   # e / (N-1)
      # logging.debug("Setting Fatigue value for (%d,%d) to %f", A, B, fatigue[A][B] + expendedEnergy)
      for i in range(numLabels):
        for j in range(numLabels):
          if (i,j) in selectedBins:
            logging.debug("Setting Fatigue value for (%d,%d) to %f", i, j, fatigue[i][j]*2)
            fmat.set(i, j, min(fatigue[i][j] * 2, 1.))
          else:
            fmat.set(i, j, max(fatigue[i][j] * 0.95, 0.))

      # logging.debug(" SIMULATION ONLY ---- NOT SAVING")
      # #  SIMULATING FOR NOW
      # sys.exit(0)


      # Control Thread requires the catalog to be accessible. Hence it starts it:
      self.catalogPersistanceState = True
      self.localcatalogserver = self.catalog.conn()

      for jcid, config in newJobCandidate.items():
        jckey = wrapKey('jc', jcid)
        # self.data['JCQueue'].append(jcid)
        # self.data[jckey] = config
        self.catalog.save({jckey: config})

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
