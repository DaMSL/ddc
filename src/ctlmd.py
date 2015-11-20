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
from random import choice

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



def generateNewJC(rawfile, frame=-1):

    logging.debug("Generating new coords from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()
    # jcuid = 'DEBUG'

    # Write out coords (TODO: should this go to catalog or to file?)
    # tmpCoord = os.path.join(DEFAULT.COORD_FILE_DIR, '%s_tmp.pdb' % jcuid)
    jobdir = os.path.join(DEFAULT.JOB_DIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)

    logging.debug("Files to use: %s, %s", coordFile, newPsfFile)


    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    # Retrieve referenced file from storage
    #   TODO: Set up historical archive for retrieval (this may need to be imported)

    #  TODO:  Use load_frame instead of loading entire trajectory

    #  Load in Historical Referenced trajectory file, filter out proteins & slice
    traj  = md.load(rawfile, top=DEFAULT.PDB_FILE)
    filt = traj.top.select('protein')    
    traj.atom_slice(filt, inplace=True)
    
    #  If no frame ref is provided, grab the middle frame
    #  TODO:  ID specific window reference point
    if frame < 0:
      frame = traj.n_fames // 2
    coord = traj.slice(frame)

    logging.debug("Working source traj: %s", str(coord))

    # Save this as a temp file to set up simulation input file
    coord.save_pdb(coordFile)

    logging.debug("Coord file saved.")

    newsimJob = dict(workdir=jobdir,
        coord = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = DEFAULT.TOPO,
        parm    = DEFAULT.PARM)

    cmd = psfgen(newsimJob)

    logging.debug("  PSFGen new simulation:\n " + cmd)

    stdout = executecmd(cmd)

    logging.debug("  PSFGen COMPLETE!!  Cleaning up\n" + stdout)

    os.remove(coordFile)

    return jcuid, newsimJob




class kv2DArray:
  def __init__(self, redis_db, name, mag=5, init=0):
    self.db = redis_db
    self.name = name
    self.mag = 0
    self.datatype = init

    # Check if the array already exists

    stored_mag = self.db.get(self.name + '_magnitude')
    if stored_mag:
      self.mag = int(stored_mag)
    else:

      # Initialize the array
      self.mag = mag
      self.db.set(self.name + '_magnitude', mag)
      if not isinstance(self.datatype, list): 
        for x in range(mag):
          for y in range(mag):
            self.db.set(self.name + '_%d_%d' % (x, y), init)

  def key (self, x, y):
    return self.name + '_%d_%d' % (x, y)

  # Only for scalar values
  def incr (self, x, y, amt=1):
    if isinstance(self.datatype, list):
      logging.error("ERROR!  Trying to set scalar to 2D K-V list Array")
      return
    self.db.incr(self.key(x, y), amt)

  def set (self, x, y, elm):
    self.db.set(self.key(x, y), elm)

  # Only for scalar values
  def add (self, x, y, elm):
    if not isinstance(self.datatype, list):
      logging.error("ERROR!  Trying to add data to scalar 2D K-V list Array")
      return
    self.db.rpush(self.key(x, y), elm)


  def get (self, x, y):
    if isinstance(self.datatype, list):
      return [elm.decode() for elm in self.db.lrange(self.key(x, y), 0, -1)]

    return self.db.get(self.key(x, y)).decode()

  def getAll(self):
    arr = arr = [[0]*self.mag]*self.mag
    for x in range(self.mag):
      for y in range(self.mag):
        elm = self.get(x, y)
        if elm == None:
          arr[x][y] = self.datatype
        elif isinstance(elm, list):
          arr[x][y] = elm
        else:
          elm = float(elm)
          arr[x][y] = int(elm) if isinstance(self.datatype, int) else float(elm)
    return arr

  def display(self):
    if isinstance(self.datatype, list):
      logging.warning("Will not displace 2D list Array")
      return

    tm = self.getAll()
    fmt = '%4d' if isinstance(self.datatype, int) else '%0.2f'

    for row in tm:
      logging.info('   ' + " ".join([fmt%x for x in row]))
      


class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('LDIndexList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('indexSize', 'JCQueue')
      self.setSplit('anlSplitParam')
      
      # exec incl hash key-name
      # TODO: wildcard loading of data

      self.modules.add('namd')

      #  This thread's execution will run "supervised"
      self.fork = False


    def term(self):
      # For now
      return False

    def split(self):
      catalog = self.getCatalog()

      # TODO:  Back Trace
      split = 1  #int(self.data['simSplitParam'])
      immed = catalog.slice('LDIndexList', split)
      return immed

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

      # TODO: Dynamically check for PC #
      engine = nearpy.Engine(454*3, 
            lshashes=[lshash], 
            storage=redis_storage)


      # Load Transition Matrix & Historical index state labels
      tmat = kv2DArray(archive, 'tmat', mag=5)
      tmat_before = tmat.getAll()

      labels = loadLabels(DEFAULT.DATA_LABEL_FILE)

      # Set initial params for index calculations
      prevState = -1    # To track state transitions
      prevTrajectory = None
      sourceHistory = None    # Holds Decision History data from source JC used to create the data
      stateCount = {}   # Index of observed states 

      # NOTE: ld_index is a list of indexed trajectories. They may or may NOT be from
      #   the same simulation (allows for grouping of multiple downstream data into 
      #   one control task). Thus, this algorithm tracks subsequent windows within one
      #   trajectories IOT track state changes. 

      for key in sorted(ld_index.keys()):

        index = np.array(ld_index[key])   # Get actual Index for this window
        sourceJC, frame = key.split(':')  # Assume colon separator

        logging.info(' Index Loaded from %s:   Shape=%s,  Type=%s', sourceJC, str(index.shape), str(index.dtype))

        # Get Decision History for the index IF it a new index not previously processed
        if sourceJC != prevTrajectory:
          sourceHistory = {}
          stateCount[sourceJC] = np.zeros(5)
          self.catalog.load({wrapKey('jc', sourceJC): sourceHistory})
          if 'state' not in sourceHistory:
            prevState = None
            logging.info("NO Historical State Data for %s", sourceJC)
          else:
            prevState = int(sourceHistory['state'])
            logging.debug("Source JC was supposed to start in state %d", startState)

          # Note:  Other Decision History is loaded here


        # Probe historical index  -- for now only probing DEShaw index
        #   TODO: Take NN distance into account when calculating stateCounts
        neigh = engine.neighbours(index)
        if len(neigh) == 0:
          logging.info ("Found no near neighbors for %s", key)
        else:
          logging.info ("Found %d neighbours:", len(neigh))
          for n in neigh:
            logging.info ("    NN:  %s   dist = %f", n[1], n[2])
            nnkey = n[1]
            trajectory, seqNum = nnkey.split(':')
            nn_state = labels[int(trajectory)].state
            stateCount[sourceJC][nn_state] += 1

            # TODO: Index State Count and factor into future decisions

          # For now, just grab the top NN & closest relative state for this index from DEShaw Data
          nnkey = neigh[0][1]
          trajectory, seqNum = nnkey.split(':')
          state = labels[int(trajectory)].state

          # Increment the transition matrix
          if prevState == None:
            prevState = state
          logging.debug("Transition from -> to ::  %d  --->  %d", prevState, state)
          tmat.incr(prevState, state, 1)
          prevState = state

      # Build Decision History Data for the Source JC's from the indices
      transitionBins = kv2DArray(archive, 'transitionBins', mag=5, init=[])      # Should this load here (from archive) or with other state data from catalog?


      for srckey, resultStates in stateCount.items():
        if len(resultStates) == 0:
          continue
        dist = resultStates / sum(resultStates)
        stateA = np.argmax(dist)

        # Source Trajectory spent most of its time in 1 state
        if max(dist) > 0.8: # <----------theta:
          transitionBins.add(stateA, stateA, srckey)

        # Observation showed some trandition 
        else:
          dist[stateA] = 0        
          stateB = np.argmax(dist)
          transitionBins.add(stateA, stateB, srckey)


      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's

      # Weight Calculation 

      # 1. Load New Transition Matrix
      tmat_after = tmat.getAll()
      logging.info("Transion Matrix:")
      tmat.display()

      # 2. Calculate new transition probably based on observed transitions
      total_t = np.sum([[tmat_after[x][y] for x in range(5)] for y in range(5)])
      # weights = sorted([[((x, y), tmat_after[x][y]/total) for x in range(5)] for y in range(5)], key=lambda x: x[1])
      probability = [[tmat_after[x][y]/total_t for x in range(5)] for y in range(5)]

      # 3. Load current "fatigue" values
      fmat = kv2DArray(archive, 'fatigue', mag=5)
      fatigue = fmat.getAll() 

      # 4. Appply weighted constant (TODO: Pull in from external or dymically set)
      alpha = 0.5   
      beta  = 0.5 

      # 5. Determine weights for job candidate bins
      wght_mat = np.zeros(shape=(5, 5))
      for x in range(5):
        for y in range(5):
          wght_mat[x][y] = alpha * probability[x][y] + beta * fatigue[x][y]

      # 6. SORT WEIGHTS  -- MIN is PREFERABLE
      weights = sorted([((x, y), wght_mat[x][y]) for x in range(5) for y in range(5)], key=lambda x: x[1])

      # Job Candidate Selection (TODO: Should we select 1 per bin or weighted # per bin??)
      tbin = transitionBins.getAll()

      newJobCandidate = {}
      for w in weights[:DEFAULT.MAX_NUM_NEW_JC]:

        #  TODO:  Bin J.C. Selection into a set of potential new J.C. params/coords
        #    can either be based on historical or a sub-state selection
        #    This is where the infinite # of J.C. is bounded to something manageable
        #     FOR NOW:  Pick a completely random DEShaw Window of the same state
        #       and go withit
        A = w[0][0]
        B = w[0][1] 
        logging.debug("Targetting transition weight, %s,  %f", str(w[0]), w[1])

        # Pick a "target bin"
        targetBin = tbin[A][B]

        # Ensure there are candidates to pick from
        if len(targetBin) == 0:
          logging.info("No DEShaw reference for transition, %s  -- going alternate", str(targetBin))

          # Flip direction of transition (only good if we're assuming transitions are non-symetric)
          targetBin = tbin[B][A]

          if len(targetBin) == 0:
            logging.info("No DEShaw reference for transition, %s  -- going alternate", str(targetBin))
            targetBin = []

            # Finally, pick any start point from the initial state (assume that state had a candidate)
            for z in range(5):
              targetBin.extend(tbin[A][z])

        # Pick a random trajectory from the bin
        sourceTraj = choice(targetBin)
        jcStartState = w[0][0]

        # Updated the "fatigue" values 
        #   The associated bin of choice is increased by a ratio of 1/N where N is total # of bins
        #   All other bins are "replenished" by a fractional amount
        #   
        #  TODO:  Flush Out these values

        expendedEnergy = fatigue[w[0][0]][w[0][1]] * 0.04  # 1/N of 
        replenishEnergy = expendedEnergy / (24.)   # e / (N-1)

        for i in range(5):
          for j in range(5):
            if i == w[0][0] and j == w[0][1]:
              fmat.set(i, j, fatigue[i][j] + expendedEnergy)
            else:
              fmat.set(i, j, fatigue[i][j] - replenishEnergy)

        logging.debug("Selected random DEShaw Trajectory # %s based on state %d", sourceTraj, jcStartState)

        # TODO: Should new Candidates come from DEShaw data or the new Sim Data?????
        # Back-project  <--- Move to separate Function tied to decision history
        archiveFile = os.path.join(DEFAULT.RAW_ARCHIVE, 'bpti-all-%04d.dcd' % int(sourceTraj))

        # frameRef = int(seqNum) * DEFAULT.HIST_SLIDE + (DEFAULT.HIST_WINDOW // 2)
        jcID, params = generateNewJC(archiveFile, 500)


        # NOTE: Update Additional JC Params and Historical Data, as needed
        jcConfig = dict(params,
            name    = jcID,
            runtime = 51000,
            temp    = 310,
            state   = jcStartState)


        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in jcConfig.items():
          logging.debug("   %s:  %s", k, str(v))

        newJobCandidate[jcID] = jcConfig

        logging.info("New JC Complete:  %s" % jcID)
          

      # Control Thread requires the catalog to be accessible. Hence it starts it:
      self.catalogPersistanceState = True
      self.localcatalogserver = self.catalog.conn()

      for jcid, config in newJobCandidate.items():
        jckey = wrapKey('jc', jcid)
        self.data['JCQueue'].append(jcid)
        # self.data[jckey] = config
        self.catalog.save({jckey: config})

      return newJobCandidate.keys()


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
