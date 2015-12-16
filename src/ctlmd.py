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
from kvadt import kv2DArray
from slurm import slurm
from random import choice, randint
from deshaw import *
from indexing import *

# import logging
# logging.basicConfig(format='%(message)s', level=logging.DEBUG)



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


def makeLogisticFunc (maxval, steep, midpt):
  return lambda x: maxval / (1 + np.exp(-steep * (midpt - x)))

skew = lambda x: (np.mean(x) - np.median(x)) / np.std(x)


def generateNewJC(rawfile, pdbfile, topo, parm, frame=None, debugstring=None):

    logging.debug("Generating new simulation coordinates from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()

    #  TODO:  Should coords be pre-fetched and pulled from pre-fetch location?
    jobdir = os.path.join(DEFAULT.JOBDIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)

    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    #  Load in Historical Referenced trajectory file, filter out proteins & slice
    traj  = md.load(rawfile, top=pdbfile)
    traj.atom_slice(DEFAULT.ATOM_SELECT_FILTER(traj), inplace=True)
    
    #  Pick a random frame from this trajectory if none is provided
    if frame is None:
      frame = randint(0, traj.n_frames)

    coord = traj.slice(frame)

    logging.debug("  Source trajectory: %s   (frame # %d)", str(coord), frame)
    if debugstring is not None:
      logging.debug("\n##%s @ %d", debugstring, frame)

    # Save this as a temp file to set up simulation input file
    coord.save_pdb(coordFile)

    newsimJob = dict(workdir=jobdir,
        coord   = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = DEFAULT.TOPO,
        srcfile = rawfile,
        srcframe = frame,
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
      self.setStream('completesim', None)

      self.addMut('jcqueue')
      self.addMut('converge')
      self.addImmut('ctlSplitParam')
      self.addImmut('ctlDelay')
      self.addImmut('numLabels')
      self.addImmut('launch')
      self.addImmut('terminate')
      self.addImmut('weight_alpha')
      self.addImmut('weight_beta')
      self.addAppend('timestep')
      self.addAppend('observe')

      # Update Base Slurm Params
      self.slurmParams['cpus-per-task'] = 24

      self.modules.add('namd')


    def term(self):
      # For now
      return False

    def split(self):

      # catalog = self.getCatalog()

      # TODO:  Provide better organization/sorting of the input queue based on weights
      # For now: just take the top N
      # split = self.data['ctlSplitParam']
      # immed = self.data['LDIndexList'][:split]

      immed = [] if len(self.data['completesim']) == 0 else ['completesim']
      return immed,None

    def fetch(self, i):
      # return {k.decode():np.fromstring(v, dtype=np.float64) for k, v in self.catalog.hgetall(wrapKey('idx', i)).items()}
      # return {i: loadNPArray(self.catalog, wrapKey('idx', i))}
      # FOR NOW Get all jobs available

      pipe = self.catalog.pipeline()
      pipe.lrange('completesim', 0, -1)
      pipe.delete('completesim')
      vals = pipe.execute()
      return vals[0]

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']


    # def execute(self, ld_index):
    def execute(self, job_list):
      logging.debug('CTL MT')

      logging.debug("============================  <PRE-PROCESS>  =============================")

      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

      numLabels = self.data['numLabels']

      delta_tmat = np.zeros(shape=(numLabels, numLabels))
      logging.debug("Processing output from %d  simulations", len(job_list))
      for job in job_list:
        logging.debug("  Loading data for simulation: %s", job)

        #  Do we need jc info here (provonance or )
        history = self.catalog.load(wrapKey('jc', job))
        traj_delta  = self.catalog.loadNPArray(wrapKey('delta', job))
        delta_tmat += traj_delta

      logging.debug('Delta Matrix: \n%s', str(delta_tmat))

      logging.debug("============================  <WEIGHT CALC>  =============================")

 
      #  TODO:  Consistency
      # Load Transition Matrix (& TODO: Historical index state labels)
      # tmat = loadNPArray(self.catalog, 'transitionmatrix')
      # if tmat is None:
      #   tmat = np.zeros(shape=(5,5))    # TODO: Move to init
      # tmat += delta_tmat


      # Weight Calculation: create a logistic function using current observation distribution
      totalBins = numLabels ** 2

      # 1. Calc "preference" part of the weight
      observe_before = np.copy(self.data['observe'])
      self.data['observe'] += delta_tmat
      observe = self.data['observe']

      # Calc convergence on probability distro (before & after)
      logging.debug("Calculating Probability Distributions for each state...")
      probDistro_before = np.zeros(shape=(numLabels, numLabels))
      probDistro        = np.zeros(shape=(numLabels, numLabels))
      for n in range(numLabels):
        numTrans_before = np.sum(observe_before[n]) - observe_before[n][n]
        probDistro_before[n] = observe_before[n] / numTrans_before
        probDistro_before[n][n] = 0
        numTrans = np.sum(observe[n]) - observe[n][n]
        probDistro[n] = observe[n] / numTrans
        probDistro[n][n] = 0
      delta        = np.zeros(shape=(numLabels, numLabels))
      delta = abs(probDistro - probDistro_before)

      bins = [(x, y) for x in range(numLabels) for y in range(numLabels)]

      logging.debug("Calculating transition rarity...")
        #  Isolate rare events (s.t. rare event seen less than mean)
      rareObserve = np.choose(np.where(observe.flatten() < np.mean(observe)), observe.flatten())

        #  Est. mid point of logistic curve by accounting for distribution skew
      midptObs = np.mean(observe) * skew(observe) + np.median(observe)

        #  Create the function
      rarityFunc = makeLogisticFunc(1, 1 / np.std(rareObserve), midptObs)

      # 2. Fatigue portion based on # times each bin was "launched"
      launch = self.data['launch']
      quota = np.sum(launch) / (totalBins - numLabels)

      # TODO:   Load New Transition Matrix  if consistency is necessary, otherwise use locally updated tmat

      #  3. Apply constants. This can be user influenced
      alpha = self.data['weight_alpha']
      beta = self.data['weight_beta']

      # 4. Calculate weight (note: follow 2 are retained for debugging only)

      #  UPDATED:  Weights CALC as a factor of rare events & instability in delta calc
      rarity = np.zeros(shape=(numLabels, numLabels))
      # fatigue = np.zeros(shape=(numLabels, numLabels))
      logging.debug("Calculating control weights...")
      weight = {}
      quotasq = quota**2
      for i in range(numLabels):
        for j in range(numLabels):
          rarity[i][j] = rarityFunc(observe[i][j] - midptObs)
          weight[(i,j)] =  0 if i == j else alpha * rarity[i][j] + beta * delta[i][j]

          # Apply a `fatigue` factor; bins are fatigued if run more than their quota
          weight[(i,j)] *= 0 if launch[i][j] > quota else (quota - launch[i][j])**2/quotasq
 
      # totalObservations = np.sum(tmat)
      # tmat_distro = {(i, j): tmat[i][j]/totalObservations for i in range(numLabels) for j in range(numLabels)}
      # quota = max(totalObservations / len(tmat_distro.keys()), len(tmat_distro.keys()))
      # pref = {(i, j): max((quota-tmat[i][j])/quota, 1/quota) for i in range(numLabels) for j in range(numLabels)}

      updatedWeights = sorted(weight.items(), key=lambda x: x[1], reverse=True)


      #  SCHEDULING   -----------------------------------------
      logging.debug("============================  <SCHEDULING>  =============================")

      #  5. Load JC Queue and all items within to get respective weights and projected target bins
      curqueue = []
      logging.debug("Loading Current Queue of %d items", len(self.data['jcqueue']))
      debug = True
      configlist = self.catalog.load([wrapKey('jc', job) for job in self.data['jcqueue']])
      for config in configlist.values():

        # Dampening Factor: proportional it currency (if no ts, set it to 1)
        jc_ts = config['timestep'] if 'timestep' in config else 1

        #  TODO: May want to consider incl convergence of sys at time job was created
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
        eqwlow = 0 if np.isnan(existingQueue[0]['weight']) else existingQueue[0]['weight']
        eqwhi  = 0 if np.isnan(existingQueue[-1]['weight']) else existingQueue[-1]['weight']
        logging.debug("Existing Queue has %d items between weights: %0.5f - %0.5f", len(existingQueue), eqwlow, eqwhi)
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

      # Rank order list of observed transition bins for each state
      rarityorderstate = np.argsort(observe.sum(axis=1))
      rarityordertrans = np.zeros(shape=(numLabels,numLabels))
      for i in range(numLabels):
        np.copyto(rarityordertrans[i], np.argsort(observe[i]))
   
      while len(jcqueue) < DEFAULT.MAX_JOBS_IN_QUEUE:

        if oldjob == None and targetBin == None:
          logging.info("No more jobs to queue.")
          break

        if (targetBin == None) or (oldjob and oldjob['weight'] > targetBin[1]) or (oldjob and np.isnan(targetBin[1])):
          jcqueue.append(oldjob['name'])
          oldjob['weight'] = 0 if np.isnan(oldjob['weight']) else oldjob['weight']
          logging.debug("Re-Queuing OLD JOB `%s`   weight= %0.5f", oldjob['name'], oldjob['weight'])
          oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()

        else:
          A, B = targetBin[0]
          logging.debug("\n\nCONTROL: Target transition bin  %s  (new job #%d,  weight=%0.5f)", str((A, B)), len(newJobCandidate), targetBin[1])

          # Identify a pool of starting points Start with Target Bin
          cpool = kv2DArray.key('candidatePool', A, B)
          selectedbin = (A,B)

          # Flip direction of transition
          if self.catalog.llen(cpool) == 0:
            logging.info("No DEShaw reference for transition, (%d, %d)  -- checking reverse direction", A, B)
            cpool = kv2DArray.key('candidatePool', B, A)
            selectedbin = (B,A)

          if self.catalog.llen(cpool) == 0:
            logging.info("No DEShaw reference for transition, (%d, %d)  -- checking wells in this order: %s", B, A, str(rarityorderstate))
            for RS in rarityorderstate:
              cpool = kv2DArray.key('candidatePool', RS, RS)
              if self.catalog.llen(cpool) == 0:
                logging.info("No DEShaw reference for transition, (%d, %d) ", RS, RS)

              else: 
                logging.info("FOUND DEShaw start point from transition, (%d, %d) ", RS, RS)
                selectedbin = (RS,RS)
                break

          logging.debug('Final Candidate Popped from Pool %s  of  %d  candidates', cpool, self.catalog.llen(cpool))

          # Pick a random trajectory from the bin  TODO:  popleft this
          candidate = self.catalog.lpop(cpool)
          sourceTraj, srcFrame = candidate.split(':')
          dstring = "####SELECT_TRAJ@ %s @ %s @ %s @ %s" % (str(targetBin[0]), str(selectedbin), sourceTraj, srcFrame)
          self.catalog.rpush(cpool, candidate)

          # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
          #   or we have a pre-fetch algorithm to get the data
          # Back-project  <--- Move to separate Function tied to decision history
          # For now:
          if isinstance(sourceTraj, int) or sourceTraj.isdigit():      # It's a DEShaw file
            pdbfile, archiveFile = getHistoricalTrajectory(sourceTraj)
          else:
            archiveFile = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.dcd' % sourceTraj)
            pdbfile     = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.pdb' % sourceTraj)

          # Generate new set of params/coords
          jcID, params = generateNewJC(archiveFile, pdbfile, DEFAULT.TOPO, DEFAULT.PARM, int(srcFrame), debugstring=dstring)

          # TODO: DERIVE RUNTIME HERE
          runtime = self.catalog.get(kv2DArray.key('runtime', A, B))
          # Update Additional JC Params and Decision History, as needed
          jcConfig = dict(params,
              name    = jcID,
              runtime = runtime,
              interval = 500,
              temp    = 310,
              state   = A,
              weight  = targetBin[1],
              timestep = self.data['timestep'],
              gc      = 1,
              application   = DEFAULT.APPL_LABEL,
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


      # Update the selection matrix
      # selObs_Store = kv2DArray(self.catalog, 'selection')
      # selObs_Store.merge(selectionTally)

      # Mark obsolete jobs for garbage collection:
      logging.info("Marking %d obsolete jobs for garbage collection", len(existingQueue))
      while len(existingQueue) > 0:
        config = existingQueue.popleft()
        config['gc'] = 0

        # Add gc jobs it to the state to write back to catalog (flags it for gc)
        self.addMut(wrapKey('jc', config['name']), config)


      # CONVERGENCE CALCUALTION  -------------------------------------
      logging.debug("============================  <CONVERGENCE -- CALC W/WEIGHTS>  =============================")

      logging.debug("OBS_MATRIX_DELTA:\n" + str(delta_tmat))
      logging.debug("OBS_MATRIX_UPDATED:\n" + str(observe))
      logging.debug("LAUNCH_MATRIX:\n" + str(launch))
      logging.debug("CONVERGENCE_PROB_DISTRO:\n" + str(probDistro))
      logging.debug("OBS_RARITY:\n" + str(rarity))
      logging.debug("CONV_DELTA:\n" + str(delta))
      logging.debug("CTL_WEIGHT:")
      for i in range(5):
        logging.debug("  %s", str(['%0.5f'% weight[(i,k)] for k in range(5)]))

      globalconvergence = np.sum(abs(probDistro - probDistro_before))
      logging.info("GLOBAL_CONVERGENCE: %f", globalconvergence)
      for i in range(5):
        logging.info("STATE_CONVERGENCE for %d: %f", i, np.sum(delta[i]))

      globalconvergence_rate = globalconvergence / len(job_list)
      logging.info("GLOBAL_CONVERGENCE_RATE: %f", globalconvergence_rate)
      for i in range(5):
        logging.info("STATE_CONVERGENCE_RATE for %d: %f", i, np.sum(delta[i])/len(job_list))

      clist_entry = {'timestep': self.data['timestep'], 'numjobs': len(job_list), 'convg': globalconvergence, 'convs': str([np.sum(delta[i]) for i in range(5)])}
      self.catalog.rpush('globalconvergelist', json.dumps(clist_entry))

      self.data['jcqueue'] = list(jcqueue)

      logging.debug("   JCQUEUE:  %s", str(self.data['jcqueue']))
      # Update Each new job with latest convergence score and save to catalog(TODO: save may not be nec'y)
      logging.debug("Updated Job Queue length:  %d", len(self.data['jcqueue']))
      for jcid, config in newJobCandidate.items():
        config['converge'] = self.data['converge']
        self.addMut(wrapKey('jc', jcid), config)
 
      return list(newJobCandidate.keys())



if __name__ == '__main__':
  mt = controlJob(__file__)
  mt.run()




# OLD COVERGENCE:
      # logging.debug("Observations MAT:\n" + str(tmat))
      # logging.debug("Fatigue:\n" + str(fatigue))

      # # Load Selection Matrix
      # smat = loadNPArray(self.catalog, 'selectionmatrix')
      # if smat is None:
      #   smat = np.full((5,5), 1.)    # SEED Selection matrix (it cannot be 0) TODO: Move to init

      # # Merge Update global selection matrix
      # smat += selectionTally
      # logging.debug("SMAT:\n" + str(smat))
      # # Load Convergence Matrix
      # cmat = loadNPArray(self.catalog, 'convergencematrix')
      # if cmat is None:
      #   cmat = np.full((5,5), 0.04)    # TODO: Move to init
      # logging.debug("CMAT:\n" + str(cmat))

      # # Remove bias from selection and add to observations 
      # #  which gives the unbiased, uniform selection 
      # inverseSelectFunc = lambda x: (np.max(x)) - x

      # #  For now, Assume a 4:1 ratio (input:output) and do not factor in output prob distro
      # unbias = 3 * inverseSelectFunc(smat) + tmat
      # logging.debug("UNBIAS:\n" + str(unbias))

      # # Calculcate "convergence" matrix as unbias(i,j) / sel(i.j)
      # updated_cmat = unbias / np.sum(unbias)
      # logging.debug("CMAT_0:\n" + str(cmat))
      # logging.debug("CMAT_1:\n" + str(updated_cmat))


      # # Calculate global convergence as summed difference:  cmat_t1 - cmat_t0
      # convergence = np.sum(abs(updated_cmat - cmat))


    # def oldcode():
    #   pass
      # TODO:  Treat Archive as an overlay service. For now, wrap inside here and connect to it


      # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
      # redis_storage = RedisStorage(archive)
      # config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
      # if not config:
      #   logging.error("LSHash not configured")
      #   #TODO: Gracefully exit
      #   return []

      # # Create empty lshash and load stored hash
      # lshash = DEFAULT.getEmptyHash()
      # lshash.apply_config(config)
      # indexSize = self.data['num_var'] * self.data['num_pc']
      # logging.debug("INDEX SIZE = %d:  ", indexSize)
      # engine = nearpy.Engine(indexSize, 
      #       lshashes=[lshash], 
      #       storage=redis_storage)

      # Load current set of known states
      #  TODO:  Injection Point for clustering. If clustering were applied
      #    this would be much more dynamic (static fileload for now)
      # labels = loadLabels()
      # labelNames = getLabelList(labels)
      # numLabels = len(labelNames)

 
      # PROBE   ------------------------
      # logging.debug("============================  <PROBE>  =============================")

      # Set initial params for index calculations
      # prevState = -1    # To track each state transition
      # prevTrajectory = None   # To check for unique trajectories
      # decisionHistory = {}    # Holds Decision History data from source JC used to create the data
      # observationDistribution = {}   #  distribution of observed states (for each input trajectory)
      # observationCount = {}
      # statdata = {}
      # newvectors = []
      # delta_tmat = np.zeros(shape=(numLabels, numLabels))


      # ### UPDATED USING RMS as probe function


      # for sourceJC in rms_list.keys():
      #   config = {}
      #   self.catalog.load({wrapKey('jc', sourceJC): config})  #TODO: Load these up front
      #   for k, v in config.items():
      #     logging.debug("  %s:  %s", k, str(v))
      #   decisionHistory[sourceJC] = config
      #   if 'state' not in config:
      #     logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
      #     prevState = None  # Do not count transition into first probed index
      #   else:
      #     logging.debug("New Index to analyze, %s: Source JC was previously in state %d", sourceJC, prevState)
      #     prevState = config['state']

      #   sourceJCKey = wrapKey('jc', sourceJC)
      #   self.addToState(sourceJCKey, config)
      #   # statdata[sourceJC] = {}


      #   # Analyze Each conform (Should this move to anlmd execute)
      #   logging.info('Checking RMS for each conform of %d length trajectory', len(rms_list[sourceJC]))
      #   for num, rms in enumerate(rms_list[sourceJC]):

      #     #  Check proximity to nearest 2 centroids
      #     #  TODO: Account for proximity to more than 2 (may be "interesting" data point)



      #   state = A
      #   labeledBin = (A, B)
      #   statdata[sourceJC]['bin'] = str(labeledBin)
      #   statdata[sourceJC]['count'] = count.tolist()
      #   statdata[sourceJC]['clust'] = clust.tolist()


      #     # Increment the transition matrix  
      #     #  (NOTE: SHOULD WE IGNORE 1ST INDEX IN A TRAJECTORY, since it doesn't describe a transition?)
      #     # if prevState == None:
      #     #   prevState = state
      #     logging.debug("    Index classified in bin:  (%d, %d)", A, B)
      #     if prevState is None:
      #       logging.debug("    First Index in Traj")
      #     else:
      #       logging.debug("    Transition (%d, %d)", prevState, state)
      #     logging.debug("      Clustered at: %s", str(clust/np.sum(clust)))
      #     logging.debug("      Counts:       %s", str(count))
      #     logging.debug("      NN Index:     %s", neigh[0][1])

      #     # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
      #     prevState = state






      # NOTE: ld_index is a list of indexed trajectories. They may or may NOT be from
      #   the same simulation (allows for grouping of multiple downstream data into 
      #   one control task). Thus, this algorithm tracks subsequent windows within one
      #   trajectories IOT track state changes. 

      #  Theta is calculated as the probability of staying in 1 state (from observed data)
      # theta = .6  #self.data['observation_counts'][1] / sum(self.data['observation_counts'])
      # logging.debug("  THETA  = %0.3f   (static for now)", theta)

      # for key in sorted(ld_index.keys()):

      #   index = np.array(ld_index[key])   # Get actual Index for this window
      #   sourceJC, frame = key.split(':')  # Assume colon separator
      #   # logging.info(' Index Loaded from %s:   Shape=%s,  Type=%s', sourceJC, str(index.shape), str(index.dtype))

      #   # Get Decision History for the index IF its a new index not previously processed
      #   #  and initialize observation distribution to zeros
      #   if sourceJC != prevTrajectory:
      #     observationDistribution[sourceJC] = np.zeros(5)
      #     observationCount[sourceJC] = np.zeros(5)

      #     #  TODO: Load indices up front with source history
      #     config = {}
      #     self.catalog.load({wrapKey('jc', sourceJC): config})  #TODO: Load these up front
      #     for k, v in config.items():
      #       logging.debug("  %s:  %s", k, str(v))
      #     decisionHistory[sourceJC] = config
      #     if 'state' not in config:
      #       logging.info("New Index to analyze, %s: NO Historical State Data", sourceJC)
      #     else:
      #       logging.debug("New Index to analyze, %s: Source JC was previously in state %d", sourceJC, prevState)

      #     prevState = None  # Do not count transition into first probed index

      #     sourceJCKey = wrapKey('jc', sourceJC)
      #     self.addToState(sourceJCKey, config)
      #     prevTrajectory = sourceJC
      #     statdata[sourceJC] = {}

      #     # Note:  Other Decision History is loaded here

      #   logging.info("  Probing `%s` window at frame # %s  (state %s)", sourceJC, frame, str(prevState))
      #   statdata[sourceJC][frame] = {}

      #   # Probe historical index
      #   neigh = engine.neighbours(index)
      #   if len(neigh) == 0:
      #     logging.info ("    Found no near neighbors for %s", key)
      #   else:
      #     logging.debug ("    Found %d neighbours:", len(neigh))

      #     #  Track the weighted count (a.k.a. cluster) for this index's nearest neighbors
      #     clust = np.zeros(5)
      #     count = np.zeros(5)
      #     for n in neigh:
      #       nnkey = n[1]
      #       distance = n[2]
      #       trajectory, seqNum = nnkey[2:].split('-')

      #       # CHANGED: Grab state direct from label (assume state is 1 char, for now)
      #       nn_state = int(nnkey[0])  #labels[int(trajectory)].state
      #       # logging.info ("    NN:  %s   dist = %f    state=%d", nnkey, distance, nn_state)
      #       count[nn_state] += 1
      #       clust[nn_state] += abs(1/distance)

      #       # Add total to the original input trajectory (for historical and feedback decisions)
      #       observationCount[sourceJC][nn_state] += 1
      #       observationDistribution[sourceJC][nn_state] += abs(1/distance)

      #     # Classify this index with a label based on the highest weighted observed label among neighbors
      #     clust = clust/np.sum(clust)
      #     order = np.argsort(clust)[::-1]
      #     A = order[0]
      #     B = A if clust[A] > theta else order[1]
      #     delta_tmat[A][B] += 1

      #     state = A
      #     labeledBin = (A, B)
      #     statdata[sourceJC][frame]['state'] = str(state)
      #     statdata[sourceJC][frame]['bin'] = str(labeledBin)
      #     statdata[sourceJC][frame]['count'] = count.tolist()
      #     statdata[sourceJC][frame]['clust'] = clust.tolist()


      #     # Increment the transition matrix  
      #     #  (NOTE: SHOULD WE IGNORE 1ST INDEX IN A TRAJECTORY, since it doesn't describe a transition?)
      #     # if prevState == None:
      #     #   prevState = state
      #     logging.debug("    Index classified in bin:  (%d, %d)", A, B)
      #     if prevState is None:
      #       logging.debug("    First Index in Traj")
      #     else:
      #       logging.debug("    Transition (%d, %d)", prevState, state)
      #     logging.debug("      Clustered at: %s", str(clust/np.sum(clust)))
      #     logging.debug("      Counts:       %s", str(count))
      #     logging.debug("      NN Index:     %s", neigh[0][1])

      #     # TODO: Consistency Decision. When does the transition matrix get updated and snych's with other control jobs????
      #     prevState = state


      # Build Decision History Data for the Source JC's from the indices
      # transitionBins = kv2DArray(archive, 'transitionBins', mag=5, dtype=str, init=[])      # Should this load here (from archive) or with other state data from catalog?
                # BUILD ARCHIVE ONLINE:
      #     if DEFAULT.BUILD_ARCHIVE:
      #       label = '%d %s-%s' % (state, sourceJC, frame)
      #       newvectors.append((index, label))

      # if DEFAULT.BUILD_ARCHIVE:
      #   for vect in newvectors:
      #     logging.info("ADDING new Index to the Archive:  %s", vect[1])
      #     engine.store_vector(vect[0], vect[1])
      # else:
      #   logging.debug('BUILD Archive is off (not storing)')

      #  DECISION HISTORY  --------------------------
      # logging.debug("============================  <DECISION HIST>  =============================")


      #  Process output data for each unque input trajectory (as produced by a single simulation)
      # launch_delta = np.zeros(shape=(numLabels, numLabels))
      # for sourceJC, cluster in observationDistribution.items():
      #   logging.debug("\nFinal processing for Source Trajectory: %s   (note: injection point here for better classification and/or move to analysis)", sourceJC)
      #   if sum(observationDistribution[sourceJC]) == 0:
      #     logging.debug(" No observed data for, %s", sourceJC)
      #     continue

      #   #  TODO: Another injection point for better classification. Here, the classification is for the input trajectory
      #   #     as a whole for future job candidate selection. 

      #   logging.debug("  Observations:      %s", str(observationCount[sourceJC]))
      #   logging.debug("  Cluster weights:   %s", str(cluster/np.sum(cluster)))
      #   index_distro = cluster / sum(cluster)
      #   # logging.debug("Observations for input index, %s\n  %s", sourceJC, str(distro))
      #   sortedLabels = np.argsort(index_distro)[::-1]    # Should this be normaized to check theta????? or is theta a global calc?
      #   statelist = [int(statdata[sourceJC][f]['state']) for f in sorted(statdata[sourceJC].keys())]

      #   stateA = statelist[0]
      #           # idxcount = np.zeros(len(numLabels))
      #   stateB = None
      #   transcount = 0
      #   for n in statelist[1:]:
      #     transcount += 1
      #     if stateA != n:
      #       stateB = n
      #       logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", sourceJC, stateA, stateB)
      #       break
      #   if stateB is None:
      #     logging.debug(" Trajectory `%s`  classified as staying in state :  %d", sourceJC, stateA)
      #     stateB = stateA
      #     # idxcount[n] += 1
      #   # sortedLabels = np.argsort(idxcount)[::-1]    # Should this be normaized to check theta????? or is theta a global calc?

      #   # stateA = sortedLabels[0]
      #   # stateB = sortedLabels[1] if idxcount[sortedLabels[1]] > 0 else stateA


      #   # Source Trajectory spent most of its time in 1 state
      #   # if max(index_distro) > theta: 
      #   #   logging.debug(" Trajectory `%s`  classified as staying in state :  %d", sourceJC, stateA)
      #   #   stateB = stateA

      #   # # Observation showed some transition 
      #   # else:
      #   #   stateB = sortedLabels[1]
      #   #   logging.debug(" Trajectory `%s`  classified as in-between states :  %d  &  %d", sourceJC, stateA, stateB)

      #   #  Add this candidate to list of potential pools (FIFO: popleft if pool at max size):
      #   if len(self.data[candidPoolKey(stateA, stateB)]) >= DEFAULT.CANDIDATE_POOL_SIZE:
      #     del self.data[candidPoolKey(stateA, stateB)][0] 
      #   logging.info("BUILD_CANDID_POOL: Added `%s` to candidate pool (%d, %d)", sourceJC, stateA, stateB)
      #   self.data[candidPoolKey(stateA, stateB)].append(sourceJC)

      #   outputBin = str((stateA, stateB))
      #   if 'targetBin' in self.data[wrapKey('jc', sourceJC)]:
      #     inputBin  = self.data[wrapKey('jc', sourceJC)]['targetBin']
      #     a, b = eval(inputBin)

      #     #  INCREMENT "LAUNCH" Counter
      #     launch_delta[a][b] += 1

      #     logging.info("  Original Target Bin was       :  %s", inputBin) 
      #     logging.info("  Actual Trajectory classified  :  %s", outputBin) 
      #     logging.debug("    TODO: ID difference, update ML weights, provonance, etc..... (if desired)")

      #   # Update other History Data
      #   key = wrapKey('jc', sourceJC)
      #   self.data[key]['actualBin'] = outputBin
      #   self.data[key]['application'] = DEFAULT.APPL_LABEL
      #   logging.debug("%s", str(statdata[sourceJC]))
      #   self.data[key]['indexList'] = json.dumps(statdata[sourceJC])

      ####   This is end of processing a single Job Candidate and here begins the cycle of finding next set of JC's
      #  Everything above  could actually co-locate with the analysis thread whereby the output of the analysis
      #  is a label or set of labels (vice a low-dim index). 


      #  WEIGHT CALCULATION ---------------------------

      # Increment the timestep

