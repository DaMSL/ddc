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

from collections import namedtuple, deque, OrderedDict

import redisCatalog
from common import *
from macrothread import macrothread
from kvadt import kv2DArray
from slurm import slurm
from random import choice, randint
from deshaw import *
from indexing import *
from kdtree import KDTree
import datareduce
import deshaw

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


def generateNewJC(trajectory, topofile=deshaw.TOPO, parmfile=deshaw.PARM, jcid=None):

    logging.debug("Generating new simulation coordinates from:  %s", str(trajectory))

    # Get a new uid
    if jcid is None:
      jcuid = getUID() 
    else:
      jcuid = jcid

    #  TODO:  Should coords be pre-fetched and pulled from pre-fetch location?
    jobdir = os.path.join(DEFAULT.JOBDIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)

    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    # Save this as a temp file to set up simulation input file
    trajectory.save_pdb(coordFile)

    newsimJob = dict(workdir=jobdir,
        coord   = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = topofile,
        parm    = parmfile)

    logging.info("  Running PSFGen to set up simulation pdf/pdb files.")
    stdout = executecmd(psfgen(newsimJob))
    logging.debug("  PSFGen COMPLETE!!\n")

    os.remove(coordFile)
    del newsimJob['coord']

    return jcuid, newsimJob



def bootstrap (source, N=1000, interval=.95):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo

  # Get unique label/category/hcube ID's
  V = set()
  for i in source:
    V.add(i)
  print ("BS: labels ", str(V))
  L = len(source)

  # Calculate mu_hat from bootstrap
  mu_hat = {}
  groupby = {v_i: 0 for v_i in V}
  for s in source:
    groupby[s] += 1
  for v_i in V:
    mu_hat[v_i] = groupby[v_i]/L

  # Iterate for each bootstrap and generate statistical distributions
  boot = {i : [] for i in V}
  for i in range(N):
    strap   = [source[np.random.randint(L)] for n in range(L)]
    groupby = {v_i: 0 for v_i in V}
    for s in strap:
      groupby[s] += 1
    for v_i in V:
      boot[v_i].append(groupby[v_i]/L)
  probility_est = {}
  for v_i in V:
    P_i = np.mean(boot[v_i])
    delta  = np.array(sorted(boot[v_i]))  #CHECK IF mu or P  HERE
    ciLO = delta[round(N*ci_lo)]
    ciHI = delta[round(N*ci_hi)]
    probility_est[v_i] = (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i)
  return probility_est



def q_select (T, value):
  idx_list = []
  for i, elm in enumerate(T):
    if elm == value:
      idx_list.append(i)
  return idx_list




class controlJob(macrothread):
    def __init__(self, fname):
      macrothread.__init__(self, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setStream('completesim', None)

      self.addMut('jcqueue')
      self.addMut('converge')
      self.addMut('ctlIndexHead')
      self.addImmut('ctlSplitParam')
      self.addImmut('ctlDelay')
      self.addImmut('numLabels')
      self.addImmut('terminate')
      self.addImmut('weight_alpha')
      self.addImmut('weight_beta')

      self.addImmut('launch')
      self.addAppend('timestep')
      # self.addAppend('observe')
      self.addMut('runtime')

      self.addImmut('pcaVectors')


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

    # DEFAULT VAL for i for for debugging
    def fetch(self, i=100):
      end = min(self.data['ctlIndexHead'] + int(i), self.catalog.llen('completesim'))
      return end

    def configElasPolicy(self):
      self.delay = self.data['ctlDelay']


    def backProjection(self, index_list):
      source_points = []
      cache_miss = []
      
      # DEShaw topology is assumed here
      ref = deshaw.deshawReference()

      #  TODO: MOVE TO CACHE ???
      logging.debug('Checking cache for %d points for reweighting', len(index_list))
      for i in index_list:
        pt = None   #  pt = cache.request(i)  
        if pt is None:
          # TODO: File retrieval
          cache_miss.append(i)
        else:
          source_points.append(pt)

      # Package all cached points into one trajectory
      logging.debug('Consolidating %d points for all cache hits', len(source_points))
      if len(source_points) > 0:
        source_traj = md.Trajectory(source_points, ref.top)
      else:
        source_traj = None

      # Archive File retrieval for all cache-miss points
      # TODO:  May need to deference to local RMS Subspace first, then Back to high-dim
      pipe = self.catalog.pipeline()
      for pt in cache_miss:
        pipe.lindex('xid:reference', pt)
      framelist = pipe.execute()
      print('framelist---> ', framelist)
      # for i in range(len(framelist)):
      #   if framelist[i] is None:
      #     dcdfile_num = cache_miss[i] // 100
      #     frame_num = (cache_miss[i] - 100*dcdfile_num) * 10
      #     framelist[i] = str((dcdfile_num, frame_num))
          
      # ID unique files and Build index filter for each unique file
      #  Account for DEShaw Files (derived from index if index not in catalog)
      atomfilter = {}
      for i, idx in enumerate(framelist):
        print ('FRAMELIST --> ', idx, type(idx))
        if idx is None:
          dcdfile_num = cache_miss[i] // 100
          frame = (cache_miss[i] - 100*dcdfile_num) * 10
          # Use negation to indicate DEShaw file (not in fileindex in catalog)
          file_index = (-1 * dcdfile_num)
        else:
          file_index, frame = eval(idx)
        if file_index not in atomfilter:
          atomfilter[file_index] = []
        atomfilter[file_index].append(frame)
      print ('atomfilter --> ', atomfilter)

      # # Get List of files
      # filelist = {}
      # for file_index in atomfilter.keys():
      #   if file_index >= 0:
      #     filelist[file_index] = self.catalog.lindex('xid:filelist', file_index)
      #   else:
      #     filelist[file_index] = deshaw.getDEShawfilename(-1 * file_index)
      # print ('filelist --> ', filelist)
        
      # Add high-dim points to list of source points in a trajectory
      for idx in atomfilter.keys():
        if idx >= 0:
          filename = self.catalog.lindex('xid:filelist', idx)
          print ('Loading recent file: ', filename)
          traj = datareduce.load_trajectory(filename)
          print ('    pre-filtered: ', str(traj))
        else:
          filename = deshaw.getDEShawfilename(-1 * idx)
          print ('Loading DEShaw file: ', filename)
          traj = deshaw.loadDEShawTraj(-1 * idx, filt='all')
          traj.atom_slice(traj.top.select('protein'), inplace=True)
          print ('    pre-filtered: ', str(traj))
        traj = traj.slice(atomfilter[idx])
        print ('    post-filtered: ', str(traj))
        if source_traj is None:
          source_traj = traj
        else:
          source_traj.join(traj, check_topology=False)
        print ('    returned: ', str(source_traj))
        print(' Pulling from file:  ', idx, source_traj.n_frames)      

      return source_traj

    # def execute(self, ld_index):
    def execute(self, thru_index):
      logging.debug('CTL MT')

    # PRE-PROCESSING ---------------------------------------------------------------------------------
      logging.debug("============================  <PRE-PROCESS>  =============================")

      self.data['timestep'] += 1
      logging.info('TIMESTEP: %d', self.data['timestep'])

    # LOAD all new subspaces (?) and values
      # Load new RMS Labels
      labeled_pts_rms = self.catalog.lrange('label:rms', 0, -1)

      # labeled_pts_rms = self.catalog.lrange('label:rms', self.data['ctlIndexHead'], thru_index)
      # varest_counts_rms = {}
      # total = 0
      # for i in labeled_pts_rms:
      #   total += 1
      #   if i not in varest_counts_rms:
      #     varest_counts_rms[i] = 0
      #   varest_counts_rms[i] += 1

      # Load PCA Subspace of hypecubes
      hcube_mapping = json.loads(self.catalog.get('hcube:pca'))
      logging.debug('# Loaded keys = %d', len(hcube_mapping.keys()))

      # TODO: accessor function is for 1 point (i) and 1 axis (j). 
      #  Optimize by changing to pipeline  retrieval for all points given 
      #  a list of indices with an axis
      func = lambda i,j: np.fromstring(self.catalog.lindex('subspace:pca', i))[j]
      logging.debug("Reconstructing the tree...")
      hcube_tree = KDTree.reconstruct(hcube_mapping, func)

    # Calculate veriable PDF estimations for each subspace via bootstrapping:
      logging.debug("=======================  <SUBSPACE CONVERENCE>  =========================")

      # FOR NOW LOAD ALL and bootstrap that.....
      logging.info("RMS Labels for _each_ point loaded. Bootstrapping.....")
      pdf_rms = bootstrap(labeled_pts_rms)
      logging.info("Bootstrap complete. PDF for each variable:")
      for k, p in pdf_rms.items():
        logging.info('  %s:  u=%0.4f  CI_lo=%0.4f  CI_hi=%0.4f  conv=%0.4f', k, p[0], p[1], p[2], p[3])

    # IMPLEMENT USER QUERY with REWEIGHTING:
      logging.debug("=======================  <QUERY PROCESSING>  =========================")
      #   Using RMS and PCA, umbrella sample transitions out of state 3

      # 1. get all points in state label=(3,2) in RMS:
      label = '(3, 2)'
      print('Projecting points in label: ', str(label))
      rms_indexlist = q_select(labeled_pts_rms, label)

      #  REWEIGHT OPERATION
      # Back-project all points to higher dimension <- as consolidatd trajectory
      source_traj = self.backProjection(rms_indexlist)
      traj = datareduce.filter_heavy(source_traj)
      logging.debug('Consolidated all RMS points in HD space: %s', str(traj))

      # 2. project into PCA space 
      rms_proj_pca = datareduce.PCA(traj.xyz, self.data['pcaVectors'], 3)
      logging.debug('Projects to PCA:  %s', str(rms_proj_pca.shape))

      # 3. Map into existing hcubes  (project only for now)
      #  A -> PCA  and B -> RMS
      #   TODO:  Insert here and deal with collapsing geometric hcubes in KDTree
      hcube_B = {}
      for i in range(len(rms_proj_pca)):
        hcube = hcube_tree.project(rms_proj_pca[i])
        # print('POJECTING ', i, '  ', rms_proj_pca[i], '  ---->  ', hcube)
        if hcube not in hcube_B:
          hcube_B[hcube] = []
        hcube_B[hcube].append(i)
      hcube_sizes = [len(k) for k in hcube_B.keys()]
      low_dim = max(hcube_sizes) + 1
      total = sum(hcube_sizes)
      # CHeck wght calc
      wgt_B = {k: len(hcube_B[k])/(total*(low_dim - len(k))) for k in hcube_B.keys()}
      comb_wgt = {k: wgt_B[k] for k in hcube_B.keys()}

      logging.debug('Projected %d points into PCA. Found the following keys:', len(rms_proj_pca))
      hcube_A = {}
      for k in hcube_B.keys():
        hcube_A[k] = hcube_tree.retrieve(k)
        logging.debug('   %15s  ->  %d A-pts    %d B-pts', k, len(hcube_A[k]), len(hcube_B[k]))
      hcube_sizes = [len(k) for k in hcube_A.keys()]
      low_dim = max(hcube_sizes) + 1
      total = sum(hcube_sizes)
      wgt_A = {k: len(hcube_A[k])/(total*(low_dim - len(k))) for k in hcube_B.keys()}
      comb_wgt = {k: wgt_A[k] for k in hcube_B.keys()}


      #  GAMMA FUNCTION
      gamma = lambda a, b : 1 - a * b
      # TODO: Factor in RMS weight
      comb_wgt = {k: gamma(wgt_A[k], wgt_B[k]) for k in hcube_B.keys()}
      total = sum(comb_wgt.values())
      print ('SUM T==== ', total)

      umbrella = OrderedDict()
      for k, v in comb_wgt.items():
        logging.debug('   %15s  ->  %4d A-pts   %4d B-pts   (wgt=%0.3f)', k, len(hcube_A[k]), len(hcube_B[k]),  v/total)
        umbrella[k] = (v/total) 
      keys = umbrella.keys()

      print ('SUM ==== ', sum(list(umbrella.values())))
    
    # EXECUTE SAMPLER
      logging.debug("=======================  <DATA SAMPLER>  =========================")

      # 1st Selection level: pick a HCube
      # Select N=20 new candidates
      #  TODO:  Number & Runtime of each job <--- Resource/Convergence Dependant
      candidates = np.random.choice(list(umbrella.keys()), 
           size=3, replace=True, p = list(umbrella.values()))

      print ('CANDIDATE HCUBES: ', candidates)
      # 2nd Selection Level: pick an index for each chosen cube (uniform)
      sampled_set = []
      for i in candidates:
        selected_index = np.random.choice(list(hcube_A[i]) + list(hcube_B[i]))
        traj = self.backProjection([selected_index])
        sampled_set.append(traj)

      # REDO CACHE CHECK FROM ABOVE!!!!!
    # Generate new starting positions
      jcqueue = OrderedDict()
      for start_traj in sampled_set:
        jcID, params = generateNewJC(start_traj)

        # TODO:  Update/check adaptive runtime, starting state
        jcConfig = dict(params,
              name    = jcID,
              runtime = 50000,
              interval = 500,
              temp    = 310,
              timestep = self.data['timestep'],
              gc      = 1,
              application   = DEFAULT.APPL_LABEL)

        logging.info("New Simulation Job Created: %s", jcID)
        for k, v in jcConfig.items():
          logging.debug("   %s:  %s", k, str(v))

        #  Add to the output queue & save config info
        jcqueue[jcID] = jcConfig
        logging.info("New Job Candidate Completed:  %s   #%d on the Queue", jcID, len(jcqueue))

    #  POST-PROCESSING  -------------------------------------
      logging.debug("============================  <POST-PROCESSING & OUTPUT>  =============================")
          
      # Clear current queue, mark previously queues jobs for GC, push new queue
      pipe = self.catalog.pipeline()
      pipe.lrange('jcqueue', 0, -1)
      pipe.delete('jcqueue')
      cursor = pipe.execute()
      existing_queue = deque(cursor[0])
      logging.info("Marking %d obsolete jobs for garbage collection", len(existing_queue))
      while len(existing_queue) > 0:
        config = existingQueue.popleft()
        config['gc'] = 0
        # Add gc jobs it to the state to write back to catalog (flags it for gc)
        self.addMut(wrapKey('jc', config['name']), config)

      self.data['jcqueue'] = list(jcqueue.keys())

      logging.debug("   JCQUEUE:  %s", str(self.data['jcqueue']))
      # Update Each new job with latest convergence score and save to catalog(TODO: save may not be nec'y)
      logging.debug("Updated Job Queue length:  %d", len(self.data['jcqueue']))
      for jcid, config in jcqueue.items():
        # config['converge'] = self.data['converge']
        self.addMut(wrapKey('jc', jcid), config)
 
      return list(jcqueue.keys())




if __name__ == '__main__':

  mt = controlJob(__file__)

  # GENDATA -- to manually generate pdb/psf files for specific DEShaw reference points
  GENDATA = False

  if GENDATA:
    wells = [(0, 2099, 684),
            (1, 630, 602),
            (2, 2364, 737),
            (3, 3322, 188),
            (4, 2108, 258)]
    print('GEN DATA!')

    #  REDO WELL
    for well in wells:
      state, win, frame = well

      logging.error(' NEED TO REDO Well start point gen')

      pdbfile, archiveFile = getHistoricalTrajectory(win)

      jcID, params = generateNewJC(archiveFile, ppdbfile, frame, jcid='test_%d'%state)

      jcConfig = dict(params,
          name    = jcID,
          runtime = 20000,
          interval = 500,
          temp    = 310,
          state   = state,
          weight  = 0.0,
          timestep = 0,
          gc      = 1,
          application   = DEFAULT.APPL_LABEL)

      print("Data Generated! ")
      print("Job = ", jcID)
      for k, v in jcConfig.items():
        logging.info('%s: %s', k, str(v))


      catalog = redisCatalog.dataStore(**DEFAULT.catalogConfig)
      catalog.hmset('jc_'+jcID, jcConfig)
    sys.exit(0)

  mt.run()



#  SOMEWHAT OLD


# #========================================

#     #  PREPROC
#       numLabels = self.data['numLabels']
#       totalBins = numLabels ** 2

#       logging.debug("Processing output from %d  simulations", len(job_list))

#       obs_delta = np.zeros(shape=(numLabels, numLabels))
      
#       #  Consolidate all observation delta's and transaction lists from recently run simulations
#       translist = {}
#       for job in job_list:
#         logging.debug("  Loading data for simulation: %s", job)

#         jkey = wrapKey('jc', job)

#         # Load Job Candidate params
#         params = self.catalog.load(jkey)

#         # Load the obsevation "delta" 
#         traj_delta  = self.catalog.loadNPArray(wrapKey('delta', job))
#         if traj_delta is not None:
#           obs_delta += traj_delta

#         # Load set of tranations actually visited for the current target bin
#         tbin = eval(params[jkey]['targetBin'])
#         trans = self.catalog.get(wrapKey('translist', job))
#         if trans is not None:
#           trans = pickle.loads(trans)
#           if tbin not in translist.keys():
#             translist[tbin] = []
#           for t in trans:
#             translist[tbin].append(t)

#       #  Update observation matrix
#       observe_before = np.copy(self.data['observe'])
#       self.data['observe'] += obs_delta
#       observe = self.data['observe']
#       launch = self.data['launch']

#     #  RUNTIME   -----------------------------------------
#       logging.info("Adapting Runtimes")
#       for tbin, tlist in translist.items():
#         A, B = tbin
#         time_s = 0
#         time_t = 0
#         num_t  = len(tlist)
#         for t in tlist:        
#           if t[0] == t[1]:
#             time_s += t[2]
#           else:
#             time_t += t[2]

#         run_step = 10000 * np.round(np.log( (num_t * max(1, time_t)) / time_s  ))
#         print("Runtime data for", A, B, ":  ", time_t, time_s, num_t, run_step)

#         currt = self.data['runtime'][A][B]
#         self.data['runtime'][A][B] = min(max(50000, currt+run_step), 500000) 

#         logging.info("Adapting runtime for (%d, %d)  from %7.0f  --->  %7.0f", 
#           A, B, currt, self.data['runtime'][A][B])

#     #  CONVERGENCE   -----------------------------------------
#       logging.debug("============================  <CONVERGENCE>  =============================")

 
#       #  TODO:  Consistency
#       # Weight Calculation: create a logistic function using current observation distribution

#       # Calc convergence on probability distro (before & after)
#       logging.debug("Calculating Probability Distributions for each state...")
#       probDistro_before = np.zeros(shape=(numLabels, numLabels))
#       probDistro        = np.zeros(shape=(numLabels, numLabels))
#       for n in range(numLabels):
#         numTrans_before = np.sum(observe_before[n]) - observe_before[n][n]
#         probDistro_before[n] = observe_before[n] / numTrans_before
#         probDistro_before[n][n] = 0

#         numTrans = np.sum(observe[n]) - observe[n][n]
#         probDistro[n] = observe[n] / numTrans
#         probDistro[n][n] = 0
#       delta        = np.zeros(shape=(numLabels, numLabels))
#       delta = abs(probDistro - probDistro_before)

#       globalconvergence = np.sum(abs(probDistro - probDistro_before))
#       globalconvergence_rate = globalconvergence / len(job_list)

#     #  WEIGHT CALC   -----------------------------------------
#       logging.debug("============================  <WEIGHT CALC>  =============================")

#       bins = [(x, y) for x in range(numLabels) for y in range(numLabels)]
#       logging.debug("Calculating transition rarity...")

#       # 1. Fatigue portion based on # times each bin was "launched"
#       quota = np.sum(launch) / totalBins
#         # Apply a `fatigue` factor; bins are fatigued if run more than their quota
#       fatigue = np.maximum( (quota-launch) / quota, np.zeros(shape=(numLabels, numLabels)))

#       # TODO:   Load New Transition Matrix  if consistency is necessary, otherwise use locally updated tmat

#       # 2. Calculate weight (note: follow 2 are retained for debugging only)
#       #  UPDATED:  Weights CALC as a factor of rare events & instability in delta calc
#       rarity = np.zeros(shape=(numLabels, numLabels))

#         #  Isolate rare events (s.t. rare event seen less than mean)
#       rareObserve = np.choose(np.where(observe.flatten() < np.mean(observe)), observe.flatten())

#         #  Est. mid point of logistic curve by accounting for distribution skew
#       midptObs = np.mean(observe) * skew(observe) + np.median(observe)

#         #  Create the function
#       rarityFunc = makeLogisticFunc(1, 1 / np.std(rareObserve), midptObs)

#       #  3. Apply constants. This can be user influenced
#       alpha = self.data['weight_alpha']
#       beta = self.data['weight_beta']

#       # fatigue = np.zeros(shape=(numLabels, numLabels))
#       logging.debug("Calculating control weights...")
#       weight = {}
#       quotasq = quota**2
#       for i in range(numLabels):
#         for j in range(numLabels):
          
#           # 4. Calculate rarity & weight incorporatin fatigue value
#           rarity[i][j] = rarityFunc(observe[i][j] - midptObs)

#           #  Old function: 
#           #       weight[(i,j)]  = alpha * rarity[i][j] + beta + delta[i][j]
#           #       weight[(i,j)] *= 0 if launch[i][j] > quota else (quota - launch[i][j])**2/quotasq

#           weight[(i,j)] =  alpha * rarity[i][j] + beta * fatigue[i][j]

#       #  5. Sort weights from high to low
#       updatedWeights = sorted(weight.items(), key=lambda x: x[1], reverse=True)

#       #  TODO:  Adjust # of iterations per bin based on weights by
#       #     replicating items in the updatedWeights list

#     #  SCHEDULING   -----------------------------------------
#       logging.debug("============================  <SCHEDULING>  =============================")

#       #  1. Load JC Queue and all items within to get respective weights and projected target bins
#       curqueue = []
#       logging.debug("Loading Current Queue of %d items", len(self.data['jcqueue']))
#       debug = True
#       configlist = self.catalog.load([wrapKey('jc', job) for job in self.data['jcqueue']])
#       for config in configlist.values():

#         # 2. Dampening Factor: proportional to its currency (if no ts, set it to 1)
#         jc_ts = config['timestep'] if 'timestep' in config else 1

#         #  TODO: May want to consider incl convergence of sys at time job was created
#         w_before    = config['weight']
#         config['weight'] = config['weight'] * (jc_ts / self.data['timestep'])
#         # logging.debug("Dampening Factor Applied (jc_ts = %d):   %0.5f  to  %05f", jc_ts, w_before, config['weight'])
#         curqueue.append(config)

#       #  3. (PreProcess current queue) for legacy JC's
#       logging.debug("Loaded %d items", len(curqueue))
#       for jc in range(len(curqueue)):
#         if 'weight' not in curqueue[jc]:
#           curqueue[jc]['weight'] = 1.0

#         if 'gc' not in curqueue[jc]:
#           curqueue[jc]['gc'] = 1

#       #  4. Sort current queue
#       if len(curqueue) > 0:
#         existingQueue = deque(sorted(curqueue, key=lambda x: x['weight'], reverse=True))
#         eqwlow = 0 if np.isnan(existingQueue[0]['weight']) else existingQueue[0]['weight']
#         eqwhi  = 0 if np.isnan(existingQueue[-1]['weight']) else existingQueue[-1]['weight']
#         logging.debug("Existing Queue has %d items between weights: %0.5f - %0.5f", len(existingQueue), eqwlow, eqwhi)
#       else:
#         existingQueue = deque()
#         logging.debug("Existing Queue is empty.")

#       #  5. Det. potential set of  new jobs  (base on launch policy)
#       #     TODO: Set up as multiple jobs per bin, cap on a per-control task basis, or just 1 per bin
#       potentialJobs = deque(updatedWeights)
#       logging.debug("Potential Job Queue has %d items between weights: %0.5f - %0.5f", len(potentialJobs), potentialJobs[0][1], potentialJobs[-1][1])

#       #  6. Prepare a new queue (for output)
#       jcqueue = deque()

#       targetBin = potentialJobs.popleft()
#       oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()
#       selectionTally = np.zeros(shape=(numLabels, numLabels))
#       newJobCandidate = {}

#       #  7. Rank order list of observed transition bins by rare observations for each state (see below)
#       rarityorderstate = np.argsort(observe.sum(axis=1))
#       rarityordertrans = np.zeros(shape=(numLabels,numLabels))
#       for i in range(numLabels):
#         np.copyto(rarityordertrans[i], np.argsort(observe[i]))

#       #  8. Continually pop new/old jobs until max queue size is attained   
#       while len(jcqueue) < DEFAULT.MAX_JOBS_IN_QUEUE:

#         #  8a:  No more jobs
#         if oldjob == None and targetBin == None:
#           logging.info("No more jobs to queue.")
#           break

#         #  8b:  Push an old job
#         if (targetBin == None) or (oldjob and oldjob['weight'] > targetBin[1]) or (oldjob and np.isnan(targetBin[1])):
#           jcqueue.append(oldjob['name'])
#           oldjob['weight'] = 0 if np.isnan(oldjob['weight']) else oldjob['weight']
#           logging.debug("Re-Queuing OLD JOB `%s`   weight= %0.5f", oldjob['name'], oldjob['weight'])
#           oldjob = None if len(existingQueue) == 0 else existingQueue.popleft()

#         #  8c:  Push a new job  (New Job Selection Algorithm)
#         else:

#           #  Job should "start" in is targetBin of (A, B)
#           A, B = targetBin[0]
#           logging.debug("\n\nCONTROL: Target transition bin  %s  (new job #%d,  weight=%0.5f)", str((A, B)), len(newJobCandidate), targetBin[1])

#           # (1)  Start with candidated in the Target Bin's candidate pool
#           cpool = kv2DArray.key('candidatePool', A, B)
#           selectedbin = (A,B)

#           # (2)  Flip direction of transition (if no candidates in that targetbin)
#           if self.catalog.llen(cpool) == 0:
#             logging.info("No DEShaw reference for transition, (%d, %d)  -- checking reverse direction", A, B)
#             cpool = kv2DArray.key('candidatePool', B, A)
#             selectedbin = (B,A)

#           # (3)  Iteratively find another candidate pool from sorted "rare obsevation" list <-- This should find at least 1
#           if self.catalog.llen(cpool) == 0:
#             logging.info("No DEShaw reference for transition, (%d, %d)  -- checking wells in this order: %s", B, A, str(rarityorderstate))
#             for RS in rarityorderstate:
#               cpool = kv2DArray.key('candidatePool', RS, RS)
#               if self.catalog.llen(cpool) == 0:
#                 logging.info("No DEShaw reference for transition, (%d, %d) ", RS, RS)

#               else: 
#                 logging.info("FOUND DEShaw start point from transition, (%d, %d) ", RS, RS)
#                 selectedbin = (RS,RS)
#                 break

#           logging.debug('Final Candidate Popped from Pool %s  of  %d  candidates', cpool, self.catalog.llen(cpool))

#           # (4). Cycle this candidate to back of candidate pool list
#           candidate = self.catalog.lpop(cpool)
#           sourceTraj, srcFrame = candidate.split(':')
#           dstring = "####SELECT_TRAJ@ %s @ %s @ %s @ %s" % (str(targetBin[0]), str(selectedbin), sourceTraj, srcFrame)
#           self.catalog.rpush(cpool, candidate)

#           # (5). Back Projection Function (using newly analzed data to identify start point of next simulation)
#           # TODO: Archive Data Retrieval. This is where data is either pulled in from remote storage
#           #   or we have a pre-fetch algorithm to get the data
#           # Back-project  <--- Move to separate Function tied to decision history
#           #  Start coordinates are either a DeShaw file (indexed by number) or a generated one
#           if isinstance(sourceTraj, int) or sourceTraj.isdigit():      # It's a DEShaw file
#             pdbfile, archiveFile = getHistoricalTrajectory(sourceTraj)
#           else:
#             archiveFile = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.dcd' % sourceTraj)
#             pdbfile     = os.path.join(DEFAULT.JOBDIR, sourceTraj, '%s.pdb' % sourceTraj)

#           # (6). Generate new set of params/coords
#           jcID, params = generateNewJC(archiveFile, pdbfile, DEFAULT.TOPO, DEFAULT.PARM, int(srcFrame), debugstring=dstring)

#     #  POST-PROC
#       logging.debug("OBS_MATRIX_DELTA:\n" + str(obs_delta))
#       logging.debug("OBS_MATRIX_UPDATED:\n" + str(observe))
#       logging.debug("LAUNCH_MATRIX:\n" + str(launch))
#       logging.debug("RUNTIMES: \n%s", str(self.data['runtime']))
#       logging.debug("CONVERGENCE_PROB_DISTRO:\n" + str(probDistro))
#       logging.debug("OBS_RARITY:\n" + str(rarity))
#       logging.debug("CONV_DELTA:\n" + str(delta))
#       # logging.debug("CTL_WEIGHT:\n" + str(np.array([[weight[(i,k)] for i in range(numLabels)] or k in range(numLabels)])))

#       logging.info("GLOBAL_CONVERGENCE: %f", globalconvergence)
#       for i in range(5):
#         logging.info("STATE_CONVERGENCE for %d: %f", i, np.sum(delta[i]))

#       logging.info("GLOBAL_RATE_CONV: %f", globalconvergence_rate)
#       for i in range(5):
#         logging.info("STATE_RATE_CONV for %d: %f", i, np.sum(delta[i])/len(job_list))
         




# REAL  OLD COVERGENCE:
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
      # obs_delta = np.zeros(shape=(numLabels, numLabels))


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
      #     obs_delta[A][B] += 1

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

