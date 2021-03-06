#!/usr/bin/env python

# from simmd import *
# from anl import *
# from ctl import *
import argparse
import math
import json
import bisect
import datetime as dt

import mdtraj as md
import numpy as np
from numpy import linalg as LA


from core.common import *
import mdtools.deshaw as deshaw
from overlay.redisOverlay import RedisClient
from core.kvadt import kv2DArray
from core.slurm import slurm
from core.kdtree import KDTree
import core.ops as ops
import datatools.datareduce as datareduce
from datatools.rmsd import *
from mdtools.simtool import generateNewJC
import mdtools.deshaw as deshaw
import bench.db as db
import plot as P

DO_COV_MAT = False

DESHAW_LABEL_FILE = 'data/deshaw_labeled_bins.txt'

# For changes to schema
def updateschema(catalog):
  settings = systemsettings()
  if not settings.configured():
    settings.applyConfig()

  dtype_map = {}
  for k, v in settings.schema.items():
    logging.debug('items %s, %s', str(k), str(v))
    dtype_map[k] = str(v)

  for k, v in settings.init.items():
    if k not in dtype_map:
      dtype_map[k] = type(v).__name__
      settings.schema[k] = type(v).__name__

  for k, v in dtype_map.items():
    logging.info('Setting schema:    %s  %s', k, str(v))

  catalog.hmset("META_schema", dtype_map)

def initializecatalog(catalog):
  """Initialize the Catalog for an experiment
  """
  settings = systemsettings()

  logging.debug("Clearing Catalog.")
  if not catalog.isconnected:
    logging.warning('Catalog is not started. Please start it first.')
    sys.exit(0)
  catalog.clear()

  logging.debug("Loading schema into catalog.")
  updateschema(catalog)
  catalog.loadSchema()

  catalog.set('name', settings.name)

  # Set defaults vals for schema
  initvals = {i:settings.init[i] for i in settings.init.keys() if settings.schema[i] in ['int', 'float', 'list', 'dict', 'str']}
  catalog.save(initvals)
  for k, v in initvals.items():
    logging.debug("Initializing data elm %s  =  %s", k, str(v))

  # New distance space based centroids
  catalog.delete('centroid')
  centfile = settings.RMSD_CENTROID_FILE
  logging.info("Loading Starting Centroids from %s", centfile)
  centroid = np.load(centfile)
  catalog.storeNPArray(centroid, 'centroid')



def centroid_bootstrap(catalog):
  centfile = settings.RMSD_CENTROID_FILE
  centroid = np.load(centfile)
  cent_npts = [1, 1, 1, 1, 1]  # TBD
  numLabels = len(centroid)
  binlist = [(a, b) for a in range(numLabels) for b in range(numLabels)]
  logging.info("Loaded Starting Centroids from %s", centfile)

  name = catalog.get('name')
  if name is None:
    logging.info('Name not configured in this catalog. Set it and try again')
    return

  # Load/Set initial (current) Configs from Catalog
  if catalog.exists('thetas'):
    thetas = catalog.loadNPArray('thetas')
  else:
    thetas = np.zeros(shape=(numLabels, numLabels))
    thetas[:] = 0.25

  if catalog.exists('transition_sensitivity'):
    trans_factor = catalog.loadNPArray('transition_sensitivity')
  else:
    trans_factor = 0.2
    
  use_gradient = True
  obs_count = {ab: 0 for ab in binlist}
  C_delta = []
  T_delta = []

  # Configure Noise Filter
  noise = int(catalog.get('obs_noise'))
  dcdfreq = int(catalog.get('dcdfreq'))
  stepsize = int(catalog.get('sim_step_size'))
  nwidth = noise//(2*stepsize)
  noisefilt = lambda x, i: np.mean(x[max(0,i-nwidth):min(i+nwidth, len(x))], axis=0)


  # Get previously Labeled data (or label data IAW current settings)
  eid = db.get_expid(name)
  obslist = [i[0] for i in db.runquery('SELECT obs FROM obs WHERE expid=%d' % eid)]
  jobs = [i[0] for i in sorted(catalog.hgetall('anl_sequence').items(), key=lambda x: x[1])]
  shape = None

  # Initialize lists for pair-wise distances (top 2 nearest centroids)
  diffList  = {}
  transList = {}
  scatPlot  = {}
  for A in range(0, numLabels-1):
    for B in range(A+1, numLabels):
      diffList[(A, B)]  = []
      transList[(A, B)] = []
      scatPlot[(A, B)]  = []
  allScat = []
  # Load trajectories & filter
  obs_global = []

  # Process learning in batches (static batch size to start)
  batch_size = 25
  max_obs = 150
  batch = 0
  while batch <= max_obs:
    logging.info("Procssing Jobs %d - %d", batch, batch+batch_size)
    exec_sim = []
    obs_list = []
    for job in jobs[batch:batch+25]:
      conf = catalog.hgetall('jc_' + job)
      traj = md.load(conf['dcd'], top=conf['pdb'])
      alpha = datareduce.filter_alpha(traj)
      conf['alpha'] = alpha.xyz
      exec_sim.append(conf)
      if shape is None:
        shape = conf['alpha'].shape[1:]

      # xyz_filtered = np.array([noisefilt(alpha.xyz, i) for i in range(alpha.n_frames)])
      rmslist = calc_rmsd(alpha, centroid)
      labels = []
      for rms in rmslist:
        # [cw[i]*LA.norm(pt - centroid[i]) for i in range(5)]
        A, B = np.argsort(rms)[:2]
        delta = np.abs(rms[B] - rms[A])
        if delta < thetas[A][B]:
          sub_state = B
        else:
          sub_state = A
        classify = (A, sub_state)
        labels.append(classify)
        obs_count[classify] += 1

        # For globally updating Thetas
        obs_global.append(classify)
        if A < B:
          diffList[(A, B)].append(rms[A] - rms[B])
        else:
          diffList[(B, A)].append(rms[B] - rms[A])

        for a in range(0, numLabels-1):
          for b in range(a+1, numLabels):
            transList[(a, b)].append(rms[a] - rms[b])
            if (a, a) == classify or (b, b) == classify:
              c = 'b'
            elif (a, b) == classify or (b, a) == classify:
              c = 'g'
            elif a == A or b == A:
              c = 'r'
            else:
              c = 'black'
            scatPlot[(a, b)].append((rms[a] - rms[b], c))
      obs_list.append(labels)

    logging.info('Bin Distribution:')
    grpby = {}
    for llist in obs_list:
      for l in llist:
        if l not in grpby:
          grpby[l] = 0
        grpby[l] += 1
    for k in sorted(grpby.keys()):
      logging.info('%s:  %5d', k, grpby[k])
    for A in range(0, numLabels-1):
      for B in range(A+1, numLabels):
        d = diffList[(A, B)]
        logging.info('Diff list for %d,%d:  %d, %5.2f, %5.2f', A, B, len(d), min(d), max(d))


    # # 6. Apply Heuristics Labeling
    # # logging.debug('Applying Labeling Heuristic. Origin:   %d, %d', srcA, srcB)
    # rmslabel = []
    # 
    # label_count = {ab: 0 for ab in binlist}
    # groupbystate = [[] for i in range(numLabels)]
    # groupbybin = {ab: [] for ab in binlist}


    # For each frame in each traj: ID labeled well pts & build avg op
    logging.info('Selecting observed Well States')
    coor_sum = {i: np.zeros(shape=shape) for i in range(numLabels)}
    coor_tot = {i: 0 for i in range(numLabels)}
    for job, obslist in zip(exec_sim, obs_list):
      # offset = int(job['xid:start'])
      # for i, frame in enumerate(job['alpha']):
      for frame, label in zip(job['alpha'], obslist):
        # A, B = eval(obslist[offset+i])
        A, B = label
        if A != B:
          continue
        coor_sum[A] += frame
        coor_tot[A] += 1

    logging.info('Calculating Avg from following stats:')
    logging.info('   Total Frames: %d', sum([len(sim['alpha']) for sim in exec_sim]))

    # Calculate New Centroids (w/deltas)
    delta = []
    for S in range(numLabels):
      if coor_tot[S] == 0:
        logging.info("   State: %d --- NO OBSERVATIONS IN THIS WELL STATE", S)
        continue
      cent_local = coor_sum[S] / coor_tot[S]
      diff_local = LA.norm(centroid[S] - cent_local)
      update = ((centroid[S] * cent_npts[S]) + (cent_local * coor_tot[S])) / (cent_npts[S] + coor_tot[S])
      delta.append(LA.norm(update - centroid[S]))
      logging.info('   State %d:  NewPts=%5d   Delta=%5.2f   LocalDiff=%5.2f', 
        S, coor_tot[S], delta[-1], diff_local)
      centroid[S] = update
      cent_npts[S] += coor_tot[S]
    centroid_change = np.mean(delta)
    if len(C_delta) > 1:
      rel_change = np.abs((centroid_change - C_delta[-1]) / C_delta[-1])
      logging.info('Centroid Change:  %5.2f   (%5.2f%%)', centroid_change, 100*rel_change)
    C_delta.append(centroid_change)
    batch += batch_size


    # Update Thetas (usig global data ?????)
    delta = []
    for A in range(0, numLabels-1):
      for B in range(A+1, numLabels):
        X = sorted(diffList[(A, B)])
        if len(X) < 100:
          logging.info('Lacking data on %d, %d', A, B)
          continue
        # logging.info('  Total # Obs: %d', len(X))
        crossover = 0
        for i, x in enumerate(X):
          if x > 0:
            crossover = i
            break
        # logging.info('  Crossover at Index: %d', crossover)
        if crossover < 50 or (len(X)-crossover) < 50:
          logging.info('  Lacking local data skipping.')
          continue

        # Find local max gradient  (among 50% of points)
        
        if use_gradient:
          thetas_updated = np.copy(thetas)
          zoneA = int((1-trans_factor) * crossover)
          zoneB = crossover + int(trans_factor * (len(X) - crossover))
          gradA = zoneA + np.argmax(np.gradient(X[zoneA:crossover]))
          gradB = crossover + np.argmax(np.gradient(X[crossover:zoneB]))
          thetaA = X[gradA]
          thetaB = X[gradB]
          thetas_updated[A][B] = np.abs(thetaA)
          thetas_updated[B][A] = np.abs(thetaB)
          tdeltA = np.abs(thetas_updated[A][B] - thetas[A][B])
          tdeltB = np.abs(thetas_updated[B][A] - thetas[B][A])
          delta.append(tdeltA)
          delta.append(tdeltB)
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', A, B, tdeltA, (100*tdeltA/thetas[A][B]))
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', B, A, tdeltB, (100*tdeltB/thetas[B][A]))
          thetas[A][B] = thetas_updated[A][B]
          thetas[B][A] = thetas_updated[B][A]
        else:
          # Classify Fixed Percent of observations as Transitional
          thetas_updated = np.copy(thetas)
          transitionPtA = int((1-trans_factor) * crossover)
          transitionPtB = crossover + int(trans_factor * (len(X) - crossover))
          thetaA = X[transitionPtA]
          thetaB = X[transitionPtB]
          thetas_updated[A][B] = np.abs(thetaA)
          thetas_updated[B][A] = np.abs(thetaB)
          tdeltA = np.abs(thetas_updated[A][B] - thetas[A][B])
          tdeltB = np.abs(thetas_updated[B][A] - thetas[B][A])
          delta.append(tdeltA)
          delta.append(tdeltB)
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', A, B, tdeltA, (100*tdeltA/thetas[A][B]))
          logging.info('  Theta Change (%d,%d):  %4.2f  (%4.1f)', B, A, tdeltB, (100*tdeltB/thetas[B][A]))
          thetas[A][B] = thetas_updated[A][B]
          thetas[B][A] = thetas_updated[B][A]

    T_delta.append(np.mean(delta))
  P.line(np.array(C_delta), 'Avg_CHANGE_Centroid_Pos_%s' % name)
  P.line(np.array(T_delta), 'Avg_CHANGE_Theta_Val_%s' % name)
  P.bargraph_simple(obs_count, 'Final_Histogram_%s' % name)
  # for k, X in diffList.items():
  #   A, B = k
  #   P.transition_line(sorted(X), A, B, title='-X', trans_factor=.5)
  # for k, X in transList.items():
  #   A, B = k
  #   P.transition_line(sorted(X), A, B, title='-ALL', trans_factor=.5)
  for k, X in scatPlot.items():
    collab = {'b': 'Well', 'g': 'Trans', 'r': 'Primary', 'brown': 'Secondary', 'black': 'None'}
    ptmap = {k: [] for k in collab.keys()}
    ordpts = sorted(X, key = lambda x : x[0])
    for i, tup in enumerate(ordpts):
      y, c = tup
      ptmap[c].append((i, y))
      # if c == 'b' or c == 'g':
      #   ptmap[c].append((i, y))
      # else:
      #   ptmap[c].append((i, 0))
    A, B = k
    P.scat_Transtions(ptmap, title='-%d_%d'%(A,B), size=1, labels=collab)



def load_PCA_Subspace(catalog):

  # HCube leaf size of 500 points
  settings = systemsettings()
  vectfile = settings.PCA_VECTOR_FILE

  logging.info("Loading PCA Vectors from %s", vectfile)
  pc_vect = np.load(vectfile)
  max_pc = pc_vect.shape[1]
  num_pc = min(settings.PCA_NUMPC, max_pc)
  pc = pc_vect[:num_pc]
  logging.info("Storing PCA Vectors to key:  %s", 'pcaVectors')
  catalog.storeNPArray(pc, 'pcaVectors')

  logging.info("Loading Pre-Calculated PCA projections from Historical BPTI Trajectory")
  pre_calc_deshaw = np.load('data/pca_applied.npy')

  # Extract only nec'y PC's
  pts = pre_calc_deshaw.T[:num_pc].T

  pipe = catalog.pipeline()
  for si in pts:
    pipe.rpush('subspace:pca', bytes(si))
  pipe.execute()
  logging.debug("PCA Subspace stored in Catalog")

  logging.info('Creating KD Tree')
  kd = KDTree(500, maxdepth=8, data=pts)
  logging.info('Encoding KD Tree')
  packaged = kd.encode()
  encoded = json.dumps(packaged)
  logging.info('Storing in catalog')
  catalog.delete('hcube:pca')
  catalog.set('hcube:pca', encoded)
  logging.info('PCA Complete')

def labelDEShaw_rmsd(store_to_disk=False):
  """label ALL DEShaw BPTI observations by state & secondary state (A, B)
  Returns frame-by-frame labels  (used to seed jobs)
  """
  settings = systemsettings()
  logging.info('Loading Pre-Calc RMSD Distances from: %s   (For initial seeding)','bpti-rmsd-alpha-dspace.npy')
  rms = np.load('bpti-rmsd-alpha-dspace.npy')
  prox = np.array([np.argsort(i) for i in rms])
  theta = 0.27
  logging.info('Labeling All DEShaw Points.')
  rmslabel = []
  skip = 100 if settings.name == 'debug' else 1
  # Only use N-% of all points
  # for i in range(0, len(rms), 100):
  for i in range(0, len(rms), skip):
    A = prox[i][0]
    proximity = abs(rms[i][prox[i][1]] - rms[i][A])    #abs
    B = prox[i][1] if proximity < theta else A
    rmslabel.append((A, B))
  if store_to_disk:
    with open(DESHAW_LABEL_FILE, 'w') as lfile:
      for label in rmslabel:
        lfile.write('%s\n' % str(label))
  return rmslabel

def seedJob_Uniform(catalog, num=1, exact=None):
  """
  Seeds jobs into the JCQueue -- pulled from DEShaw
  Selects equal `num` of randomly start frames from each bin
  to seed as job candidates
  """
  logging.info('Seeding %d jobs per transtion bin', num)
  settings = systemsettings()
  numLabels = int(catalog.get('numLabels'))
  binlist = [(A, B) for A in range(numLabels) for B in range(numLabels)]

  dcdfreq = int(catalog.get('dcdfreq'))
  runtime = int(catalog.get('runtime'))
  sim_step_size = int(catalog.get('sim_step_size'))

  if catalog.exists('label:deshaw'):
    rmslabel = [eval(x) for x in catalog.lrange('label:deshaw', 0, -1)]
  elif os.path.exists(DESHAW_LABEL_FILE):
    logging.info('Loading DEShaw Points From File....')
    with open(DESHAW_LABEL_FILE) as lfile:
      rmslabel = [eval(label) for label in lfile.read().strip().split('\n')]
    logging.info('Loaded DEShaw %d Labels from file, %s', len(rmslabel), DESHAW_LABEL_FILE)
    pipe = catalog.pipeline()
    for rms in rmslabel:
      pipe.rpush('label:deshaw', rms)
    pipe.execute()
    logging.info('DEShaw Labels stored in the catalog.')
  else:
    rmslabel = labelDEShaw_rmsd(store_to_disk=True)
    pipe = catalog.pipeline()
    for rms in rmslabel:
      pipe.rpush('label:deshaw', rms)
    pipe.execute()
    logging.info('DEShaw Labels stored in the catalog.')
  logging.info('Grouping all prelabeled Data:')
  groupby = {b:[] for b in binlist}

  for i, b in enumerate(rmslabel):
    groupby[b].append(i)

  for k in sorted(groupby.keys()):
    v = groupby[k]
    logging.info('%s %7d %4.1f', str(k), len(v), (100*len(v)/len(rmslabel)))

  if exact is None:
    source_list = sorted(groupby.keys())
  else:
    bin_list = list(groupby.keys())
    if exact <= 25:
      idx_list = np.random.choice(len(bin_list), exact, replace=False)
    else:
      idx_list = np.random.choice(len(bin_list), exact, replace=True)
    source_list = [bin_list[i] for i in idx_list]

  for binlabel in source_list:
    clist = groupby[binlabel]
    A, B = binlabel

    # No candidates
    if len(clist) == 0:
      logging.info('NO Candidates for %s', str(binlabel))
      if binlabel == (1, 3):
        logging.info('Swapping (1,2) for (1,3)')
        clist = groupby[(1,2)]
        B = 2
      elif binlabel == (3, 1):
        logging.info('Swapping (3,0) for (3,1)')
        clist = groupby[(3,0)]
        B = 0
      else:
        logging.info('Not sampling this bin')
        continue

    for k in range(num):
      logging.debug('\nSeeding Job #%d for bin (%d,%d) ', k, A, B)
      index = np.random.choice(clist)
      src, frame = deshaw.refFromIndex(index)
      logging.debug("   Selected: BPTI %s, frame: %s", src, frame)
      pdbfile, dcdfile = deshaw.getHistoricalTrajectory_prot(int(src))
      traj = md.load(dcdfile, top=pdbfile, frame=int(frame))

      # Generate new set of params/coords
      jcID, params = generateNewJC(traj)

      # Update Additional JC Params and Decision History, as needed
      config = dict(params,
          name    = jcID,
          runtime = runtime,
          dcdfreq = dcdfreq,
          interval = dcdfreq * sim_step_size,                       
          temp    = 310,
          timestep = 0,
          gc      = 1,
          origin  = 'deshaw',
          src_index = index,
          src_bin  = (A, B),
          src_hcube = 'D',
          application   = settings.APPL_LABEL)
      logging.info("New Simulation Job Created: %s", jcID)
      for k, v in config.items():
        logging.debug("   %s:  %s", k, str(v))
      catalog.rpush('jcqueue', jcID)
      catalog.hmset(wrapKey('jc', jcID), config)


def makejobconfig(catalog):
  logging.info('Seeding 1 job')
  settings = systemsettings()

  # Generate new set of params/coords
  pdbfile, dcdfile = deshaw.getHistoricalTrajectory_prot(0)
  traj = md.load(dcdfile, top=pdbfile, frame=0)
  jcID, params = generateNewJC(traj)

  dcdfreq = catalog.get('dcdfreq')
  runtime = catalog.get('runtime')
  sim_step_size = catalog.get('sim_step_size')

  # Update Additional JC Params and Decision History, as needed
  config = dict(params,
      name    = jcID,
      runtime = runtime,
      dcdfreq = dcdfreq,
      interval = dcdfreq * sim_step_size,
      temp    = 310,
      timestep = 0,
      gc      = 1,
      origin  = 'deshaw',
      src_index = 0,
      src_bin  = (0, 0),
      application   = settings.name)
  logging.info("New Simulation Job Created: %s", jcID)
  for k, v in config.items():
    logging.debug("   %s:  %s", k, str(v))
  catalog.rpush('jcqueue', jcID)
  catalog.hmset(wrapKey('jc', jcID), config)



def resetAnalysis(catalog):
  """Removes all analysis data from the database
  """
  settings = systemsettings()

  keylist0 =['completesim',
            'label:rms',
            'observe:count',
            'rsamp:rms:_dtype',
            'rsamp:rms:_shape',
            'subspace:rms',
            'xid:filelist',
            'xid:reference  ',
            'subspace:covar:fidx',
            'subspace:covar:pts',
            'subspace:covar:xid']
  keylist1 =['subspace:pca:%d',
            'subspace:pca:kernel:%d',
            'subspace:pca:updates:%d',
            'rsamp:rms:%d',
            'rsamp:rms:%d:full',
            'rsamp:rms:%d:spill']
  keylist2 =['observe:rms:%d_%d']

  logging.info("Clearing the database of recent data")
  count = 0
  for key in keylist0:
    count += catalog.delete(key)

  for key in keylist1:
    for A in range(5):
      count += catalog.delete(key % A)

  for key in keylist2:
    for A in range(5):
      for B in range(5):
        count += catalog.delete(key % (A, B))

  logging.info('Removed %d keys', count)

  jobs = catalog.hgetall('anl_sequence')
  logging.info('RE-RUN THESE JOBS')
  orderedjobs = sorted(jobs.items(), key=lambda x: x[1])
  seqnum = 1
  fileseq = 0
  jobfile = open('joblist.txt', 'w')
  for k, v in orderedjobs:
    if seqnum == 100:
      fileseq += 1
      seqnum = 1
    outline = 'src/simanl.py -a --useid="sw-%04d.%02d" -c %s -w %s' % \
      (fileseq,seqnum,settings.name, k)
    print(outline)
    jobfile.write(outline + '\n')
    seqnum += 1
  jobfile.close()

    

#############################  OLDER
DESHAW_PTS_FILE =  os.getenv('HOME') + '/work/data/debug/bpti_10p.npy'
DESHAW_SAMPLE_FACTOR = 10  # As in 1/10th of full data set

def calcDEShaw_PCA(catalog, force=False):
  numPC = 3

  numpts = catalog.llen('subspace:pca')
  if numpts == 0 or force:
    catalog.delete('subspace:pca')
    logging.debug("Projecting DEshaw PCA Vectors (assuming PC's are pre-calc'd")
    pcavect = catalog.loadNPArray('pcaVectors')
    logging.debug("Loaded PCA Vectors: %s, %s", str(pcavect.shape), str(pcavect.dtype))
    src = np.load(DESHAW_PTS_FILE)
    logging.debug("Loaded source points: %s, %s", str(src.shape), str(src.dtype))
    pcalist = np.zeros(shape=(len(src), numPC))
    start = dt.datetime.now()
    pdbfile, dcdfile = deshaw.getHistoricalTrajectory(0)
    traj = md.load(dcdfile, top=pdbfile, frame=0)
    filt = traj.top.select_atom_indices(selection='heavy')
    pipe = catalog.pipeline()
    for i, conform in enumerate(src):
      if i % 10000 == 0:
        logging.debug("Projecting: %d", i)
      heavy = np.array([conform[k] for k in filt])
      np.copyto(pcalist[i], np.array([np.dot(heavy.reshape(pc.shape),pc) for pc in pcavect[:numPC]]))
      raw_index = i * DESHAW_SAMPLE_FACTOR
      pipe.rpush('xid:reference', '(-1, %d)' % raw_index)
    end = dt.datetime.now()
    logging.debug("Projection time = %d", (end-start).seconds)

    rIdx = []
    for si in pcalist:
      rIdx.append(pipe.rpush('subspace:pca', bytes(si)))
    pipe.execute()
    logging.debug("R_Index Created (pca)")
  else:
    logging.info('PCA Already created. Retrieving existing lower dim pts')
    X = catalog.lrange('subspace:pca', 0, -1)
    pcalist = np.array([np.fromstring(si) for si in X])

  # HCube leaf size of 500 points
  logging.info('Creating KD Tree')
  kd = KDTree(500, data=pcalist)
  logging.info('Encoding KD Tree')
  encoded = json.dumps(kd.encode())
  logging.info('Storing in catalog')
  catalog.delete('hcube:pca')
  catalog.set('hcube:pca', encoded)
  logging.info('PCA Complete')

  # cacherawfile = os.path.join(DEFAULT.DATADIR, 'cache_raw')
  # np.save(cacherawfile, src)

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
      indexSize = int(indexSize)
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

#############################


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('name', default='default')
  parser.add_argument('--seedjob', action='store_true')
  parser.add_argument('--onejob', action='store_true')
  parser.add_argument('--initcatalog', action='store_true')
  parser.add_argument('--updateschema', action='store_true')
  parser.add_argument('--initpca', action='store_true')
  parser.add_argument('--reset', action='store_true')
  parser.add_argument('--all', action='store_true')
  parser.add_argument('--exact', action='store_true')
  parser.add_argument('--centroid', action='store_true')

  # parser.add_argument('--initarchive', action='store_true')
  # parser.add_argument('--seeddata', action='store_true')
  # parser.add_argument('--reindex', nargs='?', const=10, type=int)
  # parser.add_argument('--loadindex', type=int)
  # parser.add_argument('--num', type=int, default=50)
  # parser.add_argument('--winsize', type=int, default=200)
  # parser.add_argument('--slide', type=int, default=100)
  args = parser.parse_args()

  confile = args.name + '.json'

  settings = systemsettings()
  settings.applyConfig(confile)
  catalog = RedisClient(args.name)

  # TO Recalculate PCA Vectors from DEShaw (~30-40 mins at 10% of data)
  # calcDEShaw_PCA(catalog)
  # sys.exit(0)

  if args.centroid:
    centroid_bootstrap(catalog)

  if args.initcatalog or args.all:
    settings.envSetup()
    initializecatalog(catalog)

  if args.onejob:
    makejobconfig(catalog)

  if args.seedjob or args.all:
    numresources = int(catalog.get('numresources'))
    initialJobPerBin = max(1, numresources//25)
    if args.exact:
      seedJob_Uniform(catalog, num=initialJobPerBin, exact=numresources)
    else:
      seedJob_Uniform(catalog, num=initialJobPerBin)

  if args.updateschema:
    # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
    updateschema(catalog)

  if args.reset:
    resetAnalysis(catalog)

  if args.initpca:
    load_PCA_Subspace(catalog)
    # pcaVectorfile = 'data/cpca_pc3.npy'
    # logging.info("Loading cPCA Vectors from %s", pcaVectorfile)
    # pcaVectors = np.load(pcaVectorfile)
    # catalog.storeNPArray(pcaVectors, 'pcaVectors')
    # calcDEShaw_PCA(catalog)


  # if args.loadindex is not None:
  #   # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
  #   index_DEShaw(catalog, archive, args.loadindex, args.num, args.winsize, args.slide)
  #   sys.exit(0)

  # if args.reindex:
  #   # archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
  #   reindex(archive, args.reindex)


  # if args.seeddata:
  #   seedData(catalog)

  # if args.initarchive:
  #   archive = redisCatalog.dataStore(**DEFAULT.archiveConfig)
  #   logging.warning("DON'T DO THIS!")
  #   # init_archive(archive)
  #   sys.exit(0)

