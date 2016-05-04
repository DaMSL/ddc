import numpy as np
import logging
import math

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)

def makeLogisticFunc (maxval, steep, midpt):
  return lambda x: maxval / (1 + np.exp(-steep * (midpt - x)))

skew = lambda x: (np.mean(x) - np.median(x)) / np.std(x)



def bootstrap_std (series, interval=.9):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  N = len(series)
  mean = np.mean(series)
  stddev = np.std(series)
  # Z-Table look up
  Z = 1.645  # .9 CI
  err = stddev / math.sqrt(N)
  CI = Z * err
  return (mean, CI, stddev, err)




def posterior_prob (source, withcount=False):
  """
  Calculate posterior probability for set of observations
  """
  # Iterate for each item in the source
  logging.debug('Calculating Posterior Probablity on %d data observations', len(source))
  groupby = {}
  for x_i in source:
    key = eval(x_i) if isinstance(x_i, str) else x_i
    if key in groupby.keys():
      groupby[key] += 1
    else:
      groupby[key] = 1

  probability_est = {}
  for v_i in sorted(groupby.keys()):
    mu = groupby[v_i] / len(source)
    logging.debug('%10s:  %8d %6.4f', v_i, groupby[v_i], mu)
    probability_est[v_i] = mu
  if withcount:
    return groupby, probability_est
  else:
    return probility_est


def incremental_posterior_prob (histdata, batch):
  """
  Incrementally update posterior probability for set of observations
  """
  # Iterate for each item in the source
  logging.debug('Calculating Incremental Posterior Probablity on %d NEW data observations', len(batch))
  groupby = {k: int(v) for k, v in histdata.items()}
  for x_i in batch:
    key = eval(x_i) if isinstance(x_i, str) else x_i
    if key in groupby.keys():
      groupby[key] += 1
    else:
      groupby[key] = 1

  probility_est = {}
  keys = list(groupby.keys())
  for v_i in sorted(keys):
    if len(batch) == 0:
      mu = 0
    else:
      mu = groupby[v_i] / len(batch)
    logging.debug('%10s:  %8d %6.4f', v_i, groupby[v_i], mu)
    probility_est[v_i] = mu
  return probility_est

def bootstrap_sampler (source, samplesize=1, N=50, interval=.90):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo

  # Get unique label/category/hcube ID's
  V = set()
  for i in source:
    V.add(i)
  # print ("BS: labels ", str(V))

  #  EXPERIMENT #1, 4+:  SAMPLE SIZE WAS 10%
  L = round(len(source) * samplesize)
  #  EXPERIMENT #2&3:  FIXED SAMPLE SIZE OF 100K (or less)
  # L = min(len(source), 100000)

  # Calculate mu_hat from bootstrap -- REMOVED
  # mu_hat = {}
  # groupby = {v_i: 0 for v_i in V}
  # for s in source:
  #   groupby[s] += 1
  # for v_i in V:
  #   mu_hat[v_i] = groupby[v_i]/L

  # Iterate for each bootstrap and generate statistical distributions
  boot = {i : [] for i in V}
  for i in range(N):
    strap   = [source[np.random.randint(len(source))] for n in range(L)]
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
    ciHI = delta[math.floor(N*ci_hi)]
    probility_est[v_i] = (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i)
  return probility_est

def bootstrap_iter (source, size=None, interval=.90):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  # Get unique label/category/hcube ID's
  Z = 1.645  # .9 CI
  V = set()
  for i in source:
    V.add(i)

  if size is None:
    step = len(source) / 10
  else:
    step = size
  boot = {i : [] for i in V}
  for i in range(0, len(source), step):
    L = min(i+step, len(source))
    strap   = source[i:L]
    groupby = {v_i: 0 for v_i in V}
    for s in strap:
      groupby[s] += 1
    for v_i in V:
      boot[v_i].append(groupby[v_i]/L)
  probility_est = {}
  for k, v in boot.items():
    mean = np.mean(v)
    stddev = np.std(v)
    err = stddev / math.sqrt(len(v))
    CI = Z * err
    probility_est[k] = (mean, CI, stddev, err)
  return probility_est



def gen_bootstraps(all_obs, strap_size_ns, transonly=False, cumulative=False):
  """Generate Bootstrap samples from observed Data. This function uses count
  as the function to splice out the data
  """
  #  TODO Pass in label names
  obs_per_step = strap_size_ns * 1000
  ab = [(A, B) for A in range(5) for B in range(5)]
  bootstrap = {b: [] for b in ab}
  cnts = {b: 0 for b in ab}
  total = [0 for i in range(5)]
  i = 0
  print('Collecting data...')
  while i < len(all_obs):
    if i > 0 and i % obs_per_step == 0:
      for b in ab:
        A, B = b
        if total[A] == 0:
          logging.info('Bootstrap gen: No DATA for state %d  at interval:  %d' % (A, i/1000))
        else:
          bootstrap[b].append(cnts[b]/total[A])
      if not cumulative:
        cnts = {b: 0 for b in ab}
        total = [0 for i in range(5)]
    A, B = eval(all_obs[i])
    i += 1
    if transonly and A == B:
      continue
    cnts[(A, B)] += 1
    total[A] += 1
  for b in ab:
    A, B = b
    if total[A] == 0:
      logging.info('Bootstrap gen: No DATA for state %d  at interval:  %d' % (A, i/1000))
    else:
      bootstrap[b].append(cnts[b]/total[A])
  return bootstrap



def normalize_bootstraps(obs_counts, strap_size_ns, transonly=False, cumulative=False):
  """ Normalized sets of observations so they are the same size
  """
  #  TODO Pass in label names
  obs_per_step = strap_size_ns * 1000
  ab = [(A, B) for A in range(5) for B in range(5)]
  bootstrap = {b: [] for b in ab}
  cnts = {b: 0 for b in ab}
  total = [0 for i in range(5)]
  i = 0
  print('Collecting data...')
  while i < len(all_obs):
    if i > 0 and i % obs_per_step == 0:
      for b in ab:
        A, B = b
        if total[A] == 0:
          logging.info('Bootstrap gen: No DATA for state %d  at interval:  %d' % (A, i/1000))
        else:
          bootstrap[b].append(cnts[b]/total[A])
      if not cumulative:
        cnts = {b: 0 for b in ab}
        total = [0 for i in range(5)]
    A, B = eval(all_obs[i])
    i += 1
    if transonly and A == B:
      continue
    cnts[(A, B)] += 1
    total[A] += 1
  for b in ab:
    A, B = b
    if total[A] == 0:
      logging.info('Bootstrap gen: No DATA for state %d  at interval:  %d' % (A, i/1000))
    else:
      bootstrap[b].append(cnts[b]/total[A])
  return bootstrap




def bootstrap_by_state (source, samplesize=.1, N=50, interval=.95):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo

  # Get unique label/category/hcube ID's
  V = set()
  for i in source:
    V.add(i)

  #  EXPERIMENT #1, 4+:  SAMPLE SIZE WAS 10%
  L = round(len(source) * samplesize)
  #  EXPERIMENT #2&3:  FIXED SAMPLE SIZE OF 100K (or less)
  # L = min(len(source), 100000)

  # Iterate for each bootstrap and generate statistical distributions
  boot = {i : [] for i in V}
  for i in range(N):
    strap   = [source[np.random.randint(len(source))] for n in range(L)]
    groupby = {v_i: 0 for v_i in V}
    totals = [0 for i in range(5)]
    for s in strap:
      groupby[s] += 1
    for v_i in V:
      A, B = eval(v_i)
      if A != B:
        totals[A] += groupby[v_i]
    for v_i in V:
      A, B = eval(v_i)
      if A != B:  
        boot[v_i].append(groupby[v_i]/totals[A])
  probility_est = {}
  for v_i in V:
    A, B = eval(v_i)
    if A == B:
      probility_est[v_i] = (0, 0, 0, 0.)
    else:
      P_i = np.mean(boot[v_i])
      delta  = np.array(sorted(boot[v_i]))  #CHECK IF mu or P  HERE
      ciLO = delta[round(N*ci_lo)]
      ciHI = delta[math.floor(N*ci_hi)]
      probility_est[v_i] = (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i)
  return probility_est
