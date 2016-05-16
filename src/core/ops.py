import math
import numpy as np

def groupby (src):
  dest = {}
  for key, val in src.items():
    if key not in dest:
      dest[key] = []
    dest[key].append(val)
  return dest


def groupby_pair (src):
  dest = {}
  for key, val in src:
    if key not in dest:
      dest[key] = []
    dest[key].append(val)
  return dest

def groupby_cnt (src):
  dest = {}
  for val in src:
    if val not in dest:
      dest[val] = 0
    dest[val] += 1
  return dest


def flatten (src):
  dest = []
  for elm in src:
    dest.extend(elm)
  return dest



def bootstrap_replacement (source, samplesize=.1, N=100, interval=.90):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo
  N_data = len(source)

  mu = np.mean(source, axis=0)

  L = round(N_data * samplesize)
  L = max(10000, min(L, 25000))

  # Iterate for each bootstrap and generate statistical distributions
  boot = []
  for i in range(N):
    # boot   = [source[np.random.randint(N_data)] for n in range(L)]
    # strap   = [source[np.random.randint(N_data)] for n in range(L)]
    strap   = [source[n] for n in np.random.choice(np.arange(N_data), L, replace=True)]
    boot.append(np.mean(strap, axis=0))
  boot = np.array(boot)

  lamb = np.array([np.mean(strap, axis=0) - mu for strap in boot])
  ciLO = []
  ciHI = []
  err =  []
  for i, feature in enumerate(lamb.T):
    delta  = sorted(feature)
    ciLO.append(delta[round(N*ci_lo)])
    ciHI.append(delta[math.floor(N*ci_hi)])
    err.append(np.abs((ciHI[-1]-ciLO[-1])/mu[i]))

  prob_est = list(zip(mu, ciLO, ciHI, err))
  return prob_est



def bootstrap_std (series, interval=.9):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  N = len(series)
  mean = np.mean(series, axis=0)
  stddev = np.std(series, axis=0)
  # Z-Table look up
  Z = 1.645  # .9 CI
  err = stddev / math.sqrt(N)
  CI = Z * err
  return (mean, CI, stddev, err, np.abs(CI/mean))


def bootstrap_param(source, N=10):
  boot = []
  N_data = len(source)
  mu = np.mean(source, axis=0)
  for i in range(N):
    # strap   = [source[np.random.randint(len(source))] for n in range(len(source))]
    strap   = [source[n] for n in np.random.choice(np.arange(N_data), N_data, replace=True)]
    boot.append(np.mean(strap, axis=0))
  boot = np.array(boot)
  # mean = np.mean(source, axis=0)
  stddev = np.std(boot, axis=0)
  Z = 1.645  # .9 CI
  err = stddev / math.sqrt(N)
  CI = Z * err
  return (mu, CI, stddev, err, np.abs(CI/mu))


def bootstrap_mv(source, N=10):
  mu = np.mean(source, axis=0)
  boot = []
  for i in range(N):
    strap   = [source[np.random.randint(len(source))] for n in range(min(50000, len(source)))]
    boot.append(np.mean(strap, axis=0))
  boot = np.array(boot)
  mean = np.mean(boot, axis=0)
  stddev = np.std(boot, axis=0)
  Z = 1.645  # .9 CI
  err = stddev / math.sqrt(N)
  CI = Z * err
  return (mean, CI, stddev, err, np.abs(CI/mu))



def bootstrap_block(source, blocksize=5000):
  i = 0
  boot = []
  while i+blocksize < len(source):
    mu_i = np.mean(source[i:i+blocksize], axis=0)
    boot.append(mu_i)
    i += blocksize
  N = len(boot)
  boot = np.array(boot)
  mean = np.nan_to_num(np.mean(boot, axis=0))
  stddev = np.nan_to_num(np.std(boot, axis=0))
  Z = 1.645  # .9 CI
  err = np.nan_to_num(stddev / math.sqrt(N))
  CI = np.nan_to_num(Z * err)
  return (mean, CI, stddev, err, CI/mean)


