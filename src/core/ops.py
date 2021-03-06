"""Collection point for common operators used throughout the application
"""
import math
import numpy as np
import dateutil.parser as du
import datetime as dt

from collections import deque

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

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

def str2date(d):
  return du.parse(d)

def datediff(date1, date2):
  d1 = str2date(date1) if isinstance(date1, str) else d1
  d2 = str2date(date2) if isinstance(date2, str) else d2
  return np.abs((d2-d1).total_seconds())


def tri_to_square(vect):
  '''Converted an upper (lower) triangle vector to a square matrix. Each elm
  in the vect is one elm in either the lower or upper symmetric matrix'''

  K = 1
  M = np.zeros(K)
  elms = deque(reversed(vect))
  while len(elms) > 0:
    M = np.hstack((np.zeros(shape=(K+1,1)), np.vstack((np.zeros(K), M))))
    K += 1
    for i in range(K-1, 0, -1):
      M[i][0] = M[0][i] = elms.popleft()
  return M

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
  This is actually just the confidence interval estimation
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
  mu = np.mean(source, axis=0)
  while i+blocksize < len(source):
    mu_i = np.mean(source[i:i+blocksize], axis=0)
    boot.append(mu_i)
    i += blocksize
  N = len(boot)
  if N < 2:
    return [np.ones(len(mu)) for i in range(5)]
  boot = np.array(boot)
  # mean = np.nan_to_num(np.mean(boot, axis=0))
  # stddev = np.nan_to_num(np.std(boot, axis=0))
  mean = np.mean(boot, axis=0)
  stddev = np.std(boot, axis=0)
  Z = 1.645  # .9 CI
  err = stddev / math.sqrt(N)
  CI = Z * err
  return (mu, CI, stddev, err, np.abs(CI/mu))


def makeLogisticFunc (maxval, steep, midpt):
  return lambda x: maxval / (1 + np.exp(-steep * (midpt - x)))