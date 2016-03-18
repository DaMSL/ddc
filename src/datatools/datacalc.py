import numpy as np
import logging

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)
np.set_printoptions(precision=3, suppress=True)



def posterior_prob (source):
  """
  Calculate posterior probability for set of observations
  """
  # Iterate for each item in the source
  groupby = {}
  for x_i in source:
    if x_i in groupby:
      groupby[x_i] += 1
    else:
      groupby[x_i] = 1
  probility_est = {}
  for v_i in groupby.keys():
    probility_est[v_i] = groupby[x_i] / len(source)
  return probility_est



def bootstrap_sampler (source, samplesize=.1, N=50, interval=.95):
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

