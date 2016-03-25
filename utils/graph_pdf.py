import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import numpy as np
import redis
import math

from datatools.datacalc import *

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


def bootstrap_actual (series, interval=.95):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo
  N = len(series)
  delta  = np.array(sorted(series))
  P_i = np.mean(series)
  ciLO = delta[math.ceil(N*ci_lo)]
  ciHI = delta[math.floor(N*ci_hi)]
  return (P_i, ciLO, ciHI, (ciHI-ciLO)/P_i, std)


HOST = 'bigmem0003'
SAVELOC = os.environ['HOME'] + '/ddc/graph/'

r = redis.StrictRedis(host=HOST, decode_responses=True)
u = redis.StrictRedis(host='localhost', decode_responses=True)
ab = sorted([(A,B) for A in range(5) for B in range(5)])

OBS_PER_NS = 1000


def recheckStats_cumulative(ts, cumulative=False):
  TIMESTEP = ts * OBS_PER_NS
  cuml = {b: [] for b in ab}
  cnts={b: 0 for b in ab}
  i = 0
  total = [0 for i in range(5)]
  print('Collecting data...')
  while i < len(obs):
    if i > 0 and i % TIMESTEP == 0:
      for b in ab:
        A, B = b
        if total[A] == 0:
          print('No DATA for state %d  at ns interval:  %d' % (A, i/OBS_PER_NS))
        else:
          cuml[b].append(cnts[b]/total[A])
    A, B = eval(obs[i])
    cnts[(A, B)] += 1
    i += 1
    total[A] += 1
  print('Calculating Convergence...')
  for b in ab:
    A, B = b
    cuml[b].append(cnts[b]/total[A])
  bs_all = {b: [] for b in ab}
  for b in ab:
    bs_all[b] = [bootstrap_std(cuml[b][:i], interval=.9) for i in range(10, len(cuml[b]))]
  bs_ci = {b: [] for b in ab}
  bs_sd = {b: [] for b in ab}
  bs_mn = {b: [] for b in ab}
  bs_er = {b: [] for b in ab}
  for b in ab:
    bs_mn[b] = np.array([x[0] for x in bs_all[b]])
    bs_ci[b] = np.array([x[1] for x in bs_all[b]])
    bs_sd[b] = np.array([x[2] for x in bs_all[b]])
    bs_er[b] = np.array([x[3] for x in bs_all[b]])
    # bs_cv[b] = np.array([x[3] for x in bs_all[b]])
  title_pre = 'Cumulative ' if cumulative else 'Bootstrap '
  for i in range(5):
    print('Generating graphs for state ', i)
    for b in ab[i*5:i*5+5]:
      plt.plot(np.arange(len(bs_ci[b]))*(TIMESTEP/1000), bs_ci[b], label=str(b))
    plt.xlabel(title_pre + 'Conf Interval (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/cuml_conv_%dns_ci_%d.png'%(ts, i))
    plt.close()
    for b in ab[i*5:i*5+5]:
      plt.errorbar(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], yerr=bs_er[b], label=str(b))
    plt.xlabel(title_pre +'Mean with error bars (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/cuml_conv_%dns_mnerr_%d.png'%(ts, i))
    plt.close()
  print('All Done!')
  return cuml

def postgen_bootstraps(all_obs, strap_size_ns, transonly=False, cumulative=False):
  obs_per_step = strap_size_ns * OBS_PER_NS
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
          print('No DATA for state %d  at ns interval:  %d' % (A, i/OBS_PER_NS))
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
      print('No DATA for state %d  at ns interval:  %d' % (A, i/OBS_PER_NS))
    else:
      bootstrap[b].append(cnts[b]/total[A])
  return bootstrap


def postcalc_bootstraps(data, burnin=0):
  print('Calculating Convergence...')
  stats = {b: [bootstrap_std(data[b][burnin:i], interval=.9) for i in range(burnin, len(data[b]))]  for b in ab}
  bs = {}
  for i, stat in enumerate(['mn', 'ci', 'cv', 'er']):
    bs[stat] = {b: np.array([x[i] for x in stats[b]]) for b in ab}
  return bs

def get_bootstrap_data(r, burnin=0):
  """ Pull bootstrap data (in real time) from given redis catalog
  """
  print('Calculating Convergence...')
  TIMESTEP = ts * OBS_PER_NS
  bs = {}
  # Load all bootstrap data from redis
  for stat in ['mn', 'ci', 'cv', 'er']:
    data = [float(val) for val in r.lrange('boot:%s:%s'%(prefix,stat), 0, -1)]
    bs[stat] = {b: [] for b in ab}
    if len(data) % 25 != 0:
      print("ERROR. Wrong # of results", stat, len(data))
      # return
    else:
      print('Total number of samples to plot: ', len(data)//25)
      # NOTE: Fix output to specific label-keys
      skip = burnin * 25
      for i, val in enumerate(data[skip:]):
        bs[stat][ab[i%25]].append(val)
    for b in ab:
      bs[stat][b] = np.array(bs[stat][b])
  return bs
    # Plot

def plot_bootstrap_graphs(bs, ts, prefix, subdir='.', samp_type=''):
  bootmethod = 'Iterative' if prefix=='iter' else 'Cumulative'
  for i in range(5):
    print('Generating graphs for state ', i)
    for b in ab[i*5:i*5+5]:
      plt.plot(np.arange(len(bs['ci'][b]))*(ts), (bs['ci'][b]/bs['mn'][b]), label=str(b))
    plt.title('%s Sampling with %s Bootstrap' % (samp_type, bootmethod))
    plt.xlabel('Total time in ns')
    plt.ylabel('Convergence of Confidence Interval')
    plt.legend()
    plt.savefig(SAVELOC + subdir+'/%s_%dns_tc_%d.png'%(prefix, ts, i))
    plt.close()
    for b in ab[i*5:i*5+5]:
      if len(bs['mn'][b]) == 0:
        continue
      plt.errorbar(np.arange(len(bs['mn'][b]))*(ts), bs['mn'][b], yerr=bs['er'][b], label=str(b))
    plt.title('%s Sampling with %s Bootstrap' % (samp_type, bootmethod))
    plt.xlabel('Total time in ns')
    plt.ylabel('Posterior Probabilty (w/error)')
    plt.legend()
    plt.savefig(SAVELOC + subdir+'/%s_%dns_mnerr_%d.png'%(prefix, ts, i))
    plt.close()
  print('All Done!')

# For post calc
def all_plots():
  BURNIN = 10
  STEPSIZE=50
  obs_unif = u.lrange('label:raw:sm', 0, -1)[150000:]
  obs_bias = r.lrange('label:rms', 0, -1)[300000:]

  boots = postgen_bootstraps(obs_unif, STEPSIZE, cumulative=False)
  bs_u = postcalc_bootstraps(boots)
  plot_bootstrap_graphs(bs_u, STEPSIZE, 'iter', 'uniform2', samp_type='UNIFORM')

  boots = postgen_bootstraps(obs_unif, STEPSIZE, cumulative=True)
  bs_u = postcalc_bootstraps(boots)
  plot_bootstrap_graphs(bs_u, STEPSIZE, 'cuml', 'uniform2', samp_type='UNIFORM')

  boots = postgen_bootstraps(obs_bias, STEPSIZE, cumulative=False)
  bs_b = postcalc_bootstraps(boots)
  plot_bootstrap_graphs(bs_b, STEPSIZE, 'iter', 'biased1', samp_type='BIASED')

  boots = postgen_bootstraps(obs_bias, STEPSIZE, cumulative=True)
  bs_b = postcalc_bootstraps(boots)
  plot_bootstrap_graphs(bs_b, STEPSIZE, 'cuml', 'biased1', samp_type='BIASED')


  # Live Data:
  bs_r = get_bootstrap_data(r, burnin=BURNIN)
  plot_bootstrap_graphs(bs_r, STEPSIZE, 'iter', 'biased1')
  plot_bootstrap_graphs(bs_r, STEPSIZE, 'cuml', 'biased1')


def current_plots(slist=None, cumulative=False, STEPSIZE=10):
  clr = ['r', 'g', 'b', 'y', 'k']
  obs_unif = u.lrange('label:raw:sm', 0, -1)
  obs_bias = r.lrange('label:rms', 0, -1)
  boots = postgen_bootstraps(obs_unif, STEPSIZE, cumulative=cumulative)
  bs_u = postcalc_bootstraps(boots)
  boots = postgen_bootstraps(obs_bias, STEPSIZE, cumulative=cumulative)
  bs_b = postcalc_bootstraps(boots)
  # Group data by originating state:
  statecv_u = [[] for i in range(5)]
  statecv_b = [[] for i in range(5)]
  for A in range(5):
    aggu = [[] for i in range(len(bs_u['ci'][(A, 0)]))]
    aggb = [[] for i in range(len(bs_b['ci'][(A, 0)]))]
    for B in range(5):
      for k in range(len(bs_u['ci'][(A, B)])):
       aggu[k].append(bs_u['ci'][(A, B)][k] / bs_u['mn'][(A, B)][k])
      for k in range(len(bs_b['ci'][(A, B)])):
       aggb[k].append(bs_b['ci'][(A, B)][k] / bs_b['mn'][(A, B)][k])
    statecv_u[A] = [sum(k)/len(k) for k in aggu]
    statecv_b[A] = [sum(k)/len(k) for k in aggb]
  statelist = [0, 1, 2, 3, 4] if slist is None else slist
  for st in statelist:
    X = statecv_u[st]
    plt.plot(np.arange(len(X))*(STEPSIZE), X, color=clr[st], 
      label='State '+str(st)+ ' Uniform', linestyle='-')
  for st in statelist:
    X = statecv_b[st]
    plt.plot(np.arange(len(X))*(STEPSIZE), X, color=clr[st], 
      label='State '+str(st)+ ' Biased', linestyle='--')
  plt.title('BIASED vs UNIFORM Convergence')
  plt.xlabel('Total Convergence (total time in ns)')
  plt.xlim=(0, 2000)
  plt.legend()
  prefix = 'cuml' if cumulative else 'iter'
  plt.savefig(SAVELOC + '%s_tc_%dns_%s.png' % (prefix, STEPSIZE, ''.join(str(i) for i in statelist)))
  plt.close()



def plot_bootstraps(data, ts, prefix, subdir='.'):
  print('Calculating Convergence...')
  TIMESTEP = ts * OBS_PER_NS
  bs = {b: [bootstrap_std(data[b][:i], interval=.9) for i in range(len(data[b]))]  for b in ab}
  bs_mn = {b: np.array([x[0] for x in bs[b]]) for b in ab}
  bs_ci = {b: np.array([x[1] for x in bs[b]]) for b in ab}
  bs_sd = {b: np.array([x[2] for x in bs[b]]) for b in ab}
  bs_er = {b: np.array([x[3] for x in bs[b]]) for b in ab}
  for i in range(5):
    print('Generating graphs for state ', i)
    for b in ab[i*5:i*5+5]:
      plt.plot(np.arange(len(bs_cv[b]))*(ts), len(bs_cv[b]), label=str(b))
    plt.xlabel('Total Convergence (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + subdir+'/%s_%dns_tc_%d.png'%(prefix, ts, i))
    plt.close()
    for b in ab[i*5:i*5+5]:
      plt.errorbar(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], yerr=bs_er[b], label=str(b))
    plt.xlabel('Mean with error bars (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + subdir*'/%s_%dns_mnerr_%d.png'%(prefix, ts, i))
    plt.close()
  print('All Done!')





# stats=recheckStats_separate(50)



    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_ci[b]))*(TIMESTEP/1000), bs_ci[b], label=str(b))
    # plt.xlabel('Conf Interval (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/%s_%dns_ci_%d.png'%(prefix, ts, i))
    # plt.close()
    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], label=str(b))
    # plt.xlabel(title_pre + 'Mean (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/boot_%dns_mn_%d.png'%(ts, i))
    # plt.close()
    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_sd[b]))*(ts), bs_sd[b], label=str(b))
    # plt.xlabel(title_pre +'StdDev (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/boot_%dns_std_%d.png'%(ts, i))
    # plt.close()

    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_sd[b]))*(ts), (bs_ci[b]/bs_mn[b]), label=str(b))
    # plt.xlabel(title_pre +'Total Convergence (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/cuml_conv_%dns_tc_%d.png'%(ts, i))
    # plt.close()
