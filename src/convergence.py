import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import numpy as np
import redis
import math

from dateutil import parser as du
import datetime as dt

from datatools.datacalc import *
from collections import OrderedDict

import bench.db as db
import plot as P

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




# HOST = 'bigmem0003'
SAVELOC = os.environ['HOME'] + '/ddc/graph/'
# r = redis.StrictRedis(host=HOST, decode_responses=True)
# u = redis.StrictRedis(host='localhost', decode_responses=True)
s = redis.StrictRedis(port=6381, decode_responses=True)
p = redis.StrictRedis(port=6382, decode_responses=True)
u = redis.StrictRedis(port=6383, decode_responses=True)
b = redis.StrictRedis(port=6384, decode_responses=True)
r = redis.StrictRedis(port=6385, decode_responses=True)



elas = {label: redis.StrictRedis(port=6390+i, decode_responses=True) for i, label in enumerate(['elas_base', 'elas_500', 'elas_250'])}
el2 = {label: redis.StrictRedis(port=6401+i, decode_responses=True) \
  for i, label in enumerate(['elas5', 'elas10', 'elas15', 'elas25', 'elas50'])}


colormap = {'uniform': "r",'biased': 'b','parallel':'g', 'serial':'k', 'reweight':'y',
              'runtime_1000': "r", 'runtime_250': "b", 
              'elas_base': "r", 'elas_500': "g", 'elas_250': "b",
              'elas5': "r", 'elas10': "g", 'elas15': "c", 'elas25': "b", 'elas50': "k"}
pathmap = {'uniform': "uniform2",'biased': 'biased1','naive':'naive'}
ab = sorted([(A,B) for A in range(5) for B in range(5)])
OBS_PER_NS = 1000

def recheckStats_cumulative(ts, cumulative=False):
  TIMESTEP = ts * OBS_PER_NS
  cuml = {b: [] for b in ab}
  cnts = {b: 0 for b in ab}
  i = 0
  total = [0 for i in range(5)]
  print('Collecting data...')
  while i < len(obs):
    if i > 0 and i % TIMESTEP == 0:
      for b in ab:
        A, B = b
        if total[A] == 0:
          print('No DATA for state %d  at ns interval:  %d' % (A, i/OBS_PER_NS))
          cuml[b].append(0)
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
    bs_all[b] = [bootstrap_std(cuml[b][:i], interval=.9) for i in range(0, len(cuml[b]))]
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
          if len(bootstrap[b]) > 0:
            bootstrap[b].append(bootstrap[b][-1])
          else:
            bootstrap[b].append(1)
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
      if len(bootstrap[b]) > 0:
        bootstrap[b].append(bootstrap[b][-1])
      else:
        bootstrap[b].append(0)
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
  obs_unif = u.lrange('label:raw:lg', 0, -1)[150000:]
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

def current_plots(slist=None, cumulative=False, STEPSIZE=10, obs_unif=None, obs_bias=None):
  clr = ['r', 'g', 'b', 'y', 'k']
  if obs_unif is None:
    obs_unif = u.lrange('label:raw:lg', 0, -1)
  if obs_bias is None:
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
  plt.xlim=(0, 3000)
  plt.legend()
  prefix = 'cuml' if cumulative else 'iter'
  plt.savefig(SAVELOC + '%s_tc_%dns_%s_C.png' % (prefix, STEPSIZE, ''.join(str(i) for i in statelist)))
  plt.close()
  return statecv_u, statecv_b

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


def bystate(slist=None, cumulative=False, STEPSIZE=50):
  data = {'naive':{}, 'biased':{}, 'uniform':{}}
  data['uniform']['obs'] = u.lrange('label:raw:lg', 0, -1)[150000:]
  data['biased']['obs'] = b.lrange('label:rms', 0, -1)[300000:]
  data['naive']['obs'] = p.lrange('label:rms', 0, -1)
  for e in ['parallel']:
    data[e]['conv'] = [[] for i in range(5)]
    bootstrap = GP.postgen_bootstraps(data[e]['obs'], STEPSIZE, cumulative=cumulative)
    data[e]['boot'] = GP.postcalc_bootstraps(bootstrap)
    for A in range(5):
      agg = [[] for i in range(len(data[e]['boot']['ci'][(A, 0)]))]
      for B in range(5):
        for k in range(len(data[e]['boot']['ci'][(A, B)])):
          agg[k].append(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k])
      data[e]['conv'][A] = [sum(k)/len(k) for k in agg]
  maxlim = min([len(data[e]['conv'][A]) for e in data.keys() for A in range(5)])
  statelist = [0, 1, 2, 3, 4] if slist is None else slist
  for A in [0, 1, 2, 3, 4]:
    # maxlen = min([len(data[e]['conv'][A]) for e in data.keys()])
    for e in data.keys():
      X = data[e]['conv'][A]
      print(len(X))
      plt.plot(np.arange(min(40,len(X)))*(STEPSIZE), X[:min(40,len(X))], color=colormap[e], label=e)
    # plt.title('Convergence for State %d', A)
    plt.xlabel('Total Convergence (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + 'TotConv_%s.png' % (A))
    plt.close()
  return data

def histogram(slist=None, cumulative=False, STEPSIZE=50):
  data = {'serial': {}, 'parallel':{}, 'biased':{}, 'uniform':{}, 'reweight': {}}
  data['uniform']['obs'] = u.lrange('label:raw:lg', 0, -1)[150000:]
  data['biased']['obs'] = b.lrange('label:rms', 0, -1)[300000:]
  data['serial']['obs'] = s.lrange('label:rms', 0, -1)
  data['parallel']['obs'] = p.lrange('label:rms', 0, -1)
  data['reweight']['obs'] = r.lrange('label:rms', 0, -1)
  for e in ['parallel']:
    for b in ab:
      data[e][b] = 0
    for i in range(5):
      data[e]['%dt'%i] = 0
      data[e]['%dw'%i] = 0
    maxlen = min(1999000, len(data[e]['obs']))
    for o in data[e]['obs'][:maxlen]:
      A, B = eval(o)
      data[e][(A,B)] += 1
      if A == B:
        data[e]['%dw'%A] +=1
      else:
        data[e]['%dt'%A] +=1
  for b in ab:
    print('%s|%d|%d|%d|%d|%d' % (b,data['serial'][b],data['parallel'][b],data['uniform'][b],data['biased'][b],data['reweight'][b]))


def convtw(data, slist=None, cumulative=False, STEPSIZE=25):
  for e in data.keys():
    print("Post Calculating Boostraps for %s. Using stepsize of %d" % (e, STEPSIZE))
    # data[e]['conv'] = [[] for i in range(5)]
    data[e]['wtcnt'] = {'%d-Well' %A: 0 for A in range(5)}
    data[e]['wtcnt'] = {'%d-Tran' %A: 0 for A in range(5)}
    bootstrap = postgen_bootstraps(data[e]['obs'], STEPSIZE, cumulative=cumulative)
    data[e]['boot'] = postcalc_bootstraps(bootstrap)
    for A in range(5):
      aggW = [[] for i in range(len(data[e]['boot']['ci'][(A, 0)]))]
      aggT = [[] for i in range(len(data[e]['boot']['ci'][(A, 0)]))]
      for B in range(5):
        for k in range(len(data[e]['boot']['ci'][(A, B)])):
          if A == B:
            aggW[k].append(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k])
          else:
            aggT[k].append(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k])
      # data[e]['conv'][A] = [sum(k)/len(k) for k in agg]
      data[e]['wtcnt']['%d-Well' %A] = [sum(k)/len(k) for k in aggW]
      data[e]['wtcnt']['%d-Tran' %A] = [sum(k)/len(k) for k in aggT]
  return data


def plot_conv_tw(data, STEPSIZE=25, xlim=None):
  for A in [0, 1, 2, 3, 4]:
    print('Plotting graphs for state %d' % A)
    plt.clf()
    ax = plt.subplot(111)
    maxlen = min([len(data[e]['wtcnt']['%d-Well' %A]) for e in data.keys()])
    for e in data.keys():
      X = data[e]['wtcnt']['%d-Well' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[e], label=e)
    plt.title('Convergence for State %d (WELL)'% A)
    plt.xlabel('Convergence: State %d WELL (total time in ns)'%A)
    if xlim is not None:
      ax.set_xlim(xlim)
    plt.legend()
    plt.savefig(SAVELOC + 'conv-well-%s.png' % (A))
    plt.close()
    plt.clf()
    ax = plt.subplot(111)
    maxlen = min([len(data[e]['wtcnt']['%d-Tran' %A]) for e in data.keys()])
    for e in data.keys():
      X = data[e]['wtcnt']['%d-Tran' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[e], label=e)
    plt.title('Convergence for State %d (Transitions)'% A)
    # ax.set_xlim(75,600)
    plt.xlabel('Convergence: State %d TRANSITIONS (total time in ns)'%A)
    if xlim is not None:
      ax.set_xlim(xlim)
    plt.legend()
    plt.savefig(SAVELOC + 'conv-tran-%s.png' % (A))
    plt.close()


def convtw_binary(data, slist=None, cumulative=False, STEPSIZE=25):
  for e in data.keys():
    print("Post Calculating Boostraps for %s. Using stepsize of %d" % (e, STEPSIZE))
    # data[e]['conv'] = [[] for i in range(5)]
    data[e]['wtcnt'] = {'%d-Well' %A: 0 for A in range(5)}
    data[e]['wtcnt'] = {'%d-Tran' %A: 0 for A in range(5)}
    bootstrap = postgen_bootstraps(data[e]['obs'], STEPSIZE, cumulative=cumulative)
    data[e]['boot'] = postcalc_bootstraps(bootstrap)
    for A in range(5):
      aggW = [[] for i in range(len(data[e]['boot']['ci'][(A, 0)]))]
      aggT = [[] for i in range(len(data[e]['boot']['ci'][(A, 0)]))]
      for B in range(5):
        for k in range(len(data[e]['boot']['ci'][(A, B)])):
          if A == B:
            aggW[k].append(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k])
          else:
            aggT[k].append(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k])
      # data[e]['conv'][A] = [sum(k)/len(k) for k in agg]
      data[e]['wtcnt']['%d-Well' %A] = [sum(k)/len(k) for k in aggW]
      data[e]['wtcnt']['%d-Tran' %A] = [sum(k)/len(k) for k in aggT]
  statelist = [0, 1, 2, 3, 4] if slist is None else slist

  for A in [0, 1, 2, 3, 4]:
    print('Plotting graphs for state %d' % A)
    plt.clf()
    ax = plt.subplot(111)
    maxlen = min([len(data[e]['wtcnt']['%d-Well' %A]) for e in data.keys()])
    for e in data.keys():
      X = data[e]['wtcnt']['%d-Well' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[e], label=e)
    plt.title('Convergence for State %d (WELL)'% A)
    plt.xlabel('Convergence: State %d WELL (total time in ns)'%A)
    ax.set_xlim(75,600)
    plt.legend()
    plt.savefig(SAVELOC + 'TC_Comparison_Well-%s.png' % (A))
    plt.close()
    plt.clf()
    ax = plt.subplot(111)
    maxlen = min([len(data[e]['wtcnt']['%d-Tran' %A]) for e in data.keys()])
    for e in data.keys():
      X = data[e]['wtcnt']['%d-Tran' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[e], label=e)
    plt.title('Convergence for State %d (Transitions)'% A)
    ax.set_xlim(75,600)
    plt.xlabel('Convergence: State %d TRANSITIONS (total time in ns)'%A)
    plt.legend()
    plt.savefig(SAVELOC + 'TC_Comparison_Tran-%s.png' % (A))
    plt.close()
  return data

def Convergence5(stepsize):
  data = {'serial': {}, 'parallel':{}, 'biased':{}, 'uniform':{}, 'reweight': {}}
  data['uniform']['obs'] = u.lrange('label:raw:lg', 0, -1)[150000:]
  data['biased']['obs'] = b.lrange('label:rms', 0, -1)[25000:]
  data['serial']['obs'] = s.lrange('label:rms', 0, -1)
  data['parallel']['obs'] = p.lrange('label:rms', 0, -1)
  data['reweight']['obs'] = r.lrange('label:raw:lg', 0, -1)
  return convtw(data, STEPSIZE=stepsize)

def Elasticity(step=25):
  data = {}
  for k in el2.keys():
    if k == 'elas15':
      continue
    data[k] = {'obs': el2[k].lrange('label:rms', 0, -1)}
  return convtw(data, STEPSIZE=step)


def showconvlist(data):
  for run in data.keys():
    for k, v in data[run]['wtcnt'].items():
      print('%s,%s,%s' % (run, k, ','.join(['%4.2f'%i for i in v[1:]])))



def getobs(name):
  eid = db.get_expid(name)
  t = db.runquery('select idx, obs from obs where expid=%d order by idx'%eid)
  return [i[1] for i in t]

def getdistinct(obslist):
  V = set()
  for i in obslist:
    V.add(i)
  return list(V)

def calc_conv(obs, step=5000):
  binlist = [(a,b) for a in range(5) for b in range(5)]
  V = getdistinct(obs)
  results = {v: [] for v in V}
  N = len(obs)
  for m in range(step, N, step):
    conv = bootstrap_sampler(obs[:min(m,N)], samplesize=.2, N=100)
    for key in results.keys():
      if key in conv.keys():
        results[key].append(min(conv[key][3], 1.))
      else:
        results[key].append(1.)
  return results


def show_conv(conv):
  for k,v in sorted(conv.items()): 
    print(k, ['%4.2f'%i for i in v])


def obs2tw(obslist):
  out = []
  for obs in obslist:
    A,B = eval(obs)
    out.append(('well' if A==B else 'tran') + '-%d'%A)
  return out


def graphexpr(elist):
  obslist = {name: getobs(name) for name in elist}
  convlist = {k: calc_conv(v) for k,v in obslist.items()}



def conv_over_time(name, step=10000, tw=False):
  eid = db.get_expid(name)
  obs = getobs(name)
  if tw:
    obs = obs2tw(obs)

  V = getdistinct(obs)
  N = len(obs)
  plotlists = {v: [] for v in V}
  sw_list=db.runquery('select start,time,numobs from sw where expid=%d order by start'%eid)
  end_ts = lambda x: du.parse(x[0]).timestamp() + x[1]
  ts_0 = du.parse(sw_list[0][0]).timestamp()
  sw = sorted([dict(start=x[0], time=x[1], numobs=x[2], end=end_ts(x)-ts_0) for x in sw_list], key=lambda i: i['end'])
  n = 0
  snum = 0
  nextcalc = step
  while n < N and snum < len(sw):
      n += sw[snum]['numobs']
      if n > nextcalc:
        t = sw[snum]['end'] / 3600.
        # c = bootstrap_sampler(obs[:min(n,N)], samplesize=.25)
        c = bootstrap_iter(obs[:min(n,N)], size=step)
        for v in V:
          if v in c.keys():
            # plotlists[v].append((t, min(c[v][3], 1.)))
            plotlists[v].append((t, min(c[v][1]/c[v][0], 1.)))
          else:
            plotlists[v].append((t, 1.))
        nextcalc += step
      snum += 1
  return plotlists


def time_comp(elist, step=10000):
  conv = {}
  for e in elist:
    conv[e] = conv_over_time(e, step=step)
  results = {}
  for k in conv[e].keys():
    results[k] = {e: conv[e][k] for e in elist}
  return results


def total_time(name):
  end_ts = lambda x: du.parse(x[0]).timestamp() + x[1]
  eid = db.get_expid(name)
  sw_list=db.runquery('select start,time from sw where expid=%d order by start'%eid)
  start = du.parse(sw_list[0][0]).timestamp()
  sw = sorted([dict(start=x[0], time=x[1], end=end_ts(x)-start) for x in sw_list], key=lambda i: i['end'])
  return sw[-1]['end']



def do_histogram():
  # obs = {n: c for n,c in db.runquery('select expname, count(obs) from expr e, obs where e.expid=obs.expid group by expname')}
  data = OrderedDict()
  raw = OrderedDict()
  rawb = getobs('biased3_5')[47000:160000] + getobs('biased3_10')[47000:165000] + \
     getobs('biased3_25')[47000:190000] +getobs('biased3_50')[47000:173000]


  raw['Serial']   = getobs('serial')[50000:550000]
  raw['Parallel'] = getobs('uncontrol')[:500000]
  raw['Uniform']  = getobs('uniform2')[50000:550000]
  raw['Biased']   = rawb
  raw['Reweight'] = getobs('reweight4')[50000:550000]
  for key, obslist in raw.items():
    groupby = {}
    for i in obslist:
      if i not in groupby:
        groupby[i] = 0
      groupby[i] += 1
    data[key] = groupby
  P.histogram(data, 'Histogram_Event_Counts', ylim=(0,200000))
  

def plot_elas():
  rlist=['rtime100_20', 'rtime100_100', 'rtime250_10', 'rtime250_20', 'rtime250_50', 'rtime250_100', 'rtime250_200']
  costlabels={'rtime100_100': '100ps/100/%d'%(1367+32),
  'rtime100_20':  '100ps/20/%d'%(1315+32),
  'rtime250_10':  '250ps/10/%d'%(1351+95),
  'rtime250_100': '250ps/100/%d'%(1711+40),
  'rtime250_20':  '250ps/20/%d'% (825+109),
  'rtime250_200': '250ps/200/%d'%(1561+29),
  'rtime250_50':  '250ps/50/%d'%(1555+52)}

  data = time_comp(rlist, step=1000)
  for k,v in data.items():
    P.scats(v, 'ElasticityCost-'+k, xlim=(0,5), labels=costlabels, xlabel='Time (in hours)  Legend= SimTime/#Jobs/TotalCost')


if __name__=='__main__':
  do_histogram()
