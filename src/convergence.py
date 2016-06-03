import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import numpy as np
import redis
import math

from dateutil import parser as du
import datetime as dt
import datatools.feature as FL
import core.ops as op

from datatools.datacalc import *
from collections import OrderedDict

import bench.db as db
import core.ops as op
import plot as P
import features as F

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


def bootstrap_actual (series, interval=.9):
  """
  Bootstrap algorithm for sampling and confidence interval estimation
  """
  ci_lo = (1. - interval)/2
  ci_hi  = 1. - ci_lo
  N = len(series)
  delta  = np.array(sorted(series))
  P_i = np.mean(series)
  std = np.std(series)
  ciLO = delta[math.ceil(N*ci_lo)]
  ciHI = delta[math.floor(N*ci_hi)]
  return (P_i, ciHI-ciLO, std, (ciHI-ciLO)/P_i)




# HOST = 'bigmem0003'
SAVELOC = os.environ['HOME'] + '/ddc/graph/'
# r = redis.StrictRedis(host=HOST, decode_responses=True)
# u = redis.StrictRedis(host='localhost', decode_responses=True)
s = redis.StrictRedis(port=6380, decode_responses=True)
p = redis.StrictRedis(port=6381, decode_responses=True)
u = redis.StrictRedis(port=6382, decode_responses=True)
b = redis.StrictRedis(port=6383, decode_responses=True)
# b = redis.StrictRedis(port=6383, decode_responses=True)
 # = redis.StrictRedis(port=6384, decode_responses=True)
m = redis.StrictRedis(port=6385, decode_responses=True)
k = redis.StrictRedis(port=6386, decode_responses=True)
w = redis.StrictRedis(port=6387, decode_responses=True)



elas = {label: redis.StrictRedis(port=6390+i, decode_responses=True) for i, label in enumerate(['elas_base', 'elas_500', 'elas_250'])}
el2 = {label: redis.StrictRedis(port=6401+i, decode_responses=True) \
  for i, label in enumerate(['elas5', 'elas10', 'elas15', 'elas25', 'elas50'])}


colormap = {'Uniform': "red",
            'Biased': 'olive',
            'Parallel':'purple', 
            'Serial':'blue',
            'MVNN': "darkgreen", 
            'Knapsack': "slategrey", 
            'reweight':'y',
            'runtime_1000': "r",
            'runtime_250': "b", 
            'elas_base': "r", 'elas_500': "g", 'elas_250': "b",
            'elas5': "r", 'elas10': "g", 'elas15': "c", 'elas25': "b", 'elas50': "k"}

labelList = [('serial','Serial'),
             ('parallel','Parallel'), 
             ('uniform3','Uniform'), 
             ('biased4','Biased'), 
             ('feal1','MVNN')] 
labels=['C0', 'C1', 'C2', 'C3', 'C4', 'S0', 'S1', 'S2', 'S3', 'S4', '0-1', '0-2','0-3', '0-4', '1-2', '1-3','1-4','2-3','2-4','3-4']


pathmap = {'uniform': "uniform2",'biased': 'biased1','naive':'naive'}
ab = sorted([(A,B) for A in range(5) for B in range(5)])
OBS_PER_NS = 1000

nex = {}
ntr = {}

def preload_data():
  global nex, ntr
  ser = F.ExprAnl(port=6380); ser.load(76)
  para = F.ExprAnl(port=6381)
  para.loadtraj(list(range(23)), first=26500)
  _=[para.rms(i) for i in range(25)]
  unif = F.ExprAnl(port=6382); unif.load(770)
  bias = F.ExprAnl(port=6383); bias.load(770)
  rwgt = F.ExprAnl(port=6387); rwgt.load(750)
  # mvnn = F.ExprAnl(host='compute0491', port=6392); mvnn.load(1722)
  mvnn = F.ExprAnl(port=6385); mvnn.load(1722)
  # knap = F.ExprAnl(port=6386)
  # knap.load(442)
  print((dt.datetime.now()-st).total_seconds())

  labels=['C0', 'C1', 'C2', 'C3', 'C4', 'S0', 'S1', 'S2', 'S3', 'S4', '0-1', '0-2','0-3', '0-4', '1-2', '1-3','1-4','2-3','2-4','3-4']

  nex = {'Serial':ser, 'Parallel':para,'Uniform':unif,'Biased':bias, 'MVNN':mvnn, 'Reweight':rwgt}
  ntr = {'Serial':76, 'Parallel':23,'Uniform':770,'Biased':750, 'MVNN':1722, 'Reweight':750}



# DATA CONVERTING
def obs2tw(obslist):
  out = []
  for obs in obslist:
    A,B = eval(obs)
    out.append(('well' if A==B else 'tran') + '-%d'%A)
  return out

def getobs(name):
  eid = db.get_expid(name)
  t = db.runquery('select idx, obs from obs where expid=%d order by idx'%eid)
  return [i[1] for i in t]


def getdistinct(obslist):
  V = set()
  for i in obslist:
    V.add(i)
  return list(V)

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
          if data[e]['boot']['ci'][(A, B)][k] == 0:
            value = 1.
          elif data[e]['boot']['mn'][(A, B)][k] == 0:
            value = 1.
          else:
            value = min(data[e]['boot']['ci'][(A, B)][k] / data[e]['boot']['mn'][(A, B)][k], 1.)
          if A == B:
            aggW[k].append(value)
          else:
            aggT[k].append(value)
      # data[e]['conv'][A] = [sum(k)/len(k) for k in agg]
      data[e]['wtcnt']['%d-Well' %A] = [sum(k)/len(k) for k in aggW]
      data[e]['wtcnt']['%d-Tran' %A] = [sum(k)/len(k) for k in aggT]
  return data

def plotconv_tw(data, STEPSIZE=5, xlim=None, labels=None):
  global colormap
  colorList = plt.cm.brg(np.linspace(0, 1, len(data.keys())))
  labelList = [(k,k) for k in sorted(data.keys())] if labels is None else labels
  for A in [0, 1, 2, 3, 4]:
    print('Plotting graphs for state %d' % A)
    plt.clf()
    ax = plt.subplot(111)
    maxlen = min([len(data[e]['wtcnt']['%d-Well' %A]) for e in data.keys()])
    # for (e, L), C in zip(labelList, colorList):
    for e, L in labelList:
      X = data[e]['wtcnt']['%d-Well' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[L], label=L)
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
    for e, L in labelList:
      X = data[e]['wtcnt']['%d-Tran' %A][:maxlen]
      plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[L], label=L)
    plt.title('Convergence for State %d (Transitions)'% A)
    plt.xlabel('Convergence: State %d TRANSITIONS (total time in ns)'%A)
    if xlim is not None:
      ax.set_xlim(xlim)
    plt.legend()
    plt.savefig(SAVELOC + 'conv-tran-%s.png' % (A))
    plt.close()


def plottw_agg(data, STEPSIZE=5, xlim=None, labels=None):
  global colormap
  colorList = plt.cm.brg(np.linspace(0, 1, len(data.keys())))
  labelList = [(k,k) for k in sorted(data.keys())] if labels is None else labels
  cl = ['3-Tran', '1-Tran', '0-Tran', '0-Well', '4-Well', '1-Well', '4-Tran', '3-Well', '2-Tran', '2-Well']
  agg = {name: [] for name in data.keys()}  
  for name in data.keys():
    for i in range(len(data[name]['wtcnt']['0-Well'])):
      agg[name].append(np.sum([data[name]['wtcnt'][c][i] for c in cl])/10)    
  print('Plotting graphs for all states')
  plt.clf()
  ax = plt.subplot(111)
  maxlen = min([len(agg[e]) for e in data.keys()])
  for e, L in labelList:
    X = agg[e][:maxlen]
    plt.plot(np.arange(len(X))*(STEPSIZE), X, color=colormap[L], label=L)
  plt.title('Total Convergence')
  plt.xlabel('Total time (in ns)')
  if xlim is not None:
    ax.set_xlim(xlim)
  plt.legend()
  plt.savefig(SAVELOC + 'conv-total.png')
  plt.close()


elist = ['serial', 'parallel', 'uniform3', 'biased4', 'feal1']  #, 'knap1'] 
             # ('knap1','Knapsack')] 

def graphexprA(elist=elist, step=5, labels=labelList):
  data = {name: {'obs': getobs(name)} for name in elist}
  data = convtw(data, STEPSIZE=step)
  plotconv_tw(data, STEPSIZE=step, xlim=(20,200), labels=labelList)

def graphagg(elist=elist, step=5, labels=labelList):
  data = {name: {'obs': getobs(name)} for name in elist}
  data = convtw(data, STEPSIZE=step)
  plottw_agg(data, STEPSIZE=step, xlim=(20,200), labels=labelList)


def calc_feal(r):
  rmslist = [np.fromstring(i) for i in r.lrange('subspace:rms', 0, -1)]
  feal  = [FL.feal.atemporal(rms) for rms in rmslist]
  pipe = r.pipeline()
  for f in feal:
    r.rpush('subspace:feal', f.tostring())
  pipe.execute()


def plotsep(boot, step=10, tag=''):
  N = min([len(v) for v in boot.values()])
  data = {k: {} for k in labels}  
  for j,k in enumerate(labels):
    for ex, b in boot.items():
      # data[k][ex] = [min(np.abs(b[i][j][2]-b[i][j][1]),1.) for i in range(N)]
      data[k][ex] = [min(b[i][1][j],1.) for i in range(N)]
  for k, v in data.items():
    P.lines(v, 'Conv-%s-%s'%(tag,k), step=step, xlabel='Simulation Time (in ns)')

  # merge = {k: [np.mean([min(1.,np.abs(boot[i][f][2]-b[i][f][1])) for f in range(5, 20)]) for i in range(N)] for k in boot.keys()}

def plotm(boot, step=10, tag=''):
  N = min([len(v) for v in boot.values()])
  merge = {k: [np.mean([min(1., boot[k][i][1][f]) for f in range(10, 20)]) for i in range(N)] for k in boot.keys()}
  P.lines(merge, 'Conv-%s-CI-MERGE'%tag, step=step, xlabel='Simulation Time (in ns)')
  merge = {k: [np.mean([min(1., boot[k][i][2][f]) for f in range(10, 20)]) for i in range(N)] for k in boot.keys()}
  P.lines(merge, 'Conv-%s-STD-MERGE'%tag, step=step, xlabel='Simulation Time (in ns)')
  merge = {k: [np.mean([min(1., boot[k][i][3][f]) for f in range(10, 20)]) for i in range(N)] for k in boot.keys()}
  P.lines(merge, 'Conv-%s-Err-MERGE'%tag, step=step, xlabel='Simulation Time (in ns)')
  merge = {k: [np.mean([min(1., boot[k][i][4][f]) for f in range(10, 20)]) for i in range(N)] for k in boot.keys()}
  P.lines(merge, 'Conv-%s-CiMu-MERGE'%tag, step=step, xlabel='Simulation Time (in ns)')

def bootstrap(ex, trnum, size, method, state=None):
  print(ex.r.get('name'), '-', end=' ')
  feal = ex.all_feal(True, method)[:375000]
  # feal = ex.all_feal()[:410000]
  i = 0
  boot = []
  while i+size < len(feal):

    if i % 100000 == 0:
      print(i, end=' ')
    i += size
  print('done')
  return boot

def plot_boot(data, step=10, tag=''):
  N = min([len(boot[k][st]) for k in boot.keys()])
  data = {k: {} for k in labels[5:]}  
  for j,k in enumerate(labels[5:]):
    for ex, b in boot.items():
      # data[k][ex] = [min(np.abs(b[i][j][2]-b[i][j][1]),1.) for i in range(N)]
      data[k][ex] = [min(b[st][j],1.) for i in range(N)]
  for k, v in data.items():
    P.lines(v, 'Conv-%d-%s-%s'%(st,tag,k), step=step, xlabel='Simulation Time (in ns)')




MASK = [[5, 10, 11, 12, 13],
        [6, 10, 14, 15, 16],
        [7, 11, 14, 17, 18],
        [8, 12, 15, 17, 19],
        [9, 13, 16, 18, 19]]
def calc_boot(ex, size, method=None, limit=375000, state=None):
  print(ex.r.get('name'), '-', end=' ')
  if method is None:
    feal = ex.all_feal()[:limit]
  else:
    feal = ex.all_feal(True, method)[:limit]
  i = 0
  boot = [[] for k in range(5)]
  ci = [[] for k in range(5)]
  while i+size < len(feal):
    for state in range(5):
      arr = np.array(feal[i:i+size]).T
      feal_ci = []
      straps = np.array([bootstrap_std(arr[feat]) for feat in range(5, 20)])
      for feat in MASK[state]:
        # if np.isnan(straps[feat-5].any()):
        #   feal_ci.append(1.)
        # else:
          # feal_ci.append(straps[feat-5][1]/straps[feat-5][0])
          feal_ci.append(straps[feat-5][1])
      ci[state].append(feal_ci)
      feal_ci = []
      for feat in range(5):
        calc = bootstrap_std([x[feat] for x in ci[state]])
        # feal_ci.append(calc[1]/calc[0])
        feal_ci.append(calc[1])
      boot[state].append(feal_ci)
    i += size
  return boot

# def boot:
    # boot.append(op.bootstrap_block(feal[:i+size], size))


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

def showconvlist(data):
  for run in data.keys():
    for k, v in data[run]['wtcnt'].items():
      print('%s,%s,%s' % (run, k, ','.join(['%4.2f'%i for i in v[1:]])))

def calc_conv(obs, step=5000, Nsamp=10):
  V = getdistinct(obs)
  results = {v: [] for v in V}
  N = len(obs)
  for m in range(step, N, step):
    conv = bootstrap_sampler(obs[:min(m,N)], samplesize=.2, N=Nsamp)
    for key in results.keys():
      if key in conv.keys():
        results[key].append(min(conv[key][3], 1.))
      else:
        results[key].append(1.)
  return results

def show_conv(conv):
  for k,v in sorted(conv.items()): 
    print(k, ['%4.2f'%i for i in v])

def graphexprB(elist, step=5, tw=True):
  obsbin = {name: getobs(name) for name in elist}
  if tw:
    obslist = {k: obs2tw(v) for k, v in obsbin.items()}
  else:
    obslist = obsbin
  convlist = {k: calc_conv(v, step=step*1000) for k,v in obslist.items()}
  series = {}
  for k, v in convlist.items():
    for tw, data in v.items():
      if tw not in series:
        series[tw] = {}
      series[tw][k] = data
  for k, v in series.items():
    P.lines(v, 'convB-'+k, step=step)


def graphexprC(elist, step=5, tw=True):
  obsbin = {name: getobs(name) for name in elist}
  if tw:
    obslist = {k: obs2tw(v) for k, v in obsbin.items()}
  else:
    obslist = obsbin
  seriesList = getdistinct(obslist[elist[0]])
  series = {k: {} for k in seriesList}
  for name in elist:
    print('Bootstrapping on ', name)
    for k in seriesList:
      series[k][name] = []
    N = len(obslist[name])
    for n in range(0, N, step*1000):
      c = bootstrap_iter(obslist[name][:min(n,N)], size=step)
      for k in seriesList:
        if k in c.keys():
          series[k][name].append(min(c[k][1]/c[k][0], 1.))
        else:
          series[k][name].append(1.)
  for k, v in series.items():
    P.lines(v, 'convC-'+k, step=step)



#####  ELASTICITY







def Elasticity(step=25):
  data = {}
  for k in el2.keys():
    if k == 'elas15':
      continue
    data[k] = {'obs': el2[k].lrange('label:rms', 0, -1)}
  return convtw(data, STEPSIZE=step)



def plot_elas():
  rlist=['rtime250_5', 'rtime250_10', 'rtime250_20', 'rtime250_25', 'rtime250_50',  'rtime250_75', 'rtime250_100', 'rtime250_200']
  # rlist=['rtime100_20', 'rtime100_100', 'rtime250_10', 'rtime250_20', 'rtime250_50', 'rtime250_100', 'rtime250_200']
  costlabels={
  # 'rtime100_100': '100ps/100/%d'%(1367+32),
  # 'rtime100_20':  '100ps/20/%d'%(1315+32),
  'rtime250_5':  '250ps/5/%d'%(923+114),
  'rtime250_10':  '250ps/10/%d'%(1351+95),
  'rtime250_100': '250ps/100/%d'%(1711+40),
  'rtime250_20':  '250ps/20/%d'% (825+109),
  'rtime250_25':  '250ps/25/%d'% (1300+80),
  'rtime250_75':  '250ps/75/%d'% (1400+80),
  'rtime250_200': '250ps/200/%d'%(1561+29),
  'rtime250_50':  '250ps/50/%d'%(1555+52)}

  data = time_comp(rlist, step=1000)
  for k,v in data.items():
    P.scats(v, 'ElasticityCost-'+k, xlim=(0,5), labels=costlabels, xlabel='Time (in hours)  Legend= SimTime/#Jobs/TotalCost')



def elas_boot(ex, size, method=None, limit=375000, state=None):
  name = ex.r.get('name')
  print(name, '-', end=' ')
  eid = db.get_expid(name)
  # Get feature landscape (for convergence)
  if method is None:
    feal = ex.all_feal()[:limit]
  else:
    feal = ex.all_feal(True, method)[:limit]
  plotlist = []

  # Get list of all simulations
  sw_list=db.runquery('select start,time,numobs from sw where expid=%d order by start'%eid)
  print('Simulations read: ', len(sw_list))
  end_ts = lambda x: du.parse(x[0]).timestamp() + x[1]

  # Account for any gaps in execution & adjust (ensure real time is logically grouped)
  ts_0 = du.parse(sw_list[0][0]).timestamp()
  last = ts_0
  sw_seq = []
  gap = 0
  cutoff = 30*60  # 30 min gap is bad
  swbystart = sorted(sw_list, key=lambda i: i[0])
  for s, t, n in swbystart:
    sim_start = du.parse(s).timestamp() - ts_0
    if sim_start - last > cutoff:
      print('FOUND GAP:', int(sim_start - last))
      gap += sim_start - last + cutoff
    new_start = sim_start - gap
    end = new_start + t
    sw_seq.append({'start': new_start, 'end':end, 'numobs':n})
    last = sim_start

  maxobs = sum([i['numobs'] for i in sw_seq])

  # Sort by end time (for convergence)
  sw = sorted(sw_seq, key=lambda i: i['end'])
  N = min(limit, maxobs, len(feal))
  dnum = 0      # Data item # (as in stream)
  snum = 0      # Sim #
  lastcalc = 0
  nextcalc = size
  i = 0
  boot = []
  ci = []

  # Process each simulation's observations, batch into step-sizes and calc bootstrap
  last_conv = 1.
  while dnum < N and snum < len(sw):
    dnum += sw[snum]['numobs']
    if dnum > nextcalc:
      t = sw[snum]['end'] / 3600.  # get time
      arr = np.array(feal[lastcalc:dnum]).T
      feal_ci = []
      straps = np.array([bootstrap_std(arr[feat]) for feat in range(5, 20)])
      for feat in range(5, 15):
        feal_ci.append(straps[feat][1]/straps[feat][0])
        # feal_ci.append(straps[feat-5][1])
      ci.append(feal_ci)
      feal_ci = []
      for feat in range(10):
        calc = bootstrap_std([x[feat] for x in ci])
        feal_ci.append(calc[1]/calc[0])
        # feal_ci.append(calc[1])
      # conv = min(1., np.mean(feal_ci[1:]))
      conv = np.mean(np.nan_to_num(feal_ci))
      if conv == 0:
        conv = last_conv
      plotlist.append((t, conv))
      lastcalc = dnum
      nextcalc += size
      nextcalc = min(nextcalc, N)
      last_conv = conv
    snum += 1
  return plotlist


def all_elas():
  rlist = [5, 10, 25, 50, 75, 100, 200]
  rex   = {k: F.ExprAnl(port=6391+i) for i, k in enumerate(rlist)}
  for k, e in rex.items(): 
    print('Loading', k)
    e.load(min(750, len(e.conf)))

  name = 'rtime250_%d' % k
  eid = db.get_expid(name)
  db.adhoc('select swname, start, time, cpu from sw where expid=%d' % eid)

  allboot = {k: C.elas_boot(v, 1000, 'dsw', limit=151000) for k,v in rex.items()}
  plots = [{k: v[s] for k,v in allboot.items()} for s in range(5)]
  for i, p in enumerate(plots):
    P.scats(p, 'Elas_%d'%i)






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


def elas_feal(name, feal_list, max_obs, step=2000):
  eid = db.get_expid(name)
  plotlist = []
  sw_list=db.runquery('select start,time,numobs from sw where expid=%d order by start'%eid)
  end_ts = lambda x: du.parse(x[0]).timestamp() + x[1]
  ts_0 = du.parse(sw_list[0][0]).timestamp()
  sw = sorted([dict(start=x[0], time=x[1], numobs=x[2], end=end_ts(x)-ts_0) for x in sw_list], key=lambda i: i['end'])
  n = 0
  snum = 0
  nextcalc = step
  while n < max_obs and snum < len(sw):
      n += sw[snum]['numobs']
      if n > nextcalc:
        t = sw[snum]['end'] / 3600.
        # c = bootstrap_sampler(obs[:min(n,N)], samplesize=.25)
        c = op.bootstrap_block(feal_list[:n], step)
        plotlist.append((t, min(np.max(c[1]), 1.)))
        nextcalc += step
      snum += 1
  return plotlist



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











####  HISTOGRAM FOR TOTAL # OBSERVATIONS


def do_histogram():
  global nex, ntr
  ordex = ['Serial', 'Parallel','Uniform','Biased', 'MVNN', 'Reweight']
  binlist = [(a,b) for a in range(5) for b in range(5)]
  hist = {k: {ab : 0 for ab in binlist} for k in ordex}
  obs  = {k: [] for k in ordex}
  total = {k: 0 for k in ordex} 
  for k in ordex:
    print(k)
    for rmslist in nex[k].rmsd_cw.values():
      for i, rms in enumerate(rmslist):
        A, B = np.argsort(rms)[:2]
        delta = np.abs(rms[B] - rms[A])
        if delta < 0.12:
          sub_state = B
        else:
          sub_state = A
        obs[k].append((A, sub_state))
        total[k] += 1

  for k in ordex:
    for o in obs[k]:
      hist[k][o] += 1
  for k in ordex:
    for o in sorted(cnt[k].keys()):
      print(k, o, cnt[k][o])
      hist[k][o] = int(hist[k][o] * 500000 / total[k])

  cnt = {e: {k: 0 for k in ['Well-2', 'Well-3', 'Well-4',
                        'Tran-0', 'Tran-1', 'Tran-2', 'Tran-3', 'Tran-4']} for e in ordex}
  for k in ordex:
    for a in range(5):
      for b in range(5):
        if a == b:
          if a not in [0, 1]:
            cnt[k]['Well-%d'%a] = hist[k][(a,b)]
        else:
          cnt[k]['Tran-%d'%a] += hist[k][(a,b)]

  P.histogram(cnt, 'Total Observations for each state Well / Transitions')




def do_histogram_OLD():
  # obs = {n: c for n,c in db.runquery('select expname, count(obs) from expr e, obs where e.expid=obs.expid group by expname')}
  data = OrderedDict()
  raw = OrderedDict()
  raw['Serial']     = obs2tw(getobs('serial'))
  raw['Parallel']   = obs2tw(getobs('parallel')[:600000])
  raw['Uniform']    = obs2tw(getobs('uniform3')[50000:650000])
  raw['Biased']     = obs2tw(getobs('biased4')[50000:650000])
  raw['RW-Explore'] = obs2tw(getobs('reweight5_epr')[50000:650000])
  raw['RW-Exploit'] = obs2tw(getobs('reweight5_xlt')[50000:650000])
  for key, obslist in raw.items():
    groupby = {}
    for i in obslist:
      if i not in groupby:
        groupby[i] = 0
      groupby[i] += 1
    data[key] = groupby
  P.histogram(data, 'Histogram_Event_Counts', ylim=(0,200000))
  

# Older Historgram
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



### OLDER
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







if __name__=='__main__':
  do_histogram()



# def calc_boot(ex, size, method=None, limit=375000, state=None):
#   print(ex.r.get('name'), '-', end=' ')
#   if method is None:
#     feal = ex.all_feal()[:limit]
#   else:
#     feal = ex.all_feal(True, method)[:limit]
#   i = 0
#   boot = [[] for k in range(5)]
#   ci = [[] for k in range(5)]
#   while i+size < len(feal):
#     # Group By State
#     state = [[] for k in range(5)]
#     for f in feal[i:i+size]:
#       state[np.argmax(f[:5])].append(f[5:])
#     for n, st in enumerate(state):
#       print(i, n, len(st))
#       if len(st) > 0:
#         arr = np.array(st)
#         feal_ci = []
#         for feat in range(15):
#           calc = bootstrap_std(arr.T[feat])
#           feal_ci.append(calc[1])
#           ci[n].append(feal_ci)
#         feal_ci = []
#         for feat in range(15):
#           calc = bootstrap_std([x[feat] for x in ci[n]])
#           feal_ci.append(calc[1])
#         boot[n].append(feal_ci)
#       elif len(boot[n]) > 0:
#         print('NO DATA')
#         boot[n].append(boot[n][-1])
#       else:
#         boot[n].append([1]*15)
#     i += size
#   return boot
