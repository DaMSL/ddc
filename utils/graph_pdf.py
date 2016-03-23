import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import numpy as np
import redis
import math


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


HOST = 'bigmem0043'
SAVELOC = os.environ['HOME'] + '/ddc/graph/'

r = redis.StrictRedis(host=HOST, decode_responses=True)
obs = r.lrange('label:raw:lg', 0, -1)
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

def gen_bootstraps(all_obs, strap_size_ns, transonly=False, cumulative=False):
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

def plot_bootstraps(data, ts, prefix):
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
      plt.plot(np.arange(len(bs_sd[b]))*(ts), (bs_ci[b]/bs_mn[b]), label=str(b))
    plt.xlabel('Total Convergence (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/%s_%dns_tc_%d.png'%(prefix, ts, i))
    plt.close()
    for b in ab[i*5:i*5+5]:
      plt.errorbar(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], yerr=bs_er[b], label=str(b))
    plt.xlabel('Mean with error bars (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/%s_%dns_mnerr_%d.png'%(prefix, ts, i))
    plt.close()
  print('All Done!')
  return stat

stats=recheckStats_separate(50)



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








pipe = r.pipeline()
for j in jobs:
  pipe.hgetall('jc_' + j)

jobdata=pipe.execute()

groupbystart = {b:[] for b in ab}
for j in jobdata:
  if 'xid:start' not in j:
    continue
  try:
    for i in range(int(j['xid:start']), int(j['xid:end'])+1):
      groupbystart[eval(j['src_bin'])].append(obs[i])
  except IndexError as e:
    print('NoIdx', i)


def byorigin(srcbin, ts, fromstate=True):
  if fromstate:
    sA = sB = srcbin
  else:
    sA, sB = srcbin
  SAVELOC = os.environ['HOME'] + '/ddc/graph/trans/'
  TIMESTEP = ts * OBS_PER_NS
  stat = {b: [] for b in ab}
  total_obs = {b: 0 for b in ab}
  cnts = {b: 0 for b in ab}
  total = 0
  i = 0
  if fromstate:
    all_obs = []
    for b in range(5):
      all_obs.extend(groupbystart[srcbin, b])
  else:
    all_obs = groupbystart[srcbin]
  print('Collecting data for %d Observations for %s...' % (len(all_obs), str(srcbin)))
  exclude = set()
  while i < len(all_obs):
    if i > 0 and i % TIMESTEP == 0:
      for b in ab:
        stat[b].append(cnts[b]/total)
      # RESET COUNT FOR SNAPSHOT
      cnts={b: 0 for b in ab}
      total = 0
    A, B = eval(all_obs[i])
    i += 1
    if A == B:
      continue    
    cnts[(A, B)] += 1
    total_obs[(A, B)] += 1
    total += 1
  if total > 0:
    for b in ab:
      stat[b].append(cnts[b]/total)
  for b in ab:
    if total_obs[b] == 0:
      print('Transition from %s to %s NOT Observered' % (srcbin, b))
      exclude.add(b)
    if total_obs[b] < 50:
      print('Transition from %s to %s RARELY Observered #times= %d' % (srcbin, b, total_obs[b]))
      exclude.add(b)
  print('Calculating Convergence...')
  # bs = {b: bootstrap_std(stat[b]) for b in ab}
  bs = {b: [bootstrap_std(stat[b][:i], interval=.9) for i in range(len(stat[b]))]  for b in ab}
  bs_mn = {b: np.array([x[0] for x in bs[b]]) for b in ab}
  bs_ci = {b: np.array([x[1] for x in bs[b]]) for b in ab}
  bs_sd = {b: np.array([x[2] for x in bs[b]]) for b in ab}
  bs_er = {b: np.array([x[3] for x in bs[b]]) for b in ab}
  print('Generating graphs for source bin ', str(srcbin))
  for b in ab:
    if b not in exclude:
      plt.plot(np.arange(len(bs_ci[b]))*(TIMESTEP/1000), bs_ci[b], label=str(b))
  plt.xlabel('Conf Interval for State %s (total time in ns)' % str(srcbin))
  plt.legend()
  plt.savefig(SAVELOC + '/trans_%dns_ci_%d_%d.png'%(ts, sA, sB))
  plt.close()
  for b in ab:
    if b not in exclude:
      plt.plot(np.arange(len(bs_sd[b]))*(ts), (bs_ci[b]/bs_mn[b]), label=str(b))
  plt.xlabel('Total Convergence of PDF for state %s (total time in ns)' % (str(srcbin)))
  plt.legend()
  plt.savefig(SAVELOC + '/trans_%dns_tc_%d_%d.png'%(ts, sA, sB))
  plt.close()
  for b in ab:
    if b not in exclude:
      plt.errorbar(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], yerr=bs_er[b], label=str(b))
  plt.xlabel('Mean PDF of Transitions out of state %s with error (total time in ns)' % str(srcbin))
  plt.legend()
  plt.savefig(SAVELOC + '/trans_%dns_mnerr_%d_%d.png'%(ts, sA, sB))
  plt.close()
  print('All Done!')
  return bs

for a in range(5):
  z=byorigin(a, 5, True)


for a in range(5):
  for b in range(5):
    z=byorigin((a, b), 5, False)


  while i < len(groupbystart[srcbin]):
    if i > 0 and i % TIMESTEP == 0:
      for b in ab:
        stat[b].append(cnts[b]/total)
      # RESET COUNT FOR SNAPSHOT
      if not cumulative:
        cnts={b: 0 for b in ab}
        total = 0
    A, B = eval(groupbystart[srcbin][i])
    cnts[(A, B)] += 1
    i += 1
    total += 1
  if total > 0:
    for b in ab:
      stat[b].append(cnts[b]/total)
  print('Calculating Convergence...')
  bs = {b: [] for b in ab}
  bs_all = {b: [] for b in ab}
  for b in ab:
    bs_all[b] = [bootstrap_std(stat[b][:i], interval=.9) for i in range(10, len(cuml[b]))]
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
    plt.savefig(SAVELOC + '/conv_%s_%dns_ci_%d.png'%(title_pre[:5], ts, i))
    plt.close()
    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], label=str(b))
    # plt.xlabel(title_pre + 'Mean (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/conv_%s_%dns_mn_%d.png'%(title_pre[:5], ts, i))
    # plt.close()
    # for b in ab[i*5:i*5+5]:
    #   plt.plot(np.arange(len(bs_sd[b]))*(ts), bs_sd[b], label=str(b))
    # plt.xlabel(title_pre +'StdDev (total time in ns)')
    # plt.legend()
    # plt.savefig(SAVELOC + '/conv_%s_%dns_std_%d.png'%(title_pre[:5], ts, i))
    # plt.close()
    for b in ab[i*5:i*5+5]:
      plt.plot(np.arange(len(bs_sd[b]))*(ts), (bs_ci[b]/bs_mn[b]), label=str(b))
    plt.xlabel(title_pre +'Total Convergence (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/conv_%s_%dns_tc_%d.png'%(title_pre[:5], ts, i))
    plt.close()
    for b in ab[i*5:i*5+5]:
      plt.errorbar(np.arange(len(bs_mn[b]))*(ts), bs_mn[b], yerr=bs_er[b], label=str(b))
    plt.xlabel(title_pre +'Mean with error bars (total time in ns)')
    plt.legend()
    plt.savefig(SAVELOC + '/conv_%s_%dns_mnerr_%d.png'%(title_pre[:5], ts, i))
    plt.close()
  print('All Done!')
  return cuml












bk = r.keys('boot*')
boot = OrderedDict()
for b in sorted(bk):
  boot[b] = np.array(float(x) for x in r.lrange(b, 0, -1))

final_pdf = {b: bootstrap_simple(boot[b]) for b in bk}
cum_pdf = OrderedDict()
for b in sorted(bk):
  cum_pdf[b] = []
  for i in range(1, len(boot[b])):
    cum_pdf[b].append(bootstrap_simple(boot[b][:i]))

cp = {i: {} for i in range(5)}
for k, v in cum_pdf.items():
  cp[int(k[5])][k[5:]] = v

cpa = {state: {k: np.array([x[3] for x in v]) for k, v in cp[state].items()} for state in cp.keys()}
cpb = {state: {k: np.array([x[2]-x[1] for x in v]) for k, v in cp[state].items()} for state in cp.keys()}

for b in sorted(bk):
  for 


