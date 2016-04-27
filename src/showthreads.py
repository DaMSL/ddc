import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from core.slurm import slurm

import dateutil.parser as dparse


HOME = os.environ['HOME']
SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')


def splicename(fname):
  if re.match(r'\D[m,w]-\d{4}.{2}', fname):
    jtype, jnum = fname.split('-')
    m, w, _ = jnum.split('.')
    return jtype + '%06d' % (int(m)*100 + int(w))
  else:
    return None



def getjoblist(sourcedir='/home-1/bring4@jhu.edu/work/log/biased2_base/sim'):
  ls = os.listdir(sourcedir)
  ls = [i for i in ls if (i.startswith('sw') or i.startswith('cw'))]
  jobidlist = []
  joblist = []
  type_map = {}
  for logfile in sorted(ls):
    mt_type = 'sim' if logfile.startswith('sw') else 'ctl'
    with open(sourcedir + '/' + logfile) as src:
      log = src.read().split('\n')
      for l in log:
        if 'SLURM_JOB_ID' in l:
          _, jobid = l.split(':')
          jobid = int(jobid.strip())
          jobidlist.append(jobid)
          type_map[jobid] = mt_type
          break
  job_data = slurm.jobexecinfo(jobidlist)
  for i in range(len(job_data)):
    job_data[i]['type'] = type_map[job_data[i]['jobid']]
  return job_data


def makenodelist(joblist):
  nodes = set()
  for job in joblist:
    nodes.add(job['node'])
  return {n: i for i, n in enumerate(sorted(list(nodes)))}

def makebars(joblist):
  nodelist = makenodelist(joblist)
  firstjob = min(joblist, key=lambda x:x['start'])
  begin_ts = dparse.parse(firstjob['start']).timestamp()
  timetosec = lambda h,m,s: int(h)*3600 + int(m)*60 + int(s)
  timetomin = lambda h,m,s: int(h)*60 + int(m) + round(int(s)/60)
  barlist = []
  for job in joblist:
    Y = nodelist[job['node']]
    start_time = dparse.parse(job['start']).timestamp()
    X0 = int((start_time - begin_ts) // 60)
    X1 = int(X0 + timetomin(*job['time'].split(':')))
    if job['type'] == 'ctl':
      barlist.append((Y, X0, X1, 'g'))      
    else:
      barlist.append((Y, X0, X0+1, 'red'))
      barlist.append((Y, X0+1, X1-2, 'blue'))
      barlist.append((Y, X1-2, X1, 'red'))
  return barlist


def drawjobs(barlist, title, sizefactor=3):
  # Find IDle Time
  TIME_LOAD = 1
  TIME_ANL = 2
  activecol = {True: 'red', False:'black'}
  end_ts = max(barlist, key=lambda x: x[2])[2]
  maxnode = max(barlist, key=lambda x: x[0])[0] + 10
  catalog = [0 for i in range(int(end_ts))]
  parallelism = [0 for i in range(int(end_ts))]
  catalog[0] = 1
  for Y, X0, X1, col in barlist:
    catalog[int(X0)] += 1
    for i in range(int(X1)-TIME_ANL, int(X1)):
      catalog[i] += 1
    for i in range(int(X0), int(X1)):
      parallelism[i] += 1
  catbars = []
  lastactive = True
  c0 = 0
  maxparallel = max(catalog)
  for ts, numconn in enumerate(catalog):
    active = (numconn > 0)
    if not active:
      usage = (0., 0., 0.)
    else:
      usage = (min(numconn/maxparallel + .15, 1.), .15, .15)
    catbars.append((maxnode+10, ts, ts+1, usage))
    if active != lastactive:
      catbars.append((maxnode, c0, ts, activecol[lastactive]))
      c0 = ts
      lastactive = not lastactive

  for ts, p in enumerate(parallelism):
    if p == 0:
      usage = (0., 0., 0.)
    else:
      usage = (0., 0., min(p/50., 1.))
    catbars.append((maxnode+20, ts, ts+1, usage))    

  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  # fig = plt.gcf()
  fig, ax = plt.subplots()
  fig.set_dpi(300)
  fig.set_size_inches(6*sizefactor, 4*sizefactor)
  for Y, X0, X1, col in barlist:
    linewdth = 4 if col == 'g' else 2
    plt.hlines(Y, X0, X1, color=col, lw=linewdth)
    # plt.hlines(Y, X0, X0+TIME_LOAD, color='brown', lw=2)
    # plt.hlines(Y, X0+TIME_LOAD, X1-TIME_ANL, color='blue', lw=2)
    # plt.hlines(Y, X1-TIME_ANL, X1, color='red', lw=2)

  try:
    for Y, X0, X1, col in catbars:
      plt.hlines(Y, X0, X1, color=col, lw=10)
  except ValueError as e:
    print(Y, X0, X1, col)
    print(e)
  plt.xlabel("Total Wall Clock Time (in Minutes)")
  plt.ylabel("Node Number w/CATALOG node at the top")

  ax.annotate('Sims running in parallel (Brighter blue => more parallel)', xy=(10, maxnode+22), fontsize=8)
  ax.annotate('Catalog Usage (Brighter red => more used)', xy=(10, maxnode+12), fontsize=8)
  ax.annotate('Catalog Usage (Binary Only)', xy=(10, maxnode+2), fontsize=8)


  # Custom Legend:
  labels = {'g': 'Controller', 'r': 'Catalog_ACTIVE', 'black':'Catalog_IDLE', 'blue':'Simulation', 'red':'Catalog I/O'}

  patches = [mpatches.Patch(color=k, label=v) for k, v in labels.items()]

  ax.set_xlim(0, end_ts)
  ax.set_ylim(0, maxnode+25)

  plt.legend(handles=patches, loc='center right')  
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()  




##### GANTT CHART DISPLAY
def gantt():
  # print("ROWS:", rows)
  # print("COLS:", cols)
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  nodes = []
  X = np.random.random(10)-0.5
  Y = np.arange(10)
  plt.hlines(Y, 0, X, color='blue', lw=5)
  # fig, ax = plt.subplots()
  # # put the major ticks at the middle of each cell
  # ax.set_xticks(np.arange(len(rows))+.5)
  # ax.set_yticks(np.arange(len(cols))+.5)
  plt.savefig(SAVELOC + '/gantt.png')
  plt.close()  


def run():
  NUM_NODES = 5
  nodes = [[] for i in range(NUM_NODES)]

  TIME_SIM = 10
  TIME_LOAD = 1
  TIME_ANL = 2
  NUM_TIME_STEPS=50
  DB_SIZE = 1

  idlenodes = set([1,2,3,4,5])
  runnodes  = set([6,7,8,9])
  job_queue = 20
  joblist = []

  for t in range(NUM_TIME_STEPS):
    # Allocate Resources
    while len(idlenodes) > 0 and job_queue > 0:
      nextnode = idlenodes.pop()
      simtime = TIME_SIM + np.random.randint(10)-5
      job = dict(start=t, end=t+simtime,
          db=DB_SIZE, node=nextnode, name='sim', active=True)
      joblist.append(job)
      job_queue -= 1
    # Check finished jobs
    for job in joblist:
      if job['active'] and job['end'] == t:
        job['active'] = False
        idlenodes.add(job['node'])
      # stochastic change an external running node is freed or 

  fig, ax = plt.subplots()
  nodelist = set()
  for job in joblist:
    Y, X0, X1 = job['node'], job['start'], job['end']
    nodelist.add(Y)
    plt.hlines(Y, X0, X0+TIME_LOAD, color='brown', lw=5)
    plt.hlines(Y, X0+TIME_LOAD, X1-TIME_ANL, color='blue', lw=5)
    plt.hlines(Y, X1-TIME_ANL, X1, color='red', lw=5)

  ax.set_yticks(np.arange(max(nodelist)))
  plt.savefig(SAVELOC + '/gantt.png')
  plt.close()  


if __name__=='__main__':
  run()   