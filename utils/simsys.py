import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import argparse

from collections import deque


HOME = os.environ['HOME']
SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')

TIME_CTL =  3
DATA_PER_SIM_RATE = 30
TIME_LOAD = 0
TIME_ANL = 1
NUM_TIME_STEPS=600   # in mins
NUM_NODES = 100

UNIT_COST_CATALOG = 1
UNIT_COST_MTHREAD = 1



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


class jobmaker:
  def __init__(self, name, runtime, runvar):
    self.id = 0
    self.name=name
    self.runtime = runtime
    self.runvar = runvar
  def getjob(self):
    job = {'jobid': '%04d'%self.id, 
           'time' : np.round(np.random.normal(self.runtime, self.runvar)), 
           'type' : self.name}
    self.id += 1
    return job


def makebars(joblist):
  # firstjob = min(joblist, key=lambda x:x['start'])
  # begin_ts = dparse.parse(firstjob['start']).timestamp()
  # timetosec = lambda h,m,s: int(h)*3600 + int(m)*60 + int(s)
  # timetomin = lambda h,m,s: int(h)*60 + int(m) + round(int(s)/60)
  barlist = []
  for job in joblist:
    Y = job['node']
    X0 = int((start_time - begin_ts) // 60)
    X1 = int(X0 + timetomin(*job['time'].split(':')))
    barlist.append((Y, X0, X1))
  return barlist


def drawjobs(sims, ctls, catalog_activity, title, sizefactor=3):
  # Find IDle Time
  activecol = {True: 'green', False:'black'}
  end_ts = NUM_TIME_STEPS
  maxnode = max(sims, key=lambda x: x[0])[0] + 10

  catbars = []
  lastactive = True
  c0 = 0
  for ts, active in enumerate(catalog_activity):
    if active != lastactive:
      catbars.append((maxnode, c0, ts, lastactive))
      c0 = ts
      lastactive = not lastactive
  catbars.append((maxnode, c0, ts, lastactive))

  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  # fig = plt.gcf()
  fig, ax = plt.subplots()
  fig.set_dpi(300)
  fig.set_size_inches(6*sizefactor, 4*sizefactor)
  for Y, X0, X1 in sims:
    plt.hlines(Y, X0, X0+TIME_LOAD, color='brown', lw=2)
    plt.hlines(Y, X0+TIME_LOAD, X1-TIME_ANL, color='blue', lw=2)
    plt.hlines(Y, X1-TIME_ANL, X1, color='red', lw=2)

  for Y, X0, X1 in ctls:
    plt.hlines(Y, X0, X1, color='brown', lw=2)


  for Y, X0, X1, la in catbars:
    plt.hlines(Y, X0, X1, color=activecol[la], lw=12)
  plt.xlabel("Total Wall Clock Time (in Minutes)")
  plt.ylabel("Node Number w/CATALOG node at the top")

  # Custom Legend:
  labels = {'green': 'Catalog_ACTIVE', 'black':'Catalog_IDLE', 'blue':'Simulation', 'red':'Catalog I/O'}

  patches = [mpatches.Patch(color=k, label=v) for k, v in labels.items()]

  ax.set_xlim(0, end_ts)
  ax.set_ylim(0, maxnode+12)

  plt.legend(handles=patches, loc='center right')  
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()  


def run(simtime, external_usage, resamp_rate, show=False, drawgraph=False):
  DB_SIZE = 1
  resample_batch = resamp_rate * DATA_PER_SIM_RATE * simtime
  idlenodes = set([x for x in range(NUM_NODES)])
  runnodes  = set()
  usednodes = set()
  job_queue = deque()
  data_proc = 0
  data_total = 0
  joblist = []
  sim_manager = jobmaker('sim', simtime, .4*simtime)
  ctl_manager = jobmaker('ctl', TIME_CTL, 1)   
  catalog_activity = []
  mthread_cost = 0
  sim_cost = 0
  non_sim_overhead = 0

  #Init sim
  # print('Initializing simulation.')
  for i in range(int(resamp_rate * 1.2)):
    job_queue.append(sim_manager.getjob())

  # print('Executing:  %d Time Steps. TSim=%d min,  ResampRate=%d obs' % 
  #   (NUM_TIME_STEPS, simtime, resample_batch))
  for t in range(NUM_TIME_STEPS):
    catalog_access = False

    # Allocate External Resources
    for i in range(len(idlenodes)):
      if np.random.random() < external_usage:
        n = idlenodes.pop()
        usednodes.add(n)

    # Allocate Jobs -> Resources
    while len(idlenodes) > 0 and len(job_queue) > 0:
      nextnode = idlenodes.pop()
      nextjob  = job_queue.pop()
      nextjob['start'] = t
      nextjob['end'] = t + nextjob['time']
      nextjob['active'] = True
      nextjob['state'] = 'load'
      nextjob['node'] = nextnode
      joblist.append(nextjob)
      runnodes.add(nextnode)
      catalog_access = True

    # Check active jobs and catalog activity
    for job in joblist:
      if job['active']:
        state = job['state']
        if state == 'load' and job['start'] + TIME_LOAD == t: 
          job['state'] = 'execsim' if job['type'] =='sim' else 'execctl'
        elif state == 'execsim' and job['end'] - TIME_ANL == t:
          job['state'] = 'execanl'
        elif state == 'execsim':
          sim_cost += UNIT_COST_MTHREAD
        if job['state'] in ['load', 'execctl', 'execanl']:
          catalog_access = True
          non_sim_overhead += UNIT_COST_MTHREAD
        if job['end'] == t:
          job['active'] = False
          job['state'] = 'complete'
          runnodes.remove(job['node'])
          idlenodes.add(job['node'])
          if job['type'] == 'sim':
            data_proc += simtime * DATA_PER_SIM_RATE
            data_total += simtime * DATA_PER_SIM_RATE
          else:
            for i in range(resamp_rate):
              job_queue.append(sim_manager.getjob())
            data_proc -= resample_batch

    # Check for a controller launch
    if data_proc >= resample_batch:
      job_queue.append(ctl_manager.getjob())

    # TODO: stochastic change an external running nodes
    for i in range(len(usednodes)):
      if np.random.random() < (1-external_usage):
        n = usednodes.pop()
        idlenodes.add(n)

    catalog_activity.append(catalog_access)
    mthread_cost += len(runnodes) * UNIT_COST_MTHREAD

  n_idle, n_active = tuple(np.bincount(catalog_activity))
  cat_cost = len(catalog_activity) * UNIT_COST_CATALOG
  total_cost = mthread_cost + cat_cost
  total_overhead = cat_cost + non_sim_overhead

  # Compile Stats:
  if show:
    print('\nCompiling Stats:')
    print('  # Jobs Executed:       %5d' % len(joblist))
    print('  Data Produced:       %7d  (# Obs)' % data_total)
    print('  Resampled every:     %7d  (Obs)' % resample_batch)
    print('  WallClock Time:        %5d  (min, fixed)' % NUM_TIME_STEPS)
    print('  Avg Sim Time:          %5d  (min)' % simtime)
    print('  TOTAL Resource COST:   %5d' % (total_cost))
    print('  Cost sims only:        %5d  (%4.1f%% of cost)' % (sim_cost, 100*sim_cost/total_cost))
    print('  Cost non-sim mthread:  %5d  (%4.1f%% of cost)' % (non_sim_overhead, 100*non_sim_overhead/total_cost))
    print('  Catalog Active:        %5d  min (%4.1f%%)' % (n_active, 100*n_active/cat_cost))
    print('  Catalog Idle :         %5d  min (%4.1f%%)' % (n_idle, 100*n_idle/cat_cost))
    print('  Catalog Cost:          %4.1f%%' % (100*cat_cost/(total_cost)))
    print('  Catalog Waste:         %4.1f%%' % (100*n_idle/(total_cost)))
    print('  TOTAL OVERHEAD COST:   %5d  (%4.2f%%)  <-- Catalog Cost + Non-Sim Cost' % \
     (total_overhead, 100*total_overhead/total_cost))

  # Make the bars:
  if drawgraph:
    title = 'sim_test'
    print('Visualizing simulation to: ', title)
    simlist = [(j['node'],j['start'],j['end']) for j in joblist if j['type']=='sim']
    ctllist = [(j['node'],j['start'],j['end']) for j in joblist if j['type']=='ctl']
    drawjobs(simlist, ctllist, catalog_activity, title, 1)

  return dict(njobs=len(joblist), ndata=data_total, 
    tcost=total_cost, scost=sim_cost, nscost=non_sim_overhead, overhead=total_overhead)



def runmany(simtime, external_usage, resamp_rate, N=10):
  print('Executing:  %d Time Steps. TSim=%d min, Usage=%3.1f  ResampRate=%d obs' % 
    (NUM_TIME_STEPS, simtime, external_usage, resamp_rate))
  data = dict(njobs=[], ndata=[],tcost=[], scost=[], nscost=[], overhead=[])
  for i in range(N):
    output = run(simtime, external_usage, resamp_rate)
    for k, v in output.items():
      data[k].append(v)
  return {k: (np.mean(v), np.std(v)) for k, v in data.items()}


def simusage(resamp=20):
  plt.cla()
  plt.clf()
  rsamp    = (10, 20, 25, 30, 50)
  markers = ['^', 'd', 'o', '*', 's']
  sim_vals = (5, 150, 2)
  use    = [.7, .9, .95]
  colors = ['r', 'g', 'b']
  labels = ['Light External Usage', 'Moderate External Usage', 'Heavy External Usage']
  X = [i for i in range(*sim_vals)]
  Ytime = {}
  Ycost = {}
  for r in rsamp:
    Ytime[r] = {}
    Ycost[r] = {}
    for u, c, l in zip(use, colors, labels):
      Ytime[r][l] = []
      Ycost[r][l] = []
      for i in range(*sim_vals):
        d = runmany(i, u, r, N=3)
        Ytime[r][l].append(d['ndata'][0]/NUM_TIME_STEPS)
        Ycost[r][l].append(d['ndata'][0]/d['tcost'][0])
  patches = [mpatches.Patch(color=i[0], label=i[1]) for i in zip(colors, labels)]
  patches.extend([mlines.Line2D([],[],color='k', marker=i[0], label='Resample: %d Sims'%i[1]) \
    for i in zip(markers, rsamp)])

  for r, m in zip(rsamp, markers):
    for l, c in zip(labels, colors):
      plt.scatter(X, Ytime[r][l], c=c, marker=m, lw=0)
      for i in range(1, len(X)):
        plt.plot((X[i-1], X[i]), (Ytime[r][l][i-1], Ytime[r][l][i]), c=c)
      #   fit = np.polyfit(X[i-1:i+1],Ytime[r][l][i-1:i+1],1)
      #   y_fn = np.poly1d(fit)
      #   y = (y_fn(X[i-1]), y_fn(X[i+1]))
      #   x = (X[i-1], X[i+1])
      #   plt.plot(x,y, c=c)
  plt.ylabel('Data Observations Generated Per Wall-Clock Minute (more is better)')
  plt.xlabel('Single Simulation Length')
  plt.xlim(0, 180)
  plt.ylim(0, 1000)
  plt.title('Effectiveness:  Avg Data Generated Per Minute')
  plt.legend(handles=patches, loc='lower right')  
  plt.savefig(SAVELOC + '/effectiveness.png')
  plt.close()

  for r, m in zip(rsamp, markers):
    for l, c in zip(labels, colors):
      plt.scatter(X, Ycost[r][l], c=c, marker=m, lw=0)
      for i in range(1, len(X)):
        plt.plot((X[i-1], X[i]), (Ycost[r][l][i-1], Ycost[r][l][i]), c=c)
      # for i in range(1, len(X)-1, 2):
      #   fit = np.polyfit(X[i-1:i+1],Ycost[r][l][i-1:i+1],1)
      #   y_fn = np.poly1d(fit)
      #   y = (y_fn(X[i-1]), y_fn(X[i+1]))
      #   x = (X[i-1], X[i+1])
      #   plt.plot(x,y, c=c)  
  plt.ylabel('Data Observations Generated Per Unit_Cost (more is better)')
  plt.xlabel('Single Simulation Length')
  plt.xlim(0, 180)
  plt.ylim(0, 35)
  plt.title('Efficiency:  Amt of Data Generated Per Unit Cost')
  plt.legend(handles=patches, loc='lower right')  
  plt.savefig(SAVELOC + '/efficiency.png')
  plt.close()



def sim_byrsamp(resamp=20):
  plt.cla()
  plt.clf()
  rsamp  = (10, 20, 25, 30, 50)
  sim_vals = (5, 150, 2)
  # use    = [.5, .85, .95]
  use = .85
  colors = ['r', 'g', 'b', 'c', 'm']
  X = [i for i in range(*sim_vals)]
  Ytime = {}
  Ycost = {}
  for r in rsamp:
    Ytime[r] = []
    Ycost[r] = []
    for i in range(*sim_vals):
      d = runmany(i, use, r, N=3)
      Ytime[r].append(d['ndata'][0]/NUM_TIME_STEPS)
      Ycost[r].append(d['ndata'][0]/d['tcost'][0])
  patches = [mpatches.Patch(color=i[0], label='Resample Rate: %d Sims'%i[1]) 
    for i in zip(colors, rsamp)]

  for r, c in zip(rsamp, colors):
    plt.plot(X, Ytime[r], c=c)
    # for i in range(1, len(X)):
    #   plt.plot((X[i-1], X[i]), (Ytime[r][i-1], Ytime[r][i]), c=c)
  plt.ylabel('Data Observations Generated Per Wall-Clock Minute (more is better)')
  plt.xlabel('Single Simulation Length')
  plt.xlim(0, 180)
  plt.ylim(0, 1000)
  plt.title('Effectiveness:  Avg Data Generated Per Minute')
  plt.legend(handles=patches, loc='lower right')  
  plt.savefig(SAVELOC + '/effectiveness_fixedusage.png')
  plt.close()

  for r, c in zip(rsamp, colors):
    plt.plot(X, Ycost[r], c=c)
    # for i in range(1, len(X)):
    #   plt.plot((X[i-1], X[i]), (Ycost[r][i-1], Ycost[r][i]), c=c)
  plt.ylabel('Data Observations Generated Per Unit_Cost (more is better)')
  plt.xlabel('Single Simulation Length')
  plt.xlim(0, 180)
  plt.ylim(0, 35)
  plt.title('Efficiency:  Amt of Data Generated Per Unit Cost')
  plt.legend(handles=patches, loc='lower right')  
  plt.savefig(SAVELOC + '/efficiency_fixedusage.png')
  plt.close()





if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('simtime', type=int)
  parser.add_argument('-u', '--usage', type=float, default=.5)
  parser.add_argument('-r', '--resamp', type=int, default=20)
  args = parser.parse_args()
  run(args.simtime, args.usage, args.resamp)   