import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse

from collections import deque


HOME = os.environ['HOME']
SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')

TIME_CTL =  3
DATA_PER_SIM_RATE = 33
CTL_BATCH_SIZE = 20000
TIME_LOAD = 0
TIME_ANL = 1
NUM_TIME_STEPS=600   # in mins
NUM_NODES = 30

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


def run(simtime, external_usage, drawgraph=False):
  DB_SIZE = 1
  idlenodes = set([x for x in range(NUM_NODES)])
  runnodes  = set()
  usednodes = set()
  job_queue = deque()
  data_proc = 0
  data_total = 0
  joblist = []
  sim_manager = jobmaker('sim', simtime, int(.25*simtime))
  ctl_manager = jobmaker('ctl', TIME_CTL, 1)   
  catalog_activity = []
  mthread_cost = 0
  sim_cost = 0
  non_sim_overhead = 0

  #Init sim
  print('Initializing simulation.')
  for i in range(40):
    job_queue.append(sim_manager.getjob())

  print('Execting:  %d Time Steps.' % NUM_TIME_STEPS)
  for t in range(NUM_TIME_STEPS):
    catalog_access = False

    # Allocate External Resources
    for i in range(len(idlenodes)):
      if np.random.random() < external_usage:
        n = idlenodes.pop()
        usednodes.add(n)

    # Allocate Resources
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
            for i in range(25):
              job_queue.append(sim_manager.getjob())
            data_proc -= CTL_BATCH_SIZE

    # Check for a controller launch
    if data_proc >= CTL_BATCH_SIZE:
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

  # Compile Stats:
  print('\nCompiling Stats:')
  print('  # Jobs Executed:       %5d' % len(joblist))
  print('  Data Produced:       %7d' % data_total)
  print('  WallClock Time:        %5d' % NUM_TIME_STEPS)
  print('  Avg Sim Time:          %5d' % simtime)
  print('  TOTAL Resource COST:   %5d' % (total_cost))
  print('  Cost sims only:        %5d  (%4.1f%% of cost)' % (sim_cost, 100*sim_cost/total_cost))
  print('  Cost non-sim mthread:  %5d  (%4.1f%% of cost)' % (non_sim_overhead, 100*non_sim_overhead/total_cost))
  print('  Catalog Active:        %5d  (%4.1f%%)' % (n_active, 100*n_active/cat_cost))
  print('  Catalog Idle :         %5d  (%4.1f%%)' % (n_idle, 100*n_idle/cat_cost))
  print('  Catalog Cost:          %4.1f%%' % (100*cat_cost/(total_cost+cat_cost)))
  print('  Catalog Waste:         %4.1f%%' % (100*n_idle/(total_cost+cat_cost)))

  total_overhead = cat_cost + non_sim_overhead
  print('  TOTAL OVERHEAD COST:   %5d  (%4.2f%%)  <-- Catalog Cost + Non-Sim Cost' % \
   (total_overhead, 100*total_overhead/total_cost))

  # Make the bars:
  title = 'sim_test'
  if drawgraph:
    print('Visualizing simulation to: ', title)
    simlist = [(j['node'],j['start'],j['end']) for j in joblist if j['type']=='sim']
    ctllist = [(j['node'],j['start'],j['end']) for j in joblist if j['type']=='ctl']
    drawjobs(simlist, ctllist, catalog_activity, title, 1)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('simtime', type=int)
  parser.add_argument('-u', '--usage', type=float, default=.5)
  args = parser.parse_args()
  run(args.simtime, args.usage)   