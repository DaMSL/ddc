import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from collections import OrderedDict
import numpy as np

HOME = os.environ['HOME']
SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')


####  SCATTER PLOTS

def scatter(X, Y, title, size=1, L=None):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  if L is None:
    plt.scatter(X, Y, s=size, lw=0)
  else:
    plt.scatter(X, Y, c=L, s=size, lw=0)
  plt.ylabel(title)
  # plt.legend()
  plt.savefig(loc + '/' + title + '.png')
  plt.close()

def scatter3D(X, Y, Z, title, L=None):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  fig = plt.figure()
  ax = Axes3D(fig)
  if L is None:
    ax.scatter(X, Y, Z, lw=0)
  else:
    ax.scatter(X, Y, Z, c=L, lw=0)
  plt.ylabel(title)
  # plt.legend()
  plt.savefig(loc + '/' + title + '_3d.png')
  plt.close() 


#####   LINE Graph

def line(X, title):
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  if isinstance(X, dict):
    for k, v in X.items():
      print('Plotting: ', k)
      plt.plot(np.arange(len(v)), v, label=k)
  elif isinstance(X, np.ndarray):
    plt.plot(np.arange(len(X)), X)
  else:
    print("Not Implemented for:", str(type(X)))
    return
  plt.xlabel(title)
  plt.legend()
  plt.savefig(loc + '/' + title + '.png')
  plt.close()  

def linegraphcsv(X, title, nolabel=False):
  """ Plots line series for a csv list
  """



###### BAR PLOTS
def bargraph(data, title, label=None):
  colors = ['r','g','b','m','k']
  plt.cla()
  plt.clf()
  nbars = len(data[0])
  # bw = 1. / nbars
  bw = 1
  fig, ax = plt.subplots()
  X = np.arange(nbars)
  # for i in range(len(data)):
  #   if label is None:
  #     plt.bar(X, data[i], bw)
  #   else:
  #     plt.bar(X, data[i], bw, 
  #       color=colors[i], label=label[i])
  plt.bar(X, data[0], .4, color=colors[0], label=label[0])
  plt.bar(X+.4, data[1], .4, color=colors[1], label=label[1])

  ax.set_xlim(0, nbars)
  ax.set_ylim(0, 1.)
  plt.legend()
  fig.suptitle('Sampling Distribution for bin (%s): Biased vs Reweight' % title)
  fig.set_size_inches(16,6)
  plt.tight_layout()
  plt.savefig(SAVELOC + '/bar_' + title + '.png')
  plt.close()  
  plt.show()


def bargraph_simple(data, title):
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  plt.bar(np.arange(len(data)), data)
  plt.legend()
  plt.tight_layout()
  plt.savefig(SAVELOC + '/bar_' + title + '.png')
  plt.close()  
  plt.show()



#### BIPARTITE GRAPH

def addconnection(i,j,c):
  return [((-1,1),(i-1,j-1),c)]

def drawnodes(ax, s, left=True):
  if left:
    color='b'
    posx=-1
  else:
    color='r'
    posx=1
  posy=0
  for n in s:
    plt.gca().add_patch( plt.Circle((posx,posy),radius=0.05,fc=color))
    if posx==1:
      ax.annotate(n,xy=(posx,posy+0.1))
    else:
      ax.annotate(n,xy=(posx-len(n)*0.1,posy+0.1))
    posy+=1

def bipartite(nodeA, nodeB, edges, sizeA=None, sizeB=None, title='bipartite'):
  nodesizes = [.01, .03, .1, .25, .6, 1]
  if sizeA is None:
    zA = [.2]*len(nodeA)
  else:
    zA = [nodesizes[len(str(round(i)))] for i in sizeA]
  if sizeB is None:
    zB = [.2]*len(nodeB)
  else:
    zB = [nodesizes[len(str(round(i)))] for i in sizeB]

  print('Size Lists:')
  print('  ', zA)
  print('  ', zB)
  print('Edges: ', len(edges), edges)
  pad = 3
  H = max(len(nodeA),len(nodeB))

  ax=plt.figure().add_subplot(111)
  plt.axis([0,H,0,H])
  frame=plt.gca()
  frame.axes.get_xaxis().set_ticks([])
  frame.axes.get_yaxis().set_ticks([])


  # Left Nodes
  stepA = H / len(nodeA)
  posx = pad
  posy = (stepA/2)
  for n, z in zip(nodeA, zA):
    plt.gca().add_patch(plt.Circle((posx,posy),radius=z,fc='blue'))
    ax.annotate(n,xy=(posx-1,posy-.1), horizontalalignment='right')
    posy+=stepA

  # Right Nodes
  stepB = H / len(nodeB)
  posx = H-pad
  posy = (stepB/2)
  for n, z in zip(nodeB, zB):
    plt.gca().add_patch(plt.Circle((posx,posy),radius=z,fc='red'))
    ax.annotate(n,xy=(posx+1,posy-.1), horizontalalignment='left')
    posy+=stepB

  # Edges
  maxz = max([i[2] for i in edges])
  sumz = sum([i[2] for i in edges])
  for a, b, z in edges:
    W = 1
    S = ':'
    if z > 10:
      S = '--'
    if z > 50:
      S = '-'
    if z > sumz/len(edges):
      W = 2
    if z > 150:
      W = 3
    if z > 500:
      W = 4
    if z > 1000:
      W = 4 + z//1000
    plt.plot((pad,H-pad),(a*stepA+(stepA/2), b*stepB+(stepB/2)),'k', linestyle=S, linewidth=W)


  ax.annotate('GLOBAL (B)', xy=(H-4, 0), horizontalalignment='left')
  ax.annotate('LOCAL (A)', xy=(4, 0), horizontalalignment='right')
  ax.arrow(H-4, .5, -(H-8), 0, head_width=0.1, head_length=0.5, fc='k', ec='k')
  plt.ylabel("Numbers are HCube Sizes")
  plt.xlabel("Line style/width => # of projected Pts (Note: not all lines drawn)")
  plt.title(title)
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.savefig(loc + '/' + title + '.png')
  plt.close()
 
# elist = []
# for i in range(30):
#   a = np.random.randint(len(A))
#   b = np.random.randint(3)
#   c = np.random.randint(200)
#   elist.append((a, b, c))



###### HEAT Map
cdict = {
         'red':   ((0.0,  1.0, 1.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.25, 1.0, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0))}
mycmap = LinearSegmentedColormap('mycmap', cdict)
plt.register_cmap(cmap=mycmap)

def heatmap(data, rows, cols, title, vmax=None):
  # print("ROWS:", rows)
  # print("COLS:", cols)
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  # heatmap = ax.matshow(data, interpolation='nearest')
  # fig.colorbar(heatmap)
  # heatmap = ax.pcolor(data, cmap='YlGnBu', vmin=0, vmax=300)
  heatmap = ax.pcolormesh(data, cmap='YlGnBu')
  fig.colorbar(heatmap)

  # # put the major ticks at the middle of each cell
  ax.set_xticks(np.arange(len(rows))+.5)
  ax.set_yticks(np.arange(len(cols))+.5)

  vmin = 0
  if vmax is None:
    vmax = 1. if np.max(data) <= 1. else np.max(data)
      
  ax.set_xticklabels(rows, rotation='vertical', fontsize='small')
  ax.set_yticklabels(cols, fontsize='x-small')
  ax.set_xlim(0, len(rows))
  ax.set_ylim(0, len(cols))
  plt.xlabel("Pts Projected FROM each Global HCube")
  plt.ylabel("Pts Projected INTO each HCube for %s"% title[-3:])
  # plt.legend()
  fig.suptitle('Heatmap: '+title)
  plt.tight_layout()
  # fig.set_size_inches(3,6)
  plt.savefig(SAVELOC + '/heatmap_' + title + '.png')
  plt.close()  
  plt.show()






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
  plt.show()


def scrape_cw(appl_nodeAme):
  data = {}
  bench = []
  for a in range(5):
    for b in range(5):
      data[str((a, b))] = []
  state_data = [[] for i in range(5)]
  logdir = os.path.join(HOME, 'work', 'log', appl_nodeAme)
  ls = sorted([f for f in os.listdir(logdir) if f.startswith('cw')])
  for i, cw in enumerate(ls[1:]):
    filenodeAme = os.path.join(logdir, cw) 
    # print('Scanning: ', filenodeAme)
    with open(filenodeAme, 'r') as src:
      ts = None
      lines = src.read().split('\n')
      timing = []
      for l in lines:
        if 'TIMESTEP:' in l:
          elms = l.split()
          ts = int(elms[-1])
        elif '##STATE_CONV' in l:
          vals = l.split()
          state_data[int(vals[2])].append(float(vals[3]))
        elif l.startswith('##CONV'):
          if ts is None:
            print("TimeStep was not found before scrapping data. Check file: %s", cw)
            break
          label = l[11:17]
          vals = l.split()
          # print (vals)
          conv = vals[7].split('=')[1]
          # print(label, conv)
          data[label].append((ts, conv))
        elif l.startswith('##   '):
          vals = l[3:].split()
          timing.append((vals[0].strip(), vals[1].strip()))
      bench.append(timing)
  return data, bench, state_data

def printconvergence(data):
  for k, v in sorted(data.items()):
    print(k, np.mean([float(p[1]) for p in v]))
 

def plotconvergence(appl_nodeAme, data):
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  for a in range(5):
    for b in range(5):
      key = '(%a, %d)'%(a,b)
      X = [x[0] for x in data[key]]
      Y = [y[1] for y in data[key]]
      plt.plot(X, Y, label='%d'%b)
    plt.xlabel('# Transitions FROM: State %d' % a)
    plt.ylabel('Convergence')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(loc + '/' + appl_nodeAme + '_conv_%d.png' % a)
    plt.close()


def checkkey(key):
    if key.startswith('LD:pca'):
      check = 'LD:pcasubsp'
    elif key.startswith('LD:Hc'):
      check = 'LD:hcubes'
    else:
      check = key 
    if check in markpts:
      return check
    else:
      return None

def printtiming(bench):
  totals = OrderedDict()
  for i in bench[0]:
    totals[checkkey(i[1])] = []
  for i in bench:
    last = 0.
    for k in i[1:]:
      key = checkkey(k[1])
      if key is None:
        continue
      ts = float(k[0])
      tdif = ts - last
      if key.startswith('LD:pca'):
        key = 'LD:pcasubsp'
      if key.startswith('LD:Hc'):
        key = 'LD:hcubes'
      totals[key].append(tdif)
      last = ts
  for k in markpts:
    if len(totals[k]) > 0:
      print ('%-22s  %6.2f' % (k, np.mean(totals[k])))
