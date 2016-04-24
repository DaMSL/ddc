import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import numpy as np

HOME = os.environ['HOME']

appl_nodeAme = 'leastconv'

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
 

def dotgraph(X, Y, title, L=None):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  if L is None:
    plt.scatter(X, Y, s=1, lw=0)
  else:
    plt.scatter(X, Y, c=L, s=1, lw=0)
  plt.ylabel(title)
  # plt.legend()
  plt.savefig(loc + '/' + title + '.png')
  plt.close()

def dotgraph3D(X, Y, Z, title, L=None):
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



def linegraph(X, title):
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  if isinstance(X, dict):
    for k, v in X.items():
      print('Plotting: ', k)
      plt.plot(np.arange(len(v)), v, label=k)
  else:
    logging.warning("Implement for %s", str(type(X)))
    return
  plt.xlabel('title')
  plt.legend()
  plt.savefig(loc + '/' + title + '.png')
  plt.close()  



def linegraphcsv(X, title, nolabel=False):
  """ Plots line series for a csv list
  """
  


# encode = [0, 4, 2, 3, 1]
# gmm_res = [encode[g] for g in gmmfull5]
# gmmc = [np.argmax(np.bincount(gmm_res[i:i+25])) for i in range(0, 103125, 25)]
# gmmd = [gmm_res[i] for i in range(12, 103125, 25)]
# gmme = []
# for i in range(0, 103125, 25):
#   counts = np.bincount(gmm_res[i:i+25])
#   idx = np.argsort(counts)[::-1]
#   if len(idx) == 1 or counts[idx[0]] > 20:
#     gmme.append((idx[0], idx[0]))
#   else:
#     gmme.append((idx[0], idx[1]))

# stats = [0, 0, 0]
# for z in range(4125):
#   if gmme[z][0] == gmme[z][1] == labels[z]:
#     stats[0] += 1
#   elif gmme[z][0] == labels[z] or gmme[z][1] == labels[z]:
#     stats[1] += 1
#   else:
#     stats[2] += 1

#   stats[int(z[0]==z[1])] += 1
#   if z[0]!=z[1]:
#     print (z)





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

# markpts=['START',
# 'LD:hcubes',
# 'LD:pcasubsp',
# 'KDTreee_build',
# 'Bootstrap:RMS',
# 'Select:RMS_bin',
# 'BackProject:RMS_To_HD',
# 'Project:RMS_To_PCA',
# 'GammaFunc',
# 'Sampler',
# 'GenInputParams',
# 'PostProcessing']

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

# data, bench, state = scrape_cw('lc_avg')
# printconvergence(data)
# printtiming(bench)



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
