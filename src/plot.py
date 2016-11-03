import os
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from datetime import datetime as dt

from collections import OrderedDict
import numpy as np
import string
import itertools

import fca as FCA

HOME = os.environ['HOME']
SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')

 
elabels = ['Serial', 'Parallel','Uniform','Biased', 'MVNN', 'Reweight']
ecolors = {'Serial': 'darkorange', 
           'Parallel': 'maroon',
           'Uniform': 'darkgreen',
           'Biased': 'mediumblue',  
           'MVNN': 'darkmagenta', 
           'Reweight': 'red'}

arg_list = ['title', 'fname', 'xlabel', 'ylabel', 'xlim', 'ylim']
axis_list = ['xscale', 'yscale']
st_col = ['red', 'blue', 'green', 'purple', 'black']

def prep_graph():
  plt.cla()
  plt.clf()

def state_legend(loc='upper left'):
  patches = [mpatches.Patch(color=c, label='State %d'%s) for s, c in enumerate(st_col)]
  plt.legend(handles=patches, loc=loc, prop={'family': 'monospace'})

def graph_args(kwargs):
  arg = {k: kwargs.get(k, None) for k in arg_list}
  if arg['ylabel'] is not None:
    plt.ylabel(arg['ylabel'])
  if arg['xlabel'] is not None:
    plt.xlabel(arg['xlabel'])
  plt.ylim(arg['ylim'])
  plt.xlim(arg['xlim'])
  if 'vlines' in kwargs:
    for v in kwargs['vlines']:
      plt.axvline(v, color='k')
  title = 'graph' if arg['title'] is None else arg['title']
  plt.title(title)
  fname = title if arg['fname'] is None else arg['fname']
  plt.savefig(SAVELOC + '/' + fname + '.png')
  plt.close()

def graph_axis(ax, kwargs):
  arg = {k: kwargs.get(k, None) for k in axis_list}
  if arg['xscale'] is not None:
    xscale = arg['xscale']
    labels = [item.get_text() for item in ax.get_xticklabels(which='both')]
    # print(labels)
    xmin, xmax = xscale
    dx = (xmax-xmin) / len(labels)
    if isinstance(xmin, int):
      ax.set_xticklabels(['%d'%int(xmin+(i*dx)) for i,tick in enumerate(labels)])
    else:
      ax.set_xticklabels(['%.1f'%(xmin+(i*dx)) for i,tick in enumerate(labels)])
  if arg['yscale'] is not None:
    yscale = arg['yscale']
    labels = [item.get_text() for item in ax.get_yticklabels(which='both')]
    # print(labels)
    xmin, xmax = yscale
    dx = (xmax-xmin) / len(labels)
    if isinstance(xmin, int):
      ax.set_yticklabels(['%d'%int(xmin+(i*dx)) for i,tick in enumerate(labels)])
    else:
      ax.set_yticklabels(['%.1f'%(xmin+(i*dx)) for i,tick in enumerate(labels)])



# ecolors = {'Serial': '#FF8C00', 
#            'Parallel': '#803030',
#            'Uniform': '#187018',
#            'Biased': '#3030CD',  
#            'MVNN': '#C030C0', 
#            'Reweight': 'red'}

####  SCATTER PLOTS

def scatter(X, Y, title, size=1, L=None, fname=None, ylabel=None, xlabel=None, xlim=None, ylim=None, ):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  if L is None:
    plt.scatter(X, Y, s=size, lw=0)
    # plt.plot(X, Y)
  else:
    plt.scatter(X, Y, c=L, s=size, lw=0)
  col = ['red', 'blue', 'green', 'purple', 'black']
  patches = [mpatches.Patch(color=c, label='State %d'%s) for s, c in enumerate(col)]
  plt.legend(handles=patches, loc='upper left', prop={'family': 'monospace'})  

  plt.title(title)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylim is not None:
    plt.ylim(ylim)
  if xlim is not None:
    plt.xlim(xlim)
  # plt.legend()
  if fname is None:
    fname = title
  plt.savefig(loc + '/' + fname + '.png')
  plt.close()


def network2D(nodeList, edgeList, **kwargs):
  prep_graph()
  fig, ax = plt.subplots()
  for n0, n1 in edgeList:
    x0, y0 = nodeList[n0]['x'], nodeList[n0]['y']
    x1, y1 = nodeList[n1]['x'], nodeList[n1]['y']
    plt.plot((x0, x1), (y0, y1), linewidth=1, zorder=1, color='lightgrey')
  for node in nodeList.values():
    x0, y0 = node['x'], node['y']
    plt.scatter(x0, y0, c='blue', s=2*node['size'], zorder=2, lw=0)
  fig.set_size_inches(16, 16)
  graph_args(kwargs)

def nodeGraph1D(nodelist, **kwargs):
  prep_graph()
  fig, ax = plt.subplots()
  ax.axes.get_yaxis().set_visible(False)
  plt.subplot(2,1,1)
  plt.title('RMSD Distribution')
  xvals = [i['x'] for i in nodelist]
  width = (max(xvals) - min(xvals)) / (len(xvals))
  print(max(xvals), min(xvals), len(xvals), width)
  plt.axhline(0, color='lightgrey', linewidth=1, zorder=1)
  for n in nodelist:
    # plt.scatter(n['x'], 0, c='blue', s=3*n['size'], zorder=2, lw=0)
    plt.bar(n['x'], n['size'], width, color='blue', zorder=2, lw=1)
  plt.subplot(2,1,2)
  yvals = [i['y'] for i in nodelist]
  width = (max(yvals) - min(yvals)) / (len(yvals))
  print(width)
  plt.title('Backbone-DiH Distribution')
  plt.axhline(0, color='lightgrey', linewidth=1, zorder=1)
  for n in nodelist:
    plt.bar(n['y'], n['size'], width, color='blue', zorder=2, lw=1)
    # plt.scatter(n['y'], 0, c='blue', s=3*n['size'], zorder=2, lw=0)
  graph_args(kwargs)
 


# class Arrow3D(FancyArrowPatch):
#   """
#    re: http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
#   """
#   def __init__(self, xs, ys, zs, *args, **kwargs):
#       FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
#       self._verts3d = xs, ys, zs

#   def draw(self, renderer):
#       xs3d, ys3d, zs3d = self._verts3d
#       xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#       self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#       FancyArrowPatch.draw(self, renderer)


  # for n in node['edges']:
  #   x1, y1 = nodeList[n]['x'], nodeList[n]['y']
  #   plt.plot((x0, x1), (y0, y1), linewidth=1, color='k')

def scatter2D(X, Y, size=1, L=None, **kwargs):
  prep_graph()
  if L is None:
    plt.scatter(X, Y, lw=0)
  else:
    plt.scatter(X, Y, c=L, lw=0)
  graph_args(kwargs)


def scatter3D(X, Y, Z, L=None, **kwargs):
  prep_graph()
  fig = plt.figure()
  ax = Axes3D(fig)
  if L is None:
    ax.scatter(X, Y, Z, lw=0)
  else:
    ax.scatter(X, Y, Z, c=L, lw=0)
  if 'zlabel' in kwargs:
    ax.set_zlabel = kwargs['zlabel']
  graph_args(kwargs)

def scats (series, title, size=10, xlabel=None, xlim=None, ylim=None, labels=None):
  if isinstance(series, dict):
    keys = sorted(series.keys())
    if labels is None:
      labelList = keys
    else:
      labelList = [labels[i] for i in keys]
    seriesList = [series[key] for key in keys]
  else:
    print("Series must be either a list of lists or a mapping to lists")
    return
  colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  plt.clf()
  ax = plt.subplot(111)
  for D, C, L in zip(seriesList, colorList, labelList):
    X, Y = zip(*D)
    # plt.scatter(X, Y, s=size, c=C, lw=0)
    plt.plot(X, Y, c=C)
  plt.title(title)

  if xlabel is not None:
    plt.xlabel(xlabel)
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)

  patches = [mpatches.Patch(color=C, label=L) for C, L in zip(colorList, labelList)]
  plt.legend(handles=patches, loc='upper right')  

  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

def scat_Transtions (series, title, size=10, xlabel=None, xlim=None, labels=None):
  plt.clf()
  ax = plt.subplot(111)
  maxlen = 0
  maxY = 0
  minY = 0
  for C, D in series.items():
    if len(D) == 0:
      continue
    X, Y = zip(*D)
    maxlen = maxlen + len(X)
    maxY = max(maxY, max(Y))
    minY = min(minY, min(Y))
    plt.scatter(X, Y, s=size, c=C, lw=0)
  plt.title(title)
  plt.xlim(0, maxlen)
  plt.ylim(minY, maxY)
  if xlabel is not None:
    plt.xlabel(xlabel)
  patches = [mpatches.Patch(color=C, label=L) for C, L in labels.items()]
  plt.legend(handles=patches, loc='lower right')  
  plt.savefig(SAVELOC + '/scat_' + title + '.png')
  plt.close()

def scat_layered (series, title, size=10, xlabel=None, xlim=None):
  marker_list = ('o', 'v', '*', 'H', 'D', '^', '<', '>', '8', 's', 'p', 'h', 'd')
  keys = sorted(series.keys())
  labelList1 = keys
  labelList2 = sorted(series[keys[0]].keys())
  seriesList = [series[key] for key in keys]
  colorList = plt.cm.brg(np.linspace(0, 1, len(labelList1)))
  markerList = marker_list[:len(labelList2)]
  plt.clf()
  ax = plt.subplot(111)
  for S, C in zip(seriesList, colorList):
    seriesList2 = [S[k] for k in sorted(S.keys())]
    for D, M in zip(seriesList2, markerList):
      X, Y = zip(*D)
      plt.scatter(X, Y, s=size, c=C, marker=M, lw=0)
  plt.title(title)

  if xlabel is not None:
    plt.xlabel(xlabel)
  if xlim is not None:
    plt.xlim(xlim)

  markers = [mlines.Line2D([], [], color='k', marker=M, markersize=5, label=L) for M,L in zip(markerList, labelList2)]
  patches = [mpatches.Patch(color=C, label=L) for C, L in zip(colorList, labelList1)]
  plt.legend(handles=patches+markers, loc='upper right')  
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

def deshaw_rmsd(Y, title, label, size=1, fname=None):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  col = ['red', 'blue', 'green', 'purple', 'black']
  X = np.arange(len(Y))
  colors = [col[l] for l in label]
  # print(colors[230:240])
  plt.scatter(X, Y, color=colors, s=size)
  plt.xlim(0, len(Y))
  plt.title(title)
  patches = [mpatches.Patch(color=c, label='State %d'%s) for s, c in enumerate(col)]
  plt.legend(handles=patches, loc='upper left', prop={'family': 'monospace'})  
  if fname is None:
    fname = title
  plt.savefig(loc + '/' + fname + '.png')
  plt.close()

def step_rmsd(Y, title, label, fname=None, ylim=None):
  plt.cla()
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  col = ['red', 'blue', 'green', 'purple', 'black']
  colors = [col[l] for l in label]
  x0 = 0
  for (x, y), L in zip(Y, label):
    plt.plot((x0, x0+x), (y, y), color=col[L])
    x0 += x
  plt.xlim(0, x0)
  plt.title(title)
  patches = [mpatches.Patch(color=c, label='State %d'%s) for s, c in enumerate(col)]
  plt.legend(handles=patches, loc='upper left', prop={'family': 'monospace'})  
  if fname is None:
    fname = title
  plt.savefig(loc + '/' + fname + '.png')
  plt.close()

def edges(A, B, L=None, size=10, **kwargs):
  prep_graph()
  if L is None:
    L = ['k' for i in range(len(A))]
  if (A.shape[1] == 3):
      fig = plt.figure()
      ax = Axes3D(fig)
      ax.scatter(A[:,0], A[:,1], A[:,2], c=L, marker='o', s=20, lw=0)
      ax.scatter(B[:,0], B[:,1], B[:,2], c=L, marker='^', s=2, lw=0)
      for a,b,c in zip(A,B,L):
        plt.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], color=c)
  else:
      plt.scatter(A[:,0], A[:,1], c=L, s=size, marker='o', lw=0)
      plt.scatter(B[:,0], B[:,1], c=L, s=size, marker='^', lw=0)
      for a,b,c in zip(A,B,L):
        plt.plot([a[0],b[0]], [a[1],b[1]], color=c)
  graph_args(kwargs)



#####   LINE Graph 
 
def line(X, **kwargs):
  prep_graph()
  plt.rcParams['agg.path.chunksize'] = 100000
  if isinstance(X, dict):
    for k, v in X.items():
      plt.plot(np.arange(len(v)), v, label=k)
  elif isinstance(X, np.ndarray):
    plt.plot(np.arange(len(X)), X)
  else:
    print("Not Implemented for:", str(type(X)))
    return
  plt.legend(loc='upper left')
  graph_args(kwargs)

def linegraphcsv(X, title, nolabel=False):
  """ Plots line series for a csv list
  """

def step_lines(series, title, xlabel=None, scale=None):
  labelList = sorted(series.keys())
  seriesList = [series[key] for key in labelList]
  colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  plt.clf()
  ax = plt.subplot(111)
  for S, C in zip(seriesList, colorList):
    x0 = 0
    for x, y in S:
      plt.plot((x0, x0+x), (y, y), color=C)
      x0 += x
  plt.title(title)
  if xlabel is not None:
    plt.xlabel(xlabel)

  if scale is not None:
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    ax.xaxis.set_major_formatter(ticks)

  patches = [mpatches.Patch(color=C, label='%d'%L) for C, L in zip(colorList, labelList)]
  plt.legend(handles=patches, loc='upper left', prop={'family': 'monospace'})  
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

def steps(data, title, ylabel=None):
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  for d, col in zip(data, ('b', 'g', 'k')):
    x0 = 0
    for x,y in d:
      plt.plot((x0, x0+x), (y,y), c=col)
      x0 += x
  if ylabel is not None:
    plt.ylabel(ylabel)
  plt.xlabel('Time (in ps)')
  patches = [mpatches.Patch(color='b', label='Biased'),
             mpatches.Patch(color='g', label='Uniform'), 
             mpatches.Patch(color='k', label='Naive')]
  plt.legend(handles=patches, loc='lower right', prop={'family': 'monospace'})  
  plt.legend()
  plt.savefig(loc + '/' + title + '.png')
  plt.close()  

def seriesLines(X, series, title, xlabel=None):
  labelList = sorted(series.keys())
  seriesList = [series[key] for key in labelList]
  colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  plt.clf()
  ax = plt.subplot(111)
  for Y, C, L in zip(seriesList, colorList, labelList):
    plt.plot(X, Y, color=C, label=L)
  plt.title(title)
  if xlabel is not None:
    plt.xlabel(xlabel)
  plt.legend()
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

def lines(series, xseries=None, showlegend=True, dolog=False, yticks = None, **kwargs): #title, xlim=None, labelList=None, step=1, xlabel=None):
  if isinstance(series, list):
    seriesList = series
    # if labelList is None:
    labelList = ['Series %d' % (i+1) for i in range(len(seriesList))]
  elif isinstance(series, dict):
    labelList = sorted(series.keys())
    seriesList = [series[key] for key in labelList]
  else:
    print("Series must be either a list of lists or a mapping to lists")
    return
  colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  plt.clf()
  ax = plt.subplot(111)
  if xseries is None:
    for X, C, L in zip(seriesList, colorList, labelList):
      plt.plot(np.arange(len(X)), X, color=C, label=L)
  else:
    xseriesList = [xseries[k] for k in labelList]
    for X, Y, C, L in zip(xseriesList, seriesList, colorList, labelList):
      plt.plot(X, Y, color=C, label=L)
  if dolog:
    ax.set_yscale('log')
  if yticks is not None:
      print('setting yticks: ', yticks)
      ax.set_yticklabels(yticks)
  if showlegend:
    plt.legend(loc='upper left')
  graph_axis(ax, kwargs)
  graph_args(kwargs)


def show_distr(series, xscale=None, showlegend=True, states=None, **kwargs):
  labelList = sorted(series.keys())
  seriesList = [series[key] for key in labelList]
  stcolor = ['red', 'blue', 'green', 'purple', 'cyan']
  if states is None:
    colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  else:
    colorList = []
    for k in labelList:
      col = 'black' if k == 'All' else stcolor[states[k]]
      colorList.append(col)
  plt.clf()
  ax = plt.subplot(111)
  for X, C, L in zip(seriesList, colorList, labelList):
    col, lw, ls = ('black', 5, '--') if L == 'All' else (C, 1, '-')
    plt.plot(np.arange(len(X)), X, color=col, label=L, linewidth=lw, linestyle=ls)
  if xscale is not None:
    labels = [item.get_text() for item in ax.get_xticklabels(which='both')]
    # print(labels)
    xmin, xmax = xscale
    dx = (xmax-xmin) / len(labels)
    ax.set_xticklabels(['%.1f'%(xmin+(i*dx)) for i,tick in enumerate(labels)])
  if showlegend and states is not None:
    patches = [mpatches.Patch(color='black', label='All')]
    for i, c in enumerate(stcolor):
      patches.append(mpatches.Patch(color=c, label='State %d'%i))
      plt.legend(handles=patches, loc='upper right', prop={'family': 'monospace'})
  graph_args(kwargs)


def manylines(linelist, step=1, **kwargs): #title, xlim=None, labelList=None, step=1, xlabel=None):
  prep_graph()
  if isinstance(linelist, list):
    lines = np.array(linelist)
  else:
    lines = linelist
  N, D = lines.shape
  X = np.arange(D)*step
  ax = plt.subplot(111)
  for L in lines:
    plt.plot(X, L)
  graph_args(kwargs)

def conv(series, title, xlim=None, ylim=None, labelList=None, step=1, xlabel=None):
  seriesList = [series[key] for key in elabels]
  colorList  = [ecolors[key] for key in elabels]
  plt.clf()
  ax = plt.subplot(111)
  for X, C, L in zip(seriesList, colorList, elabels):
    plt.plot(np.arange(len(X))*(step), X, color=C, label=L, linewidth=2)
  plt.title(title)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if xlim is not None:
    ax.set_xlim(xlim)
  if ylim is not None:
    ax.set_ylim(ylim)
  plt.legend()
  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

def transition_line(X, A, B, title='', trans_factor=.33):
  plt.clf()
  loc = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.plot(np.arange(len(X)), X)

  # Find Crossover Point
  crossover = 0
  hgt = max(X) - min(X)
  for i, x in enumerate(X):
    if x > 0:
      crossover = i
      break
  print('Crossover at Index #', crossover)

  # Find local max gradient  (among 50% of points)
  zoneA = int((1-trans_factor) * crossover)
  zoneB = crossover + int(trans_factor * (len(X) - crossover))
  gradA = zoneA + np.argmax(np.gradient(X[zoneA:crossover]))
  gradB = crossover + np.argmax(np.gradient(X[crossover:zoneB]))
  thetaA = X[gradA]
  thetaB = X[gradB]
  # print('states',A,B, sep=',')
  # print('idxlist',zoneA, gradA, gradB, zoneB, sep=',')
  # print('theta',thetaA, thetaB, sep=',')

  # ID # Pts in each
  a_pts  = gradA
  ab_pts = crossover - gradA
  ba_pts = gradB - crossover
  b_pts  = len(X) - gradB

  # print('numpts',a_pts,ab_pts,ba_pts, b_pts,sep=',')

  plt.scatter(crossover, X[crossover], color='r', s=30, marker='o', label='Crossover')
  plt.scatter(gradA, X[gradA], color='g', s=30, marker='o', label='LocalMax (%d,%d)'%(A,B))
  plt.scatter(gradB, X[gradB], color='g', s=30, marker='o', label='LocalMax (%d,%d)'%(B,A))

  plt.axvline(crossover, color='r')
  plt.axvline(gradA, color='g')
  plt.axvline(gradB, color='g')
  plt.annotate('# A = %d' % a_pts, xy=(len(X)*.1, min(X)+hgt*.75))
  plt.annotate('# B = %d' % b_pts, xy=(len(X)*.8, min(X)+hgt*.25))
  plt.annotate('(A, B) = %d\nT=%4.2f' % (ab_pts, thetaA), xy=(crossover-ab_pts, 0), ha='right')
  plt.annotate('(B, A) = %d\nT=%4.2f' % (ba_pts, thetaB), xy=(crossover+ba_pts, 0))
  plt.title('Transitions: %d , %d  (Transition Factor = %4.2f)' % (A, B, trans_factor))
  plt.xlim(0, len(X))
  plt.ylim(min(X), max(X))
  plt.legend()
  plt.xlabel('Observations')
  plt.ylabel('Delta of Distances to Centoids %d & %d' % (A,B))
  plt.savefig(loc + '/transition%s_%d_%d'%(title,A,B) + '.png')
  plt.close()  

def bootCI(boot, step=10, tag=''):
  N = min([len(v) for v in boot.values()])
  merge = {}
  merge = {k: [np.mean([min(1., boot[k][i][1][f]) for f in range(5, 20)]) for i in range(N)] for k in boot.keys()}
  lines(merge, 'ConvCI_Merged_%s'%tag, step=step, xlabel='Simulation Time (in ns)')

def elas_graph (series, title, size=10, xlabel=None, xlim=None, ylim=None, labels=None):
  keys = sorted(series.keys())
  seriesList = [series[key] for key in keys]
  colorList = plt.cm.brg(np.linspace(0, 1, len(seriesList)))
  plt.clf()
  ax = plt.subplot(111)
  firstX = 0
  firstPt = False
  labtext = {}
  for D, C, L in zip(seriesList, colorList, keys):
    X, Y = zip(*D)
    sx, sy = X[0], Y[0]
    ex, ey = X[-1], Y[-1]
    plt.scatter(sx, sy, s=25, c=C, lw=0)
    plt.scatter(ex, ey, s=25, c=C, lw=0)
    labtext[L] = 'Step = %3d: CI=%4.2f in %3.1f hrs' % (L, ey, ex)
    plt.plot(X, Y, c=C, linewidth=2)
  plt.title('Elasticity:  Convergence vs Time')

  labelList = [labtext[key] for key in keys]
  plt.xlabel('Time (in hours)')
  plt.ylabel('Confidence Interval Width')
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)

  patches = [mpatches.Patch(color=C, label=L) for C, L in zip(colorList, labelList)]
  plt.legend(handles=patches, loc='upper right', prop={'family': 'monospace'})  

  plt.savefig(SAVELOC + '/' + title + '.png')
  plt.close()

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

def stackhisto(data, **kwargs):
  """
   Assumes a dict of labels to a list of ints representing counts
  """
  prep_graph()
  X = 0
  for k, v in sorted(data.items()):
    y0 = 0
    # print(X, k, v)
    for i, y1 in enumerate(v):
      plt.bar(X, y1, bottom=y0, color=st_col[i])
      y0 += y1
    X += 1
  plt.xticks(np.arange(X)+.5, np.arange(X))
  state_legend()
  graph_args(kwargs)

def bargraph_simple(data, err=None, **kwargs):
  prep_graph()
  fig, ax = plt.subplots()
  labels = None
  if isinstance(data, list):
    Y = data
    if err is not None:
      error = err
  elif isinstance(data, dict):
    labels = sorted(data.keys())
    Y = [data[i] for i in labels]
    if err is not None:
      error = [err[i] for i in labels]
  X = np.arange(len(Y))
  if err is None:
    plt.bar(X, Y)
  else:
    plt.bar(X, Y, yerr=error, error_kw=dict(ecolor='red'))
  plt.legend()
  if labels is not None:
      plt.xticks(X+.5, labels, rotation='vertical')
  # vmax = 1. if np.max(Y) <= 1. else np.max(Y)
  if 'title' not in kwargs:
    kwargs['title'] = 'bargraph'

  if 'ylim' not in kwargs:
    vmax = np.max(Y)
    if err is not None:
      vmax += np.max(error)
    kwargs['ylim'] = (0, vmax)
  plt.tight_layout()
  graph_args(kwargs)

def feadist(data, title, fname=None, err=None, pcount=None, norm=1, negval=False):
  plt.cla()
  plt.clf()
  numlabels = 5
  colors = ['k','grey','r','b','g',]
  labels=['C0', 'C1', 'C2', 'C3', 'C4', 'S0', 'S1', 'S2', 'S3', 'S4', '0-1', '0-2','0-3', '0-4', '1-2', '1-3','1-4','2-3','2-4','3-4']
  pairs = []
  for a in range(numlabels-1):
    for b in range(a + 1, numlabels):
      pairs.append((a,b))
  fig, ax = plt.subplots()
  Y = data
  X = np.arange(len(Y))  

  # Print first 10
  for i in range(10):
    C = i % 5
    if err is None:
      plt.bar(i, Y[i], color=colors[C])
    else:
      plt.bar(i, Y[i], color=colors[C], yerr=err[i], error_kw=dict(ecolor='red', elinewidth=2))
  for i in range(10):
    polar = Y[i+10] > norm/2
    C = pairs[i][polar]
    y = Y[i+10]*2 - norm if negval else Y[i+10]
    if err is None:
      plt.bar(i+10, y, color=colors[C])
    else:
      plt.bar(i+10, y, color=colors[C], yerr=err[i], error_kw=dict(ecolor='red', elinewidth=2))

  plt.axvline(4.9, color='k', linewidth=2)
  plt.axvline(9.9, color='k', linewidth=2)
  ymin = -norm if negval else 0
  ymax = norm
  plt.annotate('Count of Obs with\nlowest RMSD', xy=(.1, -2))
  plt.annotate('Proximity to\nCentroids', xy=(5.1, -2))
  if pcount is not None:
    plt.annotate('HC Size: %d' % pcount, xy=(5.1, -.2*norm))
  # plt.annotate('Distance Delta for each pair of RMSD', xy=(10.1, -.1*norm))

  patches = [mpatches.Patch(color=C, label= L) for C, L in zip(colors, ['State %d'%i for i in range(5)])]
  plt.legend(handles=patches, loc='lower left')  

  # plt.legend()
  ax.set_xticks(X+.5)
  ax.xaxis.set_ticks_position('none')
  ax.set_xticklabels(labels)
  if negval:
    ax.set_ylim(-norm-1, norm+1)
  else:
    ax.set_ylim(-1, norm)
  fig.suptitle(title, va='baseline')
  plt.tight_layout()
  if fname is None:
    fname = title
  plt.savefig(SAVELOC + '/' + fname + '.png')
  plt.close()  
  plt.show()

def histogram(data, title, ylim=None):
  # colorList = plt.cm.brg(np.linspace(0, 1, len(data)))
  # seriesList = list(data.keys())
  seriesList = elabels
  colorList  = [ecolors[key] for key in elabels]
  nseries = len(seriesList)

  pad = .3
  # sets   = sorted(data[seriesList[0]].keys())
  sets = ['Well-2', 'Well-3', 'Well-4', 'Tran-0', 'Tran-1', 'Tran-2', 'Tran-3', 'Tran-4']
  nbars = len(sets)
  X = np.arange(nbars)
  bw = (1-pad) / nseries

  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()

  for x, S, C in zip(np.arange(nseries), seriesList, colorList):
    Y = [data[S][i] for i in sets]
    offset = x*bw+(pad/2)
    plt.bar(X+offset, Y, bw, color=C, label=S)

  ax.set_xlim(0, nbars)
  if ylim is not None:
    ax.set_ylim(ylim)
  plt.xticks(X+.5, sets, fontsize=20)
  plt.yticks(fontsize=18)
  plt.legend(prop={'size': 16})
  # fig.suptitle(title, va='top', fontsize=32)
  fig.set_size_inches(16,6)
  plt.tight_layout()
  plt.savefig(SAVELOC + '/histogram' + '.png')
  plt.close()  


def histo_simple(data, bins=20, **kwargs):
  prep_graph()
  plt.hist(data, bins=bins)
  graph_args(kwargs)


##### PIE CHARTS
def pie(data, title):
  labelList = data.keys()
  vals = [data[k] for k in labelList]
  plt.clf()
  ax = plt.subplot(111)
  plt.pie(vals, labels=labelList, autopct='%1.1f%%', startangle=90)
  plt.title(title)
  plt.savefig(SAVELOC + '/pie_' + title + '.png')
  plt.close()


def stateviz(data, title):
  labelList = [0,1,2,3,6]
  plt.clf()
  ax = plt.subplot(111)
  circle = plt.Circle((0,0), 1, fill=False)
  plt.pie(data, labels=labelList, radius=.5, startangle=90)
  ax.add_artist(circle)
  plt.title(title)
  plt.tight_layout()
  plt.savefig(SAVELOC + '/pie_' + title + '.png')
  plt.close()


#### LATTICE
ascii_greek = ''.join([chr(i) for i in itertools.chain(range(915,930), range(931, 938), range(945, 969))])
label_domain = string.ascii_lowercase + string.ascii_uppercase + ascii_greek
def lattice_orig(obs_list, dspace=None, invert=False, **kwargs):
  st = dt.now()
  prep_graph()
  fig, ax = plt.subplots()
  N, K = obs_list.shape
  K_domain = label_domain[:K]
  frommask = lambda obs: ''.join([K_domain[i] for i, x in enumerate(obs) if x == 1])
  tomask = lambda key: [(1 if i in key else 0) for i in K_domain[i]]
  toidx = lambda key: [i for i,f in enumerate(K_domain) if f in key]
  obs = [frommask(o) for o in obs_list]
  poset = []
  max_x = 0
  count_map = {}
  idx_map = {}
  mu = {}
  for k in range(K):
    poset_k = set()
    for t, x in enumerate(obs):
      for obj in itertools.combinations(x, k):
        key = ''.join(sorted(obj))
        poset_k.add(key)
        if key not in count_map:
          count_map[key], idx_map[key] = 0, set()
        count_map[key] += 1 
        idx_map[key].add(t)
    poset.append(sorted(poset_k))
    max_x = max(max_x, len(poset_k))
  if dspace is not None:
    print(type(dspace))
    for k,v in idx_map.items():
      mu[k] = (dspace[sorted(v)])[:,toidx(k)].mean(0)

  et = dt.now()
  print('Lattice Built. Time time = %5.2f' % (et-st).total_seconds())
  print('Rendering...')
  ax.annotate('{}'+'-%d' % count_map[''], 
    xy=(max_x/2., 0), horizontalalignment='center', fontsize='xx-small')
  for y in range(1, len(poset)):
    dx = max_x if len(poset[y]) == 0 else max_x / len(poset[y])
    print(len(poset[y]), [''.join(sorted(p)) for p in poset[y]])
    for x, p in enumerate(sorted(poset[y])):
      xpos = .5*dx + x*dx
      if invert:
        ax.annotate('%s\n%s' % (','.join([str(ord(i)-97) for i in p]),''.join([chr(i+97) for i in sorted(idx_map[p])])), 
          xy=(xpos, y), horizontalalignment='center',fontsize='xx-small')
      else:
        distr = '' if p not in mu else ','.join(['%3.1f'%i for i in mu[p]])
        # ax.annotate(p+'-%d\n%s\n%s' % (count_map[p], ','.join([str(i) for i in sorted(idx_map[p])]),distr), 
        #   xy=(xpos, y), horizontalalignment='center',fontsize='xx-small')
        ax.text(xpos, y, p+'-%d\n(%s)' % (count_map[p], distr), 
          ha='center',fontsize='xx-small', wrap=True)
      if y > 1:
        for x1, inf in enumerate(poset[y-1]):
          if set(inf) < set(p):
            plt.plot((.5*dx1+x1*dx1,xpos),(y-1, y),color='lightgrey', linewidth=1)
    dx1 = dx

  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  if max_x > 10:
    fig.set_size_inches(int(.9*max_x), int(.4*max_x))
  kwargs.update(dict(xlim=(0,max_x), ylim=(0,K)))
  graph_args(kwargs)

def lattice(E, **kwargs):
  prep_graph()
  fig, ax = plt.subplots()
  N, K = E.shape
  K_domain = label_domain[:K]

  # Reduce full dataset to unique rows only
  Z = FCA.clarify_row(E)
  uniquelist = sorted(Z.keys())
  E_r = E[[Z[k][0] for k in uniquelist]]

  # Keep size for display
  Z_size = {k: len(v) for k,v in Z.items()}

  # Temporal Links
  featset = [FCA.frommask(i) for i in E]
  temporal_link = set()
  for i in range(1, len(featset)):
    a, b = featset[i-1], featset[i]
    if a == b:
      continue
    temporal_link.add((min(a,b), max(a,b)))

  print (len(temporal_link))

  # Concept mapping for each n-dimensional order feature set
  st = dt.now()
  concept = [{} for k in range(K)]
  G_size = OrderedDict()
  for k in range(K):
    elist = set(list(np.where(E_r[:,k]==1)[0]))
    concept[0][K_domain[k]] = elist
    G_size[K_domain[k]] = np.sum([Z_size[uniquelist[e]] for e in elist])
  print(Z_size)
  for k in range(1, K):
    prev = list(concept[k-1].keys())
    print('Join at order: ', k, prev)
    for i in range(0, len(prev)-1):
      for j in range(i+1, len(prev)):
        # TODO: Consider alternate meet operations
        meet = concept[k-1][prev[i]] & concept[k-1][prev[j]]
        # not optimized
        key = ''.join(sorted(set(prev[i])|set(prev[j])))
        order = len(key)-1
        try:
          if len(meet) == 0 or key in concept[order]:
            continue
        except IndexError as err:
          print(key)
        concept[order][key] = meet
        G_size[key] = np.sum([Z_size[uniquelist[e]] for e in meet])
    if sum([len(concept[i]) for i in range(k, K)]) == 0:
      # No more concepts to check
      break

  et = dt.now()
  print('Lattice Built. Time time = %5.2f' % (et-st).total_seconds())
  print('Rendering...')
  # if dspace is not None:
  #   print(type(dspace))
  #   for k,v in idx_map.items():
  #     mu[k] = (dspace[sorted(v)])[:,toidx(k)].mean(0)

  max_x = max([len(c) for c in concept])
  nodelist = {}
  ax.annotate('{}'+'-%d' % len(E), 
    xy=(max_x/2., 0), horizontalalignment='center', fontsize='xx-small')
  for k in range(len(concept)):
    dx = max_x if len(concept[k]) == 0 else max_x / len(concept[k])
    print(len(concept[k]), [''.join(sorted(p)) for p in concept[k]])
    label_list = sorted(concept[k].keys())
    for x, p in enumerate(label_list):
      xpos = .5*dx + x*dx
        # ax.annotate(p+'-%d\n%s\n%s' % (count_map[p], ','.join([str(i) for i in sorted(idx_map[p])]),distr), 
        #   xy=(xpos, y), horizontalalignment='center',fontsize='xx-small')
      ax.text(xpos, k+1, p+' (%d)\n%s' % (G_size[p], concept[k][p]), 
          ha='center',fontsize='xx-small', wrap=True)
      if p in uniquelist:
        nodelist[p] = (xpos, k+1)
        circ = plt.Circle((xpos, k+1), .5, color='red', fill=False)
        ax.add_patch(circ)
        plt.text(xpos, k+.75, '%d: %d' % (uniquelist.index(p), Z_size[p]), ha='center',color='red')
        plt.scatter(xpos+.5, k+1.2, color='red', marker='*')
      if k > 0:
        for x1, inf in enumerate(concept[k-1]):
          if set(inf) < set(p):
            plt.plot((.5*dx1+x1*dx1,xpos),(k, k+1),color='lightgrey', linewidth=1)
    dx1 = dx

  # Draw temporal Node Edges for unique observations
  for n1, n2 in temporal_link:
    (x1, y1), (x2, y2) = nodelist[n1], nodelist[n2]
    plt.plot((x1, x2), (y1, y2), color='yellow', linewidth=1)

  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  if max_x > 10:
    fig.set_size_inches(int(.9*max_x), int(.4*max_x))
  kwargs.update(dict(xlim=(0,max_x), ylim=(0,K)))
  graph_args(kwargs)

def derived_lattice(L, sublat=None, **kwargs):
  k_domain = 'abcdefghijklmno'
  tok   = lambda x: ''.join(sorted([chr(97+i) for i in x]))
  fromk = lambda x: frozenset([ord(i)-97 for i in x])

  prep_graph()
  fig, ax = plt.subplots()

  K = max([len(k) for k in L.keys()])
  key_list = [[]]
  for k in range(1, K):
    ca = [i for i in L.keys() if len(i) == k]
    key_list.append(sorted(ca))

  max_x = max([len(Lk) for Lk in key_list])

  # Concept mapping for each n-dimensional order feature set

  #  Draw null node
  nodelist = {}
  ax.annotate('{}', 
    xy=(max_x/2., 0), horizontalalignment='center', zorder=3,fontsize='xx-small')

  for k in range(1, K+1):
    dx = max_x if len(key_list[k]) == 0 else max_x / len(key_list[k])
    for x, fs in enumerate(key_list[k]):
      xpos = .5*dx + x*dx
      # distr_vect = ','.join([('%.2f'%L2[fs][i]).lstrip('0') for i in sorted(L2[fs].keys())])
      delta_vals = '\n'.join([(i+': %.3f'%L[fs][i]).lstrip('0') for i in sorted(L[fs].keys())])
      ax.text(xpos, k, fs+' (%d)\n%s' % (0, delta_vals), zorder=3,
          va='top', ha='center',fontsize='xx-small', wrap=True)
      for x1, child in enumerate(key_list[k-1]):
        if set(child) < set(fs):
          try:
            linecol = 'aquamarine' if L[child][fs] < 0.05 else 'ghostwhite'
          except KeyError as err:
            print(parent, child, L[child])
            raise KeyError
          plt.plot((.5*dx1+x1*dx1,xpos),(k-1, k),color=linecol, zorder=1,linewidth=1)
    dx1 = dx

  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  if max_x > 10:
    fig.set_size_inches(int(.9*max_x), int(.4*max_x))
  kwargs.update(dict(xlim=(0,max_x), ylim=(0,K+.25)))
  graph_args(kwargs)

def show_dlattice(L, Isize, U={}, theta=.05, clusters=[], **kwargs):
  k_domain = 'abcdefghijklmno'
  tok   = lambda x: ''.join(sorted([chr(97+i) for i in x]))
  fromk = lambda x: frozenset([ord(i)-97 for i in x])

  prep_graph()
  fig, ax = plt.subplots()
  colorList = ['blue', 'red', 'lime', 'magenta']
  K = max([len(k) for k in L.keys()])
  key_list = [[], sorted(set.union(*[set(i) for i in L.keys()]))]
  print('K=', K, key_list)
  for k in range(1, K):
    lk = set()
    for key in key_list[-1]:
      for parent in L[key].keys():
        lk.add(parent)
    key_list.append(sorted(lk))
    # key_list.append(sorted([key for key in L2.keys() if len(key) == k]))

  max_x = max([len(Lk) for Lk in key_list])

  # Concept mapping for each n-dimensional order feature set

  #  Draw null node
  nodelist = {}
  ax.annotate('{}', 
    xy=(max_x/2., 0), horizontalalignment='center', zorder=3,fontsize='xx-small')

  # Draw single feature nodes
  # fs_1d = key_list[1]
  # dx1 = max_x / len(fs_1d)
  # for x, fs in enumerate(fs_1d):
  #   xpos = .5*dx1 + x*dx1
  #   ax.text(xpos, 1, fs+' (%d)' % I[fs], 
  #         ha='center',fontsize='xx-small', wrap=True)

  # plt.rc('font', family='monospace')

  for k in range(1, K+1):
    dx = max_x if len(key_list[k]) == 0 else max_x / len(key_list[k])
    for x, fs in enumerate(key_list[k]):
      xpos = .5*dx + x*dx
      # distr_vect = ','.join([('%.2f'%L2[fs][i]).lstrip('0') for i in sorted(L2[fs].keys())])
      delta_vals = '\n'.join([(i+':%.3f'%L[fs][i]).lstrip('0') for i in sorted(L[fs].keys())])
      ax.text(xpos, k, fs+' (%d)\n%s' % (Isize[fs], delta_vals), zorder=3,
          va='top', ha='center',fontsize='7', family='monospace', wrap=True)
      plt.scatter(xpos, k, c='blue', s=5)
      if fs in U:
        nodelist[fs] = (xpos, k)
        # circ = plt.Circle((xpos, k), .5, color='red', fill=False)
        # ax.add_patch(circ)
        plt.text(xpos, k, '* %d' % (len(U[fs])), zorder=3,
          va='bottom', ha='center', fontsize='x-small', color='red')
      for x1, child in enumerate(key_list[k-1]):
        if set(child) < set(fs) and fs in L[child]:
          if L[child][fs] > 1:
            continue
          if L[child][fs] < .1:
            linecol = 'royalblue'
          elif L[child][fs] < .5:
            linecol = 'lightblue'
          else:
            linecol = 'lightcyan'
          for col, clu in enumerate(clusters):
            if fs in clu and child in clu:
              linecol = colorList[col]
              break
          plt.plot((.5*dx1+x1*dx1,xpos),(k-1, k),color=linecol, zorder=1,linewidth=1)
    dx1 = dx

  plt.text(2, .1, '* Unique Key', fontsize='x-small', color='red')
  # Draw temporal Node Edges for unique observations
  # for n1, n2 in temporal_link:
  #   (x1, y1), (x2, y2) = nodelist[n1], nodelist[n2]
  #   plt.plot((x1, x2), (y1, y2), color='yellow', linewidth=1)

  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  if max_x > 10:
    fig.set_size_inches(int(.9*max_x), int(.4*max_x))
  kwargs.update(dict(xlim=(0,max_x), ylim=(0,K+.25)))
  graph_args(kwargs)



def lattice_path_dist(linelist, step=1, **kwargs): #title, xlim=None, labelList=None, step=1, xlabel=None):
  prep_graph()
  selfs = []
  lines = []
  for L in linelist:
    lines.append([d for _,_,d in L])
    for a,b,d in L:
      if d > .1:
        selfs.append((b, d))
  lines = np.array(lines)
  N, D = lines.shape
  X = np.arange(D)*step
  ax = plt.subplot(111)
  for L in lines:
    plt.plot(X, L, linewidth=1)
  for fs, y in selfs:
    x = len(fs)-2
    plt.text(x-.1, y, fs, family='sans-serif', zorder=3, va='bottom', ha='center', fontsize='x-small')
  graph_args(kwargs)


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


def heatmap(data, cols, rows, title, vmax=None, xlabel=None, ylabel=None, colmap='gnuplot2_r'):
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  # heatmap = ax.matshow(data, interpolation='nearest')
  # fig.colorbar(heatmap)
  # heatmap = ax.pcolor(data, cmap='YlGnBu', vmin=0, vmax=300)
  # heatmap = ax.pcolormesh(data, cmap='OrRd')
  heatmap = ax.pcolormesh(data, cmap=colmap)
  fig.colorbar(heatmap)

  # # put the major ticks at the middle of each cell
  ax.set_xticks(np.arange(len(rows))+.5)
  ax.set_yticks(np.arange(len(cols))+.5)
  vmin = 0
  if vmax is None:
    vmax = 1. if np.max(data) <= 1. else np.max(data)
      
  ax.set_xticklabels(rows, rotation='vertical', fontsize=8)
  ax.set_yticklabels(cols, fontsize=8)
  ax.set_xlim(0, len(rows))
  ax.set_ylim(0, len(cols))
  if xlabel is None:
    plt.xlabel("SOURCE")
  else:
    plt.xlabel(xlabel)

  if xlabel is None:
    plt.ylabel("DESTINATION")
  else:
    plt.ylabel(ylabel)

  fig.suptitle('Heatmap: '+title)
  fig.set_size_inches(16,12)
  plt.tight_layout()
  plt.savefig(SAVELOC + '/heatmap_' + title + '.png')
  plt.close()  
  plt.show()

def heatmap_simple(data, title, labels=None, fname=None, nsscale=1, vmax=None, xlabel=None, ylabel=None, vlines=[], colmap='gnuplot2_r'):
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  vmin = min(0, np.min(data))
  if vmax is None:
    vmax = np.max(data)

  # heatmap = ax.matshow(data, interpolation='nearest')
  # fig.colorbar(heatmap)
  heatmap = ax.pcolor(data, cmap=colmap, vmin=vmin, vmax=vmax)
  # heatmap = ax.pcolormesh(data, cmap='OrRd')
  # heatmap = ax.pcolormesh(data, cmap=colmap)
  fig.colorbar(heatmap)

  # Lcolor = ['k', 'grey', 'r', 'g', 'b']
  # if labels is None:
  #   print("LABEL MARKGIN IS ON")
  #   return
  # for i, L in enumerate(labels):
  #   plt.scatter(i, 118, color=Lcolor[L], size=2)

  for v in vlines:
    plt.axvline(v, color='k', linewidth=1)
    plt.axhline(v, color='k', linewidth=1)
     
  if xlabel is not None:
    plt.xlabel(xlabel)

  if xlabel is not None:
    plt.ylabel(ylabel)

  ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*nsscale))
  ax.xaxis.set_major_formatter(ticks)
  ax.yaxis.set_major_formatter(ticks)

  fig.suptitle('Heatmap: '+title)
  # fig.set_size_inches(16,12)
  plt.tight_layout()
  if fname is None:
    fname = title
  plt.savefig(SAVELOC + '/heatmap_' + fname + '.png')
  plt.close()  
  plt.show()


def rmsd_matrix(data, labels, title, fname=None, nsscale=1, vmax=None, xlabel=None, ylabel=None, colmap='gnuplot2_r'):
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  vmin = min(0, np.min(data))
  if vmax is None:
    vmax = np.max(data)

  heatmap = ax.pcolor(data, cmap=colmap, vmin=vmin, vmax=vmax)
  fig.colorbar(heatmap)

  offset = len(data)/5

  for x, y in enumerate(labels):
    plt.scatter(x, offset*y+(offset/2), color='k')

  for i in range(5):
    plt.annotate('State %d' % i, xy=(10, 10+(offset/2)+offset*i))


  if xlabel is not None:    plt.xlabel(xlabel)
  if xlabel is not None:    plt.ylabel(ylabel)

  plt.xlim(0, len(data))
  plt.ylim(0, len(data))

  ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*nsscale))
  ax.xaxis.set_major_formatter(ticks)
  ax.yaxis.set_major_formatter(ticks)

  fig.suptitle('DEShaw: '+title)
  # fig.set_size_inches(16,12)
  plt.tight_layout()
  if fname is None:
    fname = title
  plt.savefig(SAVELOC + '/heatmap_' + fname + '.png')
  plt.close()  
  plt.show()




def heatmap_board(data, colmap='gnuplot2_r', vmax=None, **kwargs): # title, labels=None, fname=None, nsscale=1, vmax=None, xlabel=None, ylabel=None, vlines=[], ):
  prep_graph()
  fig, ax = plt.subplots()
  vmin = min(0, np.min(data))
  if vmax is None:
    vmax = np.max(data)

  # heatmap = ax.matshow(data, interpolation='nearest', cmap=colmap)
  heatmap = ax.pcolor(data, cmap=colmap, vmin=vmin, vmax=vmax)
  fig.colorbar(heatmap)

  plt.xticks(range(len(data)))
  plt.yticks(range(len(data)))

  plt.tight_layout()
  graph_args(kwargs)



def heatmap_bar(data, title, vmax=None, xlabel=None, ylabel=None, colmap='gnuplot2_r'):
  rows = [5, 10, 25, 50, 75, 100, 200]
  SAVELOC = os.path.join(os.getenv('HOME'), 'ddc', 'graph')
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots()
  heatmap = ax.pcolormesh(np.asarray(data), cmap=colmap, vmax=vmax)
  fig.colorbar(heatmap)
  ax.set_yticklabels(rows)
  ax.set_yticks(np.arange(len(rows))+.5)
  fig.set_size_inches(16,2)
  plt.tight_layout()
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

