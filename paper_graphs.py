import bench.db as db
import plot as P
import itertools as it
import matplotlib.pyplot as plt


TEXT_WIDTH=4.81
GOLDEN_RATO=1.61803398875

tblc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', \
'#17becf', '#60636A']

STD_HGT = 1.5
# EXP_COLORS = ['goldenrod','red','green','blue','purple']
EXP_COLORS = [tblc[i] for i in [1,3,2,0,4]]
EXP_NAMES  = ['Serial','Parallel','Uniform','Biased','Lattice']
EXP_N  = ['S','P','U','B','L']

STATE_COLORS = [tblc[i] for i in [0,3,10,2,4]]
# STATE_COLORS = ['red', 'blue', 'green', 'purple', 'cyan']


##############
#  SCRAPE LABEL LOG FILE
expname = 'uniform6'; data = []
with open('../results/{0}_prov.log'.format(expname)) as provfile:
  for line in provfile.read().strip().split('\n'):
    if line.startswith('BASIN'):
      _,bid,targ,actual,labelseq = line.split(',')
      data.append((bid,targ,actual,labelseq))


############################################################
#  WIDE HISTOGRAM - Rare Event (all)

data=[[   0. ,  240.7,    0. ,    0. ,    9.3],
[  96. ,   47. ,   49. ,   27. ,   31. ],
[  76.3,   54.8,   52.6,   31.8,   34.5],
[  19.3,   23.7,   72.7,   67.5,   66.8],
[   0. ,   18.8,   75.7,   81.3,   74.3]]
serieslabels=EXP_NAMES
grouplabels=['State %d'%i for i in range(5)]

W, H = TEXT_WIDTH, STD_HGT

imp.reload(P)
P.multibar(data, serieslabels=serieslabels, grouplabels=grouplabels, fname='event_histo', \
  colors=EXP_COLORS, xlabel='Observed State', no_ytick=True, ylim=(0,250), \
  ylabel='Frequency (in ns)', figsize=(W,H), latex=True)



#############################################################
#   HISTO: Transitions


denovo   = [1, 4,   15,  17,  43]
histdata = [3, 12,  19,  68,  42]
fsz = 24
fsize = (.2*TEXT_WIDTH, .2*TEXT_WIDTH)
tex_param = {"axes.labelsize": fsz, "font.size": fsz,     "xtick.labelsize": fsz,   "ytick.labelsize": fsz}
imp.reload(P)
P.bargraph_simple(histdata, labels=EXP_N, fname='tran_hist', no_ytick=True, no_xtick=True, ygrid=True,\
  colors=EXP_COLORS, figsize=fsize, yticks=[0,15,30,45,60], latex=True)
P.bargraph_simple(denovo, labels=EXP_N, fname='tran_denovo', no_ytick=True, no_xtick=True, ygrid=True,\
  colors=EXP_COLORS, figsize=fsize, yticks=[0,10,20,30,40], latex=True)




############################################################
#  RESOURCE BAR CHART - overhead metrics (4 ea)



COL2 = [tblc[i] for i in [0,3]]
COL6 = [tblc[i] for i in [2,4,8,7]]

cost_cpu={
'Simulation': [160.0438889,170.1492,159.53,157.73,161.36],
'Overhead'  : [4.74,  3.6609,  2.12, 1.15, 4.3]}

cost_wc={
'Simulation': [192, 36,  24.2,  24.2,  25.5],
'Overhead': [4.74, 3.660971806, 1.365664351, 1.660819336, 4.393268805]}

# UPDATE NOTE1:  Overlay factored to fewer CPU's (I over-assumed 2xnode wo/shutdown-- actualy ran it as .5 node
# UPDATE NOTE2:  Bias/Uniform data based on 1ns sim and 10xcontrol deci. Lattice upscaled to 10x control dec.
ovrhead = {
'Disk I/O'       : [4.74,  3.6609,  0.158289262, 0.158289262, 0.187514559],
'LocalAnalysis'  : [0,    0,        1.162674839, 1.157407407, 1.216224086],
'GlobalAnalysis' : [0,    0,        0,           0.300069444, 2.05],
'Overlay'        : [0,    0,        0.794167,    0.833333 ,   0.858333]}

# 'Sampler'        : [0, 0, 0.032986111, 0.027430556, 0.028472222],
# 'Network I/O'    : [0, 0, 0.011714139, 0.017622667, 0.052724605],


diskio = [4.74, 3.66, 0.688600289, 0.056680556]
diskio_labels = ['Off', 'Lstr', 'InS', 'Cah']


fsize = (.23*TEXT_WIDTH, .23*TEXT_WIDTH)
datalabels = ['Simulation', 'Overhead']
oplabels = ['Disk I/O', 'LocalAnalysis', 'Overlay', 'GlobalAnalysis']

# tex_param = {"axes.labelsize": fsz, "font.size": fsz,     "xtick.labelsize": fsz,   "ytick.labelsize": fsz}
imp.reload(P)

fsize_a = (.29*TEXT_WIDTH, .23*TEXT_WIDTH)
P.stackbar(cost_wc, EXP_N, datalabels, fname='cost_wc', ylabel='Time (Hrs)', colors=COL2, ylim=(0,200),\
  ygrid=True, figsize=fsize_a,  yticks=[0,100, 200], latex=True)

P.stackbar(cost_cpu, EXP_N, datalabels, fname='cost_cpu', colors=COL2, ylim=(0,200), figsize=fsize,\
  ygrid=True, yticks=[100, 150, 200], latex=True)

P.stackbar(ovrhead, EXP_N, oplabels, fname='cost_ovrhead', colors=COL6, figsize=fsize, no_xtick=True, no_ytick=True,\
  ygrid=True, ylim = (0, 6), yticks=[0,2,4,6], latex=True)

P.bargraph_simple(diskio, labels=diskio_labels, fname='cost_io', colors=COL6[:1],  figsize=fsize,yticks=[0,2,4,6], \
  ygrid=True, no_xtick=True, no_ytick=True, ylim=(0,6), latex=True)

P.cost_legend(datalabels+oplabels, COL2+COL6)



#########################################################
# DENOVO ACTIVITY GRAPH
# Uses labeled data (from RMSD) to denote activity spikes in trajectory output
def activity1(X, Y, Z=50):
  A = []
  base = np.argmax(np.bincount(X))
  for i in range(0, (Z)*(len(X)//(Z)), Z):
    w = X[i:min(len(X),i+Z)]
    cnt = np.bincount(w, minlength=5)[base]
    A.append(Y - cnt/len(w))
  return np.array(A) 

# LOAD Denovo Output
LB={}
LB['Serial'] = np.load(home+'/work/results/serial_labels.npy').reshape(1, 270402)
LB['Parallel'] = np.array([i[:15000] for i in np.load(home+'/work/results/denparallel_labels.npy')])
LB['Biased'] = np.load(home+'/work/results/denbias_labels.npy')
LB['Uniform'] = np.load(home+'/work/results/denuniform_labels.npy')
LB['Lattice'] = np.load(home+'/work/results/denlattice_labels.npy')

# Process and make graph
fsize = (TEXT_WIDTH, STD_HGT)
LBA = {k: [activity1(t, y+2, 50) for t in LB[k]] for y, k in enumerate(EXP_NAMES)}
idxlist = {'Serial': [0], 'Parallel': [2,3,4], 'Biased': [0,3,11,17,23,28,31,32,33,35], 'Uniform':list(range(10)), 'Lattice': [12,16,27,34,41,42,45,46,52,57]}
X = {k: np.array(list(it.chain(*[LBA[k][i] for i in v]))) for k,v in idxlist.items()}; xmax = min([len(v) for v in X.values()])

imp.reload(P)
P.lines({k: v[:xmax] for k,v in X.items()}, showlegend=False, labels=EXP_NAMES[::-1], lw=1, yticks=EXP_NAMES + [' '],  \
  fname='denovo_activity', ylim=(.85, 6.25), figsize=fsize, no_ytick=True, colors=EXP_COLORS[::-1], \
  xlabel='Simulation Time (normalized)', xticks=['%d'%i for i in range(0, 101, 10)], latex=True)

# idxlist = {'Serial': [0], 'Parallel': [1], 'Biased': [0,3,11,17,23,28,31,32,33,35], 'Uniform':[10,16,18,21,23,24,25,28,33,35], 'Lattice': [1,15,18,29,31,32,33,40,41,42]}



 

################
# TRADEOFF

# data,totc = [],[]; expname='biased6'
# with open(home+'/work/results/{0}_prov.log'.format(expname)) as provfile:
#   for line in provfile.read().strip().split('\n'):
#     if line.startswith('BASIN'):
#       _,bid,targ,actual,labelseq = line.split(',')
#       data.append((bid,targ,actual,labelseq))
#       totc.extend(labelseq)

fsize = (.5*TEXT_WIDTH, .4*TEXT_WIDTH)

imp.reload(P)
C = pickle.load(open('c_wc', 'rb'))
V = pickle.load(open('v_wc', 'rb'))
P.lines(V, C, False, labels=EXP_NAMES[::-1], lw=3, fname='tradeoff_wc', ylabel='VALUE: Frequency (in ns)', figsize=fsize,\
  yticks=[0, 0,50,100,150,200,250], xlabel='COST: Time (in hours)', xlim=(0,10), \
  no_xtick=True, no_ytick=True, ylim=(-10000,250000), colors = EXP_COLORS[::-1], latex=True)

C = pickle.load(open('c_cpu', 'rb'))
V = pickle.load(open('v_cpu', 'rb'))
P.lines(V, C, True, labels=EXP_NAMES[::-1], lw=3, fname='tradeoff_cpu', ylabel='VALUE: Frequency (in ns)', figsize=fsize,\
  yticks=[0, 0,50,100,150,200,250], xlabel='COST: Monetary (CPU Hours)', xlim=(0,180),
  no_xtick=True, no_ytick=True, ylim=(-10000,250000), colors = EXP_COLORS[::-1], latex=True)



# pickle.dump(C, open('c_cpu', 'wb'))
# pickle.dump(V, open('v_cpu', 'wb'))
# pickle.dump(X, open('c_wc', 'wb'))
# pickle.dump(Y, open('v_wc', 'wb'))



############################
#  HEATMAPS  -- ACCURACY
# TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]



fsize = (.2*TEXT_WIDTH, .2*TEXT_WIDTH)

# TBIN10 = ['%s%d'%(a,b) for a in ['W', 'T'] for b in range(5)]
TBIN10 = ['W0', 'T0', 'W1', 'T1', 'W2', 'T2', 'W3', 'T3', 'W4', 'T4']
BIN5 = ['State-%d'%i for i in range(5)]
BIN5 = ['%d'%i for i in range(5)]
tidx = {k: i for i, k in enumerate(TBIN10)}

def LABEL10(L, theta=0.9):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  return 'T%d' % A if A_amt < theta or (L[0] != L[-1]) else 'W%d' % A

#  SCRAPE LABEL LOG FILE
for expname in ['biased6', 'uniform6']:
  data = []
  with open('../results/{0}_prov.log'.format(expname)) as provfile:
    for line in provfile.read().strip().split('\n'):
      if line.startswith('BASIN'):
        _,bid,targ,actual,labelseq = line.split(',')
        data.append((bid,targ,actual,labelseq))
  h, w, t = np.identity(10), np.identity(5), np.identity(5)
  for _,a,b,_ in data:
    i, j = tidx[a], tidx[b]
    h[i][j] += 1
    w[int(a[1])][int(b[1])] += 1
    if a[0] == 'T':
      t[int(a[1])][int(b[1])] += 1
  h_norm = (h.T/h.sum()).T
  w_norm = (w.T/w.sum()).T
  t_norm = (t.T/t.sum()).T
  # P.heatmap(np.rot90(h_norm,3).T, TBIN10[::-1], TBIN10, expname+'_accuracy_10bin', ylabel='Start State', xlabel='Output Distribution')
  P.heatmap(np.rot90(w_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_states', figsize=fsize, latex=True)
  # P.heatmap(np.rot90(t_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_trans', ylabel='Start State', xlabel='Output Distribution')


outbin = {k:defaultdict(list) for k in ['all', 'W', 'T']}
exp = 'lattice2'
h, w, t = np.identity(10), np.identity(5), np.identity(5)
for exp in ['lattice2', 'lattrans']:
  lab_all = np.load(home+'/work/results/label_{0}.npy'.format(exp)).astype(int)
  with open(home+'/work/results/{0}/jclist'.format(exp)) as inf: 
    idlist = inf.read().strip().split('\n')
  for i, tid in enumerate(idlist):
    for a,b in TS.TimeScape.windows(home+'/work/jc/{0}/{1}/{1}_transitions.log'.format(exp,tid)):
      outbin['all'][tid].append(LABEL10(lab_all[i][a:b]))
      if exp=='lattice2': 
        outbin['W'][tid].append(LABEL10(lab_all[i][a:b]))
      else:
        outbin['T'][tid].append(LABEL10(lab_all[i][a:b]))

for k,v in outbin['all'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    h[i][j] += 1

for k,v in outbin['W'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    w[int(a[1])][int(b[1])] += 1

for k,v in outbin['T'].items():
  a = v[0]
  for b in v:
    i, j = tidx[a], tidx[b]
    t[int(a[1])][int(b[1])] += 1

expname='lattice'
h_norm = (h.T/h.sum()).T
w_norm = (w.T/w.sum()).T
t_norm = (t.T/t.sum()).T
# P.heatmap(np.rot90(h_norm,3).T, TBIN10[::-1], TBIN10, expname+'_accuracy_10bin', ylabel='Start State', xlabel='Output Distribution')
P.heatmap(np.rot90(w_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_acc_states', figsize=fsize, latex=True)
# P.heatmap(np.rot90(t_norm,3).T, BIN5[::-1], BIN5, fname=expname+'_accuracy_trans', ylabel='Start State', xlabel='Output Distribution')

ylabel='Start State', xlabel='Output Distribution', 



############################
#  HEATMAPS  -- INTRINSICS

import bench.db as db

base={'T0': 0.07874577461697177,
 'T1': 0.009405592870626454,
 'T2': 0.007441064138021862,
 'T3': 0.0009987268975811055,
 'T4': 0.009921418850695817,
 'W0': 0.4873458009570218,
 'W1': 0.2392554545853637,
 'W2': 0.12522498792747705,
 'W3': 0.018317309802888624,
 'W4': 0.02334386935335177}


SPT = [i[0] for i in db.runquery('select distinct support from latt order by support')]
NC  = [i[0] for i in db.runquery('select distinct numclu from latt order by numclu')]

for st in range(5):
  well = {k: np.array([i[0] for i in db.runquery("select score from latt where bin like '%%{0}' and support={1} group by numclu order by numclu desc".format(st,k))])/(base['W%d'%st]+base['T%d'%st]) for k in SPT}
  data = np.array([well[k] for k in SPT])
  P.heatmap(data, SPT, NC[::-1], fname='lattintin_well%d'%st, figsize=fsize, latex=True)

baseT = np.sum([v for k,v in base.items() if k in ['T1', 'T2', 'T3', 'T4']])
baseR = np.sum([v for k,v in base.items() if k in ['W3', 'W4']])

tran = {k: np.array([i[0] for i in db.runquery("select sum(score) from latt where bin in ('T1', 'T2', 'T3', 'T4') and support={0} group by numclu order by numclu desc".format(k))])/baseT for k in SPT}
rare = {k: np.array([i[0] for i in db.runquery("select sum(score) from latt where wt='W' and bin in ('W3', 'W4') and support={1} group by numclu order by numclu desc".format(st,k))])/baseR for k in SPT}
tran[91] *= 6; tran[225] *= 6
rare[91] *= 6; rare[225] *= 6

fsize = (.33*TEXT_WIDTH, .33*TEXT_WIDTH)
fsize_a = (.23*TEXT_WIDTH, .23*TEXT_WIDTH)
fsize_b = (.27*TEXT_WIDTH, .23*TEXT_WIDTH)

data = np.array([rare[k] for k in SPT])
P.heatmap(data, SPT, NC[::-1], showlegend=False, xlabel='# of Clusters', fname='lattintin_rare', \
  yticks=[], xticks=['100', '50', '25', '10', '5'], figsize=fsize_a, latex=True)

data = np.array([tran[k] for k in SPT])
P.heatmap(data, SPT, NC[::-1], showlegend=False, xlabel='# of Clusters', fname='lattintin_tran', yaxisright=True, ylabel='Support Theta', \
  yticks=['200', '2k', '20K'], xticks=['100', '50', '25', '10', '5'], figsize=fsize_b, latex=True)


# data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='T' and bin in ('T1', 'T2', 'T3', 'T4') and numclu < 101 group by numclu,support) group by numclu order by numclu desc")
# tran = {k: np.array([i[0] for i in db.runquery("select score from latt where bin like '%%{0}' and support={1} group by numclu order by numclu desc".format(st,k))])/(base['W%d'%st]+base['T%d'%st]) for k in SPT}


########################################################
# LATTICE INTRINSICS
base={'T0': 0.07874577461697177,
 'T1': 0.009405592870626454,
 'T2': 0.007441064138021862,
 'T3': 0.0009987268975811055,
 'T4': 0.009921418850695817,
 'W0': 0.4873458009570218,
 'W1': 0.2392554545853637,
 'W2': 0.12522498792747705,
 'W3': 0.018317309802888624,
 'W4': 0.02334386935335177}

fsize_a = (.27*TEXT_WIDTH, .24*TEXT_WIDTH)
fsize_b = (.24*TEXT_WIDTH, .24*TEXT_WIDTH)

# GRAPH:  RARE EVENTS by NUMCLU
baseR = sum([v for k,v in base.items() if k in ['W3', 'W4']])
data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='W' and bin in ('W3', 'W4') and support > 450  and numclu < 101 group by numclu,support) group by numclu order by numclu desc")
X, E = {k:s/baseR for k,s,e in data}, {k:e/baseR for k,s,e in data}
P.bargraph_simple(X, revsort=True, fname='RareEvents_by_numclu', ylim=(20,70), figsize=fsize_a,\
  yticks=['20x', '36x', '54x', '70x'], xlim=(0, 11), ylabel='Efficiency', 
  xticks=['100', '50', '25', '10', '5'],\
  ygrid=True, xlabel='# Clusters', latex=True)
for a,b,c in data: print(a,b/baseR,c/baseR)

# GRAPH:  TRANSITIONS by NUMCLU
baseT = sum([v for k,v in base.items() if k in ['T1', 'T2', 'T3', 'T4']])
data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='T' and bin in ('T1', 'T2', 'T3', 'T4') and numclu < 101 group by numclu,support) group by numclu order by numclu desc")
X, E = {k:s/baseT for k,s,e in data}, {k:e/baseT for k,s,e in data}
P.bargraph_simple(X, revsort=True, fname='Transitions_by_numclu', ylim=(20,32), figsize=fsize_b,\
  xticks=['100', '50', '25', '10', '5'], xlim=(0, 11), \
  no_xtick=True, ygrid=True, yticks=['20x', '24x', '28x', '32x'], xlabel='# Clusters', latex=True)
for a,b,c in data: print(a,b/baseT,c/baseT)



##############################
#  1-D Feature Distro

#  see:  cluster_de.py for full data creation
import pickle

kdistr = pickle.load(open('kdistr', 'rb'))
stlist = {str(i):i for i in range(5)}
xticks = [4,5,6,7,8,9,10]
fsize = (.4*TEXT_WIDTH, .30*TEXT_WIDTH)
fsize_a = (.45*TEXT_WIDTH, .30*TEXT_WIDTH)
DOLATEX=True
max_a, max_b = 1.2*np.max(list(kdistr['q'].values())), 1.2*np.max(list(kdistr['p'].values()))
imp.reload(P)
P.kdistr_legend(STATE_COLORS, (.15*TEXT_WIDTH, .3*TEXT_WIDTH))
P.show_distr(kdistr['q'], xscale=(4,10), show_yaxis=True, colors=STATE_COLORS, figsize=fsize_a,\
    xlabel='Measurement (Angstrom)', ylabel='Frequency', fname='distr_a', xticks=xticks, 
    no_xtick=True, no_ytick=True, xlim=(0,42), ylim=(0,max_a),latex=DOLATEX)
P.show_distr(kdistr['p'], xscale=(4,10), show_yaxis=False, colors=STATE_COLORS, figsize=fsize,\
    xlabel='Measurement (Angstrom)', fname='distr_b', xticks=xticks, no_ytick=True, no_xtick=True,
    xlim=(0,42), ylim=(0,max_b), latex=DOLATEX)


P.show_distr(kdistr['m'], xscale=(4,10), showlegend=None, show_yaxis=False, states=stlist, figsize=fsize,\
    fname='distr_c', xticks=xticks, no_yticks=True, latex=DOLATEX)
P.show_distr(kdistr['b'], xscale=(4,10), showlegend=None, show_yaxis=False, states=stlist, figsize=fsize,\
    fname='distr_d', xticks=xticks, no_yticks=True, latex=DOLATEX)




