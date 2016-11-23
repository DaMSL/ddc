import bench.db as db
import plot as P
import itertools as it

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
klist = ['Serial', 'Parallel', 'Uniform', 'Biased', 'Lattice']
LB={}
LB['Serial'] = np.load(home+'/work/results/serial_labels.npy').reshape(1, 270402)
LB['Parallel'] = np.array([i[:15000] for i in np.load(home+'/work/results/denparallel_labels.npy')])
LB['Biased'] = np.load(home+'/work/results/denbias_labels.npy')
LB['Uniform'] = np.load(home+'/work/results/denuniform_labels.npy')
LB['Lattice'] = np.load(home+'/work/results/denlattice_labels.npy')

# Process and make graph
LBA = {k: [activity1(t, y+2, 50) for t in LB[k]] for y, k in enumerate(klist)}
idxlist = {'Serial': [0], 'Parallel': [2,3,4], 'Biased': [0,3,11,17,23,28,31,32,33,35], 'Uniform':list(range(10)), 'Lattice': [12,16,27,34,41,42,45,46,52,57]}
X = {k: np.array(list(it.chain(*[LBA[k][i] for i in v]))) for k,v in idxlist.items()}; xmax = min([len(v) for v in X.values()])
P.lines({k: v[:xmax] for k,v in X.items()}, showlegend=False, yticks=[' '] + klist, xscale=(0,101), \
  title='Transition Activity over Time', fname='denovo_activity', ylim=(.75, 6), figsize=(12,4), \
  xlabel='Time (normalized)', xticks=range(0, 101, 10), latex=True)

# idxlist = {'Serial': [0], 'Parallel': [1], 'Biased': [0,3,11,17,23,28,31,32,33,35], 'Uniform':[10,16,18,21,23,24,25,28,33,35], 'Lattice': [1,15,18,29,31,32,33,40,41,42]}


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

# GRAPH:  TRANSITIONS by NUMCLU
baseT = sum([v for k,v in base.items() if k in ['T1', 'T2', 'T3', 'T4']])
data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='T' and bin in ('T1', 'T2', 'T3', 'T4') group by numclu,support) group by numclu order by numclu desc")
X, E = {k:s/baseT for k,s,e in data}, {k:e/baseT for k,s,e in data}
P.bargraph_simple(X, E, True, fname='Transitions_by_numclu', ylim=(15,32), title='Sampling Improvement: Transitions', yticks=['15x', '20x', '25x', '30x', '35x'], ylabel='Factor Increase', xlabel='Number of Clusters', latex=True)
for a,b,c in data: print(a,b/baseT,c/baseT)

# GRAPH:  RARE EVENTS by NUMCLU
baseR = sum([v for k,v in base.items() if k in ['W3', 'W4']])
data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='W' and bin in ('W3', 'W4') and support > 450 group by numclu,support) group by numclu order by numclu desc")
X, E = {k:s/baseR for k,s,e in data}, {k:e/baseR for k,s,e in data}
P.bargraph_simple(X, E, True, fname='RareEvents_by_numclu', ylim=(20,70), title='Sampling Improvement: Rare Events', yticks=['20x', '30x', '40x', '50x', '60x', '70x'], ylabel='Factor Increase', xlabel='Number of Clusters', latex=True)
for a,b,c in data: print(a,b/baseR,c/baseR)

