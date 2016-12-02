
#### ACTUAL PDF FROM DATA:
>>> pprint(decnt)
{'T0': 0.049003468106589405,
 'T1': 0.007781289784450591,
 'T2': 0.00547653540541727,
 'T3': 0.000779226480530313,
 'T4': 0.007243513762676149,
 'W0': 0.5170881074674042,
 'W1': 0.24087975767153957,
 'W2': 0.12718951666008166,
 'W3': 0.018536810219939418,
 'W4': 0.026021774441371437}
 
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

Kr = [2, 52, 56, 60, 116, 258, 311, 460, 505, 507, 547, 595, 640, 642, 665, 683, 728, 767, 851, 1244, 1485, 1629, 1636]



Km = [2, 3, 4, 23, 26, 52, 53, 54, 55, 56, 60, 79, 108, 109, 110, 111, 112, 116, 151, 164, 170, 171, 204, 217, 218, 224, 240, 241, 258, 272, 291, 292, 293, 294, 310, 311, 342, 343, 344, 360, 361, 380, 391, 392, 393, 394, 408, 409, 410, 411, 412, 430, 441, 451, 453, 457, 458, 459, 460, 461, 462, 479, 480, 499, 500, 501, 504, 505, 506, 507, 508, 509, 510, 527, 528, 529, 546, 547, 548, 549, 550, 551, 552, 553, 554, 574, 593, 594, 595, 596, 597, 598, 599, 600, 620, 621, 638, 639, 640, 641, 642, 643, 665, 682, 683, 684, 685, 687, 709, 725, 726, 728, 729, 752, 767, 768, 770, 771, 811, 851, 889, 1055, 1121, 1122, 1124, 1187, 1215, 1216, 1217, 1240, 1244, 1245, 1246, 1271, 1272, 1273, 1379, 1380, 1381, 1402, 1403, 1424, 1425, 1445, 1484, 1485, 1486, 1544, 1629, 1636, 1637, 1640, 1641, 1642, 1645, 1646, 1649]

sigma = []
deds = []
for i in range(42):
  ds = 10*np.load(home+'/work/results/DE_ds/DE_ds_full_%02d.npy' % i)[:,Kr]
  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    deds.append(ds[a:b])
    cov = np.cov(ds[a:b].T)
    ew,_= LA.eigh(cov)
    sigma.append(np.sum(ew))
  print('Loaded: ', i)

deds = np.array(deds)
sigman = np.array(sigma)
np.save(home+'/work/results/DE_sigma.npy')


Km2 = lat.reduced_feature_set2(DS, 8, .001, 400)

sigma_km2 = []
for i in range(42):
  ds = 10*np.load(home+'/work/results/DE_ds/DE_ds_full_%02d.npy' % i)
  for a,b in TS.TimeScape.windows(home+'/work/timescape/desh_%02d_transitions.log'%i):
    cov = np.cov(ds[a:b][:,Km2].T)
    ew,_= LA.eigh(cov)
    sigma_km2.append(np.sum(ew))
  print('Processed', i)

mina, minb, minc = min(sigma[:80]), min(sigma_km2[:80]), min(sigma_full[:80])
for a, b, c in zip(sigma[:80], sigma_km2, sigma_full): 
  print('%7.2f   %7.2f   %7.2f' % (a/mina,b/minb,c/minc))


iswell = lambda x: (dL[x] == dL[x][0]).sum() + 1 >= len(dL[x])
getstate  = lambda x: int(np.median(dL[x])) if iswell(x) else 6

np.bincount([getstate(i) for i in clulist[n]], minlength=6)

data = []
for log in loglist:
  with open(log) as logfile:
    src = logfile.read().strip().split('\n')
    for line in src:
      if line.startswith('SCORE'):
        elm = line.split(',')
        data.append(elm[1:])


for x in data:
  _=db.runquery("INSERT INTO latt VALUES ('%s',%s,%s,'%s',%s);"%tuple(x))


SPT = [i[0] for i in db.runquery('select distinct support from latt order by support')]
NC  = [i[0] for i in db.runquery('select distinct numclu from latt order by numclu')]

baseT = np.sum([v for k,v in base.items() if k[0]=='T'])
transition = {k: np.array([i[0] for i in db.runquery("select sum(score) from latt where wt='T' and bin like 'T%%' and support=%d group by numclu order by numclu desc"%k)]) for k in SPT}
reltrans   = {k: np.array([i[0]/baseT for i in db.runquery("select sum(score) from latt where wt='T' and bin like 'T%%' and support=%d group by numclu order by numclu desc"%k)]) for k in SPT}

well4 = {k: np.array([i[0] for i in db.runquery("select score from latt where wt='W' and bin like '%%4' and support=%d group by numclu order by numclu desc"%k)])/(base['W4']+base['T4']) for k in SPT}
well3 = {k: np.array([i[0] for i in db.runquery("select score from latt where wt='W' and bin like '%%3' and support=%d group by numclu order by numclu desc"%k)])/(base['W3']+base['T3']) for k in SPT}
well2 = {k: np.array([i[0] for i in db.runquery("select score from latt where wt='W' and bin like '%%2' and support=%d group by numclu order by numclu desc"%k)])/(base['W2']+base['T2']) for k in SPT}

well23={k: np.array([i[0] for i in db.runquery("select sum(score) from latt where wt='W' and (bin='W2' or bin='W3') and support=%d group by numclu order by numclu desc"%k)])/(base['W2']+base['W3']) for k in SPT}
well34={k: np.array([i[0] for i in db.runquery("select sum(score) from latt where wt='W' and (bin='W4' or bin='W3') and support=%d group by numclu order by numclu desc"%k)])/(base['W4']+base['W3']) for k in SPT}
well234={k: np.array([i[0] for i in db.runquery("select sum(score) from latt where wt='W' and (bin='W2' or bin='W3' or bin='W4') and support=%d group by numclu order by numclu desc"%k)])/(base['W2']+base['W3']+base['W4']) for k in SPT}

def printcsv(D):
  for k,v in sorted(D.items()):
    print('%s,%s' % (k, ','.join([str(i) for i in v])))


data = np.array([well4[k] for k in SPT])

data = db.runquery("select numclu, sum(score) as score, stdev(score) as std from latt where wt='T' and bin in ('T1', 'T2', 'T3', 'T4') group by numclu order by numclu desc")

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



P.bargraph_simple(X, E, True, fname='Transitions_by_numclu', title='Sampling Improvement For Targeting Transitions', yticks=['15x', '20x', '25x', '30x', '35x'], ylabel='Factor Increase', xlabel='Number of Clusters', ylim=(15,35), latex=True)



data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='W' and bin in ('W3', 'W4') and support > 450 group by numclu,support) group by numclu order by numclu desc"%state)
data = db.runquery("select numclu, avg(score), avg(std) from (select numclu, support, sum(score) as score, stdev(score) as std from latt where wt='W' and bin in ('W3', 'W4') and support > 450 group by numclu,support) group by numclu order by numclu desc")
