from collections import deque
import numpy as np
import sys
# import redis
import json

#Init
distro_DEshaw = np.array([
       [ 241.,   12.,    7.,    0.,    0.],
       [  14.,   79.,    0.,    0.,    2.],
       [   5.,    4.,   32.,    1.,    0.],
       [   2.,    0.,    0.,    8.,    0.],
       [   1.,    1.,    0.,    0.,    4.]])


distro_DEShaw2 = np.array(
       [ 499.,   87.,   14.,    0.,    6.],
       [  54.,  103.,   10.,    0.,    0.],
       [  11.,    8.,   12.,    1.,    0.],
       [   2.,    0.,    0.,    6.,    0.],
       [   3.,    2.,    0.,    0.,    2.]])

distro_long = np.array([
       [ 2530.,   198.,   152.,    36.,     1.],
       [  274.,   300.,    41.,     0.,     0.],
       [  193.,    57.,   413.,    21.,     0.],
       [   88.,     3.,    47.,   180.,     0.],
       [  125.,     5.,     0.,     0.,     0.]])

distro = DEShaw2

p_distro = distro / np.sum(distro)

b=[(i,j) for i in range(5) for j in range(5)]
alpha = .6
beta = .4
last_ub = np.zeros(shape=(5,5))
numLabels = 5 
totalBins = 25

def makeLogisticFunc (maxval, steep, midpt):
  return lambda x: maxval / (1 + np.exp(-steep * (midpt - x)))
skew = lambda x: (np.mean(x) - np.median(x)) / np.std(x)

def newstate():
  percellobs={k: np.zeros(shape=(5,5)) for k in b}
  select = np.zeros(shape=(5,5))
  observe = np.zeros(shape=(5,5))
  launch = np.zeros(shape=(5,5))
  pref = np.zeros(shape=(5,5))
  fatigue = np.zeros(shape=(5,5))
  weight = np.zeros(shape=(5,5))
  trand = {}
  jcqueue = deque()
  state = (jcqueue, fatigue, weight, select, launch, percellobs, observe, pref)
  for i in range(5):
    for j in range(5):
      runsim(state, i, j)
      launch[i][j] += 1
  return state

def checktrans(a, b):
  if a == b:
    res = np.random.choice(range(5), p=p_distro[a]/np.sum(p_distro[a]))
    return res, res
  else:
    # Set prob distro of transitioning to a state (or not) -- perturbed for "transition-dwel"
    spots = [p_distro[a][a]**2, 4*p_distro[a][b], p_distro[b][b]**2]
    res = np.random.choice(range(3), p=spots/np.sum(spots))
    if res == 0:
      return a, a
    if res == 1:
      return a, b
    return b, b

def runsim(state, a, b):
  jcqueue, fatigue, weight, select, launch, percellobs, observe, pref = state
  a_last = a
  b_last = b
  for t in range(4):
    a2, b2 = checktrans(a_last, b_last)
    percellobs[(a,b)][a2][b2] += 1
    observe[a2][b2] += 1
    a_last = a2
    b_last = b2
  state = (jcqueue, fatigue, weight, select, launch, percellobs, observe, pref)

def showd(l):
  for k, v in sorted(l.items()):
    print(k, v)

def show(state):
  jcqueue, fatigue, weight, select, launch, percellobs, observe, pref = state
  print('\nCurrent State of Matrices:')
  print(  'BIN      WGT  PREF   FAT       SEL  LAUNCH   OBS  Actual') ;           
  for i in range(5):
    for j in range(5):  
      print('%s  %.3f  %.3f  %.3f  %5d  %5d  %5d  %5d' % (str((i,j)), 
        weight[i][j], pref[i][j], fatigue[i][j], select[i][j], launch[i][j], observe[i][j], distro[i][j]))
  print('%s  %4.3f  %4.3f  %4.3f  %5d  %5d  %5d  %5d' % ('TOTAL:', 
    np.mean(weight), np.mean(pref), np.mean(fatigue), np.sum(select), np.sum(launch), np.sum(observe), np.sum(distro)))
  print ('Next 5 jobs:')
  for i in range(5):
    if len(jcqueue) == 0:
      break
    print ('  %s'% str(jcqueue[i]))

def go(state, num=1):
    jcqueue, fatigue, weight, select, launch, percellobs, observe, pref = state
    newqueue = deque()
    timestep = 0
    convergeList = []
    ep = 1
    print('\nConvergence Tracking:')
    print('\n   TS    C-SCORE   totalC     C-Delta')
    last = 0.
    for epoch in range(num):
      for turn in range(10):
        timestep += 1

        #  Weight Calc
        # Adjust for common event outliers
        rareObserve = np.choose(np.where(observe.flatten() < np.mean(observe)), observe.flatten())
        midptObs = np.mean(observe) * skew(observe) + np.median(observe)
        prefFunc = makeLogisticFunc(1, 1 / np.std(rareObserve), midptObs)
        quota = np.sum(launch) / totalBins
        for i in range(5):
          for j in range(5):
            pref[i][j] = prefFunc(observe[i][j] - midptObs)
            fatigue[i][j] = min((launch[i][j] / quota**2), 1)
            weight[i][j] =  alpha * pref[i][j] + beta * (1 - fatigue[i][j])
        wghtlist = {(i, j): weight[i][j] for i in range(5) for j in range(5)}
        newjobs_sorted = sorted(wghtlist.items(), key=lambda x: x[1], reverse=True)[:15]
        newjobs = deque([(k,v,timestep) for k, v in newjobs_sorted])

        #  Scheduling
        oldjob = None if len(jcqueue) == 0 else jcqueue.popleft()
        newjob = newjobs.popleft()
        newqueue.clear()
        while True:
          if oldjob is None and newjob is None:
            break
          if newjob is None:
            newqueue.append(oldjob)
            oldjob = None if len(jcqueue) == 0 else jcqueue.popleft()
          elif oldjob is None:
            newqueue.append(newjob)
            x, y = newjob[0]
            select[x][y] += 1
            newjob = None if len(newjobs) == 0 else newjobs.popleft()
          else:
            if newjob[1] > oldjob[1] * (oldjob[2] / timestep):
              newqueue.append(newjob)
              x, y = newjob[0]
              select[x][y] += 1
              newjob = None if len(newjobs) == 0 else newjobs.popleft()
            else:
              newqueue.append(oldjob)
              oldjob = None if len(jcqueue) == 0 else jcqueue.popleft()
          if len(newqueue) == 100:
            break
        jcqueue.clear()
        while len(newqueue) > 0:
          jcqueue.append(newqueue.popleft())

        #  Run 4 new simulations
        for k in range(4):
          if len(jcqueue) == 0:
            break
          job, wgt, ts = jcqueue.popleft()
          runsim(state, *job)
          launch[job[0]][job[1]] += 1

      # Convergence Calc
      totalObservations = np.sum(observe)
      totalLaunches = np.sum(launch)
      f = fanout = totalObservations / totalLaunches

      convergeScore = 0
      warr = []
      tarr = []
      harr = []
      for i in range(5):
        for j in range(5):
          l = launch[i][j]
          o = observe[i][j]
          reward = (f * l) - o
          ratio = f * l / (max(o, f * l))
          convergeScore += reward * ratio
          harr.append(o/(f*l) if l > 0 else 0)
          warr.append(reward)
          tarr.append(ratio)
          # flij = fanout * launch[i][j]
          # convergeScore += (flij* (flij - observe[i][j])) / max(observe[i][j]**2, 1)

      # r.rpush('simsim_convergelist', json.dumps({'timestep': timestep, 'converge': convergeScore}))

      delta = convergeScore - last
      print('%4d:  %9.2f   %8.2f  |  %.2f  %.2f' % 
        (timestep, convergeScore, delta, np.mean(harr), np.std(harr)))
      last = convergeScore
    state = (jcqueue, fatigue, weight, select, launch, percellobs, observe, pref)

  # show(state)
  # return state


if __name__ == '__main__':
  num = 1 if len(sys.argv) == 1 else int(sys.argv[1])
  state = newstate()
  go(state, num)
  show(state)
  # state = go(state)




#OLD
# biasdistro = {k: v/np.sum(v) for k,v in percellobs.items()}
# unbias = np.zeros(shape=(5,5))
# for k in biasdistro:
#   unbias = np.add(unbias, biasdistro[k])
# unbias  = unbias / 25
# convergence = 1 - (np.sum(abs(unbias - last_ub)))


# clist = [json.loads(score.decode()) for score in r.lrange('convergelist', 0, -1)]
# for i in sorted(clist, key=lambda x: x['timestep']):
#   print (i)
