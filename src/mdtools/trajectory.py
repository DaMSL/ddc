import numpy as np
import numpy.linalg as LA
import mdtraj as md
  


def rms_delta(X):
  A = [0]
  for i in range(1, len(X)-1):
    A.append(np.abs(X[i+1] - X[i-1]))
  A.append(0)
  return np.array(A)

  

def bin_label_25(L):
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  if A_amt < .75:
    B = A2
  elif L[0] != L[-1] and (L[0] in [A, A2] and L[-1] in [A, A2]):
    A, B = L[0], L[-1]
  else:
    B = A
  return(A,B)


def bin_label_10(L, theta=0.9):
  ''' Label Heuristic:  either most of L is in 1 state (above support theta) or 
  first and last are different (with larger mininal support theta)'''
  count = np.bincount(L, minlength=5)
  A, A2 = np.argsort(count)[::-1][:2]
  A_amt = count[A] / len(L)
  if A_amt < theta or (L[0] != L[-1]):
    return 'T%d' % A
  else:
    return 'W%d' % A



class SemiConform(object):
  glid = 0
  def __init__(self, traj, ref):
    L = len(traj)
    self.id = SemiConform.glid
    self.length = L
    rms = [rmsd(a, ref) for a in traj]
    s_rms_arg = np.argsort(rms)
    self.rms = np.median(rms)
    self.xyz = traj[s_rms_arg[L//2]]
    self.next = None
    self.prev = None
    self.label = 0
    SemiConform.glid += 1


class Conform(object):
  glid = 0
  def __init__(self, length, rms, dih, label):
    self.id = Conform.glid
    self.length = length
    self.rms = rms
    self.dih = dih
    self.label = label
    self.next = None
    self.prev = None
    Conform.glid += 1



 
def collect_ministable(traj, theta=.1):
  fnum = 0
  windows=[]
  min_size=8
  N = len(traj)
  NS = np.sqrt(len(traj))
  while fnum < N:
    ref = fnum
    fnum += 1
    while True and fnum < N:
      dens = LA.norm(traj[fnum])
      delta = LA.norm(traj[ref]-traj[fnum])/NS 
      dlast = LA.norm(traj[fnum]-traj[fnum-1])/NS 
      # print('%2d  %6.4f  %6.4f  %6.4f'%(fnum, delta, dlast, dens))
      fnum += 1
      if delta > theta:
        break
    if fnum - ref > min_size:
      windows.append((ref, fnum))
  return windows


def rmsd(A, B):
  if A.shape != B.shape:
    print("ERROR!")
    return -1
  N = len(A)
  return (LA.norm(A - B) / np.sqrt(N))   

def myrms(A, B):
  if A.shape != B.shape:
    print("ERROR!")
    return -1
  d = 0
  for a, b in zip(A,B):
    d += ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
  return np.sqrt(d/len(A))

def mdrmsd(A, B):
  if A.shape != B.shape:
    print("ERROR!")
    return -1
  d = 0
  for a, b in zip(A,B):
    d += (LA.norm(a-b))
  return np.sqrt(d/len(A))


def mdrmsd2(A, B):
  if A.shape != B.shape:
    print("ERROR!")
    return -1
  d = 0
  for a, b in zip(A,B):
    d += (LA.norm(a-b))
  return d/np.sqrt(len(A))


def rmsd_matrix(A):
  S = A.xyz if isinstance(A, md.Trajectory) else A
  N = len(S)
  M = np.zeros(shape=(N, N))
  for i in range(0, N-1):
    for j in range(i+1, N):
      M[i][j] = M[j][i] = rmsd(S[i],S[j])
  return M

def rmsd_matrix(A):
  S = A.xyz if isinstance(A, md.Trajectory) else A
  N = len(S)
  M = np.zeros(shape=(N, N))
  for i in range(0, N-1):
    for j in range(i+1, N):
      M[i][j] = M[j][i] = rmsd(S[i],S[j])
  return M



def rmsf(A, ref=None):
  S = A.xyz if isinstance(A, md.Trajectory) else A
  N, Z, _ = S.shape
  M = np.zeros(shape=(N, Z))
  if ref is None:
    ref = S[0] #np.mean(A, axis=0)
    start = 1
  else:
    start = 0
  for i in range(start, N):
    M[i] = LA.norm(S[i] - ref, axis=1)
  return M


def rmsf_agg(A, ref=None):
  suma = np.zeros(A.shape[1])
  if ref is None:
    ref = A[0] #np.mean(A, axis=0)
    start = 1
  else:
    start = 0
  for frame in A[start:]:
    suma += np.square(LA.norm(frame - ref, axis=1))
  return np.sqrt(suma/len(A))

def rmsf_mean(A):
  ref = np.mean(A, axis=0)
  N = len(A)
  Z = len(A[0])
  M = np.zeros(shape=(N, Z))
  for i in range(0, N):
    M[i] = LA.norm(A[i] - ref, axis=1)
  return M




def check(traj, frame_ref=None, wsize=5):
  last = traj[1]
  last2 = traj[0]
  if frame_ref is None:
    frame_ref = traj[0]
  res = []
  NS = np.sqrt(len(traj))
  for f in range(len(traj-5)):
    d = [LA.norm(traj[f]-frame_ref)/NS]
    fluct = [LA.norm(traj[f]-traj[f-i])/NS for i in range(1, wsize+1)]
    d.extend(fluct)
    d.append(np.std(fluct))
    d.append(LA.norm(traj[f]))
    res.append(tuple(d))
  return res

 

def rms_window(A, theta=.5):
  N = len(A)
  start = 0
  t = 2
  window_list = []
  delta_list = []
  M = np.ones(shape=(N, N))
  for i in range(N):
    M[i][i] = 0
  M[0][1] = M[1][0] = rmsd(A[0], A[1])
  while t < N:
    for i in range(start, t):
      M[t][i] = M[i][t] = rmsd(A[i], A[t])
    delta = M[t][start:t-1] - M[t-1][start:t-1]
    delta2 = M[t][start:t-2] - M[t-2][start:t-2]
    score1 = np.sum(delta)
    score2 = np.sum(delta2)

    # if t % 1 ==0:
    #   print('%4d  %4d   %5.2f %5.2f   %5.2f %5.2f    %5.2f     %5.2f  %5.2f  ' % 
    #     (start, t, np.sum(delta), np.sum(np.abs(delta)), np.sum(delta2), np.sum(np.abs(delta2)), (np.sum(delta)+np.sum(delta2)),
    #       score1,  score2))
          # np.mean(np.abs(delta)),           M[0][t], np.mean(M[start][start:t-1]), pair_diff))
    if score2 > theta:
      window_list.append((start, t-1))
      start = t
    t += 1
  window_list.append((start, t-1))
  return window_list, M




def rms_window_mean(A, theta=.5, min_length=10):
  N = len(A)
  start = 0
  t = 2
  window_list = []
  runsum = np.sum(A[:2], axis=0)
  while t < N:
    mean_rms = runsum/(t-start)
    rms_hi = rmsd(A[t], mean_rms)
    rms_lo = rmsd(A[start], mean_rms)
    if rms_lo > theta:
      runsum -= A[start]
      start += 1
      continue
    elif rms_hi < theta:
      runsum += A[t]
      t += 1
      continue
    elif t-start >= min_length:
      window_list.append((start, t-1))
    start = t
    runsum = np.sum(A[t:t+2], axis=0)
    t += 2
  if t-start >= min_length:
    window_list.append((start, t-1))
  return window_list



def spatio_temp_cluster(A, theta=.5, min_length=10):
  N = len(A)
  M = rmsd_matrix(A)
  start = 0
  t = 2
  window_list = []
  while t < N:
    rms_in = np.mean(M)
    rms_hi = rmsd(A[t], mean_rms)
    rms_lo = rmsd(A[start], mean_rms)
    if rms_lo > theta:
      runsum -= A[start]
      start += 1
      continue
    elif rms_hi < theta:
      runsum += A[t]
      t += 1
      continue
    elif t-start >= min_length:
      window_list.append((start, t-1))
    start = t
    runsum = np.sum(A[t:t+2], axis=0)
    t += 2
  if t-start >= min_length:
    window_list.append((start, t-1))
  return window_list



def read_log(fname):
  data = [0]
  with open(fname) as src:
    for line in src.readlines():
      if line.startswith('Output'):
        elm = line.split()
        if elm[6][:-1].isdigit():
          data.append(int(elm[6][:-1]))
        else:
          print("BAD Format in log file: ", line)
          break
  return data

