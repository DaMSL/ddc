import numpy as np
import numpy.linalg as LA
from sklearn.cluster import KMeans
import datetime as dt
import random

def KM_std(data, N):
  kmeans = KMeans(8, )

######
##  DATA SCIENCE LAB
# ref: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], LA.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    st = dt.datetime.now()
    # Initialize to K random centers
    oldmu = random.sample(X.tolist(), K)
    mu = random.sample(X.tolist(), K)
    print('Finding Centers... k=', K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    ed = dt.datetime.now()
    print('KMeans time = %5.1f' % (ed-st).total_seconds())
    return(mu, clusters)

def classify(data, centroids):
    labels = []
    for pt in data:
        distances = np.array([LA.norm(pt-c) for c in centroids])
        proximity = np.argsort(distances)
        labels.append(proximity[0])
    return labels

# https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def gap(X, maxk, B=10):
    bbox = list(zip(np.min(X, axis=0), np.max(X, axis=0)))
    # Dispersion for real distribution
    ks = range(2,maxk)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    centroids = []
    for indk, k in enumerate(ks):
        print("Calculating GAP for %d clusters" % k)
        mu, clusters = find_centers(X,k)
        centroids.append(mu)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([np.random.uniform(bb[0], bb[1]) for bb in bbox])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(centroids, ks, Wks, Wkbs, sk)

# cent, ks, Wkd, Wkbs, sk = gap(ptrain, 9)

# ks, logWks, logWkbs, sk = gap_statistic(X)

#######

# Alternate heurstic for K determination
# ref: https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
# ref: http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf

def fK(X, K, prev_s=0):
    Nd = len(X[0])
    a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
    mu, clusters = find_centers(X, K)
    Sk = sum([np.linalg.norm(mu[i]-c)**2 \
             for i in range(K) for c in clusters[i]])
    if K == 1 or prev_s == 0:
        fs = 1
    else:
        fs = Sk/(a(K,Nd)*prev_s)
    return fs, Sk


def scoreK(X, maxK):
    last_s = 0
    score = [1]     # 0-K & 1-K Clusters will always yield a 1
    SK = [0]
    for k in range(1, maxK):
      f, s = fK(X, k, SK[-1])
      score.append(f)
      SK.append(s)
      print(f, s)
    return score