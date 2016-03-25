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
    print('Finding Centers...')
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

# FOR 2D 
# TODO: Conver to N-D
def bounding_box_2D(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

def bounding_box_ND(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,10)
    Wks = zeros(len(ks))
    Wkbs = zeros(len(ks))
    sk = zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

# ks, logWks, logWkbs, sk = gap_statistic(X)

#######