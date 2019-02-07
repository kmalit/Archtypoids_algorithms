#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:28:02 2019

@author: keldine
"""
##############################################################################
# ARCHTYPOID ANALYSIS
##############################################################################
##############################################################################
# Library and helper functions import
##############################################################################
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

##############################################################################
# Generate Data

X = np.random.rand(300,2)
###############################################################################
# Furthest Sum
###############################################################################

def furthest_sum(X,size):
    n = X.shape[0]
    idx = []
    # INITIALIOZATION
    d = np.linalg.norm( X[0] - X, axis=1 ) # compute distances to all other points # <-  norm instead of dist
    l = d.argmax() # pick index of furthest point
    d = np.linalg.norm( X[l] - X, axis=1 ) # compute distances to all other points
    i = d.argmax() # pick index of furthest point
    idx.append(i)
    
    # create pool
    pool = list(range(n))
    pool.remove(i)
    
    while len(idx) < size:
        d = []
        for j in pool:
            # compute sum of distances to all chosen points
            d.append( np.linalg.norm( X[idx] - X[j], axis=1 ).sum() )
        # pick index of furthest point
        i = pool[ np.array(d).argmax() ]
        pool.remove(i)
        idx.append(i)
    return idx 
###############################################################################
# Get the Frame of the data
##############################################################################
def frame( X, M=1000.0 ):
    np.random.seed(10)
    # initialization
    n = X.shape[0]
    q = np.array([],dtype=np.int64)
    Q = np.vstack( ( X.T, M * np.ones(n) ) )
    for i in tqdm(range(n)):
        a, rnorm = nnls( Q, Q.T[i] )
        ind = np.where( a > 0 )[0]
        q = np.union1d(q,ind)
    return q
   
##############################################################################
# Generate Archtypoids
##############################################################################
def ADA( X, K, iteration = 10):
    N = X.shape[0]
    # BUILD PHASE - initialize a
    old_archtypoids = furthest_sum(X,K)
    curr_archtypoids = []
    # compute the alphas for the initial archtypoids
    alphas = np.random.rand(N,K)
    for n in range(N):
        alphas[n] = nnls(X[old_archtypoids].T,X[n])[0]
    # Compute RSS for initial archtypoids
    RSS_old = sqrt(mean_squared_error(X,np.dot(alphas,X[old_archtypoids])))
    #itertn = 0
    # SWAP PHASE - optimize RSS
    while (old_archtypoids != curr_archtypoids):
        #print (itertn)
        #print ("RSS",RSS_old)
        print("Archtypoids",old_archtypoids)
        # update old archtypoids
        if len(curr_archtypoids) != 0:
            old_archtypoids = curr_archtypoids
        
        # Update archtypoids to be lowest cost point. 
        for k in range(K):
            candidates  = np.setdiff1d(list(range(N)),old_archtypoids)
            # Loop over all candidate points to replace each for the current archtypoid being
            # considered for replacement
            for c in range(len(candidates)):
                # initiate new alphas
                a = np.random.rand(N,K)
                for n in range(N):
                    # Replace the archtypoid k with candidate c
                    archtypoids = old_archtypoids.copy()
                    archtypoids[k] = candidates[c]
                    # compute new alphas                
                    a[n] = nnls(X[archtypoids].T,X[n])[0]
                    
                # get value of new RSS
                RSS_new = sqrt(mean_squared_error(X,np.dot(a,X[archtypoids])))
                               
                # Test to replace archtypoid k with candidate c
                if (RSS_new <= RSS_old):
                    alphas = a.copy()
                    curr_archtypoids = archtypoids
                    RSS_old = RSS_new
        #itertn =+1        
        #if itertn == iteration:
         #   break
    return X[curr_archtypoids],RSS_old
##############################################################################
# Experiments
##############################################################################
# Compute frame
fram = X[frame(X)]

# compute ADA
a,rss = ADA(X,2)

# plot X
#color= ['red' if l == 0 else 'blue' for l in y]
plt.scatter(X[:,0],X[:,1], color='blue')
plt.title("2 d- data", loc = 'center')

# plot ADA
plt.scatter(X[:,0],X[:,1], color='b', alpha = 0.5)
plt.scatter(a[:,0],a[:,1],color='r')
plt.title("The Archtypoids", loc = 'center')
plt.savefig('ADA.png')

# plot frame
plt.scatter(X[:,0],X[:,1], color='b', alpha = 0.5)
plt.scatter(fram[:,0],fram[:,1],color='r')
plt.title("The frame", loc = 'center')
plt.savefig('Frame.png')
