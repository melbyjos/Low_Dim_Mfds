"""
This module contains models for the flat Torus and Klein bottle using a quotient metric on the
square fundamental domain of each space, as well as examples of their
persistence computations. 
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from ripser import ripser

from persim import plot_diagrams

import scipy as sp


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

import math

from scipy.spatial.distance import pdist

def minmax_subsample_distance_matrix(X, num_landmarks, seed=[]):
    """
    This function computes minmax subsampling using a square distance matrix.

    :type X: numpy array
    :param X: Square distance matrix

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type seed: list
    :param list: Default []. List of indices to seed the sampling algorith.
    """
    num_points = len(X)

    if not(seed):
        ind_L = [np.random.randint(0,num_points)] 
    else:
        ind_L = seed
        num_landmarks += 1

    distance_to_L = np.min(X[ind_L, :], axis=0)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)

        dist_temp = X[ind_max, :]

        distance_to_L = np.minimum(distance_to_L, dist_temp)
            
    return {'indices':ind_L, 'distance_to_L':distance_to_L}



######################################################
# Torus Flat Planar Model
######################################################

# This one is simple and runs particularly fast. 

def cls_Torus(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in the fundamental domain of T
    '''
    arr = np.array([[x,y],[x,y+1],[x,y-1],[x+1,y],[x+1,y+1],[x+1,y-1],[x-1,y],[x-1,y+1],[x-1,y-1]])
    return arr

def minDist_Torus(X,Y):
    '''
    Returns the geodesic distance on the fundamental domain between X and Y.
    '''
    md = np.min(sp.spatial.distance_matrix(cls_Torus(X[0],X[1]),cls_Torus(Y[0],Y[1])))
    return md

def minDist_TorusLinf(X,Y):
    '''
    Returns the geodesic distance on the fundamental domain between X and Y.
    '''
    md = np.min(sp.spatial.distance_matrix(cls_Torus(X[0],X[1]),cls_Torus(Y[0],Y[1]),p=np.inf))
    return md

# Build sampling of the flat Torus
numx = 20
numy = 20
n_landmarks = 100

xvals = np.linspace(0,1,numx)
yvals = np.linspace(0,1,numy)
x,y = np.meshgrid(xvals,yvals)
xx = x.ravel()
yy = y.ravel()
L = np.column_stack((xx,yy))

plt.scatter(L[:,0],L[:,1])
plt.show()

t0 = time.time()
dm_RR = pdist(L, minDist_Torus)
dm_RR = sp.spatial.distance.squareform(dm_RR)
print(dm_RR.shape)
t1 = time.time()-t0
print("Flat distance: %s sec" % t1)


#Compute persistence on landmarks
t0 = time.time()

sub_ind = minmax_subsample_distance_matrix(dm_RR, n_landmarks)['indices']

dm_X = dm_RR[sub_ind, :]

dm_L = dm_X[:,sub_ind]


result2 = ripser(dm_L, coeff=2, do_cocycles=True, maxdim=2, distance_matrix=True)
dgmsz2 = result2['dgms']
#print(dgmsz2) #####Sort this in reverse
result3 = ripser(dm_L, coeff=3, do_cocycles=True, maxdim=2, distance_matrix=True)
dgmsz3 = result3['dgms']
plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_diagrams(dgmsz2)
plt.title("$\mathbb{Z}/2$")
plt.subplot(122)
plot_diagrams(dgmsz3)
plt.title("$\mathbb{Z}/3$")
plt.show()


t1 = time.time()-t0
print("Persistence: %s seconds" % t1)

cocycles = result2['cocycles'] 

H_1 = cocycles[1]
H_1_diagram = dgmsz2[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)

a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

a2 = H_1_diagram[H_1_persistence_sort_ind[-2], 0] # Birth of the 2nd largest class in PH^1(R(L); Z_q)
b2 = H_1_diagram[H_1_persistence_sort_ind[-2], 1] # Death of the 2nd largest class in PH^1(R(L); Z_q)

my_eta = H_1[H_1_persistence_sort_ind[-1]] # Cochain representtive of the largest class in PH^1(R(L); Z_q)
my_eta2 = H_1[H_1_persistence_sort_ind[-2]]

cocycles3 = result3['cocycles'] 

H_13 = cocycles3[1]
H_1_diagram3 = dgmsz3[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence3 = H_1_diagram3[:,1] - H_1_diagram3[:,0]
H_1_persistence_sort_ind3 = H_1_persistence3.argsort() # index of the largest bar in PH^1(R(L); Z_q)

a3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

a32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 0] # Birth of the 2nd largest class in PH^1(R(L); Z_q)
b32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 1] # Death of the 2nd largest class in PH^1(R(L); Z_q)



epsilon = a * 1.01 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print("Birth = %s, Death = %s" % (a, b))
print("Birth2 = %s, Death2 = %s" % (a2, b2))
print("Birth3 = %s, Death3 = %s" % (a3, b3))
print("Birth32 = %s, Death32 = %s" % (a32, b32))


######################################################
# Klein Bottle Flat Planar Model
######################################################

# This one is simple and runs particularly fast. 


def cls_Klein(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in the fundamental domain of K
    '''
    arr = np.array([[x,y],[x-1,2-y],[x,y+1],[x+1,2-y],[x-1,1-y],[x+1,1-y],[x-1,-y],[x,y-1],[x+1,-y]])
    return arr

def minDist_Klein(X,Y):
    '''
    Returns the geodesic distance on the fundamental domain between X and Y.
    '''
    md = np.min(sp.spatial.distance_matrix(cls_Klein(X[0],X[1]),cls_Klein(Y[0],Y[1])))
    return md


# Build sampling of the flat Klein bottle
numx = 20
numy = 20
n_landmarks = 100

xvals = np.linspace(0,1,numx)
yvals = np.linspace(0,1,numy)
x,y = np.meshgrid(xvals,yvals)
xx = x.ravel()
yy = y.ravel()
L = np.column_stack((xx,yy))

plt.scatter(L[:,0],L[:,1])
plt.show()

t0 = time.time()
dm_RR = pdist(L, minDist_Klein)
dm_RR = sp.spatial.distance.squareform(dm_RR)
print(dm_RR.shape)
t1 = time.time()-t0
print("Flat distance: %s sec" % t1)


#Compute persistence on landmarks
t0 = time.time()

sub_ind = minmax_subsample_distance_matrix(dm_RR, n_landmarks)['indices']

dm_X = dm_RR[sub_ind, :]

dm_L = dm_X[:,sub_ind]


result2 = ripser(dm_L, coeff=2, do_cocycles=True, maxdim=2, distance_matrix=True)
dgmsz2 = result2['dgms']
#print(dgmsz2) #####Sort this in reverse
result3 = ripser(dm_L, coeff=3, do_cocycles=True, maxdim=2, distance_matrix=True)
dgmsz3 = result3['dgms']
plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_diagrams(dgmsz2)
plt.title("$\mathbb{Z}/2$")
plt.subplot(122)
plot_diagrams(dgmsz3)
plt.title("$\mathbb{Z}/3$")
plt.show()


t1 = time.time()-t0
print("Persistence: %s seconds" % t1)

cocycles = result2['cocycles'] 

H_1 = cocycles[1]
H_1_diagram = dgmsz2[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)

a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

a2 = H_1_diagram[H_1_persistence_sort_ind[-2], 0] # Birth of the 2nd largest class in PH^1(R(L); Z_q)
b2 = H_1_diagram[H_1_persistence_sort_ind[-2], 1] # Death of the 2nd largest class in PH^1(R(L); Z_q)

my_eta = H_1[H_1_persistence_sort_ind[-1]] # Cochain representtive of the largest class in PH^1(R(L); Z_q)
my_eta2 = H_1[H_1_persistence_sort_ind[-2]]

cocycles3 = result3['cocycles'] 

H_13 = cocycles3[1]
H_1_diagram3 = dgmsz3[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence3 = H_1_diagram3[:,1] - H_1_diagram3[:,0]
H_1_persistence_sort_ind3 = H_1_persistence3.argsort() # index of the largest bar in PH^1(R(L); Z_q)

a3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

a32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 0] # Birth of the 2nd largest class in PH^1(R(L); Z_q)
b32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 1] # Death of the 2nd largest class in PH^1(R(L); Z_q)



epsilon = a * 1.01 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print("Birth = %s, Death = %s" % (a, b))
print("Birth2 = %s, Death2 = %s" % (a2, b2))
print("Birth3 = %s, Death3 = %s" % (a3, b3))
print("Birth32 = %s, Death32 = %s" % (a32, b32))
