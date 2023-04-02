"""
This module contains a color-wheel model for the Moore spaces with homology H_1 = Z_r obtained by a
quotient of the unit disk, as well as examples of their
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
# Moore Space RP^2 and M(Z_4,1) Image Patch Model
######################################################
# In[47]:


#We now define a few functions which will help us to sample patches from an image and to plot a collection of patches

def getPatches(I, dim):
    """
    Given an image I, return all of the dim x dim patches in I
    :param I: An M x N image
    :param d: The dimension of the square patches
    :returns P: An (M-d+1)x(N-d+1)x(d^2) array of all patches
    """
    #http://stackoverflow.com/questions/13682604/slicing-a-numpy-image-array-into-blocks
    shape = np.array(I.shape*2)
    strides = np.array(I.strides*2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image'%dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0]*P.shape[1], dim*dim])
    return P

def imscatter(X, P, dim, zoom=1):
    """
    Plot patches in specified locations in R2

    Parameters
    ----------
    X : ndarray (N, 2)
        The positions of each patch in R2
    P : ndarray (N, dim*dim)
        An array of all of the patches
    dim : int
        The dimension of each patch

    """
    #https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    ax = plt.gca()
    for i in range(P.shape[0]):
        patch = np.reshape(P[i, :], (dim, dim))
        x, y = X[i, :]
        im = OffsetImage(patch, zoom=zoom, cmap = 'gray')
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(X)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

def plotPatches(P, zoom = 1):
    """
    Plot patches in a best fitting rectangular grid
    """
    N = P.shape[0]
    d = int(np.sqrt(P.shape[1]))
    dgrid = int(np.ceil(np.sqrt(N)))
    ex = np.arange(dgrid)
    x, y = np.meshgrid(ex, ex)
    X = np.zeros((N, 2))
    X[:, 0] = x.flatten()[0:N]
    X[:, 1] = y.flatten()[0:N]
    imscatter(X, P, d, zoom)
    
#Finally, we add a furthest points subsampling function which will help us to subsample image patches when displaying them

def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getGreedyPerm(X, M, Verbose = False):
    """
    Purpose: Naive O(NM) algorithm to do the greedy permutation
    :param X: Nxd array of Euclidean points
    :param M: Number of points in returned permutation
    :returns: (permutation (N-length array of indices), \
            lambdas (N-length array of insertion radii))
    """
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = getCSM(X[0, :][None, :], X).flatten()
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getCSM(X[idx, :][None, :], X).flatten())
        if Verbose:
            interval = int(0.05*M)
            if i%interval == 0:
                print("Greedy perm %i%s done..."%(int(100.0*i/float(M)), "%"))
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas}

# We now examine the collection of patches which hold oriented, slightly blurry line segments that are varying distances from the center of the patch.
#First, let's start by setting up the patches.
#Below, the "dim" variable sets the patch resolution, and the "sigma" variable sets the blurriness (a larger sigma means blurrier line segments).

def getLinePatchesP(dim, NAngles, NOffsets, sigma):
    """Get vertical line patches rotated through various angles and offsets with a blurriness
    parameter. 

    Args:
        dim (int): pixel size of patch. For ex, dim=10 builds a 10x10 pixel patch
        NAngles (int): number of angles
        NOffsets (int): number of offsets
        sigma (float): distortion parameter for blurry line. sigma = 0.25 is kind of a default

    Returns:
        arr: array of points in R^(dim*dim) representing the patches 
    """
    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    ps = np.linspace(-1, 1, NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            P[idx, :] = patch.flatten()
            idx += 1
    return P

def getLinePatchesQ(dim, NAngles, NOffsets, sigma):
    """Get horizontal line patches rotated through various angles and offsets with a blurriness
    parameter. 

    Args:
        dim (int): pixel size of patch. For ex, dim=10 builds a 10x10 pixel patch
        NAngles (int): number of angles
        NOffsets (int): number of offsets
        sigma (float): distortion parameter for blurry line. sigma = 0.25 is kind of a default

    Returns:
        arr: array of points in R^(dim*dim) representing the patches 
    """
    N = NAngles*NOffsets
    Q = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    ps = np.linspace(-1, 1, NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
    for i in range(NAngles):
        c = np.sin(thetas[i])
        s = np.cos(thetas[i])
        for j in range(NOffsets):
            patch = -X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            Q[idx, :] = patch.flatten()
            idx += 1
    return Q


# In[48]:


RP2 = True   ## Toggle for RP2 or MZ4. (False builds MZ4 instead of RP2)

m=20   ## Number of angles and offsets
r=4   ## Torsion order for M(Z_r,1)

n_landmarks = 300

KNN = False   ## Toggle if you want to compute the geodesic distance or the Euclidean distance

P = getLinePatchesP(dim=10, NAngles = m, NOffsets = m, sigma=0.25)

Q = getLinePatchesQ(dim=10, NAngles = m, NOffsets = m, sigma=0.25)

if RP2:
    # Only take vertical patches for RP^2
    R = P
else:
    # Take both vertical and horizontal patches and overlap them for M(Z_4,1)
    R = np.maximum(P,Q)

    
## Plot patches (slows down code significantly for data sets with >50 angles and offsets)
plt.figure(figsize=(8, 8))
plotPatches(R, zoom=1.5)  ## Adjust the zoom if you use >40 angles and offsets
ax = plt.gca()
ax.set_facecolor((0.7, 0.7, 0.7))
plt.show()

t0 = time.time()
k = 13 # parameter for KNN

dm = sp.spatial.distance.cdist(R,R, metric='euclidean') #start with euclidean distance

#Run KNN to compute geodesic distances
if KNN:
    knn_graph = np.copy(dm)

    for i in range(knn_graph.shape[0]):
        ind_remove = np.argsort(dm[i,:])[k+1:]
        knn_graph[i, ind_remove] = np.inf
        knn_graph[i,i] = 0

    knn_graph = np.minimum(knn_graph, knn_graph.T)

    # Shortest path distance:
    graph = csr_matrix(knn_graph)

    dm_RR = floyd_warshall(csgraph=graph, directed=False)
else:
    dm_RR = dm
    
t1 = time.time()-t0
print("Geodesic Distance: %s" % t1)

# Persistence on landmark set
t0 = time.time()

sub_ind = minmax_subsample_distance_matrix(dm_RR, n_landmarks)['indices']

dm_X = dm_RR[sub_ind, :]

dm_L = dm_X[:,sub_ind]

result2 = ripser(dm_L, coeff=2, do_cocycles=True, maxdim=2, distance_matrix=True)
dgmsz2 = result2['dgms']
dgmsz3 = ripser(dm_L, coeff=3, maxdim=2, distance_matrix=True)['dgms']
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


#check we get the desired topology
cocycles = result2['cocycles'] 

H_1 = cocycles[1]
H_1_diagram = dgmsz2[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort() # index of the largest bar in PH^1(R(L); Z_q)

a = H_1_diagram[H_1_persistence_sort_ind[-1], 0] # Birth of the largest class in PH^1(R(L); Z_q)
b = H_1_diagram[H_1_persistence_sort_ind[-1], 1] # Death of the largest class in PH^1(R(L); Z_q)

my_eta = H_1[H_1_persistence_sort_ind[-1]] # Cochain representtive of the largest class in PH^1(R(L); Z_q)

epsilon = a + 0.01 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print("Birth = %s, Death = %s" % (a, b))


