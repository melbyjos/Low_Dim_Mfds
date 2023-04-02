"""
This module contains models for various low-dimensional and image manifolds, as well as examples of their
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


# ########################################
# Moore Space M(Z_r,1) Disk Model
##########################################
# In[43]:


def quotient_distance(X,Y,m):
    '''
    For constructing the distance matrix of the colored disk Moore spaces M(Z_m,1).
    Takes in two data sets X and Y, as well as the integer m corresponding to the Moore space.
    '''
    if X.ndim == 1:
        X = X.reshape((1,len(X)))
    if Y.ndim == 1:
        Y = Y.reshape((1,len(Y)))

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    dimension = X.shape[1]
    
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)

    X_in_S = np.isclose(norm_X, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    X_in_D = np.logical_not(X_in_S)
    
    DX = X[X_in_D,:]
    SX = X[X_in_S,:]

    Y_in_S = np.isclose(norm_Y, 1, rtol=1e-05, atol=1e-08, equal_nan=False)
    Y_in_D = np.logical_not(Y_in_S)
    
    DY = Y[Y_in_D,:]
    SY = Y[Y_in_S,:]

    dist_DX_DY = sp.spatial.distance.cdist(DX,DY, metric='euclidean')

    dist_DX_SY = sp.spatial.distance.cdist(DX,SY, metric='euclidean')

    dist_SX_DY = sp.spatial.distance.cdist(SX,DY, metric='euclidean')    

    dist_SX_SY = sp.spatial.distance.cdist(SX,SY, metric='euclidean')

    Z_q = np.array([[np.cos(2*np.pi/m), -np.sin(2*np.pi/m)], [np.sin(2*np.pi/m), np.cos(2*np.pi/m)]])
    ZQ = np.kron(np.eye( int(dimension/2) ), Z_q)

    ZQ_k = ZQ

    for k in range(1,m):
        dist_DX_SY = np.minimum(dist_DX_SY, sp.spatial.distance.cdist(DX, SY@np.transpose(ZQ_k), metric='euclidean'))
        dist_SX_DY = np.minimum(dist_SX_DY, sp.spatial.distance.cdist(SX, DY@np.transpose(ZQ_k), metric='euclidean'))
        dist_SX_SY = np.minimum(dist_SX_SY, sp.spatial.distance.cdist(SX, SY@np.transpose(ZQ_k), metric='euclidean'))
        ZQ_k = ZQ_k@ZQ

    dist_M = np.concatenate((np.concatenate((dist_DX_DY, dist_DX_SY), axis=1), np.concatenate((dist_SX_DY, dist_SX_SY), axis=1)), axis=0)
    
    dist_M = dist_M.reshape((n_X, n_Y))

    return dist_M

 
def gen_D2(N):
    """This function generates an N-point sampling in D^2 with a slight bias towards the frontier.

    Args:
        N (int): Number of points to sample from the unit disk

    Returns:
        arr: array of points in R^2 within the unit disk 
    """
    theta = np.random.uniform(0, 2*math.pi, N)
    r0 = np.random.uniform(0, 1, int(70*N/100))
    r1 = np.random.uniform(0.7, 1, int(30*N/100))

    r = np.concatenate((r0,r1))

    x = np.multiply( r, np.cos(theta))
    y = np.multiply( r, np.sin(theta))

    x = x.flatten()
    y = y.flatten()

    return np.column_stack((x,y))

def gen_S1(N):
    """Generate an N-point sampling of the unit circle.

    Args:
        N (int): Number of points to sample from the unit circle

    Returns:
        arr: array of points in R^2 within the unit circle
    """
    theta = np.random.uniform(0, 2*math.pi, N)

    x = np.cos(theta)
    y = np.sin(theta)

    x = x.flatten()
    y = y.flatten()

    return np.column_stack((x,y))


# In[44]:


r = 3   ## Torsion order for the Moore space M(Z_r,1) (make sure it is prime)

## Adjust number of data points and/or number of subsampled landmarks. Both parameters have an effect on topology.
Dsamp = 2000   # number of points from the disk
Ssamp = 900  # number of points from the border circle
n_landmarks = 200  #number of landmarks to choose for persistence

D = gen_D2(Dsamp)
S = gen_S1(Ssamp)

# Combine into single point cloud
R = np.concatenate((D,S), axis=0)


#------------------------------------------------------------------------------
# Plot data set X
#------------------------------------------------------------------------------
fig, ax = plt.subplots()

# c = np.array([np.absolute(X[i,0] + 1j*X[i,1]) for i in range(len(X))])
#c = np.array([ X[i,0] for i in range(len(X))])
c = np.array([np.mod( np.angle(R[i,0] + 1j*R[i,1]), 2*np.pi/r ) for i in range(len(R))])

cmap = matplotlib.cm.get_cmap('hsv')
normalize = matplotlib.colors.Normalize(vmin=min(c), vmax=max(c))
colors = np.array([cmap(normalize(value)) for value in c])  # colormap for the Moore Space plot

fig = plt.scatter(R[:,0], R[:,1], color=colors)
plt.show()
plt.close()


# compute distance in the Moore space
dm = quotient_distance(R,R,r)
print(dm.shape)


# Use KNN to compute geodesic distance
knn_graph = np.copy(dm)
k = 3*r+2
for i in range(knn_graph.shape[0]):
    ind_remove = np.argsort(dm[i,:])[k+1:]
    knn_graph[i, ind_remove] = np.inf
    knn_graph[i,i] = 0

knn_graph = np.minimum(knn_graph, knn_graph.T)

# Shortest path distance:
graph = csr_matrix(knn_graph)

dm_RR = floyd_warshall(csgraph=graph, directed=False)


t0 = time.time()

#subsample dist mat for persistence on landmark set
sub_ind = minmax_subsample_distance_matrix(dm_RR, n_landmarks)['indices']

# distances from each data point to the landmark set
dm_X = dm_RR[sub_ind, :]

# dist mat for just landmarks
dm_L = dm_X[:,sub_ind]

#Compute persistence with Z_r coeff and with Z_2 coeff
result2 = ripser(dm_L, coeff=r, do_cocycles=True, maxdim=1, distance_matrix=True)
dgmsz2 = result2['dgms']
dgmsz3 = ripser(dm_L, coeff=2, maxdim=1, distance_matrix=True)['dgms']
plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_diagrams(dgmsz2)
plt.title("$\mathbb{Z}/%s$" % r)
plt.subplot(122)
plot_diagrams(dgmsz3)
plt.title("$\mathbb{Z}/2$")
plt.show()

t1 = time.time()-t0
print("Persistence: %s seconds" % t1)

## Check that we get the desired persistence cocycle
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

# We need to verify PH^1(R(L); Z_q) has a class with persitence long enough if we want Lens coordinates
if not(a<epsilon and 2*epsilon<b):
    print('{}WARNING: The largest class (a,b) in PH^1(R(L); Z_q) is not long enough: 2a is NOT smaller than b.{}'.format('\033[33m', '\033[0m'))
dist_to_L = np.min(dm_X, axis=0)
cover_r = np.max(dist_to_L)
if cover_r > epsilon:
    print('{}WARNING: Covering radius is larger than epsilon. Some points in X will be ignored.{}'.format('\033[33m', '\033[0m'))

    points_covered = dist_to_L < epsilon
    R = R[points_covered, :]
    dm_X = dm_X[:, points_covered]
    print('{}New data array shape = {}{}'.format('\033[33m', dm_RR.shape ,'\033[0m'))

##########################################
# Klein Bottle Image Patch Model
##########################################
# In[45]:


def makeDCT():
    """    Constructs the Discrete Cosine Transform basis for the Klein bottle image patch model

    Returns:
        arrays: four arrays encoding the DCT basis vectors
    """
    m1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    m2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    m3 = np.array([[1,-2,1],[1,-2,1],[1,-2,1]])
    m4 = np.array([[1,1,1],[-2,-2,-2],[1,1,1]])
    m5 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    m6 = np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
    m7 = np.array([[1,-2,1],[0,0,0],[-1,2,-1]])
    m8 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])


    D = np.array([[2,-1,0,-1,0,0,0,0,0], 
                  [-1,3,-1,0,-1,0,0,0,0], 
                  [0,-1,2,0,0,-1,0,0,0], 
                  [-1,0,0,3,-1,0,-1,0,0],
                  [0,-1,0,-1,4,-1,0,-1,0], 
                  [0,0,-1,0,-1,3,0,0,-1],
                  [0,0,0,-1,0,0,2,-1,0],
                  [0,0,0,0,-1,0,-1,3,-1],
                  [0,0,0,0,0,-1,0,-1,2]])

    v1 = m1.flatten()
    v1 = v1-np.mean(v1)
    v1 = v1/np.sqrt((v1.dot(D).dot(v1.T)))

    v2 = m2.flatten()
    v2 = v2-np.mean(v2)
    v2 = v2-np.sqrt((v2.dot(D).dot(v1.T)))*v1
    v2 = v2/np.sqrt((v2.dot(D).dot(v2.T)))

    v3 = m3.flatten()
    v3 = v3-np.mean(v3)
    v3 = v3-np.sqrt((v3.dot(D).dot(v2.T)))*v2-np.sqrt((v3.dot(D).dot(v1.T)))*v1
    v3 = v3/np.sqrt((v3.dot(D).dot(v3.T)))

    v4 = m4.flatten()
    v4 = v4-np.mean(v4)
    v4 = v4-np.sqrt((v4.dot(D).dot(v3.T)))*v3-np.sqrt((v4.dot(D).dot(v2.T)))*v2-np.sqrt((v4.dot(D).dot(v1.T)))*v1
    v4 = v4/np.sqrt((v4.dot(D).dot(v4.T)))
    
    return v1,v2,v3,v4

def Klein(numa,numt):
    """    Builds the Klein bottle image patch model with numa directional angles and numt bar angles. Ideal when numt = 2*numa.
    See Figure 6 in 
    https://fds.duke.edu/db/attachment/2638
    
    Note: the data set is actually of size (numa+1)*(numt+1) for convenience, probably better ways to implement this.

    Args:
        numa (int): Number of horizontal angles
        numt (int): number of vertical angles

    Returns:
        arr: point cloud modeling the Klein bottle. Note this point cloud respects the euclidean
        metric
    """
    K = []
    alphas = np.linspace(np.pi/4,5*np.pi/4,numa+1)[:numa+1]
    thetas = np.linspace(-np.pi/2,3*np.pi/2,numt+1)[:numt+1]
    for t in thetas:
        for a in alphas:
            vec = np.cos(t)*np.cos(a)*v1-np.cos(t)*np.sin(a)*v2+np.sin(t)*abs(np.cos(2*a)*v1+np.sin(2*a)*(-v2))
            K.append(vec)
    return np.round_(np.array(K),2), np.round_(alphas,2), np.round_(thetas,2)


# In[52]:

# Build the DCT basis
v1,v2,v3,v4 = makeDCT()

## Adjust number of data points and/or number of subsampled landmarks. Both parameters have an effect on topology.

numalphas = 50
numthetas = 2*numalphas
n_landmarks = 300  # generally this model needs at least 150 LMs, and gets better the more you have

L, alphas,thetas = Klein(numalphas,numthetas)
L = np.squeeze(L)

# Compute distance (this is geodesic, no need for KNN)
dm_RR = sp.spatial.distance.pdist(L,'euclidean')
#make it square
dm_RR = sp.spatial.distance.squareform(dm_RR)
print(dm_RR.shape)

t0 = time.time()

sub_ind = minmax_subsample_distance_matrix(dm_RR, n_landmarks)['indices']

dm_X = dm_RR[sub_ind, :]

dm_L = dm_X[:,sub_ind]

#Compute persistence in dim 1 over Z_2 and Z_3. Klein bottle should have two classes over Z_2 but
#just one class over Z_3. In dim 2, there should be a bar for Z_2 but no bar for Z_3
result2 = ripser(dm_L, coeff=2, do_cocycles=True, maxdim=1, distance_matrix=True)
dgmsz2 = result2['dgms']
#print(dgmsz2) #####Sort this in reverse
result3 = ripser(dm_L, coeff=3, do_cocycles=True, maxdim=1, distance_matrix=True)
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

## Check to see if this is really the Klein bottle:
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

# Also check for Z_3: 
cocycles3 = result3['cocycles'] 

H_13 = cocycles3[1]
H_1_diagram3 = dgmsz3[1] # dgm(PH^1(R(L); Z_q))
H_1_persistence3 = H_1_diagram3[:,1] - H_1_diagram3[:,0]
H_1_persistence_sort_ind3 = H_1_persistence3.argsort() # index of the largest bar in PH^1(R(L); Z_3)

a3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 0] # Birth of the largest class in PH^1(R(L); Z_3)
b3 = H_1_diagram3[H_1_persistence_sort_ind3[-1], 1] # Death of the largest class in PH^1(R(L); Z_3)

a32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 0] # Birth of the 2nd largest class in PH^1(R(L); Z_3)
b32 = H_1_diagram3[H_1_persistence_sort_ind3[-2], 1] # Death of the 2nd largest class in PH^1(R(L); Z_3)



epsilon = a * 1.01 # Epsilon is the radius e usefor the balls with centers in the landmarks.
print("Birth = %s, Death = %s" % (a, b))
print("Birth2 = %s, Death2 = %s" % (a2, b2))
print("Birth3 = %s, Death3 = %s" % (a3, b3))
print("Birth32 = %s, Death32 = %s" % (a32, b32))




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


######################################################
# Klein Bottle Flat Planar Model
######################################################

# This one is simple and runs particularly fast. 
# In[49]:


def cls(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in the fundamental domain of K
    '''
    arr = np.array([[x,y],[x-1,2-y],[x,y+1],[x+1,2-y],[x-1,1-y],[x+1,1-y],[x-1,-y],[x,y-1],[x+1,-y]])
    return arr
def minDist(X,Y):
    '''
    Returns the geodesic distance on the fundamental domain between X and Y.
    '''
    md = np.min(sp.spatial.distance_matrix(cls(X[0],X[1]),cls(Y[0],Y[1])))
    return md


# In[51]:

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
dm_RR = pdist(L, minDist)
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

