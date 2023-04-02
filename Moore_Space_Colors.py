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