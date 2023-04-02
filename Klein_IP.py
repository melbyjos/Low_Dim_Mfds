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
