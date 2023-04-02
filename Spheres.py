import numpy as np
import scipy as sp
from ripser import ripser
import math
import time
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from persim import plot_diagrams

np.random.seed( 42 )

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

def gen_S1(N):
    theta = np.random.uniform(0, 2*math.pi, N)

    x = np.cos(theta)
    y = np.sin(theta)

    x = x.flatten()
    y = y.flatten()

    return np.column_stack((x,y))

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(2000)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_wireframe(x, y, z, color='g', rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
plt.show()

S = np.concatenate((np.reshape(xi, (xi.shape[0],1)),np.reshape(yi, (yi.shape[0],1)),np.reshape(zi, (zi.shape[0],1))),axis=1)
dm_RR = sp.spatial.distance.cdist(S,S, metric='euclidean')

n_landmarks = 100


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
