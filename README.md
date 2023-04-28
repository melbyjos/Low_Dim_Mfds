# Low_Dim_Mfds
Models for some low-dimensional manifolds constructed in Python.

Each of the *.py files in this repository contains models for certain spaces and surfaces as point
clouds. They are constructed using well-understood geometry, and the pairwise distance matrix is
also constructed for each model point cloud. In addition, persistent cohomology computations are
implemented using Ripser as a topological verification of each model. Those computations also
contain plots of persistence modules/barcodes and print out birth/death times and lifecycle
information for the longest two classes.

Spheres.py contains models for the circle (S^1) as well as fuzzy point clouds modeling any
higher-dimensional unit sphere as a subset of Euclidean space. The number of necessary points and
landmarks increases as you increase dimension, as expected.

Flat_Torus_Klein.py contains models for the torus and Klein bottle based on the standard quotient of
the unit square in R^2.

Moore_Space_Colors.py contains  model for the Moore space M(Z_n, 1) as a quotient of the unit disk.
These spaces are the unique topological spaces up to homotopy with first cohomology H^1(M(Z_n,1)) =
Z_n and all other cohomology groups zero. This model is a point cloud with ~3,000 points in the
plane. Persistence computations are accurate for subsamplings of ~100 points.

Moore_Space_IP.py contains models for two particular Moore spaces: M(Z_2,1) = RP^2 and M(Z_4,1) as
rotations of image patches. These image patches are grayscale 10x10 pixel patches viewed as points
in 100-dimensional space. The base image patch is rotated and pixel patterns offset so that it is
perturbed throughout the plane. The resulting collection of perturbed patches models the Moore space
under the Euclidean distance/metric in R^100. Persistence computations tend to be accurate for ~200
or more landmarks. 

Klein_IP.py contains an image patch model for the Klein bottle based on the Discrete Cosine
Transform (DCT) basis image patches. The model is based on https://fds.duke.edu/db/attachment/2638,
and it is a point cloud of ~5,000 points in R^9 and its underlying distance is given by the
Euclidean metric as well. Persistence is accurate for ~200 or more landmarks, but more is better.
