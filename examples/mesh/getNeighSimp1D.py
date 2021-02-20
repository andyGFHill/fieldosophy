#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script show how to find neighboring nearest simplices to a given simplex in one dimension.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy.GRF import FEM
from fieldosophy.GRF import GRF
from fieldosophy import mesh as mesher

from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize

        
plt.figure(1)
plt.clf()

print("Running one-dimensional FEM test case")


# %% Create mesh

print("Creating system")

# Number of nodes
N = int(3e1)

# Define the Matérn random field
r = 0.55 # Set correlation range (range for which two points have approximately 0.1 correlation)
nu = 1.51   # Set smoothness 
sigma = 1   # Set standard deviation
sigmaEps = 1e-2     # Set noise level

# Create 1D mesh
nodes = np.linspace(0-1*r,1+1*r, num=N, dtype=np.double).reshape((-1,1))
triangles = np.array([np.arange(0,N-1), np.arange(1,N)], dtype = np.uintc).transpose().copy()
triangles = triangles[:, [1,0]].copy()
mesh = mesher.Mesh( triangles = triangles, nodes = nodes )
mesh.getBoundary()

# Get edges with new method
edges = mesher.Mesh.getEdges( mesh.triangles, mesh.topD, mesh.topD, libInstance = mesh._libInstance )
neighs = mesher.Mesh.getSimplexNeighbors( edges["simplicesForEdges"], edges["edgesForSimplices"], libInstance = mesh._libInstance )



plt.plot( nodes, 0 * nodes, 'bo-' )

# Plot original simplex and its neighbors
chosenSimplex = np.random.randint( 0, mesh.NT )
meshPlotter = mesher.MeshPlotter( mesh )
edges = meshPlotter.getSimplicesLines( neighs[chosenSimplex,:] )
plt.plot(edges[0], edges[0]*0, color="green", linewidth=6)
edges = meshPlotter.getSimplicesLines( np.array([chosenSimplex]) )
plt.plot(edges[0], edges[0]*0, color="black", linewidth=6)

