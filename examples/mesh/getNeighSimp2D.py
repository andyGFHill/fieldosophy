#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script show how to find neighboring nearest simplices to a given simplex in 2 dimensions.

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


        
print("Running two-dimensional FEM test case")
print("")


plt.figure(1)
plt.clf()


# %% Create 2D mesh

# Limits of coordinates
coordinateLims = np.array( [ [0,1], [0,1] ] )

print("Compute Mesh in 2D")
meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims, 0.1 )


print("Plot mesh")

plt.figure(1)
ax = plt.subplot(221)
plt.cla()
ax.set_title( "Mesh" )    
meshPlotter = mesher.MeshPlotter( meshPlane )
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")


# %%

# Get edges
edges = mesher.Mesh.getEdges( meshPlane.triangles, meshPlane.topD, meshPlane.topD, libInstance = meshPlane._libInstance )
# Get neighbors if simplices
neighs = mesher.Mesh.getSimplexNeighbors( edges["simplicesForEdges"], edges["edgesForSimplices"], libInstance = meshPlane._libInstance )

# Plot original simplex and its neighbors
chosenSimplex = np.random.randint( 0, meshPlane.NT )
edges = meshPlotter.getSimplicesLines( neighs[chosenSimplex,:] )
plt.plot(edges[0], edges[1], color="green", linewidth=6)
edges = meshPlotter.getSimplicesLines( np.array([chosenSimplex]) )
plt.plot(edges[0], edges[1], color="black", linewidth=6)

