#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script highlight how to define an implicitly extended mesh in 1,2, and 3 dimensions.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy import mesh as mesher


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



        
print("Running test case: Implicit mesh")
print("")


fig = plt.figure(1)
plt.clf()



# %% 1D case


# Create 1D mesh
nodes = np.array([0,2,3,4,6]).reshape((-1,1))
triangles = np.array([np.arange(0,4), np.arange(1,5)], dtype = np.uintc).transpose().copy()
triangles = triangles[:, [1,0]].copy()
mesh1 = mesher.Mesh( triangles = triangles, nodes = nodes )
mesh1.getBoundary()
# Get edges and neighbors
edges1 = mesher.Mesh.getEdges( mesh1.triangles, mesh1.topD, mesh1.topD, libInstance = mesh1._libInstance )
neighs1 = mesher.Mesh.getSimplexNeighbors( edges1["simplicesForEdges"], edges1["edgesForSimplices"], libInstance = mesh1._libInstance )


# Get implicit mesh of mesh0
offset = np.array([-5], dtype=np.float64)
numPerDimension = np.array([3], dtype=np.uintc)
implicitMesh = mesher.ImplicitMesh( mesh1, offset, numPerDimension, neighs1 )
mesh2 = implicitMesh.toFullMesh()
neighs2 = implicitMesh.getFullNeighs()

ax = plt.subplot(3,2,1)
ax.cla()
ax.set_title( "Explicit mesh 1D" )
plt.plot( mesh1.nodes[:,0], 0 * mesh1.nodes[:,0], marker="x" )

ax = plt.subplot(3,2,2)
ax.cla()
ax.set_title( "Implicit mesh 1D" )
plt.plot( mesh2.nodes[:,0], 0 * mesh2.nodes[:,0], marker="x" )

# # Plot a randomly chosen simplex and its neighbors
# chosenSimplex = np.random.randint( 0, mesh2.NT )
# meshPlotter = mesher.MeshPlotter( mesh2 )
# edges = meshPlotter.getSimplicesLines( neighs2[chosenSimplex,:] )
# plt.plot(edges[0], edges[0]*0, color="green", linewidth=3)
# edges = meshPlotter.getSimplicesLines( np.array([chosenSimplex]) )
# plt.plot(edges[0], edges[0]*0, color="black", linewidth=3)

# Plot a randomly chosen point and its neighboring simplices
chosenPoint = mesh2.getBoundingBox()
chosenPoint = np.random.uniform( size = mesh2.embD ) * (chosenPoint[:,1]-chosenPoint[:,0]) + chosenPoint[:,0]
chosenPoint = mesh2.nodes[10, :]
# chosenPoint = np.array([-1])
meshPlotter = mesher.MeshPlotter( mesh2 )
partOfSimplices = np.array([implicitMesh.pointInSimplex( chosenPoint, iter ) for iter in range(mesh2.NT)])
edges = meshPlotter.getSimplicesLines( partOfSimplices )
plt.plot(edges[0], edges[0]*0, color="green", linewidth=3)
plt.scatter(chosenPoint[0], chosenPoint[0]*0, color="black")



# %% 2D case


# Create unrefined mesh
mesh3 = mesher.Mesh( np.array([ [0,1,2], [1,2,3]], dtype=int), np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) )
# Set maximum diameter for each node
maxDiamArray = 1
# Refine mesh
mesh3, neighs3 = mesh3.refine( maxDiam = maxDiamArray, maxNumNodes = mesh3.N + 100 )



# Set offset
offset = np.array( [-8,17], dtype=np.float64 )
# Set numper of multiplications in each dimension
numPerDimension = np.array( [3,2] )
# Get implicit mesh of mesh0
implicitMesh2D = mesher.ImplicitMesh( mesh3, offset, numPerDimension, neighs3 )
[implicitMesh2D.fromSectorAndExplicit2Node(3,iter) for iter in range(4)]
implicitMesh2D.fromNodeInd2SectorAndExplicit(10)
mesh4 = implicitMesh2D.toFullMesh()
neighs4 = implicitMesh2D.getFullNeighs()


ax = fig.add_subplot(323 )
ax.cla()
ax.set_title( "Explicit mesh 2D" )    
meshPlotter = mesher.MeshPlotter( mesh3 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")

ax = fig.add_subplot(324 )
ax.cla()
ax.set_title( "Implicit mesh 2D" )    
meshPlotter = mesher.MeshPlotter( mesh4 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")


# # Plot a randomly chosen simplex and its neighboring simplices
# chosenSimplex = 9#np.random.randint( 0, mesh4.NT )
# edges = meshPlotter.getSimplicesLines( neighs4[chosenSimplex,:] )
# plt.plot(edges[0], edges[1], color="green", linewidth=3)
# edges = meshPlotter.getSimplicesLines( np.array([chosenSimplex]) )
# plt.plot(edges[0], edges[1], color="black", linewidth=3)

# Plot a randomly chosen point and its neighboring simplices
chosenPoint = mesh4.getBoundingBox()
chosenPoint = np.random.uniform( size = mesh4.embD ) * (chosenPoint[:,1]-chosenPoint[:,0]) + chosenPoint[:,0]
chosenPoint = mesh4.nodes[10, :]
chosenPoint = np.array([-6, 18.7])
partOfSimplices = np.array([implicitMesh2D.pointInSimplex( chosenPoint, iter ) for iter in range(mesh4.NT)])
edges = meshPlotter.getSimplicesLines( partOfSimplices )
plt.plot(edges[0], edges[1], color="green", linewidth=3)
plt.scatter(chosenPoint[0], chosenPoint[1], color="black")




# %% 3D case

# Extend mesh3 into three dimensions
mesh5 = mesher.Mesh( np.array([ [0,1,2], [1,2,3]], dtype=int), np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) )
mesh5 = mesher.regularMesh.extendMeshRegularly( mesh5, spacing = 1, num = 1 )
# Get edges and neighbors
edges5 = mesher.Mesh.getEdges( mesh5.triangles, mesh5.topD, mesh5.topD, libInstance = mesh5._libInstance )
neighs5 = mesher.Mesh.getSimplexNeighbors( edges5["simplicesForEdges"], edges5["edgesForSimplices"], libInstance = mesh5._libInstance )


# Set offset
offset = np.array( [-5,-8,17], dtype=np.float64 )
# Set numper of multiplications in each dimension
numPerDimension = np.array( [3,3,3] )
# Get implicit mesh of mesh0
implicitMesh3D = mesher.ImplicitMesh( mesh5, offset, numPerDimension, neighs5 )
mesh6 = implicitMesh3D.toFullMesh()
neighs6 = implicitMesh3D.getFullNeighs()


# Plot mesh5
plt.figure(1)
ax = plt.subplot(325, projection="3d")
plt.cla()
ax.set_title( "Explicit mesh 3D" )
meshPlotter = mesher.MeshPlotter( mesh5 )
edges = meshPlotter.getLines()
ax.plot3D(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot3D(edges[0], edges[1], edges[2], color="red")

# Plot mesh6
plt.figure(1)
ax = plt.subplot(326, projection="3d")
plt.cla()
ax.set_title( "Implicit mesh 3D" )
meshPlotter = mesher.MeshPlotter( mesh6 )
edges = meshPlotter.getLines()
# ax.plot3D(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot3D(edges[0], edges[1], edges[2], color="red")

# Plot a randomly chosen simplex and its neighbors
chosenSimplex = np.random.randint( 0, mesh6.NT )
meshPlotter = mesher.MeshPlotter( mesh6 )
edges = meshPlotter.getSimplicesLines( neighs6[chosenSimplex,:] )
ax.plot3D(edges[0], edges[1], edges[2], color="green", linewidth=3)
edges = meshPlotter.getSimplicesLines( np.array([chosenSimplex]) )
ax.plot3D(edges[0], edges[1], edges[2], color="black", linewidth=3)


plt.show()