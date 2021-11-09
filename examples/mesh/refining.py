#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates refining of meshes. 
In the top row a two simplices mesh is refined such that the bottom left part has finer and finer resolution.
The refinement is chosen such that a chosen node will not be part of a simplex with longer diameter than 0.2.

In the bottom row a sphere is approximated by a box. 
The approximation is then refined until no simplex has a longer diamter than 0.4.


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



        
print("Running mesh refinement example")
print("")


fig = plt.figure(1)
plt.clf()

print("Plot mesh")


# %% Create 1D mesh

nodes = np.linspace(0,1,4, dtype=np.float64).reshape((-1,1))
triangles = np.stack( (np.arange(0, nodes.size-1), np.arange(1,nodes.size)) ).transpose().astype(np.uintc)
meshm0 = mesher.Mesh( triangles, nodes )

ax = fig.add_subplot(321, )
ax.set_title( "Mesh refine 0" )    
plt.plot( meshm0.nodes[:,0], 0 * meshm0.nodes[:,0], marker="x" )

# Set maximum diameter for each node
maxDiamArray = 0.2 * np.ones( (meshm0.N) )
maxDiamArray[2] = 0.05
# Create refined sphere
meshm1 = meshm0.refine( maxDiam = maxDiamArray, maxNumNodes = meshm0.N + 50 )

ax = fig.add_subplot(322, )
ax.set_title( "Mesh refine 1" )    
plt.plot( meshm1.nodes[:,0], 0 * meshm1.nodes[:,0], marker="x" )


# %% Create simplest 2D mesh


# Create unrefined sphere
mesh0 = mesher.Mesh( np.array([ [0,1,2], [1,2,3]], dtype=np.uintc), np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) )



ax = fig.add_subplot(323, )
ax.set_title( "Mesh refine 0" )    
meshPlotter = mesher.MeshPlotter( mesh0 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")

# Set maximum diameter for each node
maxDiamArray = 0.8 * np.ones( (mesh0.N) )
maxDiamArray[0] = 0.1

# Create refined mesh
mesh1 = mesh0.refine( maxDiam = maxDiamArray, maxNumNodes = mesh0.N + 50 )

ax = fig.add_subplot(324)
ax.set_title( "Mesh refine 1" )   
meshPlotter = mesher.MeshPlotter( mesh1 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")

# simpId = 6
# for iter in neighs1[simpId, :]:
#     if iter < mesh1.NT:
#         plt.plot( mesh1.nodes[ mesh1.triangles[iter, [0,1,2,0]], 0 ], mesh1.nodes[ mesh1.triangles[iter, [0,1,2,0]], 1 ] , color = "magenta", linewidth=5 )
# plt.plot( mesh1.nodes[ mesh1.triangles[simpId, [0,1,2,0]], 0 ], mesh1.nodes[ mesh1.triangles[simpId, [0,1,2,0]], 1 ] , color = "green", linewidth=5 )

# %% Create spherical mesh


# Create unrefined sphere
meshSphere0 = mesher.Mesh.meshOnSphere( maxDiam = np.inf, maxNumNodes = int(1e3), radius = 1)


fig = plt.figure(1)

ax = fig.add_subplot(325, projection='3d')
ax.set_title( "Mesh refine 0" )    
meshPlotter = mesher.MeshPlotter( meshSphere0 ) 
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")

# Set maximum diameter for each node
maxDiamArray = 2 * np.sin( 45 / 180.0 * np.pi / 2.0 ) * np.ones( (meshSphere0.N) )
maxDiamArray[0] = 2 * np.sin( 1 / 180.0 * np.pi / 2.0 )


# Create refined sphere
meshSphere1 = meshSphere0.refine( maxDiam = maxDiamArray, maxNumNodes = meshSphere0.N + 200, transformation = mesher.geometrical.mapToHypersphere )

ax = fig.add_subplot(326, projection='3d')
ax.set_title( "Mesh refine 1" )    
meshPlotter = mesher.MeshPlotter( meshSphere1 ) 
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")




plt.show()