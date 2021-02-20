#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script highlight:
    * Generating a 2D regular grid.
    * Refining a 2D regular grid at only a few places.
    * Extending a 2D mesh to 3 dimensions.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy import mesh as mesher

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np



fig = plt.figure(1)
fig.clf()


# Set bounding box
coordinateLims = np.array([ [0,1], [0,1], [0,1] ])
# Get 2D mesh
mesh = mesher.regularMesh.meshInPlaneRegular( coordinateLims[:2, :], np.array([0.3, 0.2]) )
# # Remove one of the triangles
# mesh.triangles = mesh.triangles[:1,:]
# mesh = mesher.Mesh( mesh.triangles, mesh.nodes )

# Refine some points to create irregular mesh
refineArray = np.ones(mesh.nodes.shape[0])
refineArray[[0, 11, 27]] = 0.05
mesh = mesh.refine( maxDiam = refineArray, maxNumNodes = mesh.N + 65 )


ax = plt.subplot(221)
ax.cla()        
meshPlotter = mesher.MeshPlotter( mesh )
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="blue")

# for iter in range(mesh.triangles.shape[0]):   
#     triangles = np.concatenate( (mesh.triangles, mesh.triangles[:,0:1]), axis = 1)
#     plt.plot( mesh.nodes[triangles[iter, :], 0], mesh.nodes[triangles[iter, :], 1], color="blue" )



# Extend to 3D mesh
mesh3D = mesher.regularMesh.extendMeshRegularly( mesh, spacing = 1, num = 1 )
    
ax = fig.add_subplot(222, projection='3d')
ax.cla()        
meshPlotter = mesher.MeshPlotter( mesh3D )
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")


# Plot inner faces

# # Get edges
# edges = mesher.Mesh.getEdges(mesh3D.triangles, 3, 3, libInstance = (mesh3D._libInstance) )["edges"]
# # Get boundary edges
# bndEdges = mesh3D.computeBoundary()["edges"]
# # Get edges not on boundary
# innerEdges = [ set(edges[iter,:]) not in [set(bndEdges[iter2,:]) for iter2 in range(bndEdges.shape[0])] for iter in range(edges.shape[0])]
# # Plot
# for iter in np.array(innerEdges).nonzero()[0]:
#     ax.add_collection3d( Poly3DCollection( [list(mesh3D.nodes[edges[iter, :], :]) ], alpha=0.2 ) )
    
    

# %% Refine in 3D
    
# # Refine some points to create irregular mesh
# refineArray = 2 * np.ones(mesh3D.nodes.shape[0])
# refineArray[0] = 1
# mesh3D = mesh3D.refine( maxDiam = refineArray, maxNumNodes = mesh3D.N + 13 )

# ax = fig.add_subplot(223, projection='3d')
# ax.cla()        
# edges = mesh3D.prepareForPlotting()
# # ax.plot(edges[0][:,0], edges[0][:,1], edges[0][:,2], color="blue")
# ax.plot(edges[1][:,0], edges[1][:,1], edges[1][:,2], color="red")        
    
    
    
    
# %% Visualize 3D
    
# ax = fig.add_subplot(222, projection='3d')
# ax.cla()

# nodes = mesh3D.nodes

# ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2])
# ax.plot( nodes[[0,1,2,0],0], nodes[[0,1,2,0],1], nodes[[0,1,2,0],2], 'b' )
# ax.plot( nodes[[4,5,6,4],0], nodes[[4,5,6,4],1], nodes[[4,5,6,4],2], 'b' )

# ax.plot( nodes[[0,4],0], nodes[[0,4],1], nodes[[0,4],2], 'b' )
# ax.plot( nodes[[1,5],0], nodes[[1,5],1], nodes[[1,5],2], 'b' )
# ax.plot( nodes[[2,6],0], nodes[[2,6],1], nodes[[2,6],2], 'b' )

# ax.plot( nodes[[0,5],0], nodes[[0,5],1], nodes[[2,6],2], 'm' )
# ax.plot( nodes[[2,4],0], nodes[[2,4],1], nodes[[2,4],2], 'm' )
# ax.plot( nodes[[5,2],0], nodes[[5,2],1], nodes[[5,2],2], 'm' )

# ax.add_collection3d( Poly3DCollection( [list(nodes[[0,5,2]]) ], alpha=0.5 ) )
# ax.add_collection3d( Poly3DCollection( [list(nodes[[4,5,2]]) ], alpha=0.5 ) )


