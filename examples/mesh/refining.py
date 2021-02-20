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


plt.figure(1)
plt.clf()



# %% Create simplest 2D mesh


# Create unrefined sphere
mesh0 = mesher.Mesh( np.array([ [0,1,2], [1,2,3]], dtype=int), np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) )

print("Plot mesh")
fig = plt.figure(1)
plt.clf()

ax = fig.add_subplot(221, )
ax.set_title( "Mesh refine 0" )    
meshPlotter = mesher.MeshPlotter( mesh0 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")

# Set maximum diameter for each node
maxDiamArray = 1 * np.ones( (mesh0.N) )
maxDiamArray[0] = 0.2

# Create refined sphere
mesh1 = mesh0.refine( maxDiam = maxDiamArray, maxNumNodes = mesh0.N + 2000 )

ax = fig.add_subplot(222)
ax.set_title( "Mesh refine 1" )   
meshPlotter = mesher.MeshPlotter( mesh1 ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")



# %% Create spherical mesh


# Create unrefined sphere
meshSphere0 = mesher.Mesh.meshOnSphere( None, maxDiam = np.inf, maxNumNodes = np.inf, radius = 1)

print("Plot mesh")
fig = plt.figure(1)

ax = fig.add_subplot(223, projection='3d')
ax.set_title( "Mesh refine 0" )    
meshPlotter = mesher.MeshPlotter( meshSphere0 ) 
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")

# Set maximum diameter for each node
maxDiamArray = 0.4 * np.ones( (meshSphere0.N) )
#maxDiamArray[0] = 0.4

# Define spherical transformation
def transformation(x):
    return x / np.linalg.norm(x)

# Create refined sphere
meshSphere1 = meshSphere0.refine( maxDiam = maxDiamArray, maxNumNodes = meshSphere0.N + 1000, transformation = transformation )

ax = fig.add_subplot(224, projection='3d')
ax.set_title( "Mesh refine 1" )    
meshPlotter = mesher.MeshPlotter( meshSphere1 ) 
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")



plt.show()