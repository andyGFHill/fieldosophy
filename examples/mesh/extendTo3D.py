#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script show how to extend a 2D mesh to 3D.

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
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize


        
print("Running two-dimensional FEM test case")
print("")


plt.figure(1)
plt.clf()


# %% Create 3D mesh


# Limits of coordinates
coordinateLims = np.array( [ [0,1], [0, 1] ] )
# Define original minimum corelation length
corrMin = 0.8
extension = 0.0#corrMin*1
# Create fake data points to force mesh
lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( np.ceil( np.diff(coordinateLims[1,:])[0] / (corrMin/7) ) ) )
lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( np.ceil( np.diff(coordinateLims[0,:])[0] / (corrMin/7) ) ) )
dataGrid = np.meshgrid( lons, lats )
dataPoints = np.hstack( (dataGrid[0].reshape(-1,1), dataGrid[1].reshape(-1,1)) )

# Mesh
print("Compute Mesh in 2D")
meshPlane = None
meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin/1/np.sqrt(2) )


print("Plot 2D mesh")

plt.figure(1)
ax = plt.subplot(221)
plt.cla()
ax.set_title( "Mesh" )    
meshPlotter = mesher.MeshPlotter( meshPlane )
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")

# plt.scatter(dataPoints[:,0], dataPoints[:,1], c="red")




# %% Extend mesh to 3D


print("Extend mesh in 3D")

# Extend mesh in third dimension
meshSpace = mesher.regularMesh.extendMeshRegularly( meshPlane, spacing = corrMin/5, num = 1 )

# Acquire edges between all nodes


print("Plot 3D mesh")
# Plot
plt.figure(1)
ax = plt.subplot(222, projection="3d")
plt.cla()
ax.scatter3D( meshSpace.nodes[:,0], meshSpace.nodes[:,1], meshSpace.nodes[:,2], color="blue" )

meshPlotter = mesher.MeshPlotter( meshSpace )
edges = meshPlotter.getLines()
ax.plot3D(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot3D(edges[0], edges[1], edges[2], color="red")


plt.show()