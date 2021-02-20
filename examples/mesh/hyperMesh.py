#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script highlight how to define a hyper rectangular mesh.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy import mesh as mesher

import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



        
print("Running test case: Hyper-rectangular mesh extension")
print("")


fig = plt.figure(1)
plt.clf()



# %% 2D case


# Create 1D mesh
nodes = np.array([0,1,2], dtype=np.float64).reshape((-1,1))
triangles = np.array([[0,1], [1,2]], dtype = np.uintc).reshape((-1,2))
mesh1 = mesher.Mesh( triangles = triangles, nodes = nodes )

# Get implicit mesh of mesh1
offset = np.array([-5], dtype=np.float64)
numPerDimension = np.array([1000], dtype=np.uintc)
implicitMesh = mesher.ImplicitMesh( mesh1, offset, numPerDimension, np.array([[1], [0]], dtype=np.uintc) )
startTime = time.time()
mesh2 = implicitMesh.toFullMesh()
neighs2 = implicitMesh.getFullNeighs()
endTime = time.time()
print( str(endTime-startTime) + " seconds")

ax = plt.subplot(3,2,1)
ax.cla()
ax.set_title( "Explicit mesh 1D" )
plt.plot( mesh1.nodes[:,0], 0 * mesh1.nodes[:,0], marker="x" )

ax = plt.subplot(3,2,2)
ax.cla()
ax.set_title( "Implicit mesh 1D" )
plt.plot( mesh2.nodes[:,0], 0 * mesh2.nodes[:,0], marker="x" )



# Get hyper rectangular mesh of mesh2
offsetHyper = np.array([4], dtype=np.float64)
stepLengths = np.array([2], dtype=np.float64)
numSteps = np.array([0], dtype=np.uintc)
hyperRectMesh = mesher.HyperRectMesh(implicitMesh, offsetHyper, stepLengths, numSteps)
hyperRectMesh._storeMeshInternally()



plt.show()
