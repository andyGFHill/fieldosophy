#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script highlight how curvature can be added to get observation matrices of points that are on a curved submanifold.

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



        
print("")


fig = plt.figure(1)
plt.clf()



# %% 1D curved submanifold


meshCircle, neighsCircle = mesher.Mesh.meshOnCircle( maxDiam = 0.3, maxNumNodes = int(6), radius = 1 )

meshPlotter = mesher.MeshPlotter( meshCircle ) 
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")



# %% Define three points of observation


points = np.array([ [np.cos(angle), np.sin(angle)] for angle in np.linspace( 90, 120, num=3 ) * np.pi/180.0 ])
points[0,:] = np.sin( 2*np.pi/6 ) * points[0,:]
points[2,:] = 1.1 * points[2,:]
plt.scatter(points[:,0], points[:,1])


# %% Get observation matrix

obsMat = meshCircle.getObsMat( points, embTol = 0.1 ).toarray()
obsMat2 = meshCircle.getObsMat( points, embTol = 0.1, centersOfCurvature = np.zeros( (1,2) ) ).toarray()



