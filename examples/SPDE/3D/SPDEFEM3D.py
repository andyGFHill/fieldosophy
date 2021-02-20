#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating an Matérn FEM approximation model in 3 dimensions.
    * Generate samples from this model.
    * Visualize realizations.

It should be mentioned that the extension is not long enough to get rid of boundary effects. This is due to limiting the execution time of the example script.
Extension can be increased below if wanted.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy.GRF import FEM
from fieldosophy import mesh as mesher
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from fieldosophy.misc.misc_spatiotemporal import SpatiotemporalViewer
from mpl_toolkits import mplot3d
        
print("Running three-dimensional FEM test case")
print("")



# %% Create 3D mesh


# Limits of coordinates
coordinateLims = np.array( [ [0,1], [0,1], [0,1] ] )
# Define original minimum corelation length
corrMin = 0.8
extension = 0.0#corrMin*1

# Mesh
# coordinateLims = np.array( [ [0,1], [0,1] ] )
# meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin/5/np.sqrt(3) )
# meshSpace = mesher.regularMesh.extendMeshRegularly( meshPlane, spacing = .4, num = 3 )
print("Compute Mesh in 3D")
meshSpace = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin / 5 / np.sqrt(3) )


print("Plot 3D mesh")
# Plot
fig = plt.figure(1)
ax = plt.subplot(221, projection="3d")
plt.cla()

# ax.scatter3D( meshSpace.nodes[:,0], meshSpace.nodes[:,1], meshSpace.nodes[:,2], color="green" )
meshPlotter = mesher.MeshPlotter( meshSpace )
boundaryEdges = meshPlotter.getBoundaryLines()
ax.plot3D(boundaryEdges[0], boundaryEdges[1], boundaryEdges[2], color="red")



# %% Create FEM system

print("Set up FEM system")

# Define the random field
r =  corrMin
nu = 2
sigma = 1

BCDirichlet = np.NaN * np.ones((meshSpace.N))
BCDirichlet[meshSpace.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshSpace.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with function
# BCRobin = None

# Create FEM object
fem = FEM.MaternFEM( mesh = meshSpace, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )



# %% Sample

# Acquire realizations
print("Generate realizations")

M = int(2e0)
sigmaEps = 1e-3

Z = fem.generateRandom( M )

# Set observation points
lats = np.linspace(coordinateLims[1,0]+1e-3, coordinateLims[1,-1]-1e-3, num = int( 80 ) )
lons = np.linspace(coordinateLims[0,0]+1e-3, coordinateLims[0,-1]-1e-3, num = int( 80 ) )
obsPoints = np.meshgrid( lons, lats )
obsPointsList = np.array( [  \
     (np.stack( ( obsPoints[0].flatten(), obsPoints[1].flatten(), iterZOffset * np.ones(obsPoints[0].size) ), axis=1 )) \
         for iterZOffset in np.linspace(1e-3,1-1e-3, num=80) ] ) 
obsPointsList = obsPointsList.reshape( (-1,3) )
time = np.sort(np.unique(obsPointsList[:,2]))
time = np.array([ np.datetime64( 0, 'h') + np.timedelta64( int(time[iter] * time.size ), 'h' ) for iter in range(time.size) ], dtype = np.datetime64)

# Get observation matrix
print("Acquire observation matrix")
obsMat = fem.mesh.getObsMat( obsPointsList )
ZObs = obsMat.tocsr() * Z + stats.norm.rvs( loc = 0, scale = sigmaEps, size = M*obsMat.shape[0] ).reshape((obsMat.shape[0], M))




# %% Visualize realisation


def plotInstant( self, ind ):
    # function for plotting one time instance
    
    img = ZObs[ (obsPoints[0].size * ind):(obsPoints[0].size * (ind+1)), int(self.userControllers[0]["controllers"]["string"].get()) ].reshape(obsPoints[0].shape)
    
    # Plot first images
    if self.userdata["pc"] is None:            
            self.userdata["pc"] = self.userdata["ax"].pcolormesh( obsPoints[0][0,:], obsPoints[1][:,0], img )
    else:
        self.userdata["pc"].set_array( img[:,:-1].ravel())
    

def indCallback(self, event):
    self.controls.slider.set( 0 )

fig = plt.figure()
ax = plt.subplot(111)

viewer = SpatiotemporalViewer( time = time, title = "Visualize 3D", fig = fig, plotInstant = plotInstant, \
    userdata = { "pc":None,"ax":ax }, \
    userControllers = [{"label":"Realization: ", "type":"entry", "state":"normal", "text":"0", "callback":indCallback}])
viewer.mainloop()


