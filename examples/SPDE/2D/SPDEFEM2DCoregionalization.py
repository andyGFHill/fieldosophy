#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating an Matérn FEM approximation model in 2 dimensions.
    * Generate samples from this model.
    * Acquire a model of non.stationary coregionalization between sample 0 and sample 1.


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


        
print("Running two-dimensional FEM test case")
print("")


plt.figure(1)
plt.clf()
plt.figure(2)
plt.clf()


# %% Create 2D mesh


# Limits of coordinates
coordinateLims = np.array( [ [0,1], [0, 1] ] )
# Define original minimum corelation length
corrMin = 0.4
extension = corrMin*1.5
# Create fake data points to force mesh
lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( np.ceil( np.diff(coordinateLims[1,:])[0] / (corrMin/7) ) ) )
lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( np.ceil( np.diff(coordinateLims[0,:])[0] / (corrMin/7) ) ) )
dataGrid = np.meshgrid( lons, lats )
dataPoints = np.hstack( (dataGrid[0].reshape(-1,1), dataGrid[1].reshape(-1,1)) )

# Mesh
print("Compute Mesh")
meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin/5/np.sqrt(2) )
# Remove all nodes too far from active points    
meshPlane = meshPlane.cutOutsideMesh( dataPoints.transpose(), extension )

print("Plot mesh")

plt.figure(1)
ax = plt.subplot(221)
plt.cla()
ax.set_title( "Mesh" )    
meshPlotter = mesher.MeshPlotter(meshPlane)
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")
plt.scatter(dataPoints[:,0], dataPoints[:,1], c="red")




# %% Create FEM system

print("Set up FEM system")

# Define the random field
r =  0.48
nu = 1.3
sigma = 1
sigmaEps = 2e-2
BCDirichlet = np.NaN * np.ones((meshPlane.N))
BCDirichlet[meshPlane.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with function
# BCRobin = None

# Create FEM object
fem = FEM.MaternFEM( mesh = meshPlane, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )









# %% Sample

# Acquire realizations
print("Generate realizations")

M = int(2e3)

Z = fem.generateRandom( M )

# Set observation points
lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( 60 ) )
lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( 60 ) )
obsPoints = np.meshgrid( lons, lats )

# Get observation matrix
print("Acquire observation matrix")
obsMat = fem.mesh.getObsMat( np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ))
ZObs = obsMat.tocsr() * Z + stats.norm.rvs( loc = 0, scale = sigmaEps, size = M*obsMat.shape[0] ).reshape((obsMat.shape[0], M))





print("Plot covariances")


plt.figure(1)

ax = plt.subplot(222)
plt.cla()
ax.set_title( "Stationary covariance plot" )

# Get node closest to middle
midPoint = np.mean( coordinateLims, axis = 1 )
runx = np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ) - midPoint
runx = np.sqrt(np.sum(runx**2, axis=1))
orderInd =  np.argsort(runx)
runx = np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ) - np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) )[orderInd[0], :]
runx = np.sqrt(np.sum(runx**2, axis=1))
orderInd =  np.argsort(runx)
runx = runx[orderInd]

# Compute estimated covariance from realization
runy = ( ZObs[orderInd[0], :] - np.mean(ZObs[orderInd[0], :]) ).reshape((1,-1)) * (ZObs - np.mean(ZObs, axis=1).reshape((-1,1)))
runy = np.mean(runy, axis=1)
runy = runy[orderInd]

# Plot empirical covariance
plt.plot(runx, runy, label = "SPDE empirical", color="green", linestyle="dashed")

# Compute SPDE correlation
runy = obsMat.tocsr()[orderInd, :] * fem.multiplyWithCovariance(obsMat.tocsr()[orderInd[0], :].transpose())
# Plot true covariance from model
plt.plot(runx, runy, label = "SPDE", color="red", linewidth=2)
      
# Compute theoretical Matérn correlation
runy = GRF.MaternCorr( runx, nu = nu, kappa = np.sqrt(8*nu)/r )
plt.plot(runx, runy, label = "Matern", color="blue")
plt.legend()
plt.xlabel("Time [s]")




ax = plt.subplot(2,2,3)
plt.cla()
ax.set_title( "A realization" )
# temp = obsMat * np.sqrt(np.sum(fem.mesh.nodes**2, axis=1))
# plt.imshow( temp.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
#             extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.imshow( ZObs[:,0].reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.colorbar()




ax = plt.subplot(224)
plt.cla()
ax.set_title( "Covariance" )
# Compute SPDE covariance
runy = fem.mesh.getObsMat( midPoint.reshape((1,-1)) )
runy = runy.transpose()
runy = fem.multiplyWithCovariance(runy)
runy = obsMat.tocsr() * runy
plt.imshow( runy.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
           extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.colorbar()











        



    
# %% Define model of coregionalization

# Define correlation between fields
rho = 0.1 + (1-np.abs(meshPlane.nodes[:,0])**1) * 0.9

# Randomly choose 10 points in the domain
condPoints = np.random.uniform(size=10*2).reshape((-1,2))
# Get observation matrix
condObsMat = fem.mesh.getObsMat( condPoints ).tocsc()
# set conditional values
condVal = condObsMat.tocsr() * Z[:,0] \
    + stats.norm.rvs( loc = 0, scale = 1e-3, size = condObsMat.shape[0] )
# Get real values of y
condObsVal = (obsMat.tocsr() * rho) * ZObs[:,0] + np.sqrt(np.abs(1 - (obsMat.tocsr() * rho)**2)) * ZObs[:,1]

# Compute conditional distribution
condDistr = fem.cond(condVal, condObsMat, sigmaEps = 1e-3)
# Get conditional mean at observation points
condMean =  obsMat.tocsr() * (rho * condDistr.mu)
# Get conditional standard deviation
condStd = obsMat * sparse.diags(rho)
condStd = condStd.tocsr() * condDistr.multiplyWithCovariance( condStd.transpose().tocsc() )
condStd = condStd + obsMat.tocsr() * (1-rho**2)
condStd = np.sqrt(condStd.diagonal())


# Plot 
plt.figure(2)
plt.clf()

# Plot rho
ax = plt.subplot(3,2,1)
plt.cla()
ax.set_title( "Marginal cross-correlation" )
plt.imshow( ( obsMat.tocsr() * rho ).reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.colorbar()


# Plot error-field
ax = plt.subplot(3,2,2)
plt.cla()
ax.set_title( "Error-field" )
plt.imshow( ZObs[:,1].reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.clim((2,-2))    
plt.colorbar()


# Plot x-field
ax = plt.subplot(3,2,3)
plt.cla()
ax.set_title( "X-field" )
plt.imshow( ZObs[:,0].reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.clim((2,-2))    
plt.colorbar()

plt.scatter( condPoints[:,0], condPoints[:,1], color = "red", s = 30 )


# Plot y-field
ax = plt.subplot(3,2,4)
plt.cla()
ax.set_title( "Y-field" )
plt.imshow( condObsVal.reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.clim((2,-2))    
plt.colorbar()



# Plot conditional std
ax = plt.subplot(3,2,5)
plt.cla()
ax.set_title( "Conditional standard deviation" )
plt.imshow( condStd.reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.clim((1,0))    
plt.colorbar()

# Plot conditional mean 
ax = plt.subplot(3,2,6)
plt.cla()
ax.set_title( "conditional mean" )
plt.imshow( condMean.reshape(obsPoints[0].shape), origin="lower", aspect="auto", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.clim((1,-1))    
plt.colorbar()








plt.show()

