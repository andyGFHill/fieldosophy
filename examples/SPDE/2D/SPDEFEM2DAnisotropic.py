#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating an anisotropic Matérn FEM approximation model in 2 dimensions.
    * Generate samples from this model.
    * Compute correlation (and compare with theoretical correlation).
    * Conditional distribution give observations of two points.
    * Estimating the range parameter.


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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize
from scipy import linalg
import random


        
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
corrMin = 0.2
extension = corrMin*2
# Create fake data points to force mesh
lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( np.ceil( np.diff(coordinateLims[1,:])[0] / (corrMin/7) ) ) )
lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( np.ceil( np.diff(coordinateLims[0,:])[0] / (corrMin/7) ) ) )
dataGrid = np.meshgrid( lons, lats )
dataPoints = np.hstack( (dataGrid[0].reshape(-1,1), dataGrid[1].reshape(-1,1)) )

# Mesh
print("Compute Mesh")
meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin/7/np.sqrt(2) )
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
nu = 2.7
sigma = 1
sigmaEps = 1e-3
r = np.array([corrMin, corrMin*2])
angle = (45) * np.pi/180

BCDirichlet = np.NaN * np.ones((meshPlane.N))
BCDirichlet[meshPlane.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with function
# BCRobin = None

# Create FEM object
fem = FEM.anisotropicMaternFEM( mesh = meshPlane, childParams = {'angle':angle, "r":r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )









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



# %% Plot covariances



print("Plot covariances")


plt.figure(1)
#plt.clf()

ax = plt.subplot(222)
plt.cla()
ax.set_title( "Stationary covariance plot" )

# Get a line of observations at an angle 
midPoint = np.mean( coordinateLims, axis = 1 )
cov_obs = np.linspace( -0.5 * np.diff(coordinateLims[0,:]), 0.5 * np.diff(coordinateLims[0,:]), num = 400 )
angle_obs = angle+np.pi/2#(45) * np.pi/180

cov_obs = midPoint + FEM.angleToVecs2D(angle_obs)[:,0] * cov_obs
cov_obs_ind = (cov_obs[:,0] < coordinateLims[0,0]) | (cov_obs[:,0] > coordinateLims[0,1]) | (cov_obs[:,1] < coordinateLims[1,0]) | (cov_obs[:,1] > coordinateLims[1,1])
cov_obs = cov_obs[ ~cov_obs_ind, : ]
# Compute distances
distances = cov_obs - midPoint.reshape((1,-1))
distances = np.sqrt(np.sum(distances**2, axis=1))
orderInd =  np.argsort(distances)
distances = distances[orderInd]
cov_obs = cov_obs[orderInd, :]

# Compute observation matrix
obsMat_obs = fem.mesh.getObsMat( cov_obs )
# Compute SPDE correlation
runy = obsMat_obs.tocsr() * fem.multiplyWithCovariance(obsMat_obs.tocsr()[0, :].transpose())
# Plot true covariance from model
plt.plot(distances, runy, label = "SPDE", color="red", linewidth=2)
      
# Compute theoretical Matérn correlation
runy = GRF.MaternCorr( distances, nu = nu, kappa = np.sqrt(8*nu)/r[1] )
plt.plot(distances, runy, label = "Matern", color="blue")
plt.legend()
plt.xlabel("Time [s]")



ax = plt.subplot(223)
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
midPoint = np.mean( np.concatenate( [ obsPoints[iter].reshape((-1,1)) for iter in range(len(obsPoints)) ], axis=1 ), axis=0)
runy = fem.mesh.getObsMat( midPoint.reshape((1,-1)) )
runy = runy.transpose()
runy = fem.multiplyWithCovariance(runy)
runy = obsMat.tocsr() * runy
plt.imshow( runy.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
           extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.colorbar()











        




# %% Conditional distribution

# Set condition points
condPoints = np.array( [ [0.4,0.4], [0.55,0.45], [0.83, 0.9] ] )
# Get observation matrix
condObsMat = fem.mesh.getObsMat( condPoints ).tocsc()
# set conditional values
condVal = np.array( [1.2, 1.1, -0.5] )
# Compute conditional distribution
condDistr = fem.cond(condVal, condObsMat, sigmaEps)
# Get conditional mean at observation points
condMean = obsMat.tocsr() * condDistr.mu

# Plot
plt.figure(2)
ax = plt.subplot(2,2,1)
plt.cla()
ax.set_title( "Conditional mean" )
plt.imshow( condMean.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
            extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.colorbar()










# %% Learn spatial values (anisotropic)

print( "--- Start optimizing for pointwise anisotropic Matérn ---" )



def loglik( data, directionTensor, params ):
    # Compute log-likelihood function for data
    
    # Get number of replicates
    M = data.shape[1]
    
    # Acquire nu 
    nu = params["nu"]
    # Acquire vectors and ranges
    v1 = params["mainVec"]
    r1 = np.linalg.norm(v1)
    v1 = v1.reshape((-1,1)) / r1
    v2 = np.matmul( np.array([[0,-1], [1,0]]), v1 )
    r2 = params["orthR"]
    # Acquire standard deviation of error
    sigmaEps = params["sigmaEps"]
    
    if nu <= 0:
        return -np.inf
    
    # Acquire anisotropic matern covariances
    Sigma = GRF.anisotropicMaternCorr( directionTensor.transpose((2,0,1)).reshape( (2,-1) ), 
          nu, np.concatenate( (v1,v2), axis=1 ), np.array([r1,r2]) ) 
    Sigma = Sigma.reshape( (directionTensor.shape[0], directionTensor.shape[1]) )
        
    # Adjust for nugget effect
    Sigma = Sigma / ( 1 + sigmaEps**2 )
    Sigma[range(directionTensor.shape[0]), range(directionTensor.shape[0])] = Sigma[range(directionTensor.shape[0]), range(directionTensor.shape[0])] + sigmaEps**2
    assert(not np.any(np.isnan(Sigma)))
    # Decompose
    chol = linalg.cholesky(Sigma, lower=True)
    # Compute log determinant of covariance matrix
    lDet = 2 * np.sum(np.log(np.diag(chol)))    
    y = linalg.cho_solve( (chol, True), data )        
    # Compute log-likelihood
    l = - 0.5 * M * np.log(2*np.pi) - 0.5 * lDet - 0.5/M * np.sum(y * data)
    
    return l
    

def optimTrans( x ):
    # Function transforming parameters to actual values from unconstrained ones 
    return [ x[0], x[1], np.exp(x[2]), np.exp(x[3]), np.exp(x[4]) ]
def optimTransInv( x ):
    # Function transforming parameters to unconstrained ones from actual values 
    return [ x[0], x[1], np.log(x[2]), np.log(x[3]), np.log(x[4]) ]
def convertToStruct( x ):
    nu = x[3]
    # Acquire vectors and ranges
    v1 = np.array(x[0:2])
    r1 = np.linalg.norm(v1)
    v1 = v1.reshape((-1,1)) / r1
    v2 = np.asarray(np.matmul( mesher.geometrical.getRotationMatrix(np.pi/2), v1 ))
    r2 = x[2]
    sigmaEps = x[4]
    
    return { "v" : np.concatenate((v1,v2), axis=1), "r" : np.array([r1,r2]), "nu" : nu, "sigmaEps" : sigmaEps }

def getAnisotropicPoly( x, v ):    
    out = x.reshape((1,2)) + np.array([ v[:,0], v[:,1], -v[:,0], -v[:,1]])
    return Polygon(out)



# Choose random set of points
useIndex = random.sample( range(ZObs.shape[0]), 60 )
# Acquire difference between all points
diff = np.nan * np.ones( (len(useIndex), len(useIndex), 2) )
for iter in range(diff.shape[0]):
    diff[iter, :, 0] = obsPoints[0].flatten()[useIndex[iter]] - obsPoints[0].flatten()[useIndex]
    diff[iter, :, 1] = obsPoints[1].flatten()[useIndex[iter]] - obsPoints[1].flatten()[useIndex]
# Select chosen parts of sample    
data = ZObs[ useIndex, : ]  

# Define function to minimize
def optimFunc( x ):
    out = optimTrans( x )
    out = {"mainVec": np.array(out[0:2]), "orthR":out[2], "nu":out[3], "sigmaEps":out[4]}
    return - loglik( data, diff, out )
    

v = np.asarray( np.matmul( mesher.geometrical.getRotationMatrix(angle), np.array([[1,0], [0,1]]) ) )

# Set initial value
x0 = optimTransInv( [ r[1] , 0, r[0], 0.3, 1e-4 ] )    
print( "start value: " + str(optimTrans(x0)) )
print( "Initial log-lik: " + str( optimFunc( x0 ) ) )

    
# Optimize in two ways
resultOptim1 = None
optimalParams = None
try:
    resultOptim1 = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': True, "maxiter":200, "gtol": 1e-8} )
    optimalParams = optimTrans( resultOptim1.x )
except: 
    pass
resultOptim2 = None
try:
    resultOptim2 = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': True, "maxiter":2000} )    
    if optimalParams is None:
        optimalParams = optimTrans( resultOptim2.x )    
    elif resultOptim2.fun < resultOptim1.fun:
        optimalParams = optimTrans( resultOptim2.x )    
except:
    pass
optimalParams = convertToStruct(optimalParams)
    
# Plot
plt.figure(2)
ax = plt.subplot(222)
plt.cla()
ax.set_title( "Pointwise estimate" )

# Compute SPDE covariance
midPoint = np.mean( np.concatenate( [ obsPoints[iter].reshape((-1,1)) for iter in range(len(obsPoints)) ], axis=1 ), axis=0)
runy = fem.mesh.getObsMat( midPoint.reshape((1,-1)) )
runy = runy.transpose()
runy = fem.multiplyWithCovariance(runy)
runy = obsMat.tocsr() * runy
plt.imshow( runy.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
           extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )

# Draw initial
initialParams = convertToStruct(optimTrans(x0))
poly = [ getAnisotropicPoly(midPoint, initialParams["r"].reshape((1,2)) * initialParams["v"] ) ]
ax.add_collection(PatchCollection( poly, facecolor='blue', edgecolor='k', alpha=0.2, linewidths=0.5))

# Draw found
poly = [ getAnisotropicPoly(midPoint, optimalParams["r"].reshape((1,2)) * optimalParams["v"] ) ]
ax.add_collection(PatchCollection( poly, facecolor='red', edgecolor='k', alpha=0.2, linewidths=0.5))






plt.show()