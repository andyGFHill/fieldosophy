#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating a Matérn FEM approximation model in 1 dimensions.
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
import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize

        
plt.figure(1)
plt.clf()

print("Running one-dimensional FEM test case")


# %% Create system

print("Creating system")

# Number of nodes
N = int(2e2)

# Define the Matérn random field
r = 0.55 # Set correlation range (range for which two points have approximately 0.1 correlation)
nu = 1.51   # Set smoothness 
sigma = 1   # Set standard deviation
sigmaEps = 1e-2     # Set noise level

# Create 1D mesh
nodes = np.linspace(0-1*r,1+1*r, num=N, dtype=np.double).reshape((-1,1))
triangles = np.array([np.arange(0,N-1), np.arange(1,N)], dtype = np.uintc).transpose().copy()
triangles = triangles[:, [1,0]].copy()
mesh = mesher.Mesh( triangles = triangles, nodes = nodes )
mesh.getBoundary()

# Create Dirichlet boundary condition
BCDirichlet = np.NaN * np.ones((mesh.N))
BCDirichlet[mesh.getBoundary()["nodes"]] = 0
BCDirichlet = None
# Create Robin boundary condition
BCRobin = np.ones( (mesh.boundary["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 0.3 # Association with function
# BCRobin = None
# Create MaternFEM object 
fem = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )

# %% Generate samples

# Number of realizations
M = int(2e3)
# Number of observation points
J = 200

print("Simulate " + str(M) + " realizations")

# Acquire realizations on nodes
Z = fem.generateRandom( M )

# Set observation points
obsPoints = np.linspace( 0, 1, num = J, dtype=np.float64 ).reshape((-1,1))
# obsPoints = mesh.nodes[(mesh.nodes[:,0] < 1) & (mesh.nodes[:,0] > 0), : ]
# obsPoints = mesh.nodes
# J = obsPoints.shape[0]

# Get observation matrix
obsMat = fem.mesh.getObsMat( obsPoints )
# Acquire observations at observation points
Z = obsMat.tocsr() * Z
Z = Z + stats.norm.rvs( loc = 0, scale = sigmaEps, size = J*M ).reshape((J, M))



ax = plt.subplot(2,2,1)
ax.cla()

ax.set_title( "Realization" )
plt.plot( obsPoints, Z[:,0] )
plt.xlabel("Time [s]")



# %% Plot correlation function



ax = plt.subplot(2,2,2)
ax.cla()
ax.set_title( "Correlation function" )

# find mid point
midPointInd = np.argsort(obsPoints.flatten())[int(J/2)]
runx = obsPoints.flatten()
distance = np.sqrt( np.abs( runx[midPointInd]-runx )**2 )
orderIndex = np.argsort( distance )
runx = distance[orderIndex]

# Handle estimated correlation from realizations
runy = Z[midPointInd, :].reshape((1,-1)) * Z 
runy = np.mean(runy, axis=1)
runy = runy / np.std(Z[midPointInd, :])
runy[1:-1] = runy[1:-1] / np.std( Z[1:-1, :], axis=1 )
runy = runy[orderIndex]
plt.plot(runx, runy, label = "SPDE emprirical", color="green", linestyle="dashed")

# Plot SPDE correlation
runy = sparse.coo_matrix( (np.array([1]), ( np.array([midPointInd]), np.array([0])) ), \
          shape = (obsPoints.shape[0], 1) )
runy = obsMat.transpose() * runy
runy = fem.multiplyWithCovariance(runy)
runy = obsMat * runy 
runy = runy[orderIndex]
plt.plot(runx, runy, label = "SPDE", color="red", linewidth=2)

# Compute theoretical Matérn correlation
runy = GRF.MaternCorr( runx, nu = nu, kappa = np.sqrt(8*nu)/r )
plt.plot(runx, runy, label = "Matern", color="blue")

plt.legend()
plt.xlabel("Time [s]")
# plt.xlim((0,0.5))
plt.ylim((0,1.2))





# %% Conditional distribution

print("Compute conditional distribution")

# Set condition points
condPoints = np.array( [0.4, 0.85 ] )
# Get observation matrix
condObsMat = fem.mesh.getObsMat( condPoints ).tocsc()
# set conditional values
condVal = np.array( [1.4, -0.85] )
# Compute conditional distribution
condDistr = fem.cond(condVal, condObsMat, sigmaEps = 0.3)
# Get conditional mean at observation points
condMean = obsMat.tocsr() * condDistr.mu
# Get conditional standard deviation
condVar = obsMat.tocsr() * condDistr.multiplyWithCovariance( obsMat.transpose().tocsc() ) 
condVar = np.diag(condVar)


print( "Original log-lik: " + str(fem.loglik( condVal, condObsMat.tocsc(), sigmaEps=0.3)) )
print( "Conditional log-lik: " + str(condDistr.loglik( condVal, condObsMat.tocsc(), sigmaEps=0.3)) )





ax = plt.subplot(2,2,3)
ax.cla()
ax.set_title( "Conditional distribution" )
# Plot conditional mean
plt.plot( obsPoints, condMean )
# Plot conditional marginal interquartile range
plt.fill_between( obsPoints.flatten(), \
     stats.norm.ppf( 0.25, loc = condMean, scale = np.sqrt(condVar) ), \
     stats.norm.ppf( 0.75, loc = condMean, scale = np.sqrt(condVar) ), color = [0,1,0], label="STDs" )
# Plot points to condition on
plt.scatter( condPoints, condVal, color="red")
plt.xlabel("Time [s]")



# %% Parameter estimation

print("Estimate range parameter")



# Get a copy of the original distribution
paramEst = fem.copy()
rangeLim = np.array( [np.max(np.diff(mesh.nodes[:,0]))*5, 1] )
def optimTrans( x ):
    return stats.logistic.cdf(x)*np.diff(rangeLim)[0] + rangeLim[0]
def optimTransInv( x ):
    return stats.logistic.ppf((x-rangeLim[0])/np.diff(rangeLim)[0])
# Define function to optimize
def optimFunc( x ):
    # Transform from unconstrained to constrained value
    r = optimTrans( x[0] )
    # Update current system
    paramEst.updateSystem( {'r':r}, nu=nu, sigma=sigma, BCRobin = BCRobin )
    # Compute log-lik
    logLik = paramEst.loglik( Z, obsMat.tocsc(), sigmaEps=sigmaEps)
    # Return minus log-likelihood
    return - logLik
# Set initial value
x0 = [ optimTransInv( 0.3 ) ]
# Optimize ("BFGS")
# resultOptim = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': True, "maxiter":20, "gtol": 1e-1} )
resultOptim = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': True, "maxiter":200} )
# Get result
rEst = optimTrans( resultOptim.x[0] )
print( "Found range: " + str(rEst) )


ax = plt.subplot(2,2,4)
ax.cla()
ax.set_title( "Correlation function estimated" )

# find mid point
midPointInd = np.argsort(obsPoints.flatten())[int(J/2)]
runx = obsPoints.flatten()
distance = np.sqrt( np.abs( runx[midPointInd]-runx )**2 )
orderIndex = np.argsort( distance )
runx = distance[orderIndex]

# Plot SPDE correlation
runy = sparse.coo_matrix( (np.array([1]), ( np.array([midPointInd]), np.array([0])) ), \
          shape = (obsPoints.shape[0], 1) )
runy = obsMat.transpose() * runy
runy = paramEst.multiplyWithCovariance(runy)
runy = obsMat * runy 
runy = runy[orderIndex]
plt.plot(runx, runy, label = "SPDE", color="red", linewidth=2)


# Compute theoretical Matérn correlation
runy = GRF.MaternCorr( runx, nu = nu, kappa = np.sqrt(8*nu)/r )
plt.plot(runx, runy, label = "True", color="blue")
# Compute estimated 
runy = GRF.MaternCorr( runx, nu = nu, kappa = np.sqrt(8*nu)/rEst )
plt.plot(runx, runy, label = "Estimated", color="green")

plt.legend()
plt.xlabel("Time [s]")






plt.show()



