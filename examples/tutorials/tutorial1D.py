#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script corresponds to the code in the tutorial on 1D modeling using the SPDE-approach that can be found in the documentation, see "https://andygfhill.github.io/fieldosophy".

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""




import numpy as np
np.random.seed = 1


# %% Constructing the mesh

# Create mesh
import numpy as np
N = 500
nodes = np.linspace( 0,1, N ).reshape((-1,1))
simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
from fieldosophy import mesh as mesher
mesh = mesher.Mesh( triangles = simplices, nodes = nodes )


# %% Constructing a Matérn model


# Define the Matérn random field
r = 0.4 # Set correlation range (range for which two points have approximately 0.13 correlation)
nu = 1.5   # Set smoothness (basically the Hölder constant of realizations)
sigma = 2   # Set standard deviation
# Create MaternFEM object 
from fieldosophy.GRF import FEM
fem = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma )

# Acquire realizations on nodes
Z = fem.generateRandom( 2 )
# Plot realizations
from matplotlib import pyplot as plt
plt.figure(1)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z[:,0] )
plt.title( "Realization 1" )
# plt.savefig(figpath+"realization1.png")
plt.figure(2)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z[:,1] )
plt.title( "Realization 2" )
# plt.savefig(figpath+"realization2.png")



# %% Adding boundary conditions

# Create Dirichlet boundary condition
BCDirichlet = np.NaN * np.ones((mesh.N))
BCDirichlet[[0, 300]] = np.array([3, -0.5])
# Create new fem model
fem2 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet )
# Acquire realizations on nodes
Z2 = fem2.generateRandom( 2 )

plt.figure(1)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z2[:,0] )
plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
plt.title( "Realization 1 with Dirichlet conditions" )
# plt.savefig(figpath+"realizationWithBCDirichlet1.png")
plt.figure(2)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z2[:,1] )
plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
plt.title( "Realization 2 with Dirichlet conditions" )
# plt.savefig(figpath+"realizationWithBCDirichlet2.png")


# Create Robin boundary condition
BCRobin = np.ones( (2, 2) )
BCRobin[1, 0] = 0 # Association with constant
BCRobin[1, 1] = -1 # Association with function
# Update new fem model
fem3 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )
# Acquire realizations on nodes
Z3 = fem3.generateRandom( 2 )

plt.figure(1)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z3[:,0] )
plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
plt.title( "Realization 1 with Dirichlet and Robin conditions" )
# plt.savefig(figpath+"realizationWithBCDirRobin1.png")
plt.figure(2)
plt.clf()
plt.plot( mesh.nodes.flatten(), Z3[:,1] )
plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
plt.title( "Realization 2 with Dirichlet and Robin conditions" )
# plt.savefig(figpath+"realizationWithBCDirRobin2.png")

# %% Extending the mesh

# Create new mesh extended by r in each direction
N = int( 500 * ( (2*r+1)/1 ) )
nodes = np.linspace( 0-r,1+r, N ).reshape((-1,1))
simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
extendedMesh = mesher.Mesh( triangles = simplices, nodes = nodes )

# Create Robin boundary condition
BCRobin = np.ones( (2, 2) )
BCRobin[:, 0] = 0 # Association with constant
BCRobin[:, 1] = -1 # Association with function

# Create fem model
fem4 = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCRobin = BCRobin )
# Acquire realizations on nodes
Z4 = fem4.generateRandom( 2 )

plt.figure(1)
plt.clf()
plt.plot( extendedMesh.nodes.flatten(), Z4[:,0] )
plt.title( "Realization 1 with extended mesh" )
# plt.savefig(figpath+"realizationExtended1.png")
plt.figure(2)
plt.clf()
plt.plot( extendedMesh.nodes.flatten(), Z4[:,1] )
plt.title( "Realization 2 with extended mesh" )
# plt.savefig(figpath+"realizationExtended2.png")



# %% Compute covariances on extended

# Compute covariance
referenceNode = np.zeros((extendedMesh.N,1))
referenceNode[500] = 1
covSPDE = fem4.multiplyWithCovariance( referenceNode )

# Compare with actual matern covariance
from fieldosophy.GRF import GRF
covMatern = sigma**2 * GRF.MaternCorr( np.abs(extendedMesh.nodes[500,0] - extendedMesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )

# Plot covariances
plt.figure(1)
plt.clf()
plt.plot( extendedMesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
plt.vlines( extendedMesh.nodes[500,0], 0, 4, linestyle='--' )
plt.title( "Covariance between point 0.601 and all other" )
plt.plot( extendedMesh.nodes.flatten(), covMatern, color="red", label="Matern" )
plt.legend()
# plt.savefig(figpath+"covariancesExtendedComparison.png")


# %% Compute covariances on original mesh

# Compute covariance
referenceNode = np.zeros((mesh.N,1))
referenceNode[300] = 1
covSPDE = fem.multiplyWithCovariance( referenceNode )

# Compare with actual matern covariance
from fieldosophy.GRF import GRF
covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[300,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )

# Plot covariances
plt.figure(1)
plt.clf()
plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
plt.vlines( mesh.nodes[300,0], 0, 4, linestyle='--' )
plt.title( "Covariance between point 0.601 and all other" )
plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
plt.legend()
# plt.savefig(figpath+"covariancesComparisonMiddle.png")

# Compute covariance
referenceNode = np.zeros((mesh.N,1))
referenceNode[0] = 1
covSPDE = fem.multiplyWithCovariance( referenceNode )

# Compare with actual matern covariance
from fieldosophy.GRF import GRF
covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[0,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )

# Plot covariances
plt.figure(2)
plt.clf()
plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
plt.vlines( mesh.nodes[0,0], 0, 4, linestyle='--' )
plt.title( "Covariance between point 0 and all other" )
plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
plt.legend()
# plt.savefig(figpath+"covariancesComparisonLeft.png")


# %% Compare covariances with Dirichlet and Robin

# Create Dirichlet boundary condition
BCDirichlet = np.NaN * np.ones((mesh.N))
BCDirichlet[0] = 0
# Create Robin boundary condition
BCRobin = np.ones( (2, 2) )
BCRobin[1, 0] = 0 # Association with constant
BCRobin[1, 1] = -0.26 # Association with function
# Update new fem model
fem5 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )

temp = 300

# Compute covariance
referenceNode = np.zeros((mesh.N,1))
referenceNode[temp] = 1
covSPDE = fem5.multiplyWithCovariance( referenceNode )

# Compare with actual matern covariance
from fieldosophy.GRF import GRF
covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[temp,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )

# Plot covariances
plt.figure(2)
plt.clf()
plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
plt.vlines( mesh.nodes[temp,0], 0, 4, linestyle='--' )
plt.title( "Covariance between point 0.601 and all other" )
plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
plt.legend()
# plt.savefig(figpath+"covariancesComparisonDirichletAndRobin.png")



# %% Conditional distribution

# Set points of measurement
condPoints = np.array( [0.2, 0.9 ] ).reshape((-1,1))
# Get observation matrix for measurement points
condObsMat = fem4.mesh.getObsMat( condPoints ).tocsc()
# Set conditional values
condVal = np.array( [2.0, -1.0] )
# Set measurement noise
sigmaEps = np.sqrt(np.array([0.3, 0.05]))
# Compute conditional distribution
condDistr = fem4.cond(condVal, condObsMat, sigmaEps = sigmaEps)

# Define analysis points on the unit interval
anPoints = np.linspace(0,1,100).reshape((-1,1))
# Get observation matrix for analysis points
anObsMat = fem4.mesh.getObsMat( anPoints ).tocsc()
# Get posterior mean mean at analysis points
condMean = anObsMat.tocsr() * condDistr.mu
# Generate from the posterior random field and get values at analysis points
condZ = anObsMat.tocsr() * condDistr.generateRandom( 5 )
# Get covariance matrix for posterior Gaussian random field at analysis points
condVar = anObsMat.tocsr() * condDistr.multiplyWithCovariance( anObsMat.transpose().tocsc() ) 
# Use only the variances, i.e., the diagonal of the covariance matrix
condVar = np.diag(condVar)

# Plot conditional distribution
plt.figure(1)
plt.clf()
# Plot conditional marginal 90% prediction interval
from scipy import stats
plt.fill_between( anPoints.flatten(), \
     stats.norm.ppf( 0.05, loc = condMean, scale = np.sqrt(condVar) ), \
     stats.norm.ppf( 0.95, loc = condMean, scale = np.sqrt(condVar) ), color = [0,1,0], label="STDs" )
# Plot realizations
plt.plot( anPoints.flatten(), condZ[:,0], color="gray", label="Realization 1", linestyle="--", linewidth=1)
plt.plot( anPoints.flatten(), condZ[:,1], color="gray", label="Realization 2", linestyle="--", linewidth=1)
plt.plot( anPoints.flatten(), condZ[:,2], color="gray", label="Realization 3", linestyle="--", linewidth=1)
plt.plot( anPoints.flatten(), condZ[:,3], color="gray", label="Realization 4", linestyle="--", linewidth=1)
plt.plot( anPoints.flatten(), condZ[:,4], color="gray", label="Realization 5", linestyle="--", linewidth=1)
# Plot posterior mean
plt.plot( anPoints.flatten(), condMean, color="red", label="Mean", linewidth=2)
# Plot measured values
plt.scatter( condPoints.flatten(), condVal, label="Measurements", color="blue" )

plt.legend( loc='upper right' )
# plt.savefig(figpath+"conditionalRandomField.png")




# %% likelihood

# Compute log-likelihood of fem4 model
logLik = fem4.loglik( condVal, condObsMat.tocsc(), sigmaEps=sigmaEps)


# Create Robin boundary condition
BCRobin = np.ones( (2, 2) )
BCRobin[:, 0] = 0 # Association with constant
BCRobin[:, 1] = -0.26 # Association with function
# Create FEM object
femLoglik = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCRobin = BCRobin, mu = condDistr.mu )

# compute
logLik2 = femLoglik.loglik( condVal, condObsMat.tocsc(), sigmaEps=sigmaEps)




# %% Estimate parameters using maximum likelihood


# Create Robin boundary condition
BCRobin = np.ones( (2, 2) )
BCRobin[:, 0] = 0 # Association with constant
BCRobin[:, 1] = -0.26 # Association with function

# Define new Matérn random field
rTrue = 0.2 # Set correlation range (range for which two points have approximately 0.13 correlation)
nuTrue = 2.5   # Set smoothness (basically the Hölder constant of realizations)
# Create MaternFEM object 
femTrue = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':rTrue}, nu = nuTrue, sigma = sigma, BCRobin = BCRobin )

# Generate 100 realizations
ZTrue = femTrue.generateRandom( 100 )

# Define 20 "measurement points" randomly over the unit interval
measPoints = np.random.rand( (20) ).reshape((-1,1))
# Get observation matrix 
measObsMat = femTrue.mesh.getObsMat( measPoints ).tocsc()
# Get observations to measurement points
ZMeas = measObsMat.tocsr() * ZTrue
# Define measurement standard deviation (the same for all points)
sigmaEpsTrue = 0.2
# Add measurement noise
ZMeas = ZMeas + stats.norm.rvs( loc = 0, scale = sigmaEpsTrue, size = 20*100 ).reshape((20, 100))

plt.figure(1)
plt.clf()
# Plot realizations
plt.plot( extendedMesh.nodes.flatten(), ZTrue[:,0], label="Z" )
plt.scatter( measPoints.flatten(), ZMeas[:,0], label = "ZMeas")
plt.legend( loc='upper right' )
plt.title("Realization 1")
# plt.savefig(figpath+"trueRealization1.png")
plt.figure(2)
plt.clf()
# Plot realizations
plt.plot( extendedMesh.nodes.flatten(), ZTrue[:,1], label="Z" )
plt.scatter( measPoints.flatten(), ZMeas[:,1], label = "ZMeas")
plt.legend( loc='upper right' )
plt.title("Realization 2")
# plt.savefig(figpath+"trueRealization2.png")



# Create new mesh
N = 70 * 3
nodes = np.linspace( -1,2, N ).reshape((-1,1))
simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
mesh = mesher.Mesh( triangles = simplices, nodes = nodes )
# Create observation matrix for points
obsMat = mesh.getObsMat( measPoints ).tocsc()

# Define the Matérn random field
r = 0.4 # Set correlation range (range for which two points have approximately 0.13 correlation)
nu = 1.5   # Set smoothness (basically the Hölder constant of realizations)
sigma = 2   # Set standard deviation
sigmaEps = 0.1 # Set the measurement noise standard deviation
femNew = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCRobin = BCRobin )

# loglik
loglik = femNew.loglik( ZMeas, obsMat.tocsc(), sigmaEps=sigmaEps )
loglik2 = femTrue.loglik( ZMeas, measObsMat.tocsc(), sigmaEps=sigmaEpsTrue )


def optimTrans( x ):
    # Function for mapping nu and r back to original range
    
    y = x.copy()
    
    y[0] = stats.logistic.cdf(y[0]) * 3.0 + 1.0
    y[1] = stats.logistic.cdf(y[1]) * 0.9 + 0.1
    y[2] = stats.logistic.cdf(y[2]) * 0.99 + 0.01
    
    return y
    
def optimTransInv( x ):
    # Function for mapping nu and r to an unbounded space
    
    y = x.copy()
    
    y[0] = stats.logistic.ppf( (y[0] - 1.0 ) / 3.0 )
    y[1] = stats.logistic.ppf( (y[1] - 0.1 ) / 0.9 )
    y[2] = stats.logistic.ppf( (y[2] - 0.01 ) / 0.99 )
    
    return y
    

def optimFunc( x ):
    # function to optmize, in this case the log-likelihood after transformation
    
    # Transform from unconstrained to constrained value
    y = optimTrans( x )
    nuCur = y[0]
    rCur = y[1]
    sigmaEpsCur = y[2]
    
    # Update current system
    femNew.updateSystem( childParams = {'r':rCur}, nu=nuCur, sigma=sigma, BCRobin = BCRobin )
    # Compute log-lik
    logLik = femNew.loglik( ZMeas, obsMat.tocsc(), sigmaEps=sigmaEpsCur)
    # Return minus log-likelihood
    return - logLik    


from scipy import optimize

# Set initial value
x0 = optimTransInv( [ nu, r, sigmaEps ] )
# Optimize ("BFGS")
# resultOptim = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': True, "maxiter":20, "gtol": 1e-1} )
resultOptim = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': True, "maxiter":200} )
# Get result
nuEst, rEst, sigmaEpsEst = optimTrans( resultOptim.x )


plt.figure(1)
plt.clf()
# Plot realizations
plt.plot( np.linspace(0.01, 0.4, 500), GRF.MaternCorr( np.linspace(0.01, 0.4, 500), nu = nuEst, kappa = np.sqrt(8*nuEst)/rEst ), label="Estimated", color = "black", linewidth = 2 )
plt.plot( np.linspace(0.01, 0.4, 500), GRF.MaternCorr( np.linspace(0.01, 0.4, 500), nu = nuTrue, kappa = np.sqrt(8*nuTrue)/rTrue ), label="True", color="red" )
plt.legend()
# plt.savefig(figpath+"comparisonCovarianceTrueEstimated.png")



# %% non-stationary

def mapFEMParamsToG( params ):
    
    GInv = [ (params["r"] / np.sqrt(8*nu) )**2 ]
    logGSqrt = - 0.5 * np.log( GInv[0] )
    
    return (logGSqrt, GInv)

# Create FEM object
nonstatfem = FEM.nonStatFEM( mesh = extendedMesh, childParams = {"r":r, "f":mapFEMParamsToG}, nu = nu, sigma = sigma, BCRobin = BCRobin )


# Compute the middle point in each simplex 
simplexMeanPoints = np.mean( extendedMesh.nodes[ extendedMesh.triangles , 0], axis=1 )
# Use simplixes middle points to set the local correlation range
rLocal = simplexMeanPoints - 0.1
# Set the extensions to the original value of 0.4
rLocal[simplexMeanPoints < 0 ] = 0.4
rLocal[simplexMeanPoints > 1 ] = 0.4
# Create FEM object
nonstatfem = FEM.nonStatFEM( mesh = extendedMesh, childParams = {"r":rLocal, "f":mapFEMParamsToG}, nu = nu, sigma = sigma, BCRobin = BCRobin )


# Get covariance between the point in the middle of the interval and all other points
referenceNode = np.zeros((extendedMesh.N,1))
referenceNode[450] = 1
covSPDE = nonstatfem.multiplyWithCovariance( referenceNode )
covSPDE = nonstatfem.multiplyWithCovariance( referenceNode )
# Compare with actual matern covariance
covMatern = sigma**2 * GRF.MaternCorr( np.abs(extendedMesh.nodes[450,0] - extendedMesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )


# Plot covariances
plt.figure(1)
plt.clf()
plt.plot( extendedMesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
plt.vlines( extendedMesh.nodes[450,0], 0, 4, linestyle='--' )
plt.title( "Covariance between point 0.5 and all other" )
plt.plot( extendedMesh.nodes.flatten(), covMatern, color="red", label="Matern" )
plt.legend()
# plt.savefig(figpath+"nonStatCovariance.png")

# Plot non-stationarity
plt.figure(2)
plt.clf()
plt.plot( simplexMeanPoints, rLocal, color="black")
plt.title( "Local correlation ranges" )
# plt.savefig(figpath+"localCorelationRange.png")



# Acquire realizations on nodes
ZNonStat = anObsMat.tocsr() * nonstatfem.generateRandom( 2 )
# Plot realizations
plt.figure(1)
plt.clf()
plt.plot( anPoints.flatten(), ZNonStat[:,0] )
plt.title( "Realization 1" )
# plt.savefig(figpath+"realization1Nonstat.png")
plt.figure(2)
plt.clf()
plt.plot( anPoints.flatten(), ZNonStat[:,1] )
plt.title( "Realization 2" )
# plt.savefig(figpath+"realization2Nonstat.png")





