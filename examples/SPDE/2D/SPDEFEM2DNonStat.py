#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating an non-stationary Matérn FEM approximation model in 2 dimensions.
    * Generate samples from this model.
    * Compute covariances.
    * Compute conditional distributions.
    * Estimate non-stationarity.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy.GRF import FEM
from fieldosophy import mesh as mesher
from fieldosophy.misc import misc_functions as misc


from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import random
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
corrMin = 0.1
extension = corrMin
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
nu = 2.7
sigma = 1
sigmaEps = 1e-3

# Get mid points of triangles
triPoints = np.mean( meshPlane.nodes[ meshPlane.triangles ], axis=1 )
angle = (45 + triPoints[:,0] * 90) * np.pi/180
r = np.array([corrMin, corrMin*3])

# Enforce barrier method
r = r.reshape((1,-1)) * np.ones((meshPlane.NT, 2))
r[np.any( (triPoints > 1) | (triPoints < 0), axis=1 ), :] = corrMin


def mapFEMParams( params ):
    # Function to map own parameters to FEM parameters   
    
    # Get basis of tangent spaces
    vectors = FEM.angleToVecs2D(params["angle"])
    
    # Compute kappa and H
    logGSqrt, GInv = FEM.orthVectorsToG( vectors.transpose((0,2,1)), params["r"] / np.sqrt(8*nu) )
    
    return (logGSqrt, GInv)



BCDirichlet = np.NaN * np.ones((meshPlane.N))
BCDirichlet[meshPlane.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with function
# BCRobin = None

# Create FEM object
fem = FEM.nonStatFEM( mesh = meshPlane, childParams = {'angle':angle, "r":r, "f":mapFEMParams}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )









# %% Sample

# Acquire realizations
print("Generate realizations")

M = int(2e2)

Z = fem.generateRandom( M )

# Set observation points
lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( 100 ) )
lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( 100 ) )
obsPoints = np.meshgrid( lons, lats )

# Get observation matrix
print("Acquire observation matrix")
obsMat = fem.mesh.getObsMat( np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ))
ZObs = obsMat.tocsr() * Z + stats.norm.rvs( loc = 0, scale = sigmaEps, size = M*obsMat.shape[0] ).reshape((obsMat.shape[0], M))



# %% Plot covariances



print("Plot covariances")


plt.figure(1)






ax = plt.subplot(223)
plt.cla()
ax.set_title( "A realization" )
# temp = obsMat * np.sqrt(np.sum(fem.mesh.nodes**2, axis=1))
# plt.imshow( temp.reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
#             extent = ( coordinateLims[0,0], coordinateLims[0,1], coordinateLims[1,0], coordinateLims[1,1] ) )
plt.imshow( ZObs[:,0].reshape(obsPoints[0].shape), origin="lower", aspect="equal", \
            extent = ( np.min(obsPoints[0]), np.max(obsPoints[0]), np.min(obsPoints[1]), np.max(obsPoints[1]) ) )
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








# %% Estimate values (non-stationary global)

print("Set up initial FEM system")

# Number of sine components in each direction
numXSinComps = 2
numYSinComps = 1

# Define the random field
nuEst = 2
sigmaEpsEst = sigmaEps*5

# Initial guess
angleEst = np.zeros(numXSinComps*numYSinComps)
r1Est = np.zeros(numXSinComps*numYSinComps)
r2Est = np.zeros(numXSinComps*numYSinComps)
angleEst[0] = 0
# angleEst[1] = -np.pi/4
r1Est[0] = np.log(corrMin*3)
r2Est[0] = -4


# Choose a subset of observation points
subsetObs = random.sample( range(ZObs.shape[0]), 600 )
ZTrain = ZObs[subsetObs, :]
obsMatTrain = obsMat[subsetObs,:].tocsc()


# Acquire coefficient basis matrix 
sinMatrix = misc.cosinBasisMatrix( triPoints, \
    np.stack( ( np.tile( np.array(range(numXSinComps)), numYSinComps ), np.repeat( np.array(range(numYSinComps)), numXSinComps ) ), axis=1 ).transpose(), \
    np.array([[0,1],[0,1]]) )

actionZone = (triPoints[:,0] <= 1) & (triPoints[:,0] >= 0) & (triPoints[:,1] <= 1) & (triPoints[:,1] >= 0)


def mapFEMParams( params ):
    # Function to map parameters to FEM parameters
    
    # Get angle
    curAngle = np.asarray( np.matmul( sinMatrix, params["angle"] ) )
    # Get ranges
    curR1 = corrMin + np.exp( np.asarray( np.matmul( sinMatrix, params["r1"] ) ) )
    curR2 = corrMin + (curR1-corrMin) * stats.logistic.cdf( np.asarray( np.matmul( sinMatrix, params["r2"] ) ) )
    curR = np.stack( (curR1, curR2), axis=1 )
    curR[actionZone, :] = corrMin    
    # Compute vectors
    vectors = FEM.angleToVecs2D(curAngle)
    vectors[actionZone, :, :] = np.eye(2).reshape((1,2,2))
    
    # Compute kappa and H
    logGSqrt, GInv = FEM.orthVectorsToG( vectors.transpose(), curR / np.sqrt(8*params["nu"]) )
    
    return (logGSqrt, GInv)


BCDirichlet = np.NaN * np.ones((meshPlane.N))
BCDirichlet[meshPlane.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with function
# BCRobin = None

# Create FEM object
femEst = FEM.nonStatFEM( mesh = meshPlane, childParams = {'nu':nuEst, "angle":angleEst, "r1":r1Est, "r2":r2Est, "f":mapFEMParams}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )



def optimTransInv( x ):
    out = x.copy()
    out[-2] = np.log(out[-2])
    out[-1] = np.log(out[-1])
    return(out)
    
def optimTrans( x ):
    out = x.copy()
    out[-2] = np.exp(out[-2])
    out[-1] = np.exp(out[-1])
    return(out)
# Define function to optimize
def optimFunc( x ):
    # Transform from unconstrained to constrained value
    x = optimTrans( x )
    # Update current system
    femEst.updateSystem( {'nu':x[-2], "angle":x[np.array(range(numXSinComps*numYSinComps))], \
        "r1":x[numXSinComps*numYSinComps + np.array(range(numXSinComps*numYSinComps))], \
        "r2":x[2*numXSinComps*numYSinComps + np.array(range(numXSinComps*numYSinComps))], "f":mapFEMParams}, \
        nu=x[-2], sigma=sigma/np.sqrt(sigma**2+x[-1]**2), BCRobin = BCRobin )
    # Return minus log-likelihood
#    return - paramEst.loglik( ZObs, obsMat.tocsc(), sigmaEps=x[-1])
    return - femEst.loglik( ZTrain, obsMatTrain, sigmaEps=x[-1])


# Set initial value
x0 = optimTransInv( np.concatenate( ( angleEst, r1Est, r2Est, np.array([nuEst]), np.array([sigmaEpsEst]) ), axis=0 ) )
# Compute initial log-likelihood
print( "Initial log-likelihood: " + str(-optimFunc(x0)) )


# Optimize 
resultOptim = None
rEst = x0
try:
    Nfeval = 1
    def callbackF(Xi):
        global Nfeval
        print( "Optim1 Iteration " + str(Nfeval) + " log-likelihood: " + str(-optimFunc(Xi)) )
        Nfeval += 1
    resultOptim = optimize.minimize( optimFunc, rEst, method='Nelder-Mead', options={'disp': True, "maxiter":5}, callback = callbackF )
    rEst = resultOptim.x
    print( "Found range: " + str(optimTrans(rEst)) )
    print( "Optim1 found log-likelihood: " + str(-resultOptim.fun) )
except:
    print( "Failed optimization 1!" )
try:
    Nfeval = 1
    def callbackF(Xi):
        global Nfeval
        print( "Optim2 Iteration " + str(Nfeval) + " log-likelihood: " + str(-optimFunc(Xi)) )
        Nfeval += 1
    resultOptim = optimize.minimize( optimFunc, rEst, method='BFGS', options={'disp': True, "maxiter":2, "gtol":1e-4}, callback = callbackF )
    rEst = resultOptim.x
    print( "Found range: " + str(optimTrans(rEst)) )
    print( "Optim2 found log-likelihood: " + str(-resultOptim.fun) )
except:
    print( "Failed optimization 2!" )    
try:
    Nfeval = 1
    def callbackF(Xi):
        global Nfeval
        print( "Optim3 Iteration " + str(Nfeval) + " log-likelihood: " + str(-optimFunc(Xi)) )
        Nfeval += 1
    resultOptim = optimize.minimize( optimFunc, rEst, method='Nelder-Mead', options={'disp': True, "maxiter":5}, callback = callbackF )
    rEst = resultOptim.x
    print( "Found range: " + str(optimTrans(rEst)) )
    print( "Optim3 found log-likelihood: " + str(-resultOptim.fun) )
except:
    print( "Failed optimization 3!" )


# Transform parameters
tempEst = optimTrans(rEst)
angleEst2 = tempEst[0*numXSinComps*numYSinComps + np.array(range(numXSinComps*numYSinComps))]
r1Est2 = tempEst[1*numXSinComps*numYSinComps + np.array(range(numXSinComps*numYSinComps))]
r2Est2 = tempEst[2*numXSinComps*numYSinComps + np.array(range(numXSinComps*numYSinComps))]
nuEst2 = tempEst[-2]
sigmaEpsEst2 = tempEst[-1]
    
fig = plt.figure(2)
ax = plt.subplot(222)
ax.cla()
ax.set_title( "Estimated" )  

def getAnisotropicPoly( x, v ):    
    out = x.reshape((1,2)) + np.array([ v[:,0], v[:,1], -v[:,0], -v[:,1]])
    return Polygon(out)

poly = []
for iter in np.arange(0, triPoints.shape[0], step = 50):
    
    if not actionZone[iter]:
        continue
    
    curAngle = np.asarray( np.matmul( sinMatrix[iter,:], angleEst2 ) )
    curR1 = corrMin + np.exp( np.asarray( np.matmul( sinMatrix[iter,:], r1Est2 ) ) )
    curR2 = corrMin + (curR1-corrMin) * stats.logistic.cdf( np.asarray( np.matmul( sinMatrix[iter,:], r2Est2 ) ) )
    
    # Compute vectors
    vectors = FEM.angleToVecs2D(curAngle)
    vectors = vectors * np.array([curR1, curR2]).reshape((1, -1))
    
    # Get polygon
    poly.append( getAnisotropicPoly( np.array([triPoints[iter, 0],triPoints[iter, 1]]), 1e-1 * vectors ) )
    ax.add_collection(PatchCollection( poly, facecolor='red', edgecolor='k', alpha=0.2, linewidths=0.5))

ax.axis('equal')





plt.show()
