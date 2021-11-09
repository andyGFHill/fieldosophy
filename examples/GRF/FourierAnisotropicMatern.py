#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating a Matérn spectral approximation model in 2 dimensions in the plane.
    * Generate samples from this model.
    * Compute correlation (and compare with theoretical correlation).
    * Estimating the smoothness parameter.
    * Using indexing to avoid uncessesary points in space


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

from matplotlib import pyplot as plt

import numpy as np
from scipy import stats
from scipy import optimize

from fieldosophy import mesh as mesher
from fieldosophy import GRF


plt.figure(1)
plt.clf()


nu = 0.8
shape = np.array([150,150])
region = np.array([ [0, 20], [0,20] ])

rotMat = np.asarray( mesher.geometrical.getRotationMatrix( 45.0 * np.pi / 180.0 ) )
D = np.sqrt(8*nu)/np.array([7, 3])
G = np.matmul( rotMat, np.matmul( np.diag( D**(-2.0) ), rotMat.transpose() ) )

Fourier = GRF.spectral.Fourier( shape = shape, region = region )
Fourier.setSpectralDensity( Fourier.anisotropicMatern( nu = nu, G = G ) )


# %% Simulate

samples = Fourier.generate( size = 100 )

plt.subplot(221)
plt.imshow(samples[:,:,0], extent = region[-1::-1,:].flatten() )
plt.title("Realization")



# %% Compute covariance
corrKernel = np.zeros( shape )
corrKernel[int(np.ceil(shape[0]/2)):int(np.ceil(shape[0]/2)+1), int(np.ceil(shape[1]/2)):int(np.ceil(shape[1]/2)+1)] = 1.0

corrKernel = Fourier.multiplyCov( corrKernel )[:,:,0]

plt.subplot(222)
plt.imshow(corrKernel, extent = region[-1::-1,:].flatten() )
plt.title("Covariance kernel")


# Compute theoretical Matérn correlation
runx = np.linspace( 0, np.diff(region[0,:])[0]/2, num = int(np.floor(corrKernel.shape[0]/2)) )
runy = GRF.GRF.MaternCorr( runx, nu = nu, kappa = np.dot(np.linalg.solve(G, np.eye(2)[:,0:1]).flatten(), np.eye(2)[:,0])**(0.5) )

plt.subplot(223)
plt.plot(runx, runy, label = "Matern", color="blue")
plt.plot(runx, corrKernel[int(np.ceil(shape[0]/2)):, int(np.ceil(shape[1]/2))], label = "Fourier", color="red")
plt.title("Comparison with true Matern")
plt.legend()



# %% Estimate parameters

print("Optimizing parameters")

# Get a copy of the original distribution
nuLim = np.array( [0.1,10] )
def optimTrans( x ):
    return stats.logistic.cdf(x)*np.diff(nuLim)[0] + nuLim[0]
def optimTransInv( x ):
    return stats.logistic.ppf((x-nuLim[0])/np.diff(nuLim)[0])
# Define function to optimize
def optimFunc( x ):
    # Transform from unconstrained to constrained value
    nuTemp = optimTrans( x[0] )
    # Update current system
    Fourier.setSpectralDensity( Fourier.anisotropicMatern( nu = nuTemp, G = G ) )
    # Compute log-lik
    logLik = Fourier.logLik( samples )
    # Return minus log-likelihood
    return - logLik
# Set initial value
x0 = [ optimTransInv( 0.9 ) ]
# Optimize ("BFGS")
# resultOptim = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': True, "maxiter":10, "gtol": 1e-1} )
# # resultOptim = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': True, "maxiter":200} )
# # Get result
# nuEst = optimTrans( resultOptim.x[0] )
# print( "Found smoothness: " + str(nuEst) )



# %% Use indices to mask away ninteresting region


mask = np.zeros(shape, dtype=bool)
mask[0:int(np.ceil(shape[0]/2)), 0:int(np.ceil(shape[1]/2))] = True

plt.subplot(224)
plt.imshow(mask)

corrKernel2 = np.zeros( shape )
# corrKernel2[int(shape[0]/4):int(shape[0]/4+1), int(shape[1]/4):int(shape[1]/4+1)] = 1.0
corrKernel2[0, 0] = 1.0
corrKernel2 = corrKernel2[ mask ].reshape( (int(np.ceil(shape[0]/2)), int(np.ceil(shape[1]/2))) )


# corrKernel2 = Fourier.multiplyCov( corrKernel2.flatten(), input_indices = mask.flatten())[:,:,0]
corrKernel2 = Fourier.multiplyCov( corrKernel2.flatten(), input_indices = mask.flatten(), output_indices = mask.flatten() ).reshape( (int(np.ceil(shape[0]/2)), int(np.ceil(shape[1]/2))) )
plt.imshow(corrKernel2)#, extent = region[-1::-1,:].flatten() )








