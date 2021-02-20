# -*- coding: utf-8 -*-
"""
This script highlight:
    * How to define a normal-inverse Gaussian distribution.
    * How to sample from the defined distribution.
    * How to estimate the parameters of the distribution using method of moments, Gaussian approximation, expectation-maximization, and gradient descent.
    * How to access to functions such as the pdf and the transformation between standard Gaussian and the given NIG distribution.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""



from fieldosophy.marginal import NIG

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import stats


# %% Create distributions
    
# Create NIG distribution
NIGDistr1 = { "alpha":5/15, "beta":-4/15, "mu":4, "delta":15 }
NIGDistr1 = NIG.NIGDistribution(NIGDistr1)

NIGDistr2 = { "alpha":30/15, "beta":9/15, "mu":4, "delta":15 }
NIGDistr2 = NIG.NIGDistribution(NIGDistr2)


# %% Sample from distributions

NIGX = np.array( NIGDistr1.sample( size = 2*int(1e3)) ).reshape((-1,1))
NIGX = np.column_stack( (NIGX, np.array( NIGDistr2.sample( size = 2*int(1e3) ) ).reshape((-1,1)) ) )


# %% Estimate distributions 


paramsMOM = NIG.NIGEstimation.MOM( NIGX )
paramsGauss = NIG.NIGEstimation.Gauss( NIGX )
paramsMLE1 = NIG.NIGEstimation.EMMLE( NIGX, [paramsGauss, paramsMOM], maxIter = int(2e2), tol = 1e-10 )
paramsMLE2 = NIG.NIGEstimation.EMMLE( NIGX, { \
        "alpha": np.array([paramsMLE1[iter]["alpha"] for iter in range(len(paramsMLE1)) ]), \
        "beta": np.array([paramsMLE1[iter]["beta"] for iter in range(len(paramsMLE1)) ]), \
        "mu": np.array([paramsMLE1[iter]["mu"] for iter in range(len(paramsMLE1)) ]), \
        "delta": np.array([paramsMLE1[iter]["delta"] for iter in range(len(paramsMLE1)) ]) \
        } , maxIter = int(2e2), tol = 1e-10 )

paramsGradient = NIG.NIGEstimation.gradientMLE( NIGX, init = [paramsGauss, paramsMOM], maxIter = 1000, tol = 1e-8 )




# %% Plot histograms and compare with pdfs

def plotResult(x, params):
    
    # Loop through each dimension in data
    for iter in np.arange(0, x.shape[1]):
        print( "Dimension: " + str(iter) )
        
        plt.subplot(2,2,iter+1)
        plt.cla()
        plt.title("Hist and PDFs")
        
        plt.hist(x[:,iter], bins = 30, density = True)
        runx = np.linspace(start = np.min(x[:,iter]), stop = np.max(x[:,iter]), num = 200 )
        
        # Loop through each parameter set
        for iterParams, param in enumerate(params.values()):
            distr = NIG.NIGDistribution( param[iter] )
            y = distr.PDF(runx)
            plt.plot( runx, y, color = cm.jet(float(iterParams)/len(params)), label = list(params.keys())[iterParams] )
            plt.legend()
            print( "Log-likelihood from " + list(params.keys())[iterParams] + " is: " + str(np.sum(distr.lPDF(x[:,iter])/x.shape[0])) )
            
        plt.xlim(runx[0], runx[-1])
            
            


plt.clf()

# Plot pdfs    
plotResult( NIGX, {"Gauss" : [{"alpha":paramsGauss["alpha"][0,iter], "beta":paramsGauss["beta"][0,iter], "mu":paramsGauss["mu"][0,iter], "delta":paramsGauss["delta"][0,iter]} for iter in range(NIGX.shape[1])], \
                   "MOM" : [{"alpha":paramsMOM["alpha"][0,iter], "beta":paramsMOM["beta"][0,iter], "mu":paramsMOM["mu"][0,iter], "delta":paramsMOM["delta"][0,iter]} for iter in range(NIGX.shape[1])], \
                   "MLE1" : paramsMLE1, "MLE2" : paramsMLE2, "Gradient" : paramsGradient} )


# %% Plot normplots of transformed data
    
    
distrMLE2 =  [NIG.NIGDistribution( { \
        "alpha": paramsMLE2[iter]["alpha"], \
        "beta": paramsMLE2[iter]["beta"], \
        "mu":paramsMLE2[iter]["mu"], \
        "delta": paramsMLE2[iter]["delta"] \
        } ) for iter in range(len(paramsMLE1))]
    
    
plt.subplot(2,2,3)
plt.cla()
plt.title("Q-Q-plot to Normal")
(quantiles, values), (slope, intercept, r) = stats.probplot( distrMLE2[0].NIG2Gauss( NIGX[:,0] ) , dist='norm')
plt.plot(values, quantiles,'ob')
plt.plot(quantiles * slope + intercept, quantiles, 'r')

plt.subplot(2,2,4)
plt.title("Q-Q-plot to Normal")
(quantiles, values), (slope, intercept, r) = stats.probplot( distrMLE2[1].NIG2Gauss( NIGX[:,1] ) , dist='norm')
plt.plot(values, quantiles,'ob')
plt.plot(quantiles * slope + intercept, quantiles, 'r')





plt.show()