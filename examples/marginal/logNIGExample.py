# -*- coding: utf-8 -*-
"""
This script highlight how to create, sample, estimate, and plot a log-NIG distribution.

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


# %% Create distribution
    
# Create NIG distribution
NIGDistr = { "alpha":4, "beta":-2, "mu":-1, "delta":1 }
NIGDistr = NIG.NIGDistribution(NIGDistr, log = True)



# %% Sample from distributions

NIGX = NIGDistr.sample( size = int(4e3)).reshape((-1,1))


# %% Estimate distributions 


paramsMOM = NIG.NIGEstimation.MOM( np.log(NIGX) )
paramsGauss = NIG.NIGEstimation.Gauss( np.log(NIGX) )
paramsMLE1 = NIG.NIGEstimation.EMMLE( np.log(NIGX), [paramsGauss, paramsMOM], maxIter = int(2e2), tol = 1e-10 )
paramsMLE2 = NIG.NIGEstimation.EMMLE( np.log(NIGX), { \
        "alpha": np.array([paramsMLE1[iter]["alpha"] for iter in range(len(paramsMLE1)) ]), \
        "beta": np.array([paramsMLE1[iter]["beta"] for iter in range(len(paramsMLE1)) ]), \
        "mu": np.array([paramsMLE1[iter]["mu"] for iter in range(len(paramsMLE1)) ]), \
        "delta": np.array([paramsMLE1[iter]["delta"] for iter in range(len(paramsMLE1)) ]) \
        } , maxIter = int(2e2), tol = 1e-10 )




# %% Plot histograms and compare with pdfs

plt.clf()

plt.subplot(223)
plt.cla()
plt.scatter(np.arange(0,NIGX.size), NIGX)



plt.subplot(221)
plt.cla()
plt.title("Hist and PDFs in original scale")

plt.hist(NIGX, bins = 30, density = True)
runx = np.linspace(start = np.min(NIGX), stop = np.max(NIGX), num = 200 )

dictParams = { "Gauss" : {"alpha": paramsGauss["alpha"][0,0], "beta": paramsGauss["beta"][0,0], "mu": paramsGauss["mu"][0,0], "delta": paramsGauss["delta"][0,0]}, \
        "MOM" : {"alpha": paramsMOM["alpha"][0,0], "beta": paramsMOM["beta"][0,0], "mu": paramsMOM["mu"][0,0], "delta": paramsMOM["delta"][0,0]}, \
        "MLE1" : paramsMLE1[0], "MLE2": paramsMLE2[0] }

# Loop through each parameter set
for iterParams, param in enumerate( dictParams ):
    distr = NIG.NIGDistribution( dictParams[param], log = True )
    y = distr.PDF(runx)
    plt.plot( runx, y, color = cm.jet(float(iterParams)/len(dictParams.values())), label = list(dictParams.keys())[iterParams])
    plt.legend()


plt.subplot(222)
plt.cla()
plt.title("Hist and PDFs in log-scale")

plt.hist(np.log(NIGX), bins = 30, density = True)
runx = np.linspace(start = np.min(np.log(NIGX)), stop = np.max(np.log(NIGX)), num = 200 )

dictParams = { "Gauss" : {"alpha": paramsGauss["alpha"][0,0], "beta": paramsGauss["beta"][0,0], "mu": paramsGauss["mu"][0,0], "delta": paramsGauss["delta"][0,0]}, \
        "MOM" : {"alpha": paramsMOM["alpha"][0,0], "beta": paramsMOM["beta"][0,0], "mu": paramsMOM["mu"][0,0], "delta": paramsMOM["delta"][0,0]}, \
        "MLE1" : paramsMLE1[0], "MLE2": paramsMLE2[0] }

# Loop through each parameter set
for iterParams, param in enumerate( dictParams ):
    distr = NIG.NIGDistribution( dictParams[param], log = False )
    y = distr.PDF(runx)
    plt.plot( runx, y, color = cm.jet(float(iterParams)/len(dictParams.values())), label = list(dictParams.keys())[iterParams])
    plt.legend()






# %% Plot normplots of transformed data
    
    
distrMLE2 = NIG.NIGDistribution( paramsMLE2[0], log=True )
    
    
plt.subplot(224)
plt.cla()
plt.title("Q-Q-plot to Normal")
(quantiles, values), (slope, intercept, r) = stats.probplot( distrMLE2.NIG2Gauss( NIGX[:,0] ) , dist='norm')
plt.plot(values, quantiles,'ob')
plt.plot(quantiles * slope + intercept, quantiles, 'r')




plt.show()