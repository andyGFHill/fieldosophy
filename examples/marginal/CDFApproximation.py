#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script show the spline approximation of the CDF and quantile function.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy.marginal.NIG import NIGDistribution as NIG

import numpy as np
from matplotlib import pyplot as plt





# Define NIG distribution
distr = NIG( {"alpha":3, "beta":-2.6, "mu":0, "delta": 20} )
# Initiate probability approximations
distr.initProbs()
probs = distr.getProbs()
    
x = np.linspace( probs["x"][0] - 2* distr.getStats()["std"], probs["x"][-1] + 2 * distr.getStats()["std"], num = int(2e3) )
y = distr.CDF(x)

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
plt.plot(x,y)
plt.plot(x, distr.approximateCDF(x) )
plt.title("CDF")

plt.subplot(1,2,2)
plt.plot(y,x)
plt.plot(y, distr.approximateQ(y) )
plt.title("Quantile")

# for tck in distr.getProbs()["TCKList"]:
#     # SPLINE-interpolation
#     yintrp = interpolate.splev( x, tck, der=0 )    
#     plt.plot(x, yintrp)

# plt.plot(x, probs["polys"](x))
# xL = x[x < probs["x"][0]]
# yL = distr.extrapolateCDFLeft( xL )
# plt.plot( xL, yL )       
# xR = x[x > probs["x"][-1]]
# yR = distr.extrapolateCDFRight( xR )
# plt.plot( xR, yR )   