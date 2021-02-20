#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for marginal module of Fieldosophy.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy.marginal.NIG import NIGDistribution as NIG

import unittest
import numpy as np



class test_marginalModule( unittest.TestCase ):
    # Class for testing marginal module
    
    
    def test_splineApproximation(self):
        # Test case to make sure that the spline approximation of the CDF and quantile functions are good enough

        # Define NIG distribution
        distr = NIG( {"alpha":3, "beta":-2.6, "mu":0, "delta": 20} )
        # Initiate probability approximations
        distr.initProbs()
        probs = distr.getProbs()

        # Get interval to test over    
        x = np.linspace( probs["x"][0] - 1 * distr.getStats()["std"], probs["x"][-1] + 1 * distr.getStats()["std"], num = int(2e3) )
        # Compute actual CDF values
        y = distr.CDF(x)
        # Compare with approximation
        yHat = distr.approximateCDF(x)
        # Make sure that difference is not too large anywhere
        self.assertTrue( np.max(np.abs(y-yHat)) < 1e-4 )
        
        # Get quantile approximation
        xHat = distr.approximateQ(y)
        # Make sure that difference is not too large anywhere
        self.assertTrue( np.quantile(np.abs(x-xHat), 0.75) < 1e-6 )



if __name__ == '__main__':
    unittest.main()