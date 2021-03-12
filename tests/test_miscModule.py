#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for misc module of Fieldosophy.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy.misc import Cheb

import unittest
import numpy as np



class test_miscModule( unittest.TestCase ):
    # Class for testing marginal module
    
    
    def test_simplexNeighbors2D(self):
        # Test case to make sure that the neighboring simplices are actually neighboring (in 2D).

        print("Testing approximating power function with rational function!")
        
        # Define the order of the numerator polynomial
        m = 2
        # Define the order of the denominator polynomial
        n = 2
        
        domain = (1e-6,1)
        
        # Define function
        f = lambda x : ((x) ** 4.4)
        # f = lambda x : ((x+2) ** 2 - 5 * (x+2) ** 1)
        
        # Truncated Chebyshev polynomial approximation
        c = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg = 20, domain = domain )
        b = np.polynomial.polynomial.Polynomial( np.polynomial.chebyshev.cheb2poly(c.coef), domain = domain )
        
        runx = np.linspace(domain[0],domain[1],num=int(1e3))        
        pk, qk = Cheb.ClenshawLord( f, 20, domain, m, n )
                
        maxError = np.max( np.abs( pk(runx)/qk(runx) - f(runx) )  )
        RMSE = np.sqrt(np.mean( np.abs( pk(runx)/qk(runx) - f(runx) )**2  ))
                        
        # Make sure that error is in correct range
        self.assertTrue( (maxError < 0.1) and (maxError > 0.09) )
        self.assertTrue( (RMSE< 0.04) and (RMSE > 0.03) )
                




if __name__ == '__main__':
    unittest.main()