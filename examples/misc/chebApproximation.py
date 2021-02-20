#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates approximation arbitrary function on given interval by rational functions using the Clenshaw-Lord-Chebyschev-Pade algorithm.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

from fieldosophy.misc import Cheb
import numpy as np    
from matplotlib import pyplot as plt

# Define the order of the numerator polynomial
m = 3
# Define the order of the denominator polynomial
n = 3

domain = (1e-6,1)

# Define function
f = lambda x : ((x) ** 4.4)
# f = lambda x : ((x+2) ** 2 - 5 * (x+2) ** 1)

# Truncated Chebyshev polynomial approximation
c = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg = 20, domain = domain )
b = np.polynomial.polynomial.Polynomial( np.polynomial.chebyshev.cheb2poly(c.coef), domain = domain )


plt.figure(1)
plt.clf()

runx = np.linspace(domain[0],domain[1],num=int(1e3))
plt.plot( runx, f(runx), color="blue" )
plt.plot( runx, b(runx), color="red" )

pk, qk = Cheb.ClenshawLord( f, 20, domain, m, n )
plt.plot( runx, pk(runx)/qk(runx), color="green" )

print( "Chebyshev coefficient of order m+n: " + str( c.coef[m+n] ) )
print( "Max error: " + str(np.max( np.abs( pk(runx)/qk(runx) - f(runx) ) ) ) )