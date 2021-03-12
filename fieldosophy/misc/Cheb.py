#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Port of some of the functionality of the "chebpade" function in the "chebfun" library.

Copyright 2017 by The University of Oxford and The Chebfun Developers.
All rights reserved. See http://www.chebfun.org/ for Chebfun information.

This file is part of Fieldosophy, a toolkit for random fields.

"""

import numpy as np
from scipy import linalg
from scipy import signal



# %% Define functions

def ClenshawLord( f, N, domain, m, n ):
    """
    Function for computing Clenshaw-Lord approximation of chebychev polynomial on -1,1
    
    A port of the function "chabpade" from the Matlab-library "chebfun".
    """
    
    
    # Acquire Chebyshev approximation of f
    chebPoly = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg = N, domain = domain )
        
    # Get chebyshev coefficients
    c = chebPoly.coef
    
    # Zero pad if needed
    if c.size < m + 2*n:
        c = np.concatenate( (c, np.zeros(m+2*n-c.size)) )
    
    # Setup system of linear equations to solve for beta
    c[0] = 2*c[0]    
    top = c[ np.abs( np.arange(m-n+1,m+1) ) ]
    bottom = c[ np.abs( np.arange(m,m+n) ) ]
    rhs = c[ np.arange(m+1,m+n+1) ]
    
    beta = np.array([1])
    if n > 0:
        beta = linalg.hankel(top, bottom)
        beta = linalg.solve( beta, rhs )
        beta = np.flipud( np.concatenate( (-beta , np.array([1])) ) )
    
    # Compute alpha
    l = np.max((m,n))
    c[0] = c[0]/2
    alpha = signal.convolve(c[:l+1], beta)
    alpha = alpha[:l+2]
    
    # Compute numerator polynomial
    pk = np.zeros( (m+1) )
    D = np.outer( alpha, beta )
    pk[0] = np.sum(D.diagonal())
    for k in range(m):
        pk[k+1] = np.sum( np.concatenate( ( D.diagonal(k+1), D.diagonal(-k-1) ) ) )
    
    # Compute denominator
    qk = np.zeros( (n+1) )
    qk[0] = np.dot(beta, beta)
    for k in range(n):
        u = beta[:-(k+1)]
        v = beta[(k+1):]
        qk[k+1] = np.dot(u,v)
    
    # # Normalize the coefficients
    pk = pk/qk[0]
    qk = 2*qk/qk[0]
    qk[0] = 1    
    
    # Get coefficients of corresponding power series
    pk = np.polynomial.chebyshev.cheb2poly(pk)
    qk = np.polynomial.chebyshev.cheb2poly(qk)
    
    # Return polynomial objects
    pk = np.polynomial.polynomial.Polynomial( pk, domain = domain )
    qk = np.polynomial.polynomial.Polynomial( qk, domain = domain )
    
    
    return (pk, qk)



