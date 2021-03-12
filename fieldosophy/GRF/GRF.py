#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for Gaussian random fields.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


import ctypes
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt


c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int32)

# Representation of Eigen sparse vector
class EigenSparseStruct(ctypes.Structure):
    _fields_ = [ 
        ("numRows", ctypes.c_int), ("numCols", ctypes.c_int), ("numNonZeros", ctypes.c_int), 
        ("outerIndex", c_int_p), ("innerIndex", c_int_p), ("values", c_double_p) 
        ]

matPtr = ctypes.POINTER(EigenSparseStruct)


def dense2EigenRepr( A, eigenRepr ):
    
    # Initialize
    outerIndices = np.array( [0], dtype="int" )
    innerIndices = np.array( [], dtype="int" )
    values = np.array( [], dtype="float64" )

    # Loop through all rows
    for iterI in range(A.shape[0]):
        
        # Get indices of non zeroes in current column
        curNonZeroes = ~(A[iterI, :] == 0)
        # Get number of non zeroes in current row
        numCurRow = np.sum( curNonZeroes )
        # Add to outerIndices
        outerIndices = np.append( outerIndices, outerIndices[-1] + numCurRow)
        # Add to inner indices
        innerIndices = np.append( innerIndices, np.where( curNonZeroes )[0] )
        # Add to values
        values = np.append( values, A[iterI, curNonZeroes] )            
    
    # Insert in eigenRepr    
    for iter in range(len(values)):
        eigenRepr.contents.values[iter] = values[iter]
        eigenRepr.contents.innerIndex[iter] = innerIndices[iter]
    for iter in range(len(outerIndices)):        
        eigenRepr.contents.outerIndex[iter] = outerIndices[iter]
            
    return( )
 
def EigenRepr2Dense( A, rowMajor = True ):
    
    numNonZeros = A.numNonZeros
    numCols = A.numCols
    numRows = A.numRows
    
    data = np.zeros( (numNonZeros), dtype="float64" )
    rows = np.zeros( (numNonZeros), dtype="int32" )
    cols = np.zeros( (numNonZeros), dtype="int32" )
    
    numOuter = numRows
    if not rowMajor:
        numOuter = numCols
    
    # Get data values
    for iter in range(numNonZeros):
        data[iter] = A.values[iter]    
        
    # Loop through rows
    curInd = 0    
    for iter in range(numOuter):
        curLength = A.outerIndex[iter+1] - A.outerIndex[iter]
        if ( curLength > numNonZeros ):
            print(str(curLength))
        if rowMajor:            
            cols[curInd:(curInd + curLength)] = A.innerIndex[curInd:(curInd+curLength)]
            rows[curInd:(curInd + curLength)] = iter
        else:
            rows[curInd:(curInd + curLength)] = A.innerIndex[curInd:(curInd+curLength)]
            cols[curInd:(curInd + curLength)] = iter
        curInd = curInd + curLength
        
    out = np.zeros( (numRows, numCols), dtype="float64" )
    for iter, iterPair in enumerate( zip( *(rows, cols) ) ):
        out[ iterPair[0], iterPair[1] ] = data[iter]
        
    return( out )
        


def kv(v, z):
    
    eps = 1e-8
    out = special.kv( v, z )
    
    zeroInds = (out < eps)
    out[zeroInds] = np.sqrt( np.pi / (2*z[zeroInds]) ) * np.exp( - z[zeroInds])
    
    return( out )



def MaternCorr( x, nu, kappa ):
    """
    The Matérn correlation function, i.e. normalized such that the value at 0 is 1.
    
    :param x: The distances to evaluate.
    :param nu: Smoothness parameter, correspond to the Hölder constant almost everywhere of realizations.
    :param kappa: The scaling constant.
    
    :return The Matérn correlation evaluated at x.
    """
    
    zeroInd = (x==0)    
    
    # If kappa is array
    if type(kappa) is np.ndarray:
        if kappa.size == x.size:
            kappa = kappa[~zeroInd]
    
    h = kappa * x[~zeroInd]
        
    out = np.ones( x.shape )
    
    constant = 2.0 / ( (2 ** nu) * special.gamma(nu) )
    out[~zeroInd] = (h ** nu) * kv( nu, h ) * constant
    
    out = out
    
    return( out )


def MaternCorrParamDer( x, nu, kappa ):
    """
    The derivative of the Mat'ern correlation function.
    
    :param x: At what distances to evaluate the derivative.
    :param nu: The smoothness parameter.
    :param kappa: The scaling constant.
    
    :return An 2 x n array where first row is the derivative w.r.t. nu for each x and the second row the derivative w.r.t. kappa for each x. Also provide the correlation value for each x as a separate output.
    
    """
    
    zeroInd = (x==0)
    
    h = kappa * x[~zeroInd]    
    
    besselknu = kv( nu, h )    
    besselknu1 = kv( nu+1, h )
        
    # Compute correlation
    corr = np.ones(x.shape)
    corr[~zeroInd] = (h ** nu) * besselknu * (2.0 / ( (2 ** nu) * special.gamma(nu)) )
    # Approximate derivative of corr w.r.t. nu
    eps = 1e-4
    dlcorrnu = np.zeros(x.shape)
    dlcorrnu[~zeroInd] = ( MaternCorr(x[~zeroInd], nu*(1+eps), kappa) - corr[~zeroInd]) / (nu*eps)
    
    # Compute derivative of besselk for kappa    
    dlcorrkappa = np.zeros(x.shape)
    if ( np.any(h == 0) or np.any(besselknu == 0) ):
        print("Galet på gång")
    dlcorrkappa[~zeroInd] = ( 2 * nu / h - besselknu1 / besselknu ) * x[~zeroInd] * corr[~zeroInd]
    
    J = np.array( [ dlcorrnu, dlcorrkappa ] )
    
    return( J, corr )



def anisotropicMaternCorr(x, nu, v, r):    
    """ Anisotropic Matern
    
    x - array D x N. Here, x is not only distance but vector values (also direction).
    D is dimensionality an N are number of 'distances'.
    nu - smoothness.
    v - orthogonal matrix where each column is the corresponding vector for ranges r.
    r - correlation range in each v direction.
    """
    
    # map x to deformed vectors
    out = np.matmul( v.transpose(), x )
    out = np.matmul( np.diag(1/r), out )
    # ACquire distance in deformed
    out = np.sqrt( np.sum( np.asarray(out) ** 2, axis = 0 ) )
    
    assert(not np.any(np.isnan(out)))
    assert(not np.any(np.isinf(out)))
    
    # Assemble covariances    
    out = MaternCorr( out, nu, np.sqrt(8*nu) )  

    return out




