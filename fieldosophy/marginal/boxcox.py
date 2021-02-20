#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for Box-Cox transformation.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np

# function for transforming using the Box-Cox transform
def boxCoxTransform( data, paramLambda, useGrads = False ):
    # Logarithmize
    lData = np.log( data )    
    # Init output
    out = [np.zeros( data.shape, dtype="float64" )]
    if useGrads:
        out.append( np.zeros( data.shape, dtype="float64" ) )
    
    # Get machine precision
    eps = np.finfo(np.float64).eps
    # Get indices of  log normal
    indexLN = paramLambda <= eps * 100
    # Get indices of  non-log normal
    indexNLN = paramLambda > eps * 100
    
    # Handle log-normal
    if np.any( indexLN ):                  
        out[0][indexLN, :] = lData[indexLN, :]
        
    # Handle non-log-normal
    if np.any( indexNLN ):
        # (data^paramLambda - 1) / paramLambda        
        out[0][indexNLN, :] = ( data[indexNLN, :] ** paramLambda[indexNLN].reshape((-1,1)) - 1 ) / paramLambda[indexNLN].reshape((-1,1))
        if useGrads:
            # ( ln(data) - 1 / paramLambda ) + ln(data) / paramLambda
            out[1][indexNLN, :] = data[indexNLN, :] * ( lData[indexNLN, :] - 1 / paramLambda[indexNLN].reshape((-1,1)) ) + lData[indexNLN, :] / paramLambda[indexNLN].reshape((-1,1))
    
    # Return output
    return out
    
    

def boxCoxInverseTransform( data, paramLambda, useGrads = False ):
     
    # Init output
    out = [np.zeros( data.shape, dtype="float64" )]
    if useGrads:
        out.append( np.zeros( data.shape, dtype="float64" ) )
    
    # Get machine precision
    eps = np.finfo(np.float64).eps
    # Get indices of  log normal
    indexLN = paramLambda <= eps * 100
    # Get indices of  non-log normal
    indexNLN = paramLambda > eps * 100
    
    # Handle log-normal
    if np.any( indexLN ):
        out[0][indexLN, :] = np.exp( data[indexLN, :] )
        
    # Handle non-log-normal
    if np.any( indexNLN ):
        # ((lambda * x + 1)^(1/lambda)
        out[0][indexNLN, :] = ( data[indexNLN, :] * paramLambda[indexNLN].reshape((-1,1)) + 1 ) ** ( 1 / paramLambda[indexNLN].reshape((-1,1)) )

    
    # Return output
    return out


def testBoxCox():
    N = 10
    M = 3
    temp = np.float64( np.random.rand(M, N) )
    tempParam = np.float64( np.random.rand( M,1 ) ).flatten()
    temp2 = boxCoxTransform( temp, tempParam, useGrads = True )
    temp3 = boxCoxInverseTransform( temp2[0], tempParam, useGrads = True )
    
    print( np.max( np.abs( temp[0:3,0:2] - temp3[0][0:3,0:2] ) ) )
    
    return



