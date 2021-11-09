#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous functions

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np




def cosinusBasis( points, coefficients, boundingBox ):
    """
    Function for evaluating superpositioned sinus function
    
    :points: A N x d array where d are the number of dimensions and N are the number of points
    :coefficients: A d x n array where each column correspond to a specific coefficient. The value in the rows correspond to the harmonic in each of the dimensions (which are multiplied with each other)
    :boundingBox: Gives a bounding box for where the functions continue with their next period.    
    
    """
    
    # Preallocate output
    out = np.zeros( (points.shape[0]) )
    
    # Loop through all elements of coefficients
    with np.nditer(coefficients, flags=['multi_index']) as ndIterObj:
        for iter in ndIterObj:
            curOut = iter * np.ones(out.shape)
            # Go through all multiindices
            for enumIter, subIter in enumerate(ndIterObj.multi_index):
#                print( "iter: " + str(iter) + " subIter: " + str(subIter) + " enumIter: " + str(enumIter) )
                if subIter >= 1:
                    curOut = curOut * np.cos( \
                          subIter * np.pi * ( points[:,enumIter]-boundingBox[enumIter,0] ) / ( boundingBox[enumIter,1]-boundingBox[enumIter,0] ) \
                          )
            
            # Fill up on super-position
            out = out + curOut
    
    # Return output
    return out

def cosinBasisMatrix( points, coefSetup, boundingBox ):
    """
    Matrix giving evaluated superpositioned sinus functions for chosen points 
    
    :points: A N x d array where d are the number of dimensions and N are the number of points
    :coefSetup: A d x n array where each column correspond to a specific coefficient. The value in the rows correspond to the harmonic in each of the dimensions (which are multiplied with each other)
    :boundingBox: Gives a bounding box for where the functions continue with their next period.
    
    """
    
    out = np.ones( (points.shape[0], coefSetup.shape[1]) )
    
    # Loop through all combinations of cosines for each dimension
    for iterSetup in range( coefSetup.shape[1] ):
        # Loop through all factors
        for iterEnum in range(coefSetup.shape[0]):
            if coefSetup[iterEnum, iterSetup] >= 1:
                out[:, iterSetup] = out[:, iterSetup] * np.cos( \
                   coefSetup[iterEnum, iterSetup] * np.pi * ( points[:,iterEnum]-boundingBox[iterEnum,0] ) / ( boundingBox[iterEnum,1]-boundingBox[iterEnum,0] ) \
                   )
            
    return out


