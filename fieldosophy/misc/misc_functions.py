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
    
    """
    
    out = np.ones( (points.shape[0], coefSetup.shape[1]) )
    
    for iterSetup in range( coefSetup.shape[1] ):
        for iterEnum in range(coefSetup.shape[0]):
            if coefSetup[iterEnum, iterSetup] >= 1:
                out[:, iterSetup] = out[:, iterSetup] * np.cos( \
                   coefSetup[iterEnum, iterSetup] * np.pi * ( points[:,iterEnum]-boundingBox[iterEnum,0] ) / ( boundingBox[iterEnum,1]-boundingBox[iterEnum,0] ) \
                   )
            
    return out


