#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for matching templates using cross-correlation.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


import numpy as np
import ctypes
import os

from scipy import interpolate
from scipy import ndimage


class TemplateMatching:
    """
    class for template matching
    """
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    c_bool_p = ctypes.POINTER(ctypes.c_bool)
    
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" )
    _libInstance = None
    
    
    def __init__( self, libPath = None):
        
        if libPath is not None:
            self._libPath = libPath
            
        # Instantiate C library
        self._libInstance = ctypes.CDLL(self._libPath)
        
        # Declare mapTrivals2Mat function
        self._libInstance.misc_localMaxCrossCorr2D.restype = ctypes.c_int
        self._libInstance.misc_localMaxCrossCorr2D.argtypes = [ \
           self.c_double_p, self.c_double_p, \
           ctypes.c_uint, ctypes.c_uint, \
           ctypes.c_uint, ctypes.c_uint, \
           ctypes.c_uint, ctypes.c_uint, \
           ctypes.c_uint, ctypes.c_uint, \
           self.c_uint_p, self.c_bool_p, self.c_double_p ]

        return    


    def griddedTemplateMatching( self, img1, img2, templateRadius, searchRadius, \
        estimateInds = None, templateSkip = 0, searchSkip = 0, templateStart = 0, searchStart = 0 ):
        """
        Function for performing template matching between two sets of bitmapped images
        """
        
        # Convert input
        img1 = np.ascontiguousarray(img1, dtype=np.double)
        img2 = np.ascontiguousarray(img2, dtype=np.double)    
        if estimateInds is not None:
            estimateInds = np.ascontiguousarray(estimateInds, dtype=np.bool)    
            
        # Preallocate output
        mapIndex = np.zeros( img1.shape, dtype = np.uintc, order = 'C' )
        maxCrossCorr = np.nan * np.ones( img1.shape, dtype = np.double, order = 'C' )
            
        
        # Acquire pointers
        img1_p = img1.ctypes.data_as( self.c_double_p )
        img2_p = img2.ctypes.data_as( self.c_double_p )
        mapIndex_p = mapIndex.ctypes.data_as( self.c_uint_p )
        if estimateInds is not None:
            estimateInds_p = estimateInds.ctypes.data_as( self.c_bool_p )
        else:
            estimateInds_p = None
        maxCrossCorr_p = maxCrossCorr.ctypes.data_as( self.c_double_p )
        
        # Perform template matching
        status = self._libInstance.misc_localMaxCrossCorr2D( \
            img1_p, img2_p, \
            ctypes.c_uint( img1.shape[1] ), ctypes.c_uint( img1.shape[0] ), \
            ctypes.c_uint( templateRadius ), ctypes.c_uint( searchRadius ), \
            ctypes.c_uint( np.uintc(templateSkip) ), ctypes.c_uint( np.uintc(searchSkip) ), \
            ctypes.c_uint( np.uintc(templateStart) ), ctypes.c_uint( np.uintc(searchStart) ), \
            mapIndex_p, estimateInds_p, maxCrossCorr_p )          
        if status != 0:            
            raise Exception( "Uknown error occured! Error status: " + str(status) ) 
        
        # Return index
        return (mapIndex, maxCrossCorr)
    
    



    def resampleOnGrid( points, values, resolution, boundingBox = None ):
        """
        Function for resampling unstructured (or structured) spatial observations
        """
        
        # Get resolution
        h = resolution[0]
        w = resolution[1]
        
        # Get bounding box
        if boundingBox is None:
            boundingBox = np.array( [ \
             [np.min(points[:,0]), np.max(points[:,0])], \
             [np.min(points[:,1]), np.max(points[:,1])] \
             ] )
        
        # Interpolate onto grid
        y = np.linspace( boundingBox[1,0], boundingBox[1,1], num = h )
        x = np.linspace( boundingBox[0,0], boundingBox[0,1], num = w )
        X,Y = np.meshgrid( x,y )
        img = interpolate.griddata( points, values, (X,Y), method = 'linear' )
        
        return { "image":img, "x":x, "y":y, "boundingBox":boundingBox }


    def acquireVectorField( templateMatch, maxCrossCorr, X, Y, crossCorrThresh = 1, medianSize = None, meanSize = None ):
        """
        Function for computing vector field representation of template matching
        """
        
        # Preallocate vector field
        vectorField = np.nan * np.ones( (templateMatch.size, 2) )
        # Acquire indices which are recognized
        estimInd = templateMatch < np.prod(templateMatch.shape)        
        # Acquire vectors from template matching        
        vectorField[estimInd.flatten('C'),0] = X.flatten('C')[templateMatch[estimInd]] - X[estimInd].flatten('C')
        vectorField[estimInd.flatten('C'),1] = Y.flatten('C')[templateMatch[estimInd]] - Y[estimInd].flatten('C')
        
        # Filter out too small cross-correlation
        mask = ~np.isnan(maxCrossCorr.flatten())
        mask[mask] &= maxCrossCorr.flatten()[mask] < crossCorrThresh
        vectorField[ mask , :] = np.nan
        
        # Get X and Y components
        vectorFieldX = vectorField[:,0].reshape(X.shape)
        vectorFieldY = vectorField[:,1].reshape(X.shape)
        
        # Median filter to acquire more consistent matching
        if medianSize is not None:
            vectorFieldX = ndimage.median_filter( vectorFieldX, size = medianSize )
            vectorFieldY = ndimage.median_filter( vectorFieldY, size = medianSize )
            
        # Mean filter to acquire more consistent matching
        if meanSize is not None:
            vectorFieldX = ndimage.generic_filter( vectorFieldX, np.nanmean, size = meanSize, mode = 'nearest' )
            vectorFieldY = ndimage.generic_filter( vectorFieldY, np.nanmean, size = meanSize, mode = 'nearest' )
        
        return ( vectorFieldX, vectorFieldY )


