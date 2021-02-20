#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functionality for the mesh graph, i.e., a graph that divides the mesh up into chunks.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import os
import ctypes
import numpy as np



class MeshGraph:
    # Class representing the mesh graph
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  

    # Store the boundaries of the nodes
    boundaries = None
    triangleList = None

    _libInstance = None
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" )    


    def __init__(self, triangles, nodes, minGraphDiam, maxNumGraphNodes = 10, minNumTrianglesInGraph = 1, libPath = None):
    
        if libPath is not None:
            self._libPath = libPath
        
        # Instantiate C library
        self._libInstance = ctypes.CDLL(self._libPath)
        
        # triangles = triangles.copy().astype(ctypes.c_uint)
        # nodes = nodes.copy().astype(ctypes.c_double)
        
        # Maximum number of graph nodes
        maxNumGraphNodes = ctypes.c_uint( maxNumGraphNodes )
        # Minimum graph node diameter
        minGraphDiam = ctypes.c_double( minGraphDiam )
        # Minimum number of triangles in each graph node
        minNumTrianglesInGraph = ctypes.c_uint( minNumTrianglesInGraph )        
        
        # Represent the triangles
        triangles_p = triangles.ctypes.data_as(self.c_uint_p) 
        # Represent the nodes
        points_p = nodes.ctypes.data_as(self.c_double_p) 
        # Init number of graph nodes
        numGraphNodes = ctypes.c_uint( np.uintc(0) )
        # Init graph index
        idx = ctypes.c_uint( np.uintc(0) )
        
        # Create graph
        self._libInstance.MeshGraph_createGraph.restype = ctypes.c_int
        self._libInstance.MeshGraph_createGraph.argtypes = \
            [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
             ctypes.c_uint, ctypes.c_double, ctypes.c_uint, \
             self.c_uint_p, self.c_uint_p ]
        status = self._libInstance.MeshGraph_createGraph( \
            points_p, ctypes.c_uint( nodes.shape[0] ), ctypes.c_uint( nodes.shape[1] ), \
            triangles_p, ctypes.c_uint( triangles.shape[0] ), ctypes.c_uint( triangles.shape[1] - 1 ), \
            maxNumGraphNodes, minGraphDiam, minNumTrianglesInGraph, \
            ctypes.byref( numGraphNodes ), ctypes.byref( idx ) )   
        if status != 0:
            raise Exception( "Uknown error occured!" )                
        
        
        
        # Preallocate boundaries
        self.boundaries = np.NaN * np.ones( ( np.uintc(numGraphNodes), nodes.shape[1], 2 ), dtype=np.float64 )
        boundaries_p = self.boundaries.ctypes.data_as(self.c_double_p)
        
        # Get node boundaries
        self._libInstance.MeshGraph_getNodeBoundaries.restype = ctypes.c_int
        self._libInstance.MeshGraph_getNodeBoundaries.argtypes = \
            [ ctypes.c_uint, self.c_double_p, ctypes.c_uint, ctypes.c_uint ]
        status = self._libInstance.MeshGraph_getNodeBoundaries( idx, boundaries_p, numGraphNodes, ctypes.c_uint( nodes.shape[1] ) )
        if status != 0:
            raise Exception( "Uknown error occured!" )
        
        
        
        
        # Create a list of list of triangles (one for each graph node)
        self.triangleList = [None] * np.uintc(numGraphNodes)
        numTriangles = ctypes.c_uint( 0 )
        
        # Define functions for acquiring triangle lists
        self._libInstance.MeshGraph_getNodeNumTriangles.restype = ctypes.c_int
        self._libInstance.MeshGraph_getNodeNumTriangles.argtypes = [ ctypes.c_uint, ctypes.c_uint, self.c_uint_p ]
        
        self._libInstance.MeshGraph_getNodeTriangles.restype = ctypes.c_int
        self._libInstance.MeshGraph_getNodeTriangles.argtypes = [ ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, \
                                             ctypes.c_uint, self.c_uint_p ]        
        
        # Loop through all nodes and populate
        for iterNodes in range( np.uintc(numGraphNodes) ):
            # Get number of triangles
            status = self._libInstance.MeshGraph_getNodeNumTriangles( idx, ctypes.c_uint( iterNodes ), ctypes.byref( numTriangles ) )
            if status != 0:
                raise Exception( "Uknown error occured!" )
            # Preallocate space for the triangles
            self.triangleList[iterNodes] = np.zeros( np.uintc(numTriangles) , dtype=np.uintc)
            triangles_p = self.triangleList[iterNodes].ctypes.data_as(self.c_uint_p) 
            # Acquire the triangles
            status = self._libInstance.MeshGraph_getNodeTriangles( \
                  idx, ctypes.c_uint( iterNodes ), ctypes.c_uint( nodes.shape[1] ), numTriangles, triangles_p )
            if status != 0:
                raise Exception( "Uknown error occured!" )
            
        
        # Free graph
        self._libInstance.MeshGraph_freeGraph.restype = ctypes.c_int
        self._libInstance.MeshGraph_freeGraph.argtypes = [ ctypes.c_uint ]
        status = self._libInstance.MeshGraph_freeGraph( idx )
        if status != 0:
            raise Exception( "Uknown error occured!" )
            
            
            
            
    def prepareForPlotting(self):
        # Function for generating vectors for plotting the graph using pyplot.plot
        
        d = self.boundaries.shape[1]
        N = self.boundaries.shape[0]
            
        # Create vector for boundary segments
        outputVector = np.zeros( ( d, N, 2**d+2 ) )
        
        for iterDim in range(d):
            temp = np.repeat( self.boundaries[:, iterDim, :], repeats = iterDim+1, axis = 1 )
            if (d > 1):
                if (d-iterDim > 1 ):
                    temp = np.concatenate( (temp, np.flip(temp, axis=1)), axis=1 )
                    temp = np.tile( temp, (1, iterDim+1))
            temp = np.concatenate( (temp, temp[:, 0:1], np.NaN * np.ones((N, 1)) ), axis=1 )
            # Insert into output
            outputVector[iterDim, :, :] = temp
        
        # flatten
        outputVector = outputVector.reshape( (d, N*(2**d+2)) )
        
        return outputVector