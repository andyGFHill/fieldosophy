#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for creating regularly spaced meshes.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


import numpy as np
import ctypes

from .Mesh import Mesh







    

def extendMeshRegularly( mesh, spacing, num = 1 ):
    ''' Extends mesh to new dimension by regular spacings '''
    libInstance = mesh._libInstance

    # Preallocate space for new simplices
    newTriangles = np.empty( (mesh.NT * (mesh.topD+1), (mesh.topD+2)) , dtype=np.uintc, order = 'C' )
    newTriangles_p = newTriangles.ctypes.data_as(Mesh.c_uint_p)
    oldTriangles = np.ascontiguousarray(mesh.triangles)
    oldTriangles_p = oldTriangles.ctypes.data_as(Mesh.c_uint_p)
    
    # Acquire simplices between layers
    libInstance.mesh_extendMesh.restype = ctypes.c_int
    libInstance.mesh_extendMesh.argtypes = \
        [ Mesh.c_uint_p, ctypes.c_uint, \
         Mesh.c_uint_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint ]
    status = libInstance.mesh_extendMesh( \
        newTriangles_p, ctypes.c_uint( newTriangles.shape[0] ), \
        oldTriangles_p, ctypes.c_uint( mesh.NT ), ctypes.c_uint( mesh.topD ), ctypes.c_uint( mesh.N ) )
        
    if status != 0:
        
        if status == 1:
            raise Exception( "Wrong number given for number of new simplices in mesh!" ) 
        if status == 2:
            raise Exception( "computeNewSubSimplices failed!" ) 
            
        raise Exception( "Uknown error occured! Error code: " + str(status) ) 
        
        
    if not isinstance(spacing, np.ndarray):
        spacing = spacing * np.ones(num)
    
    
    # Extend dimension from old node
    newNodes = np.concatenate( (mesh.nodes, np.zeros((mesh.N,1))), axis = 1 )
    # Loop through all number of extensions
    for iterExtension in range(spacing.size):
        # Copy last extension
        newNodes = np.concatenate( (newNodes, newNodes[-mesh.N:, :]), axis=0 )
        # Increase last dimension
        newNodes[-mesh.N:, -1] = newNodes[-mesh.N:, -1] + spacing[iterExtension]
        
        # If iteration is greater than zero
        if iterExtension > 0:
            # Extend triangles
            newTriangles = np.concatenate( (newTriangles, \
                newTriangles[ 0:( mesh.NT * (mesh.topD+1) ) , :] + iterExtension * mesh.N  \
                ), axis = 0 )
        
    # Create output mesh
    out = Mesh( nodes = newNodes, triangles = newTriangles )
    
    return out
        
   


def meshInPlaneRegular( boundaryPolygon, maxDiam ):
    """ Creates triangular mesh in plane. """    

    d = boundaryPolygon.shape[0]
    
    if not type(maxDiam) is np.array:
        maxDiam = maxDiam * np.ones((d))
    
    curRange = np.diff(boundaryPolygon[0,:])[0]
    numIntervals = int(np.ceil(curRange/maxDiam[0]))
    
    # Get one dimensional mesh
    nodes = np.linspace(boundaryPolygon[0,0], boundaryPolygon[0,1], numIntervals+1).reshape((-1,1))
    triangles = np.array( [np.arange(0,numIntervals), np.arange(1,numIntervals+1)], dtype=int ).transpose()
    
    # Loop through dimensions
    for iterD in range(d-1):
        
        curRange = np.diff(boundaryPolygon[iterD+1,:])[0]
        numIntervals = int(np.ceil(curRange/maxDiam[iterD+1]))
        extrusion = curRange/numIntervals
        
        mesh = Mesh(nodes = nodes, triangles = triangles)        
        mesh = extendMeshRegularly( mesh, extrusion, numIntervals )
        mesh.nodes[:,-1] = mesh.nodes[:,-1] + boundaryPolygon[iterD+1, 0]
        nodes = mesh.nodes
        triangles = mesh.triangles
        
    mesh.convexHull = boundaryPolygon
       
    return mesh
       

    