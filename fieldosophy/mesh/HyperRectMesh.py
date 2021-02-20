#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for hyper rectangular mesh extensions into new dimensions.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import ctypes
from .Mesh import Mesh
from .ImplicitMesh import ImplicitMesh




    
    
class HyperRectMesh:
    # A class for representing an implicit mesh
    
    _mesh = None            # The implicit mesh to derive the hyper rectangular mesh from
    _offset = None          # The offset of the extended dimensions
    _stepLengths = None     # The step lengths in the extended dimensions
    _numSteps = None        # The number of steps in the extended dimensions
    _libInstance = None     # Library instance
    _internalID = None
    N = None
    NT = None
    embD = None
    topD = None
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    c_bool_p = ctypes.POINTER(ctypes.c_bool)  
    
    
    def __init__(self, mesh, offset = None, stepLengths = None, numSteps = None):
        # Constructor
        
        # Assign
        self._mesh = mesh
        if not isinstance(self._mesh, ImplicitMesh):
            if not isinstance(self._mesh, Mesh):
                raise Exception( "mesh was not of type ImplicitMesh or Mesh!" )
            self._mesh = ImplicitMesh(self._mesh)
            
            
        self._offset = offset
        self._stepLengths = stepLengths
        self._numSteps = numSteps
        self._libInstance = mesh._libInstance
        
        
        # Enforce format
        if self._offset is not None and self._offset is not np.dtype(np.float64):
            self._offset = self._offset.astype(np.float64)
        if  self._stepLengths is not None and self._stepLengths is not np.dtype(np.uintc):
            self._stepLengths = self._stepLengths.astype(np.uintc)
        if  self._numSteps is not None and self._numSteps is not np.dtype(np.uintc):
            self._numSteps = self._numSteps.astype(np.uintc)
            
            
        self.embD = self._mesh._mesh.embD
        self.topD = self._mesh._mesh.topD
        if (self._offset is not None):
            self.embD = self.embD + self._offset.size
            self.topD = self.topD + self._offset.size
            
            
            
        # Define functions
        
        self._libInstance.hyperRectExtension_createMesh.restype = ctypes.c_int
        self._libInstance.hyperRectExtension_createMesh.argtypes = \
            [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_double_p, self.c_uint_p, \
             self.c_double_p, self.c_double_p, self.c_uint_p, ctypes.c_uint, \
             self.c_uint_p, self.c_uint_p, self.c_uint_p, \
             self.c_uint_p ]    
                
        self._libInstance.hyperRectExtension_eraseMesh.restype = ctypes.c_int
        self._libInstance.hyperRectExtension_eraseMesh.argtypes = [ ctypes.c_uint ]    
            
            
            
            
        
           
        
    def __del__(self):
        # Destructor
        
        # Erase stored mesh
        self.logout()
        
        
    def _storeMeshInternally(self):
        # Store mesh internally
        
        # Get pointers to actual mesh
        nodes_p = self._mesh._mesh.nodes.ctypes.data_as(self.c_double_p)
        simplices_p = self._mesh._mesh.triangles.ctypes.data_as(self.c_uint_p)
        neighs_p = None
        if (self._mesh._neighs is not None):
            neighs_p = self._mesh._neighs.ctypes.data_as(self.c_uint_p)
        offset_p = None
        if (self._mesh._offset is not None):
            offset_p = self._mesh._offset.ctypes.data_as(self.c_double_p)
        numPerDimension_p = None
        if (self._mesh._numPerDimension is not None):
            numPerDimension_p = self._mesh._numPerDimension.ctypes.data_as(self.c_uint_p)
        
        offsetHyper_p = None
        if self._offset is not None:
            offsetHyper_p = self._offset.ctypes.data_as(self.c_double_p)
        stepLengths_p = None
        if self._stepLengths is not None:
            stepLengths_p = self._stepLengths.ctypes.data_as(self.c_double_p)
        numSteps_p = None
        if self._numSteps is not None:
            numSteps_p = self._numSteps.ctypes.data_as(self.c_uint_p)
        hyperDim = np.uintc( 0 )
        if (offsetHyper_p is not None and stepLengths_p is not None and numSteps_p is not None):
            hyperDim = np.uintc(self._offset.size)
        
        # Preallocate output
        meshId = ctypes.c_uint( 0 )
        newNumNodes = ctypes.c_uint( 0 )
        newNumSimplices = ctypes.c_uint( 0 )
        
        # Create implicit mesh
        status = self._mesh._libInstance.hyperRectExtension_createMesh( \
           nodes_p, ctypes.c_uint( self._mesh._mesh.nodes.shape[0]) , ctypes.c_uint( self._mesh._mesh.embD ), \
           simplices_p, ctypes.c_uint( self._mesh._mesh.triangles.shape[0]) , ctypes.c_uint( self._mesh._mesh.topD ), \
           offset_p, numPerDimension_p, \
           offsetHyper_p, stepLengths_p, numSteps_p, ctypes.c_uint(hyperDim), \
           ctypes.byref( meshId ), ctypes.byref( newNumNodes ), ctypes.byref( newNumSimplices ), \
           neighs_p )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from hyperRectExtension_createMesh()" ) 

        # Store mesh internally            
        self._internalID = meshId.value
        self.N = newNumNodes.value
        self.NT = newNumSimplices.value
        
        
        
        
        
        
        
        
    def logout(self):
        # Erase mesh internally
        
        # If saved mesh internally
        if self._internalID is not None:
            # Erase
            self._libInstance.hyperRectExtension_eraseMesh( self._internalID )
            self._internalID = None

            
        
        

    