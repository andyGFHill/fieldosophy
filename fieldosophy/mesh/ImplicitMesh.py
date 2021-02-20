#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for implicit extension of mesh.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import ctypes
from .Mesh import Mesh




    
    
class ImplicitMesh:
    # A class for representing an implicit mesh
    
    _mesh = None    # The mesh to derive the implicit mesh from
    _offset = None  # The offset of the implicit mesh
    _numPerDimension = None  # Number of multiples of original mesh in each dimension
    _internalID = None # ID to internally stored internal mesh
    _neighs = None  # The neighborhood structure of the explicit mesh
    _libInstance = None     # Library instance 
    
    N = None # Number of nodes in implicit mesh
    NT = None # Number of simplices in implicit mesh
    
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    c_bool_p = ctypes.POINTER(ctypes.c_bool)  
    
    
    def __init__(self, mesh, offset = None, numPerDimension = None, neighs = None):
        # Constructor
        
        if not isinstance(mesh, Mesh):
            raise Exception( "mesh was not of type Mesh!" )
        
        # Assign
        self._mesh = mesh
        self._offset = offset
        self._numPerDimension = numPerDimension
        self._neighs = neighs
        self._libInstance = mesh._libInstance
        
        # Enforce format
        if self._offset is not None and self._offset is not np.dtype(np.float64):
            self._offset = self._offset.astype(np.float64)
        if self._numPerDimension is not None and self._numPerDimension is not np.dtype(np.uintc):
            self._numPerDimension = self._numPerDimension.astype(np.uintc)
            
        # Define functions
        self._libInstance.implicitMesh_createImplicitMesh.restype = ctypes.c_int
        self._libInstance.implicitMesh_createImplicitMesh.argtypes = \
            [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_double_p, self.c_uint_p, \
             self.c_uint_p, self.c_uint_p, self.c_uint_p, \
             self.c_uint_p ]
                
        self._libInstance.implicitMesh_retrieveFullMeshFromImplicitMesh.restype = ctypes.c_int
        self._libInstance.implicitMesh_retrieveFullMeshFromImplicitMesh.argtypes = \
            [ ctypes.c_uint, \
             self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
             self.c_uint_p, ctypes.c_uint, ctypes.c_uint ]
                
        self._libInstance.implicitMesh_retrieveFullNeighsFromImplicitMesh.restype = ctypes.c_int
        self._libInstance.implicitMesh_retrieveFullNeighsFromImplicitMesh.argtypes = \
            [ ctypes.c_uint, self.c_uint_p, ctypes.c_uint, ctypes.c_uint ]
                
        self._libInstance.implicitMesh_eraseImplicitMesh.restype = ctypes.c_int
        self._libInstance.implicitMesh_eraseImplicitMesh.argtypes = [ ctypes.c_uint ]
        
        self._libInstance.implicitMesh_nodeInd2SectorAndExplicit.restype = ctypes.c_int
        self._libInstance.implicitMesh_nodeInd2SectorAndExplicit.argtypes = [ ctypes.c_uint, ctypes.c_uint, self.c_uint_p, self.c_uint_p ]
        
        self._libInstance.implicitMesh_nodeSectorAndExplicit2Ind.restype = ctypes.c_int
        self._libInstance.implicitMesh_nodeSectorAndExplicit2Ind.argtypes = [ ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, self.c_uint_p ]
        
        self._libInstance.implicitMesh_pointInSimplex.restype = ctypes.c_int
        self._libInstance.implicitMesh_pointInSimplex.argtypes = [ ctypes.c_uint, self.c_double_p, ctypes.c_uint, ctypes.c_uint, self.c_bool_p ]
        
        self._libInstance.implicitMesh_nodeInSimplex.restype = ctypes.c_int
        self._libInstance.implicitMesh_nodeInSimplex.argtypes = [ ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, self.c_bool_p ]
        
        
    def __del__(self):
        # Destructor
        
        # Erase stored mesh
        self.logout()
        
        
    def _storeMeshInternally(self):
        # Store mesh internally
        
        # Get pointers to actual mesh
        nodes_p = self._mesh.nodes.ctypes.data_as(self._mesh.c_double_p)
        simplices_p = self._mesh.triangles.ctypes.data_as(self._mesh.c_uint_p)
        neighs_p = None
        if (self._neighs is not None):
            neighs_p = self._neighs.ctypes.data_as(self._mesh.c_uint_p)
        offset_p = None
        if (self._offset is not None):    
            offset_p = self._offset.ctypes.data_as(self._mesh.c_double_p)
        numPerDimension_p = None
        if (self._numPerDimension is not None):    
            numPerDimension_p = self._numPerDimension.ctypes.data_as(self._mesh.c_uint_p)
        
        # Preallocate output
        meshId = ctypes.c_uint( 0 )
        newNumNodes = ctypes.c_uint( 0 )
        newNumSimplices = ctypes.c_uint( 0 )
        
        # Create implicit mesh
        status = self._libInstance.implicitMesh_createImplicitMesh( \
           nodes_p, ctypes.c_uint( self._mesh.nodes.shape[0]) , ctypes.c_uint( self._mesh.embD ), \
           simplices_p, ctypes.c_uint( self._mesh.triangles.shape[0]) , ctypes.c_uint( self._mesh.topD ), \
           offset_p, numPerDimension_p, \
           ctypes.byref( meshId ), ctypes.byref( newNumNodes ), ctypes.byref( newNumSimplices ), \
           neighs_p )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_createImplicitMesh()" ) 

        # Store mesh internally            
        self._internalID = meshId
        self.N = newNumNodes.value
        self.NT = newNumSimplices.value
        
        
        
        
        
        
        
        
    def logout(self):
        # Erase mesh internally
        
        # If saved mesh internally
        if self._internalID is not None:
            # Erase
            self._libInstance.implicitMesh_eraseImplicitMesh( self._internalID )
            self._internalID = None
        
        
        
    def toFullMesh(self):
        # Function for returning a full mesh corresponding to the ImplicitMesh
        
        # If not saved mesh internally
        if self._internalID is None:
            # do that
            self._storeMeshInternally()
        
        # Preallocate for full mesh            
        newNodes = np.zeros( (self.N, self._mesh.embD), dtype = np.float64 )
        newSimplices = np.zeros( (self.NT, self._mesh.topD+1), dtype = np.uintc )
        
        newNodes_p = newNodes.ctypes.data_as(self._mesh.c_double_p)
        newSimplices_p = newSimplices.ctypes.data_as(self._mesh.c_uint_p)
        
        # Retrieve implicit mesh
        status = self._libInstance.implicitMesh_retrieveFullMeshFromImplicitMesh( self._internalID, \
           newNodes_p, ctypes.c_uint(np.uintc(self.N)) , ctypes.c_uint( np.uint(self._mesh.embD) ), \
           newSimplices_p, ctypes.c_uint(np.uintc(self.NT)) , ctypes.c_uint( np.uintc(self._mesh.topD) ) )
            
        if status != 0:
            # Try to save internally again
            self._storeMeshInternally()
            # Try again to retrieve implicit mesh
            status = self._libInstance.implicitMesh_retrieveFullMeshFromImplicitMesh( self._internalID, \
               newNodes_p, ctypes.c_uint(np.uintc(self.N)) , ctypes.c_uint( np.uintc(self._mesh.embD) ), \
               newSimplices_p, ctypes.c_uint(np.uintc(self.NT)) , ctypes.c_uint( self._mesh.topD ) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_retrieveFullMeshFromImplicitMesh()" ) 
        
        # Return
        return Mesh( newSimplices, newNodes, libPath = self._mesh._libPath )
        
      
    def getFullNeighs(self):
        # Get whole neighborhood structure
        
        if self._neighs is None:
            return None
        
        newNeighs = np.zeros( (self.NT, self._mesh.topD+1), dtype = np.uintc )  
        newNeighs_p = newNeighs.ctypes.data_as(self._mesh.c_uint_p)
        
        status = self._libInstance.implicitMesh_retrieveFullNeighsFromImplicitMesh( self._internalID, \
               newNeighs_p, ctypes.c_uint(np.uintc(self.NT) ), ctypes.c_uint( np.uintc(self._mesh.topD) ) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_retrieveFullNeighsFromImplicitMesh()" ) 
            
        return newNeighs
        
    
    def fromNodeInd2SectorAndExplicit(self, nodeInd):
        # Function for returning sector and explicit node index from implicit node index

        # Enforce formating        
        nodeInd = np.uintc(nodeInd)
        # If not saved mesh internally
        if self._internalID is None:
            # do that
            self._storeMeshInternally()
        sector = ctypes.c_uint(0)
        explicitInd = ctypes.c_uint(0)        
        status = self._libInstance.implicitMesh_nodeInd2SectorAndExplicit( self._internalID, ctypes.c_uint( nodeInd ), \
                                                                        ctypes.byref(sector), ctypes.byref(explicitInd) )
        if status != 0:
            # Try to save internally again
            self._storeMeshInternally()
            # Retry call
            status = self._libInstance.implicitMesh_nodeInd2SectorAndExplicit( self._internalID, ctypes.c_uint( nodeInd ), \
                                                                        ctypes.byref(sector), ctypes.byref(explicitInd) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_nodeInd2SectorAndExplicit()" ) 
            
        return {"sector":sector.value, "explicitInd":explicitInd.value}
    

    def fromSectorAndExplicit2Node(self, sector,explicitInd):
        # Function for returning implicit node index from sector and explicit node index

        # Enforce formating        
        sector = np.uintc(sector)
        explicitInd = np.uintc(explicitInd)
        # If not saved mesh internally
        if self._internalID is None:
            # do that
            self._storeMeshInternally()
        nodeInd = ctypes.c_uint(0)
        status = self._libInstance.implicitMesh_nodeSectorAndExplicit2Ind( self._internalID, ctypes.c_uint( sector ), \
                                                                        ctypes.c_uint( explicitInd ), ctypes.byref(nodeInd) )    
        if status != 0:
            # Try to save internally again
            self._storeMeshInternally()
            # Retry call
            status = self._libInstance.implicitMesh_nodeSectorAndExplicit2Ind( self._internalID, ctypes.c_uint( sector ), \
                                                                        ctypes.c_uint( explicitInd ), ctypes.byref(nodeInd) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_nodeSectorAndExplicit2Ind()" )         
            
        return nodeInd.value
            
    
    def nodeInSimplex(self, nodeInd, simplexInd):
        # Function for checking if node is in simplex
        
        # If not saved mesh internally
        if self._internalID is None:
            # do that
            self._storeMeshInternally()
        
        # Enforce formating        
        nodeInd = np.uintc(nodeInd)
        simplexInd = np.uintc(simplexInd)
        
        out = ctypes.c_bool(False)
        status = self._libInstance.implicitMesh_nodeInSimplex( self._internalID, \
                ctypes.c_uint( nodeInd ), ctypes.c_uint( simplexInd ), ctypes.byref(out) )
            
        if status != 0:
            # Try to save internally again
            self._storeMeshInternally()
            # Retry call
            status = self._libInstance.implicitMesh_nodeInSimplex( self._internalID, \
                ctypes.c_uint( nodeInd ), ctypes.c_uint( simplexInd ), ctypes.byref(out) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_nodeInSimplex()" )         
            
        return out.value
    
    def pointInSimplex(self, point, simplexInd):
        # Function for checking if node is in simplex
        
        if (point.size != self._mesh.embD):
            raise Exception( "Wrong dimensionality of point in input" )         
        
        # If not saved mesh internally
        if self._internalID is None:
            # do that
            self._storeMeshInternally()
        
        # Enforce formating        
        if point.dtype is not np.dtype("float64"):
            point = point.astype(np.float64)
        point_p = point.ctypes.data_as(self.c_double_p) 
        simplexInd = np.uintc(simplexInd)
        
        out = ctypes.c_bool(False)
        status = self._libInstance.implicitMesh_pointInSimplex( self._internalID, \
                 point_p, np.uintc(point.size), simplexInd, ctypes.byref(out) )   
        
        if status != 0:
            # Try to save internally again
            self._storeMeshInternally()
            # Retry call
            status = self._libInstance.implicitMesh_pointInSimplex( self._internalID, \
                 point_p, np.uintc(point.size), simplexInd, ctypes.byref(out) )   
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from implicitMesh_pointInSimplex()" )         
            
        return out.value

    