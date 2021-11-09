#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for plotting meshes.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np

from .HyperRectMesh import HyperRectMesh
from .Mesh import Mesh



class MeshPlotter:
    # Class dedicated to plotting of mesh
       
    _mesh = None
    
    edges = None
    boundaryEdges = None
    
    
    def __init__(self, mesh):
        
        # Initiate object
        self._mesh = mesh
        
        # # If mesh is not hyper rect
        # if isinstance( mesh, HyperRectMesh ):
        #     # Initiate object
        #     self._mesh = mesh
        # else:
        #     # Initiate object
        #     self._mesh = HyperRectMesh( mesh )
        
        
        
    def computeEdges(self):
        
        # Get all lines of mesh
        self.edges = Mesh.getEdges(self._mesh.triangles, self._mesh.topD, 2, simplicesForEdgesOutput = False, libInstance = self._mesh._libInstance )
        # self.edges = self._mesh.getEdges()
        
        return
    
    def computeBoundaryEdges(self):
        
        # Get boundary
        boundarySimplices = self._mesh.getBoundary()
        
        if self._mesh.topD > 2:
            # Get all lines on the boundary
            self.boundaryEdges = Mesh.getEdges(boundarySimplices["edges"], self._mesh.topD-1, 2, simplicesForEdgesOutput = False, libInstance = self._mesh._libInstance )
            # self.boundaryEdges = Mesh.getEdges(boundarySimplices["edges"], self._mesh.topD-1, 2, simplicesForEdgesOutput = False, libInstance = self._mesh.getLibInstance() )
        else:
            self.boundaryEdges = boundarySimplices
        
        return
        
        
    def getLines(self):
        # Get all lines of mesh
        
        if self._mesh.topD == 1:
            inds = self._mesh.triangles
            # inds = self._mesh.getTriangles()
        else:
            if self.edges is None:
                self.computeEdges()        
            inds = self.edges["edges"]
        
        output = []
        # curNodes = self._mesh.getNodes( inds )
        for iterD in range(self._mesh.embD):
            vals = self._mesh.nodes[ inds , iterD ]
            # vals = curNodes[ :, iterD ]
            vals = np.concatenate( (vals, np.NaN * np.ones((vals.shape[0],1)) ), axis=1 ).flatten()
            output.append( vals )
        
        return output
    
    def getBoundaryLines(self):
        # Get all lines of boundary of mesh
        
        if self.edges is None:
            self.computeEdges()  
        
        if self._mesh.topD == 1:
            inds = self._mesh.triangles
            # inds = self._mesh.getTriangles()
        
        if self.boundaryEdges is None:
            self.computeBoundaryEdges()
        
        inds = self.boundaryEdges["edges"]
        
        output = []
        for iterD in range(self._mesh.embD):
            vals = self._mesh.nodes[ inds , iterD ]
            vals = np.concatenate( (vals, np.NaN * np.ones((vals.shape[0],1)) ), axis=1 ).flatten()
            output.append( vals )
        
        return output
    
    
    def getSimplicesLines(self, simpInds = None ):
        # Get all lines of given simplices
        
        if simpInds is None:
            simpInds = np.ones(self._mesh.NT, dtype=bool)
            
        simpInds = simpInds[simpInds < self._mesh.NT]
        
        # Get lines of given simplices 
        inds = None
        if self._mesh.topD > 1:
            inds = Mesh.getEdges(self._mesh.triangles[simpInds, :], self._mesh.topD, 2, simplicesForEdgesOutput = False, edgesForSimplicesOutput = False, libInstance = self._mesh._libInstance )["edges"]
            # inds = Mesh.getEdges(self._mesh.getTriangles(simpInds), self._mesh.topD, 2, simplicesForEdgesOutput = False, edgesForSimplicesOutput = False, libInstance = self._mesh.getLibInstance() )["edges"]
        else:
            inds = self._mesh.triangles[simpInds, :]
            # inds = self._mesh.getTriangles(simpInds)
        
        output = []
        # curNodes = self._mesh.getNodes(inds)
        for iterD in range(self._mesh.embD):
            vals = self._mesh.nodes[ inds , iterD ]
            # vals = curNodes[: , iterD ]
            vals = np.concatenate( (vals, np.NaN * np.ones((vals.shape[0],1)) ), axis=1 ).flatten()
            output.append( vals )
            
        return output
    
    
    
