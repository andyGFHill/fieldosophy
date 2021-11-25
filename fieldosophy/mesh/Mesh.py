#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for meshes.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import ctypes
import meshio
import os

from scipy import sparse
from scipy import special

from . import geometrical_functions as geom




    
    
class Mesh:
    # A class for representing an explicit mesh
    
    topD = None # Dimensionality of manifold, i.e., topological dimension
    embD = None # Dimensionality of space embedded in (dimensionality of nodes)
    N = None # Number of nodes
    NT = None # Number of simplices
    
    nodes = None # The nodes of the mesh
    triangles = None # The triangles (defined by connection of nodes) of the mesh
    
    boundary = None   # Boundary of mesh
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    c_bool_p = ctypes.POINTER(ctypes.c_bool)  

    _libInstance = None
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" )
    
    def __init__(self, triangles, nodes, libPath = None):

        if libPath is not None:
            self._libPath = libPath
        
        # Instantiate C library
        self._libInstance = ctypes.CDLL(self._libPath)
        
        # Get dimensionality of manifold
        self.topD = triangles.shape[1]-1
        # Get dimensionality of embedded space
        self.embD = nodes.shape[1]
        
        # Check sizes
        if self.topD > self.embD:
            raise Exception( "sub-manifold dimension cannot be smaller than the space it is embedded in!" )
        
        # get number of nodes
        self.N = nodes.shape[0]
        # Get number of simplices
        self.NT = triangles.shape[0]
        
        # Get topological mesh        
        if triangles.dtype is not np.dtype(np.uintc):
            triangles = triangles.astype(np.uintc)
        if not triangles.flags['C_CONTIGUOUS']:
            triangles = np.ascontiguousarray(triangles)
        self.triangles = triangles
        
        
        # Get nodes
        if nodes.dtype is not np.dtype("float64"):
            nodes = nodes.astype(np.float64)
        if not nodes.flags['C_CONTIGUOUS']:
            nodes = np.ascontiguousarray(nodes)
        self.nodes = nodes

    
        
        
    def copy(self):
        return Mesh( self.triangles, self.nodes, self._libPath)
    
        
    # %% member functions        
    
    
    def refine( self, maxDiam, maxNumNodes, numLevels = None, transformation = None ):
        # Refine mesh or simplices thereof
        
        # If maxDiam is not an array
        if not isinstance(maxDiam, np.ndarray ):
            maxDiam = np.array([maxDiam])
        if maxDiam.dtype is not np.float64:
            maxDiam = maxDiam.astype(np.float64)
        # If max diam does not have the right size
        if (maxDiam.size != 1) and (maxDiam.size != self.N):
            raise Exception( "maxDiam did not have the right size" )
            
        nodes_p = self.nodes.ctypes.data_as(self.c_double_p) 
        maxDiam_p = maxDiam.flatten().ctypes.data_as(self.c_double_p) 
        triangles_p = self.triangles.ctypes.data_as(self.c_uint_p) 
        
        numLevels_p = None
        if numLevels is not None:
            numLevels = ctypes.c_uint( numLevels )
            numLevels_p = ctypes.byref(numLevels)
        
        newNumNodes = ctypes.c_uint( 0 )
        newNumSimplices = ctypes.c_uint( 0 )
        meshId = ctypes.c_uint( 0 )
        
        def transformationWrapper( pointer, lengthOfPointer ):
            
            if transformation is not None:
                array = np.ctypeslib.as_array(pointer, shape=(lengthOfPointer*self.embD, ))
                outdata = array.reshape( (lengthOfPointer, self.embD) ).copy()
                outdata = transformation(outdata)
                array[:] = outdata.flatten()
                # outer = np.ctypeslib.as_array(output, shape=(lengthOfPointer * self.embD, ))
                # outer[:] = outdata.flatten()
            # for iter in range(lengthOfPointer):
            #     outer[iter, :] = outdata[iter,:]
            
            return 0
                
                
        
        CMPFUNC = ctypes.CFUNCTYPE( ctypes.c_int,  self.c_double_p, ctypes.c_uint )
        cmp_func = CMPFUNC( transformationWrapper )
        
        # Call low-level mesh refinement
        self._libInstance.mesh_refineMesh.restype = ctypes.c_int
        self._libInstance.mesh_refineMesh.argtypes = [ \
              self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
              self.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
              self.c_uint_p, self.c_uint_p, self.c_uint_p, \
              ctypes.c_uint, self.c_double_p, ctypes.c_uint,\
              self.c_uint_p, CMPFUNC ]
        status = self._libInstance.mesh_refineMesh( \
            nodes_p, ctypes.c_uint( self.N ), ctypes.c_uint( self.embD ), \
            triangles_p, ctypes.c_uint( self.NT ), ctypes.c_uint( self.topD ), \
            ctypes.byref(newNumNodes), ctypes.byref(newNumSimplices), ctypes.byref(meshId), \
            ctypes.c_uint(np.uintc(maxNumNodes)), maxDiam_p, ctypes.c_uint(np.uintc(maxDiam.size)), \
            numLevels_p, cmp_func )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from mesh_refineMesh()" ) 
            
        newNodes = np.zeros( (newNumNodes.value, self.embD), dtype=np.float64)
        newNodes_p = newNodes.ctypes.data_as(self.c_double_p)
        newSimplices = np.zeros( (newNumSimplices.value, self.topD+1), dtype=np.uintc )
        newSimplices_p = newSimplices.ctypes.data_as(self.c_uint_p)
            
        # Acquire low-level mesh
        self._libInstance.mesh_acquireMesh.restype = ctypes.c_int
        self._libInstance.mesh_acquireMesh.argtypes = [ ctypes.c_uint, \
              self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
              self.c_uint_p, ctypes.c_uint, ctypes.c_uint ]
        status = self._libInstance.mesh_acquireMesh( meshId, \
            newNodes_p, newNumNodes, ctypes.c_uint( self.embD ), \
            newSimplices_p, newNumSimplices, ctypes.c_uint( self.topD ) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from mesh_acquireMesh()" ) 
            
            
        # Return new mesh
        return Mesh(newSimplices, newNodes)
    
    
    
    
    def getObsMat( self, points, embTol = 0.0, centersOfCurvature = None ):
        # Acquire observation matrix (in non-manifold space)
            
        if points.dtype is not np.dtype(np.float64):
            points = points.astype(np.float64)
            
        # Represent the triangles
        triangles_p = self.triangles.ctypes.data_as(self.c_uint_p) 
        # Represent the nodes
        nodes_p = self.nodes.ctypes.data_as(self.c_double_p) 
        # Represent the points
        points_p = points.ctypes.data_as(self.c_double_p) 
        # Represent the centers of curvature
        centersOfCurvature_p = None
        numCentersOfCurvature = 0
        if centersOfCurvature is not None:
            if not isinstance(centersOfCurvature, np.ndarray ):
                raise Exception("'centersOfCurvature' is not a numpy array")
            if centersOfCurvature.ndim != 2:
                raise Exception("'centersOfCurvature' is not a 2-dimensional array")
            
            if centersOfCurvature.shape[1] != self.embD:
                raise Exception("'centersOfCurvature' did not have the right dimensionality")
            numCentersOfCurvature = centersOfCurvature.shape[0]
            centersOfCurvature = centersOfCurvature.astype(np.float64)
            centersOfCurvature_p = centersOfCurvature.ctypes.data_as(self.c_double_p)
        
        # Store observation matrix
        data = np.zeros( ( points.shape[0] * (self.topD+1) ), dtype=np.float64 )
        row = np.zeros( ( points.shape[0] * (self.topD+1) ), dtype=np.uintc )
        col = np.zeros( ( points.shape[0] * (self.topD+1) ), dtype=np.uintc )
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p) 
        
        # Compute observation matrix
        self._libInstance.mesh_getObservationMatrix.restype = ctypes.c_int
        self._libInstance.mesh_getObservationMatrix.argtypes = \
            [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint, \
              self.c_double_p, ctypes.c_uint, \
              self.c_double_p, ctypes.c_uint, \
              self.c_uint_p, ctypes.c_uint, \
              ctypes.c_uint, ctypes.c_uint, ctypes.c_double, \
              self.c_double_p, ctypes.c_uint]  
        status = self._libInstance.mesh_getObservationMatrix( \
            data_p, row_p, col_p, ctypes.c_uint( data.size ), \
            points_p, ctypes.c_uint( points.shape[0] ), \
            nodes_p, ctypes.c_uint( self.nodes.shape[0] ), \
            triangles_p, ctypes.c_uint( self.triangles.shape[0] ), \
            ctypes.c_uint( self.embD ), ctypes.c_uint( self.topD ), ctypes.c_double( embTol ), \
            centersOfCurvature_p, ctypes.c_uint( numCentersOfCurvature ) )
        if status != 0:
            if status == 1:
                raise Exception( "TODO" ) 
            raise Exception( "Uknown error occured! Error code " + str(status) + " from getObservationMatrix()" ) 
    
        # Remove unused
        row = row[ data != 0 ]
        col = col[ data != 0 ]
        data = data[ data != 0 ]
        out = sparse.coo_matrix( (data, (row, col)), shape=(points.shape[0], self.N) )
        out = out.tocsr()
        
        return( out )
    
    
    
    
    def S2N( self  ):
        # Get matrix of which simplices that are associated with which nodes
        
        # Get matrix mapping which nodes each simplex is associated to (simplex to nodes matrix) 
        cols = np.arange( self.NT ).repeat(self.topD + 1)        
        rows = self.triangles.reshape( (-1) )
        S2N = sparse.coo_matrix( (np.ones((self.NT * (self.topD+1))), (rows, cols)), shape=(self.N, self.NT) )
        
        return S2N
    
    
    def S2NByArea( self, areas  ):
        # Get simplex values to nodes by weighting area
                
        # Get matrix mapping which nodes each simplex is associated to (simplex to nodes matrix) 
        cols = np.arange( self.NT ).repeat(self.topD + 1)        
        rows = self.triangles.reshape( (-1) )
        S2N = sparse.coo_matrix( (np.ones((self.NT * (self.topD+1))), (rows, cols)), shape=(self.N, self.NT) )
        
        # Acquire total area for each vertex
        totalArea = S2N * areas
        # Acquire relative area for each simplex in each node
        areas = areas.repeat(self.topD + 1) / totalArea[rows]
        
        # Acquire simplex values weighted by their relative area matrix
        S2N = sparse.coo_matrix( (areas, (rows, cols)), shape=(self.N, self.NT) )
        
        return S2N

        
        
        
    
    
    def gradAndAreaForSimplices( self, grads = True, areas = True ):
        # Acquire the gradients coefficient matrix of faces in the mesh
            
        # Represent the triangles
        triangles_p = self.triangles.ctypes.data_as(self.c_uint_p) 
        # Represent the nodes
        nodes_p = self.nodes.ctypes.data_as(self.c_double_p) 
        
        maxNumNonNull = self.embD * self.NT * (self.topD+1) + 100
        
        
        data_p = None
        areas_p = None
        row_p = None
        col_p = None
        data = None
        areasVals = None
        row = None
        col = None
        
        if grads:
            data = np.NaN * np.ones( (maxNumNonNull) , dtype=np.float64 )
            row = np.zeros( (maxNumNonNull) , dtype=np.uintc )
            col = np.zeros( (maxNumNonNull) , dtype=np.uintc )
            data_p = data.ctypes.data_as(self.c_double_p)
            row_p = row.ctypes.data_as(self.c_uint_p)
            col_p = col.ctypes.data_as(self.c_uint_p) 
        
        if areas:
            areasVals = np.NaN * np.ones( (self.NT) , dtype=np.float64 )
            areas_p = areasVals.ctypes.data_as(self.c_double_p)
        
        
        areas
        # Compute observation matrix
        self._libInstance.mesh_getGradientCoefficientMatrix.restype = ctypes.c_int
        self._libInstance.mesh_getGradientCoefficientMatrix.argtypes = \
            [ ctypes.c_uint, 
             self.c_double_p, self.c_uint_p, self.c_uint_p, self.c_uint_p, \
             self.c_double_p, ctypes.c_uint, \
             self.c_uint_p, ctypes.c_uint, \
             ctypes.c_uint, ctypes.c_uint, \
             self.c_double_p ]
        index = ctypes.c_uint(0)
                
        status = self._libInstance.mesh_getGradientCoefficientMatrix( ctypes.c_uint(maxNumNonNull), \
            data_p, row_p, col_p, ctypes.byref( index ), \
            nodes_p, ctypes.c_uint( self.N ), \
            triangles_p, ctypes.c_uint( self.NT ), \
            ctypes.c_uint(self.embD), ctypes.c_uint( self.topD ), \
            areas_p)

        if status != 0:
            if status == 1:
                raise Exception( "Not enough non null elements given!" ) 
            raise Exception( "Uknown error occured! Error code " + str(status) + " from mesh_getGradientCoefficientMatrix()" ) 
    
        out = {}
    
        if grads:
            # Remove unused
            row = row[~np.isnan(data)]
            col = col[~np.isnan(data)]
            data = data[~np.isnan(data)]
            gradMat = sparse.coo_matrix( (data, (row, col)), shape=(self.NT * self.embD, self.N) )
            gradMat = gradMat.tocsr()
            out["gradMat"] = gradMat

        if areas:
            out["areas"] = areasVals
        
        return out
    
    
    
    def getBoundary(self):
        # Acquire the boundary of current mesh
        
        if self.boundary is None:
            self.boundary = self.computeBoundary()
            
        return self.boundary
    
    def getNeighs(self):
        # Get neighborhood of mesh
        
        if self.topD > 1:
            # Get all edges
            edges = Mesh.getEdges( self.triangles, self.topD, self.topD, libInstance = self._libInstance )
            # Get all neighbors
            neighs = Mesh.getSimplexNeighbors( edges["simplicesForEdges"], edges["edgesForSimplices"], libInstance = self._libInstance )
        else:
            
            # Get all simplices sharing node indices with current simplices (excluding current simplex)
            neighs = [ np.setdiff1d( np.nonzero( np.any( np.isin( self.triangles, self.triangles[iter,:]), axis = 1 ) )[0], np.array([iter]) )  for iter in range(self.NT) ]
            # The simplices which are not sharing all of their edges
            for iter in range(self.NT):
                if neighs[iter].size < 2:
                    neighs[iter] = np.append( neighs[iter], self.NT * np.ones( 2 - neighs[iter].size) )
            neighs = np.array(neighs, dtype=np.uintc)            
        
        return neighs
        
    
    def computeBoundary(self):
        # Compute boundary of current mesh
        
        # Get all edges and which simplices they belong to
        boundary = Mesh.getEdges( self.triangles, self.topD, self.topD, libInstance = self._libInstance )
        # Get edges on the boundary by index (can be found since they only have one simplex)
        boundaryEdgesIndices = np.any( boundary["simplicesForEdges"] == self.NT, axis = 1 )
        # Get boundary simplices 
        boundarySimplices = boundary["simplicesForEdges"][boundaryEdgesIndices,:].flatten()
        boundarySimplices = boundarySimplices[boundarySimplices != self.NT]
        # Get actually array of boundary edges
        boundaryEdges = boundary["edges"][boundaryEdgesIndices, :]
        # Get boundary nodes
        boundaryNodes = np.unique( boundaryEdges )
        
        return { "nodes":boundaryNodes, "edges":boundaryEdges, "simplices":boundarySimplices }


    def getStatistics(self, calculate = []):
        # Some statistics of mesh
                
        # Get all vertices of mesh
        verts = None
        if self.topD == 1:
            verts = self.triangles
        else:
            verts = Mesh.getEdges( self.triangles, self.topD, 2, libInstance = self._libInstance )["edges"]
                
        vertLengths = np.sqrt(np.sum( np.diff( self.nodes[verts, :], axis=1 ).reshape( (-1,self.embD) )**2, axis=1 )).flatten()
        diamMin = np.min(vertLengths)
        diamMax = np.max(vertLengths)
        
        
        return { "diamMin":diamMin, "diamMax":diamMax }
    

    
    def saveMesh(self, msh_filename = None, vtk_filename = None):
        # Saves current mesh to file
        Mesh.saveMeshToFile(self, msh_filename, vtk_filename)
    


    def cutMesh( self, outside, includeBorderTriangles = True ):
        """ Remove nodes in mesh. It is possible to keep nodes which faces also includes nodes which explicitly should be kept if the 'includeBorderTriangles' flag is set. """
        
        # copy current mesh
        mesh = self.copy()
        
        numNodes = mesh.N

        # Get triangles with outside nodes
        outsideTriangles = np.any( np.isin( mesh.triangles, np.where(outside) ), axis=1 )
        # Get triangles with inside nodes
        insideTriangles = np.any( np.isin( mesh.triangles, np.where(~outside) ), axis=1 )
        # Get triangles with both inside and outside nodes
        bothTriangles = insideTriangles & outsideTriangles
        
        # Get nodes which are part of bothTriangles
        connected = np.full(numNodes, False, dtype=bool)
        connected[ np.unique( mesh.triangles[ bothTriangles, : ].flatten() ) ] = True        
        
        # Acquire which triangles and nodes to use
        useTriangles = insideTriangles & ~bothTriangles
        useNodes = ~outside
        if includeBorderTriangles:
            useTriangles = insideTriangles | bothTriangles
            useNodes = ~outside | connected
        
        # Remove all triangles that are purely outside
        mesh.triangles = mesh.triangles[useTriangles, :]    
        # Get index of points not to remove
        keepPointsIndex = np.sum(useNodes) * np.ones( (numNodes) )
        keepPointsIndex[useNodes] = np.array(range(np.sum(useNodes)))
        # Go through each triangle and rename index
        for iter in range(mesh.triangles.shape[0]):
            mesh.triangles[iter, :] = keepPointsIndex[ mesh.triangles[iter, :] ].astype(int)
        # Remove points
        mesh.nodes = mesh.nodes[useNodes]
        
        return Mesh(mesh.triangles, mesh.nodes)
     
        
    
    
    def cutOutsideMeshOnSphere( self, activePoints, distance, includeBorderTriangles = True ):
        """ Remove nodes in mesh outside of specified spherical distance """

        # Get minimum distance between nodes and active points
        minDistInd, minDist = geom.smallestDistanceBetweenPointsOnSphere( self.nodes, activePoints.transpose().copy(), self._libInstance )
        # Mark all nodes too far away as outside
        outside = minDist > distance        
        # cut away the unecessary nodes and triangles of mesh
        return self.cutMesh( outside, includeBorderTriangles )
    
    
    
    def cutOutsideMesh( self, activePoints, distance, includeBorderTriangles = True ):
        """ Remove nodes in mesh outside of specified planar distance """
        
        # Get minimum distance between nodes and active points
        minDistInd, minDist = geom.smallestDistanceBetweenPoints( self.nodes, activePoints.transpose().copy(), self._libInstance )
        # Mark all nodes too far away as outside
        outside = minDist > distance
        # cut away the unecessary nodes and triangles of mesh
        return self.cutMesh( outside, includeBorderTriangles )
        
    
    
    def saveMeshToFile(self, msh_filename = None, vtk_filename = None):
        ''' Saves a mesh to file '''
        
        nodes = self.nodes
        cells = { "triangle" : self.triangles }
        output = meshio.Mesh( nodes, cells )
            
        if msh_filename is not None:
            meshio.write( msh_filename, output )
        
        if vtk_filename is not None:
            meshio.write( vtk_filename, output )    
        
        return
    
    
    
    def getSimplicesForNodes( self, nodeIndices ):
        # Acquire simplices including given node index
        
        # Preallocate output
        output = [None] * nodeIndices.size
        # Loop through all node indices
        for iter in range(nodeIndices.size):
            # Get logical indices to simplices including current node index
            tempInds = np.any( self.triangles == nodeIndices[iter], axis=1 )
            output[iter] = np.where(tempInds)[0]
        
        return output
    
    def getSimplicesForPoints( self, points ):
        # Acquire simplices including given points
    
        # Get observation matrix of points
        obsMat = self.getObsMat( points )
        
        # Preallocate output
        output = self.NT * np.ones( points.shape[0], dtype=np.uintc )
        # Loop through all points
        for iter in range(points.shape[0]):
            
            # go trough simplices and find the simplex that includes the most of the given nodes
            tempSimplex = np.array( [ np.isin( self.triangles[:,iterDim], np.nonzero( obsMat[iter,:] )[1] ) for iterDim in range(self.topD+1) ] )
            tempSimplex = np.sum( tempSimplex, axis=0)
            tempInd = np.argmax(tempSimplex)
            if (tempSimplex[tempInd] > 0):
                output[iter] = tempInd
            
        return output
    
    def getBoundingBox(self):
        # Get bounding box of mesh
        
        boundingBox = np.zeros((self.embD, 2))
        boundingBox[:, 0] = np.min(self.nodes, axis = 0)
        boundingBox[:, 1] = np.max(self.nodes, axis = 0)
        
        return boundingBox
    
    
    
    
    
    # %% Static functions    
    
    def getEdges(triangles, topD, edgeD, \
              edgesOutput = True, simplicesForEdgesOutput = True, edgesForSimplicesOutput = True, \
              libPath = "./meshLIB.so", libInstance = None ):
        '''
        Acquire array of edges
        
        :param triangles: simplices as a 2D array where each row is a separate simlex and the columns represent the indices of nodes in the simplex
        :param topD: The dimensionality of the simplex (topD = 2 means that the simplex is a triangle, and hence have 3 nodes)
        :param edgeD: Number of elements in an edge (edgeD = 2 correspond to an edge being a pair of points)
        :param edgeOutput: True if an explicit list of edges should be acquired. (Default is True)
        :param simplicesForEdgesOutput: True if an explicit list of which simplices that are associated to each edge should be acquired. (Default is True)
        :param edgeForSimplicesOutput: True if an explicit list of which edges each simplex has should be acquired. (Default is True)
        :param libPath: The path to the dynamically linked library to use for computing the edges
        :param libInstance: A possible instance of the dynamically linked library. 
        
        '''
        
        if topD == 1:
            
            edges = np.unique(triangles).reshape((-1,1))
            simplicesForEdges = [ np.nonzero(np.any(triangles == edges[iter], axis=1))[0] for iter in range(edges.size) ]
            for iter in range(edges.size):
                if (simplicesForEdges[iter].size < 2):
                    simplicesForEdges[iter] = np.append( simplicesForEdges[iter], triangles.shape[0] * np.ones( (2 - simplicesForEdges[iter].size) ) )
            simplicesForEdges = np.array(simplicesForEdges, dtype=np.uintc)
            
            edgesForSimplices = triangles
            
            return { "edges":edges, "simplicesForEdges":simplicesForEdges, "edgesForSimplices":edgesForSimplices }
        
        
        if libInstance is None:
            libInstance = ctypes.CDLL(libPath)
        
        # Get number of possible combinations of edges for each simplex
        numCombinations = np.uintc(special.binom( topD+1, edgeD ))
        
        # Preallocate space for output
        numEdges = ctypes.c_uint(np.uintc(0))
        edgeId = ctypes.c_uint(np.uintc(0))
        maxSimplicesPerEdge = ctypes.c_uint(np.uintc(0))
        
        # Set pointers
        triangles_p = triangles.ctypes.data_as(Mesh.c_uint_p)
        
        # Call computation of edges
        libInstance.mesh_computeEdges.restype = ctypes.c_int
        libInstance.mesh_computeEdges.argtypes = \
            [ ctypes.c_uint, Mesh.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
             Mesh.c_uint_p, Mesh.c_uint_p, Mesh.c_uint_p ]                      
        status = libInstance.mesh_computeEdges( \
              ctypes.c_uint( edgeD ), triangles_p, ctypes.c_uint(triangles.shape[0]), ctypes.c_uint( topD ), \
              ctypes.byref( numEdges ), ctypes.byref( edgeId ), ctypes.byref( maxSimplicesPerEdge ) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) ) 
        
        edges = None
        edges_p = None
        simplicesForEdges = None
        simplicesForEdges_p = None
        edgesForSimplices = None
        edgesForSimplices_p = None
        
        # If should provide edges as output
        if edgesOutput:
            # Preallocate edges
            edges = np.empty( (numEdges.value, edgeD) , dtype=np.uintc )        
            # Set pointer to edges
            edges_p = edges.ctypes.data_as(Mesh.c_uint_p)
        # If should provide simplices for each edge as output
        if simplicesForEdgesOutput:
            # Preallocate
            simplicesForEdges =  triangles.shape[0] * np.ones( (numEdges.value, maxSimplicesPerEdge.value) , dtype=np.uintc )        
            # Set pointer
            simplicesForEdges_p = simplicesForEdges.ctypes.data_as(Mesh.c_uint_p)
        # If should provide edges for each simplex as output
        if edgesForSimplicesOutput:
            # Preallocate
            edgesForSimplices = np.empty( (triangles.shape[0], numCombinations) , dtype=np.uintc )      
            # Set pointer
            edgesForSimplices_p = edgesForSimplices.ctypes.data_as(Mesh.c_uint_p)
            
        # Call retrieval of edges
        libInstance.mesh_populateEdges.restype = ctypes.c_int
        libInstance.mesh_populateEdges.argtypes = \
            [ Mesh.c_uint_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, \
             Mesh.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
             Mesh.c_uint_p, ctypes.c_uint ]
        status = libInstance.mesh_populateEdges( \
            edges_p, ctypes.c_uint( edgeD ), numEdges, edgeId, \
            simplicesForEdges_p, maxSimplicesPerEdge, ctypes.c_uint( triangles.shape[0] ), \
            edgesForSimplices_p, ctypes.c_uint( numCombinations ) )
        if status != 0:
            if status == 1:
                raise Exception( "Edges not available!" ) 
            else:
                raise Exception( "Uknown error occured! Error code: " + str(status) ) 
            
        # Clear edges
        libInstance.mesh_clearEdges.restype = ctypes.c_int
        libInstance.mesh_clearEdges.argtypes = [ ctypes.c_uint ]
        status = libInstance.mesh_clearEdges( edgeId )
        if status != 0:
            raise Exception( "Edges not available!" ) 
            
        return { "edges":edges, "simplicesForEdges":simplicesForEdges, "edgesForSimplices":edgesForSimplices }
        

    def getSimplexNeighbors( simplicesForEdges, edgesForSimplices, libPath = "./meshLIB.so", libInstance = None ):
        '''
        ' Compute neighboring simplices for every simplex in mesh
        '
        ' simplicesForEdges: matrix where row correspond to edge index and columns correspond to simplex indices associated with edge in corresponding row.
        ' edgesForSimplices: matrix where row correspond to simplex index and columns correspond to edge indices associated with simplex in corresponding row.
        ' libPath : The path to the dynamically linked library to use for computing the edges
        ' libInstance : A possible instance of the dynamically linked library. 
        '
        '''
        
        if libInstance is None:
            libInstance = ctypes.CDLL(libPath)
            
        if ( simplicesForEdges.shape[1] != 2 ):
            raise Exception("Error! More than two simplices sharing edges. This should be impossible when considering borders of simplice. ")
            
        # Preallocate neighbors
        neighs = np.empty( (edgesForSimplices.shape[0], edgesForSimplices.shape[1]) , dtype=np.uintc )        
        # Set pointer to neighbors
        neighs_p = neighs.ctypes.data_as(Mesh.c_uint_p)
        # Set pointer
        simplicesForEdges_p = simplicesForEdges.ctypes.data_as(Mesh.c_uint_p)
        # Set pointer
        edgesForSimplices_p = edgesForSimplices.ctypes.data_as(Mesh.c_uint_p)
        
        # Call computation of neighbors
        libInstance.mesh_getSimplexNeighborhood.restype = ctypes.c_int
        libInstance.mesh_getSimplexNeighborhood.argtypes = \
            [ ctypes.c_uint, ctypes.c_uint, 
             Mesh.c_uint_p, ctypes.c_uint, \
             Mesh.c_uint_p, ctypes.c_uint, \
             Mesh.c_uint_p ]
        status = libInstance.mesh_getSimplexNeighborhood( \
             simplicesForEdges.shape[0], edgesForSimplices.shape[0], \
             simplicesForEdges_p, simplicesForEdges.shape[1], \
             edgesForSimplices_p, edgesForSimplices.shape[1], \
             neighs_p )
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) ) 
            
        return neighs
         
    
    
    
    def loadMeshFromFile( filename = None ):
            # Loads a mesh from a file using meshio
            
            # Open mesh file
            mesh = meshio.read(filename)
            
            nodes = mesh.points
            triangles = None
            if type(mesh.cells) is dict:
                triangles = mesh.cells["triangle"]
            else:
                triangles = mesh.cells[0][1]
            
            mesh = Mesh( triangles, nodes )
            
            return mesh    
        
        
        
    def meshOnCircle( maxDiam, maxNumNodes, radius = 1 ):
        """ Creates 1D mesh on circle. """ 

    
        # ---- Create original rectangle ----
        numberOfNodes = np.min( ( maxNumNodes, np.ceil( np.pi / np.arcsin(np.min( (np.abs(maxDiam)/2.0 / radius, 1.0) )) ) ) ).astype(int)
        
        nodes = np.linspace( 0, 360, num = numberOfNodes + 1 )
        nodes = nodes[0:-1] 
        nodes = radius * np.stack( ( np.cos( nodes * np.pi/180 ), np.sin( nodes * np.pi/180 ) ) ).transpose().copy()
        
        triangles = np.stack( (np.arange(0, nodes.shape[0]), np.arange(1, nodes.shape[0] + 1)) ).transpose().copy()
        triangles[-1, 1] = 0
        
        
#        def transformation(x):
#            return radius * geom.mapToHypersphere(x)
#        
#        # Transform nodes of box
#        nodes = transformation(nodes)
#    
        # Create mesh of box
        mesh = Mesh(nodes = nodes, triangles = triangles)
        
        # Get neighbors
        neighs = np.arange(0,triangles.shape[0]).reshape((-1,1)) * np.ones((1,2))
        neighs[:,0] = (neighs[:,0]-1) % triangles.shape[0]
        neighs[:,1] = (neighs[:,1]+1) % triangles.shape[0]
        
#    
#        # Refine to perfection
#        mesh, neighs = mesh.refine( maxDiam = maxDiam, maxNumNodes = maxNumNodes, transformation = transformation )
        
        
           
        return (mesh, neighs)
    
    
    
    def meshInSquare( scale ):
        """ Creates simplicial mesh in box. """    
        
        triangles = np.array([ [0,1,2], [1,2,3]], dtype=int)
        nodes = np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) * scale.reshape((1,-1))
        
        mesh = Mesh( triangles, nodes )
        
        return mesh
    
    

    
    
    def meshOnBox( maxDiam, maxNumNodes ):
        """ Creates triangular mesh on box surface. """    
        
        
        nodes = np.zeros((8, 3))
        # Loop through each dimension
        nodes[:, 0] = np.tile( np.array([-1,1]), 4 )
        nodes[:, 1] = np.tile( np.repeat(np.array([-1,1]), 2), 2 )
        nodes[:, 2] = np.repeat(np.array([-1,1]), 4)
        
        triangles = np.zeros( (12, 3), dtype = np.uint64 )
        triangles[0, :] = np.array([0,1,2])
        triangles[1, :] = np.array([1,2,3])
        triangles[2, :] = np.array([4,5,6])
        triangles[3, :] = np.array([5,6,7])
        
        triangles[4, :] = np.array([0,1,4])
        triangles[5, :] = np.array([1,4,5])
        triangles[6, :] = np.array([2,3,6])
        triangles[7, :] = np.array([3,6,7])
        
        triangles[8, :] = np.array([0,2,4])
        triangles[9, :] = np.array([2,4,6])
        triangles[10, :] = np.array([1,3,5])
        triangles[11, :] = np.array([3,5,7])
        
        # Create mesh of box
        mesh = Mesh(nodes = nodes, triangles = triangles)
        
        # Refine to perfection
        mesh, neighs = mesh.refine( maxDiam = maxDiam, maxNumNodes = maxNumNodes )
        
        return mesh
    
    
    
    def meshOnSphere( maxDiam, maxNumNodes, radius = 1 ):
        """ Creates triangular mesh on sphere surface. """    
    
        def transformation(x):
            return radius * geom.mapToHypersphere(x)
    
        
        # ------- Create original icosahedron --------
        mesh = Mesh.icosahedronSurface(radius)
        mesh = Mesh( nodes = mesh["nodes"], triangles = mesh["triangles"] )
        
        # Refine to perfection
        mesh = mesh.refine( maxDiam = maxDiam, maxNumNodes = maxNumNodes, transformation = transformation )
           
        return mesh
    
    
    
    def icosahedronSurface(r):
        """
        Returns a icosahedral mesh with radius r
        
        :return: A dictionary with 'nodes' and 'triangles'
        """
    
        c = (1.0+np.sqrt(5))/2.0
    
        nodes = np.array( [ \
            [-1.0, c, 0.0], \
            [1.0, c, 0.0], \
            [-1.0, -c, 0.0], \
            [1.0, -c, 0.0], \
            [0.0, -1.0, c], \
            [0.0, 1.0, c], \
            [0.0, -1.0, -c], \
            [0.0, 1.0, -c], \
            [c, 0.0, -1.0], \
            [c, 0.0, 1.0], \
            [-c, 0.0, -1.0], \
            [-c, 0.0, 1.0] \
           ] )
            
        nodes = nodes * r / np.sqrt(c**2+1)
            
        triangles = np.array( [ \
            [0, 11, 5], \
            [0, 5, 1], \
            [0, 1, 7], \
            [0, 7, 10], \
            [0, 10, 11], \
            [1, 5, 9], \
            [5, 11, 4], \
            [11, 10, 2], \
            [10, 7, 6], \
            [7, 1, 8], \
            [3, 9, 4], \
            [3, 4, 2], \
            [3, 2, 6], \
            [3, 6, 8], \
            [3, 8, 9], \
            [4, 9, 5], \
            [2, 4, 11], \
            [6, 2, 10], \
            [8, 6, 7], \
            [9, 8, 1] \
            ] )
            
        return {"nodes":nodes, "triangles":triangles}
    
    
