#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for finding distances between points.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import ctypes
from .HyperRectMesh import HyperRectMesh



class MeshDistanceFinder:
    # Class dedicated to finding distances for mesh
       
    _mesh = None
    _metricTensors = None
    _tensorMode = None
    _internalID = None
    _internalTightenedPathID = None
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    c_bool_p = ctypes.POINTER(ctypes.c_bool)  
    
    def __init__(self, mesh, metricTensors, tensorMode):
        
        # Initiate object
        self._mesh = mesh
        if not isinstance(self._mesh, HyperRectMesh):
            self._mesh = HyperRectMesh(self._mesh)
        
        
        self._metricTensors = metricTensors.astype( np.float64 )
        self._tensorMode = np.int(tensorMode)
        
        
        self._mesh._libInstance.GraphPath_getShortestPathDijkstra.restype = ctypes.c_int
        self._mesh._libInstance.GraphPath_getShortestPathDijkstra.argtypes = \
            [ ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, \
             self.c_double_p, \
             self.c_uint_p, ctypes.c_uint, \
             self.c_double_p, ctypes.c_uint, \
             self.c_uint_p, \
             self.c_double_p, self.c_uint_p, \
             self.c_double_p, self.c_uint_p,
             ctypes.c_uint, ctypes.c_double]
                
        self._mesh._libInstance.GraphPath_getShortestPathAStar.restype = ctypes.c_int
        self._mesh._libInstance.GraphPath_getShortestPathAStar.argtypes = \
            [ ctypes.c_uint, ctypes.c_double, \
             ctypes.c_uint, ctypes.c_uint, self.c_double_p, \
             ctypes.c_uint, ctypes.c_uint, self.c_double_p, \
             self.c_double_p, self.c_uint_p, self.c_uint_p, \
             self.c_uint_p ]
                
        self._mesh._libInstance.GraphPath_retrievePathAStar.restype = ctypes.c_int
        self._mesh._libInstance.GraphPath_retrievePathAStar.argtypes = [ ctypes.c_uint, ctypes.c_uint, self.c_uint_p, self.c_uint_p ]
        
        self._mesh._libInstance.meshAndMetric_create.restype = ctypes.c_int
        self._mesh._libInstance.meshAndMetric_create.argtypes = \
            [ ctypes.c_uint, self.c_uint_p, \
             self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
             ctypes.c_int, \
             self.c_uint_p, self.c_uint_p, \
             self.c_uint_p, self.c_uint_p ]
                
        self._mesh._libInstance.meshAndMetric_erase.restype = ctypes.c_int
        self._mesh._libInstance.meshAndMetric_erase.argtypes = [ ctypes.c_uint ]
        
        self._mesh._libInstance.meshAndMetric_check.restype = ctypes.c_int
        self._mesh._libInstance.meshAndMetric_check.argtypes = [ ctypes.c_uint ]
                                                  
        self._mesh._libInstance.FreePath_createTightenedPath.restype = ctypes.c_int
        self._mesh._libInstance.FreePath_createTightenedPath.argtypes = [ self.c_uint_p, ctypes.c_uint, ctypes.c_uint, \
              self.c_uint_p, self.c_uint_p, self.c_double_p, \
              self.c_double_p, self.c_double_p ]
        
        self._mesh._libInstance.FreePath_optimizeTightenedPath.restype = ctypes.c_int
        self._mesh._libInstance.FreePath_optimizeTightenedPath.argtypes = [ ctypes.c_uint, ctypes.c_uint, self.c_uint_p, self.c_double_p ]
        
        self._mesh._libInstance.FreePath_modifyTightenedPath.restype = ctypes.c_int
        self._mesh._libInstance.FreePath_modifyTightenedPath.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint ]
        
        self._mesh._libInstance.FreePath_retrieveTightenedPath.restype = ctypes.c_int
        self._mesh._libInstance.FreePath_retrieveTightenedPath.argtypes = [ self.c_double_p, ctypes.c_uint, self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
               self.c_uint_p, self.c_uint_p]

        self._mesh._libInstance.FreePath_eraseTightenedPath.restype = ctypes.c_int
        self._mesh._libInstance.FreePath_eraseTightenedPath.argtypes = [ ctypes.c_uint ]



        
     
        
    def __del__(self):
        # Destructor
        
        # Erase stored mesh
        self.logout()
        
        
    def _storeMeshInternally(self):
        # Store mesh internally
        
        # If is stored internally already
        if (self.checkInternal()):
            return

        self._mesh._storeMeshInternally()
        if (self._mesh._internalID is None):
            raise Exception( "No internal ID was found!" )
        
        numElementsPerTensor = np.uintc(np.prod(self._metricTensors.shape[1:]))
        numTensors = np.uintc(self._metricTensors.shape[0])
        tensorMode = np.int(self._tensorMode)
        metricTensors_p = self._metricTensors.ctypes.data_as(self.c_double_p)
        sectors_p = None
        numSectorDimensions_p = None
        
        # Preallocate output
        ID = ctypes.c_uint( 0 )
        numNodes = ctypes.c_uint( 0 )
        numSimplices = ctypes.c_uint( 0 )
        
        # Create implicit mesh
        status = self._mesh._libInstance.meshAndMetric_create( ctypes.c_uint(self._mesh._internalID), ctypes.byref(ID), \
                      metricTensors_p, ctypes.c_uint( numElementsPerTensor ), ctypes.c_uint( numTensors ), \
                      ctypes.c_int( tensorMode ), \
                      ctypes.byref(numNodes), ctypes.byref(numSimplices), \
                      sectors_p, numSectorDimensions_p )
            
        if status != 0:
            raise Exception( "Uknown error occured! Error code " + str(status) + " from meshAndMetric_create()" ) 

        # Store mesh internally            
        self._internalID = ID.value
        self._mesh.N = numNodes.value
        self._mesh.NT = numSimplices.value
        
        
        
        
        
        
    def logout(self):
        # Erase mesh internally
        
        # If saved mesh internally
        if self._internalID is not None:
            # Erase
            self._mesh._libInstance.meshAndMetric_erase( ctypes.c_uint(self._internalID) )
            self._internalID = None
        # If saved internal tightened path internally
        if self._internalTightenedPathID is not None:
            self._mesh._libInstance.FreePath_eraseTightenedPath( self._internalTightenedPathID )
            self._internalTightenedPathID = None
        
        
    def checkInternal(self):
        # Check if internal representation is still existing
        
        if self._internalID is None:
            return False
        
        status = self._mesh._libInstance.meshAndMetric_check( ctypes.c_uint(self._internalID) )
        
        # If existed internally
        if status == 0:
            return True
        else:
            return False
        
        
        
        
        
    def findShortestGraphPathDijkstra(self, startNodeInd, startNodeSimplex, startPoint = None, \
          endNodeInds = np.array([]), endPoints = None, endPointsSimplices = None ):
        '''
        ' Find shortest path between a start point and one or several end points
        '
        '''
        
        self._storeMeshInternally()
        
        # Make sure that nodes exists
        if ( startNodeInd >= self._mesh.N ):
            raise Exception( "Start node was out of range!" )
        
        
        if not isinstance(startNodeInd, np.uintc):
            startNodeInd = np.uintc(startNodeInd)
        if not isinstance(startNodeSimplex, np.uintc):
            startNodeSimplex = np.uintc(startNodeSimplex)
        if startPoint is not None:
            if startPoint.dtype is not np.float64:
                startPoint = np.float64(startPoint)
            
        if endNodeInds.dtype is not np.dtype(np.uintc):
            endNodeInds = endNodeInds.astype(np.uintc)
        if endPoints is not None:            
            if endPoints.dtype is not np.dtype(np.float64):
                endPoints = endPoints.astype(np.float64)
        if endPointsSimplices is not None:            
            if endPointsSimplices.dtype is not np.dtype(np.uintc):
                endPointsSimplices = endPointsSimplices.astype(np.uintc)
        
        # Preallocate space for output
        distances = np.inf * np.ones( (self._mesh.N), dtype = np.double )
        priorNodes = self._mesh.N * np.ones( (self._mesh.N), dtype = np.uintc )
        
        # Set pointers
        distances_p = distances.ctypes.data_as(self.c_double_p)
        priorNodes_p = priorNodes.ctypes.data_as(self.c_uint_p)
        
        startPoint_p = None
        if (startPoint is not None ):
            startPoint_p = startPoint.ctypes.data_as(self.c_double_p)
        
        endNodeIndices_p = None
        if (endNodeInds.size > 0 ):
            endNodeIndices_p = endNodeInds.ctypes.data_as(self.c_uint_p)
        
        endPoints_p = None
        endSimplices_p = None
        endPointsDistances_p = None
        endPointsPriorNodes_p = None
        if ( (endPoints is not None) and (endPointsSimplices is not None)  ):
            endPoints_p = endPoints.ctypes.data_as(self.c_double_p)
            endSimplices_p = endPointsSimplices.ctypes.data_as(self.c_uint_p)
            
            endPointsDistances = np.inf * np.ones( (endPoints.shape[0]), dtype = np.double )
            endPointsDistances_p = endPointsDistances.ctypes.data_as(self.c_double_p)
            endPointsPriorNodes = self._mesh.N * np.ones( (endPoints.shape[0]), dtype = np.uintc )
            endPointsPriorNodes_p = endPointsPriorNodes.ctypes.data_as(self.c_uint_p)
            
        
        
        # Call 
        status = self._mesh._libInstance.GraphPath_getShortestPathDijkstra( self._internalID, \
              ctypes.c_uint(startNodeInd), ctypes.c_uint(startNodeSimplex), \
              startPoint_p, \
              endNodeIndices_p, ctypes.c_uint(endNodeInds.size), \
              endPoints_p, ctypes.c_uint(endPoints.shape[0]), \
              endSimplices_p, \
              distances_p, priorNodes_p, \
              endPointsDistances_p, endPointsPriorNodes_p,
              ctypes.c_uint( np.uintc(1e9)  ), ctypes.c_double(np.float64('inf')) )
            
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) ) 
            
            
        output = { "distances":distances, "priorNodes":priorNodes }
        if (endPointsDistances is not None):
            output["endPointsDistances"] = endPointsDistances
            output["endPointsPriorNodes"] = endPointsPriorNodes
            
        return output
    
    
    
    
    
    
    def findShortestGraphPathAStar(self, startNodeInd, startNodeSimplex, \
          endNodeInd, endPointSimplex, startPoint = None, endPoint = None ):
        '''
        ' Find shortest path between a start point and a end point
        '
        '''
        
        self._storeMeshInternally()
        
        
        if startPoint is None:
            # Make sure that node exist
            if ( startNodeInd >= self._mesh.N ):
                raise Exception( "Start node was out of range!" )
        if endPoint is None:
            # Make sure that node exist
            if ( endNodeInd >= self._mesh.N ):
                raise Exception( "End node was out of range!" )
        
        
        if not isinstance(startNodeInd, np.uintc):
            startNodeInd = np.uintc(startNodeInd)
        if not isinstance(startNodeSimplex, np.uintc):
            startNodeSimplex = np.uintc(startNodeSimplex)
        if startPoint is not None:
            if startPoint.dtype is not np.float64:
                startPoint = np.float64(startPoint)
        if not isinstance(endNodeInd, np.uintc):
            endNodeInd = np.uintc(endNodeInd)
        if not isinstance(endPointSimplex, np.uintc):
            endPointSimplex = np.uintc(endPointSimplex)
        if endPoint is not None:
            if endPoint.dtype is not np.float64:
                endPoint = np.float64(endPoint)
                
        smallestMetric = ctypes.c_double(np.float64(1))
        distance = ctypes.c_double( np.float64('inf') )
        numNodes = ctypes.c_uint( 0 )
        retrievalID = ctypes.c_uint( 0 )

        startPoint_p = None
        if (startPoint is not None ):
            startPoint_p = startPoint.ctypes.data_as(self.c_double_p)
        endPoint_p = None
        if (endPoint is not None ):
            endPoint_p = endPoint.ctypes.data_as(self.c_double_p)    

        # Call 
        status = self._mesh._libInstance.GraphPath_getShortestPathAStar( self._internalID, smallestMetric, \
              ctypes.c_uint(startNodeInd), ctypes.c_uint(startNodeSimplex), startPoint_p, \
              ctypes.c_uint(endNodeInd), ctypes.c_uint(endPointSimplex), endPoint_p, \
              ctypes.byref(distance), ctypes.byref(numNodes), ctypes.byref(retrievalID), \
              ctypes.c_uint( np.uintc(1e9)  ) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) ) 
        
        # Preallocate path 
        path = np.zeros( numNodes.value, dtype=np.uintc)
        path_p = path.ctypes.data_as(self.c_uint_p)
        pathSimplices = np.zeros( numNodes.value, dtype=np.uintc)
        pathSimplices_p = pathSimplices.ctypes.data_as(self.c_uint_p)
            
        # Call 
        status = self._mesh._libInstance.GraphPath_retrievePathAStar( retrievalID, numNodes, path_p, pathSimplices_p )    
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) ) 
                    
        output = { "distance":distance.value, "path":path, "simplices":pathSimplices }
            
        return output
    
    

    def tightenGraphPath( self, pathNodes, startPoint = None, endPoint = None ):
        # Function for finding a "true" shortest path given a graph path
        
        self._storeMeshInternally()
        
        pathNodes = np.uintc(pathNodes.copy())
        
        if startPoint is not None:
            startPoint = np.float64(startPoint.copy())
        if endPoint is not None:
            endPoint = np.float64(endPoint.copy())
                

        # Preallocate output
        pathNodes_p = pathNodes.ctypes.data_as(self.c_uint_p)
        startPoint_p = None
        if (startPoint is not None ):
            startPoint_p = startPoint.ctypes.data_as(self.c_double_p)
        endPoint_p = None
        if (endPoint is not None ):
            endPoint_p = endPoint.ctypes.data_as(self.c_double_p)
        
        
        pathTightenerID = ctypes.c_uint( 0 )
        numPathPoints = ctypes.c_uint( 0 )
        distance = ctypes.c_double( 0 )
        
        # Call computation of tightened path
        status = self._mesh._libInstance.FreePath_createTightenedPath( pathNodes_p, ctypes.c_uint(np.uintc(pathNodes.size)), self._internalID, \
               ctypes.byref(pathTightenerID), ctypes.byref(numPathPoints), ctypes.byref(distance), \
               startPoint_p, endPoint_p)            
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) + " in FreePath_createTightenedPath" ) 

        
        # Call tightening of path
        for iter in range(1):
            status = self._mesh._libInstance.FreePath_optimizeTightenedPath( pathTightenerID, ctypes.c_uint(np.uintc(100)), ctypes.byref(numPathPoints), ctypes.byref(distance) )                            
            if status != 0:
                raise Exception( "Uknown error occured! Error code: " + str(status) + " in FreePath_optimizeTightenedPath" ) 
                
        # Get number of nodes in path
        numNodes = ctypes.c_uint(numPathPoints.value)
        if startPoint is not None:
            numNodes.value = numNodes.value - 1
        if endPoint is not None:
            numNodes.value = numNodes.value - 1

#        # Modify
#        numPureNodes = numPathPoints.value
#        if startPoint is not None:
#            numPureNodes = numPureNodes - 1
#        if endPoint is not None:
#            numPureNodes = numPureNodes - 1
#        for iter in range(numPureNodes):
#            coefs = np.array( [0.2], dtype=np.float64 )
#            if (iter == 0 and startPoint is None):
#                coefs = coefs * 0
#            if (iter == numPureNodes-1 and endPoint is None):
#                coefs = coefs * 0
#            coefs_p = coefs.ctypes.data_as(self.c_double_p)
#            # modify tightened path
#            status = self._mesh._libInstance.FreePath_modifyTightenedPath( coefs_p, ctypes.c_uint(iter), ctypes.c_uint(2), pathTightenerID )
#            if status != 0:
#                raise Exception( "Uknown error occured! Error code: " + str(status) + " in FreePath_modifyTightenedPath" ) 
            
        # Preallocate output
        pathPoints = np.zeros( (numPathPoints.value, self._mesh.embD), dtype = np.float64 )                                  
        pathPoints_p = pathPoints.ctypes.data_as(self.c_double_p)
        pathSimplices = np.zeros( (numNodes.value), dtype = np.uintc )                                  
        pathSimplices_p = pathSimplices.ctypes.data_as(self.c_uint_p)
                                          
        # retrieve tightened path
        status = self._mesh._libInstance.FreePath_retrieveTightenedPath( pathPoints_p, numPathPoints, ctypes.byref(distance), \
                    ctypes.c_uint(np.uintc(self._mesh.embD)), pathTightenerID, \
                    pathSimplices_p, ctypes.byref(numNodes) )
        if status != 0:
            raise Exception( "Uknown error occured! Error code: " + str(status) + " in FreePath_retrieveTightenedPath" ) 
            
        # Get internal tightened path number
        self._internalTightenedPathID = pathTightenerID
        
        
        return {"path":pathPoints, "distance":distance.value, "simplices":pathSimplices}            
        
        
        


    def getNodesFromGraphPath( destinations, origin, priorNodes ):
        # Get lists of nodes in graph path between origin and destination
        
        output = [None] * destinations.size
        
        
        for iter in range(destinations.size):
            numIters = 0
            # Insert destination first
            index = destinations[iter]
            tempList = [index]
            while (index != origin and numIters <= priorNodes.size):
                index = priorNodes[index]
                tempList.append( index )
                numIters = numIters + 1
            output[iter] = np.array(tempList)
    
        return output
    
    
    
    
