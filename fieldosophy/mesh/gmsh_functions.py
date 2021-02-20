#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for the GMSH mesher. 
Should only be used if GMSH is installed on your computer.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


import meshio
import subprocess
import numpy as np
from scipy import spatial as scispat

from . import geometrical_functions as geom
from .Mesh import Mesh







    
    


    
class MeshError(Exception):
    ''' Class for errors when using GMSH '''

    class ErrorEnum:
        general = 0
        gmshLost = 1        

    mErrorCodes = ( "General Error!", "Could not allocate gmsh! Is GMSH installed?" )
    
    def __init__(self, value):
        self.value = value
        self.msg = self.mErrorCodes[ value ]
        
    def __str__(self):
        return repr(self.value)  




def addGMSHPoint( number, point, charLength ):
    """ Add a point to GMSH """    
    
    if isinstance( point, list ):
        point = np.array(point)    
    point = point.flatten()
    
    string = "Point(" + str(number) + ") = {"
    string = string + str(point[0]) + ", "
    string = string + str(point[1]) + ", "
    if point.size > 2:
        string = string + str(point[2]) + ", "
    else:
        string = string + "0.0, "
        
    string = string + str(charLength) + "};"
    # Return string
    return string

def addGMSHCircle( number, startPoint, centerPoint, endPoint ):
    """ Add a circle to GMSH """            
    string = "Circle(" + str(number) + ") = {" + str(startPoint) + ", " + str(centerPoint) + ", " + str(endPoint) + "};"
    return string

def addGMSHLine( number, startPoint, endPoint ):
    """ Add a line to GMSH """            
    string = "Line(" + str(number) + ") = {" + str(startPoint) + ", " + str(endPoint) + "};"
    return string

def addGMSHLineLoop( number, points ):
    """ Add a line loop to GMSH """            
    string = "Line Loop(" + str(number) + ") = {" 
    # Go through the points
    for iter in range(np.size(points)):
        string = string + str(points[iter])
        # Evaluate if a ", " should be added afterwards
        if iter < np.size(points) - 1:
            string = string + ", "    
    # Add tail
    string = string +  "};"
    return string    


def meshInPlane( boundaryPolygon, maxDiam, geo_filename, msh_filename = None, vtk_filename = None, gmshExec = "gmsh", activePoints = None):
    """ Creates triangular mesh in plane. """    
    
    # Define smallest mesh size
    lcar = maxDiam
    
    # Add Header
    gmshString = ["// Meshing for metoc"]        
    gmshString.append( "Mesh.CharacteristicLengthMax = " + str(lcar) + ";" )
    gmshString.append( "Mesh.CharacteristicLengthMin = " + str(lcar/5.0) + ";" )    
    gmshString.append( "Mesh.CharacteristicLengthFromPoints = 1;" )
    gmshString.append( "Mesh.CharacteristicLengthExtendFromBoundary = 1;" )
    gmshString.append( "Mesh.CharacteristicLengthFromCurvature = 0;" )   
    
    
    # Add boundary points
    gmshPointList = [None] * boundaryPolygon.shape[1]        
    for iter in range(np.size(gmshPointList)):
        gmshPointList[iter] = addGMSHPoint( iter+1, boundaryPolygon[:, iter], lcar )                
    gmshString.extend( gmshPointList ) 
    # Keep track of number of points
    numPoints = np.size(gmshPointList)
    
    # Make lines from points
    gmshLineList = [None] * boundaryPolygon.shape[1]
    for iter in range(boundaryPolygon.shape[1]):
        gmshLineList[iter] = addGMSHLine( iter+1, (iter) % 4 + 1, (iter+1) % 4 + 1 )           
    gmshString.extend( gmshLineList )     
    # Line loop
    gmshString.append( addGMSHLineLoop(1, [1, 2, 3, 4]) )    
            
    # If active points are defined     
    if activePoints is not None:        
        # Compute convex hull        
        convexHullVerts = scispat.ConvexHull( activePoints.transpose() )
        convexHull = activePoints[:, convexHullVerts.vertices]
        activePoints = activePoints[:, ~np.isin( np.arange(activePoints.shape[1]), convexHullVerts.vertices )  ]
        # Insert points on convex hull to force the exterior to have the correct characteristic length
        gmshConvexHullList = [None] * convexHull.shape[1]
        for iter in range(convexHull.shape[1]):
            gmshConvexHullList[iter] = addGMSHPoint( numPoints + iter + 1, convexHull[:, iter], lcar )
        gmshString.extend( gmshConvexHullList )            
        # Add line
        gmshConvexHullLineList = [None] * convexHull.shape[1]
        for iter in range(convexHull.shape[1]):
            gmshConvexHullLineList[iter] = addGMSHLine( boundaryPolygon.shape[1] + iter + 1, (iter) % convexHull.shape[1] + numPoints + 1, (iter+1) % convexHull.shape[1] + numPoints + 1 )
        gmshString.extend( gmshConvexHullLineList )
        # Add number of points
        numPoints = numPoints + convexHull.shape[1]        
        
        # Get minimum distance between a point and its closest neighbor
        minDist = np.array( [None] * activePoints.shape[1] )
        for iter in range(np.size(minDist)):
            # Find distance between points
            dist = geom.distanceBetweenPoints( activePoints[:, iter].reshape([2,1]), activePoints )            
            minDist[iter] = np.amin( dist[dist != 0.0] )
        # Set the value to the smallest of distance and characteristic length
        minDist = np.minimum( minDist, lcar )
        
        # Insert points
        gmshActivePointsList = [None] * activePoints.shape[1]
        for iter in range(activePoints.shape[1]):            
            gmshActivePointsList[iter] = addGMSHPoint( numPoints + iter+1, activePoints[:, iter], minDist[iter] )
        gmshString.extend( gmshActivePointsList )
        numPoints = numPoints + activePoints.shape[1]        

    gmshString.append( "Surface(1) = {1};" )
    gmshString.append( "Physical Surface(\"meshSurface\") = {1};" )
    # print("\n".join(gmshString)) 
    
    # Mesh
    mesh = None
    try:   
        with open(geo_filename, "w") as f:
            f.write( "\n".join(gmshString) )   
        if msh_filename is not None: 
            command = [ gmshExec, geo_filename, '-2', '-format', 'msh', '-bin', '-o', msh_filename]    
            p = subprocess.Popen( (command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT )  
            sdtout, stderr = p.communicate()                
            # Read msh file
            mesh = meshio.read(msh_filename)
        
    except FileNotFoundError:        
        # Raise exception
        raise MeshError( MeshError.ErrorEnum.gmshLost )
        
    except:
        raise MeshError( MeshError.ErrorEnum.general )

        
    if (mesh is not None) and (vtk_filename is not None):
        # Write mesh file to vtk
        meshio.write( vtk_filename, mesh) 
    
    # Acquire mesh object
    try:
        mesh = Mesh.loadMeshFromFile( msh_filename )         
    except FileNotFoundError:                
        raise MeshError( MeshError.ErrorEnum.general )
        
        
    if boundaryPolygon.shape[0] < mesh.nodes.shape[1]:
        mesh.nodes = mesh.nodes[:, 0:boundaryPolygon.shape[0]]
        mesh.embD = mesh.topD
        
    return mesh




def meshOnSphere( boundaryRect, corrMin, geo_filename, msh_filename = None, vtk_filename = None, gmshExec = "gmsh", activePoints = None):
    """ Creates triangular mesh on the sphere of the rectangular region specified by lon and lat. """    
    
    # Define smallest mesh size
    lcar = corrMin / 5.0
    
    # Add Header
    gmshString = ["// Meshing for metoc"]
    gmshString.append( "Mesh.CharacteristicLengthMax = " + str(lcar) + ";" )
    gmshString.append( "Mesh.CharacteristicLengthMin = " + str(lcar/5.0) + ";" )          
    gmshString.append( "Mesh.CharacteristicLengthFromPoints = 0;" )
    gmshString.append( "Mesh.CharacteristicLengthExtendFromBoundary = 0;" )
    gmshString.append( "Mesh.CharacteristicLengthFromCurvature = 1;" )
    gmshString.append( "Mesh.MinimumCirclePoints = 100;" )
    
    # Add boundary points
    gmshPointList = [None] * (4+3)
    gmshPointList[0] = addGMSHPoint( 1, [0.0,0.0,0.0], lcar )    
    gmshPointList[1] = addGMSHPoint( 2, [0.0,0.0, np.amax(boundaryRect[2, :])], lcar )     
    gmshPointList[2] = addGMSHPoint( 3, [0.0,0.0, np.amin(boundaryRect[2, :])], lcar )                
    for iter in range(4):        
        gmshPointList[iter+3] = addGMSHPoint( iter+4, boundaryRect[:, iter], lcar )        
    gmshString.extend( gmshPointList )     
    # Keep track of number of points
    numPoints = np.size(gmshPointList)
    
    # Make Circles from points
    gmshCircleList = [None] * 4
    gmshCircleList[0] = addGMSHCircle( 1, 4, 2, 5 )
    gmshCircleList[1] = addGMSHCircle( 2, 5, 1, 6 )
    gmshCircleList[2] = addGMSHCircle( 3, 6, 3, 7 )
    gmshCircleList[3] = addGMSHCircle( 4, 7, 1, 4 )    
    gmshString.extend( gmshCircleList ) 
    # Line loop
    gmshString.append( addGMSHLineLoop(1, [1, 2, 3, 4]) )    
                
    # If active points are defined     
    if activePoints is not None:                
        # Get minimum distance between a point and its closest neighbor
        minDist = np.array( [None] * activePoints.shape[1] )
        for iter in range(np.size(minDist)):
            # Find distance between points
            dist = geom.distanceBetweenPointsOnSphere( activePoints[:, iter].reshape([3,1]), activePoints )            
            minDist[iter] = np.amin( dist[dist != 0.0] )
        # Set the value to the smallest of distance and characteristic length
        minDist = np.minimum( minDist, lcar )
        
        # Insert points
        gmshActivePointsList = [None] * activePoints.shape[1]
        for iter in range(activePoints.shape[1]):
            gmshActivePointsList[iter] = addGMSHPoint( numPoints + iter+1, activePoints[:, iter], minDist[iter] )
        gmshString.extend( gmshActivePointsList )
        numPoints = numPoints + activePoints.shape[1]        
            
    # Create the surfaces            
    gmshString.append( "Surface(1) = {1};" )
    gmshString.append( "Physical Surface(\"meshSurface\") = {1};" )
    print("\n".join(gmshString))
    # Mesh
    mesh = None
    try:   
        with open(geo_filename, "w") as f:
            f.write( "\n".join(gmshString) )    
            
        if msh_filename is not None:            
            command = [ gmshExec, geo_filename, '-2', '-format', 'msh', '-bin', '-o', msh_filename]    
            p = subprocess.Popen( (command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT )  
            sdtout, stderr = p.communicate()                
            print("Mesh file created!\n")
            # Read msh file
            mesh = meshio.read(msh_filename)
        
    except FileNotFoundError:        
        # Raise exception
        raise MeshError( MeshError.ErrorEnum.gmshLost )

        
    if (mesh is not None) and (vtk_filename is not None):
        # Write mesh file to vtk
        meshio.write( vtk_filename, mesh) 

    # Acquire mesh object
    try:
        mesh = Mesh.loadMeshFromFile( msh_filename )         
    except FileNotFoundError:                
        raise MeshError( MeshError.ErrorEnum.general )
                
    return mesh




        
        