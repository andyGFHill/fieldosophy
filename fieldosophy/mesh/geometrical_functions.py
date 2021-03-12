#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometrical functionality.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import ctypes



# Function for mapping from long-lat to 3d coordinates on a unit sphere
def lonlat2Sphere( lonlat ):  
    # If more than one point was given    
    if np.ndim(lonlat) > 1:
        coord = np.empty( [3, lonlat.shape[1]] )
    else:
        coord = np.empty( 3 )
    coord[0] = np.cos( lonlat[0] / 180.0 * np.pi ) * np.cos( lonlat[1] / 180.0 * np.pi )
    coord[1] = np.sin( lonlat[0] / 180.0 * np.pi ) * np.cos( lonlat[1] / 180.0 * np.pi )
    coord[2] = np.sin( lonlat[1] / 180.0 * np.pi )
    return coord

# Function for mapping from 3d coordinates on a unit sphere to lon-lat
def sphere2Lonlat( coord ):
    if coord.ndim == 1:
        coord = coord.reshape((1,-1))
    
    lat = np.arcsin( coord[:, 2] )    
    lon = coord[:, 0] / np.cos(lat)
    lon[ lon < -1] = -1
    lon[ lon > 1] = 1
    lon = np.arccos(lon)
    lon[ coord[:, 1] < 0 ] = - lon[ coord[:, 1] < 0 ]
    
    lon[np.isnan(lon)] = np.sign(lat[np.isnan(lon)]) * np.pi/2
    
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    
    return [lon, lat]

# Distance between points on unit sphere
def distanceOnSphere( point1, point2 ):
    
    if point1.ndim == 1:
        point1 = point1.reshape((-1,1))
    if point2.ndim == 1:
        point2 = point2.reshape((-1,1))
    
    # Get  angular distance
    point1 = point1 / np.linalg.norm( point1, axis = 0 )
    point2 = point2 / np.linalg.norm( point2, axis = 0 )    
    
    angle = np.sum( point1 * point2, axis = 0 )
    angle = np.minimum( angle, 1 )
    angle = np.maximum( angle, -1 )
    angle = np.arccos( angle )
    
    return angle

# Acquire 2-dimensional rotation matrix
def getRotationMatrix( angle ):
    return np.matrix( [ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ] )

# Get local planar coordinate system of current point on sphere
def getLocalCoordOnSphere( point, north = [0,0,1] ):
    xbar = -np.cross( point, north )           
    xbar = xbar / np.linalg.norm(xbar)
    ybar = -np.cross( xbar, point )
    ybar = ybar / np.linalg.norm(ybar)    
    zbar = point / np.linalg.norm(point)
    return (xbar, ybar, zbar)

# Spherical addition in angle/vector
def addSphereDistance( point, vec, angle, north = [0,0,1] ):
    # If vec is an angle
    if np.ndim(vec) == 0:
        vec = np.transpose(np.matmul( getRotationMatrix( vec / 180.0 * np.pi ), [0,1] ))        
        vec = np.ravel( np.matmul( np.stack(getLocalCoordOnSphere(point, north), axis=-1), vec ) )        
                
    # Return new point    
    return point * np.cos(angle) + vec * np.sin(angle)
        

def extendRectBoundInPlane( lon, lat, corrMin ):
    """ Extend the boundary of (an assumed rectangular) domain in lon-lat coordinates on the sphere """
    
    boundCoord = np.empty([2, 4])
    boundCoord[:, 0] = np.array([lon[0], lat[1]])
    boundCoord[:, 1] = np.array([lon[1], lat[1]])
    boundCoord[:, 2] = np.array([lon[1], lat[0]])
    boundCoord[:, 3] = np.array([lon[0], lat[0]])
    
    # Extend boundary
    boundaryPolygon = np.empty( [2, 4] )
    boundaryPolygon[:, 0] = boundCoord[:, 0] + corrMin* np.array([-1,1])
    boundaryPolygon[:, 1] = boundCoord[:, 1] + corrMin* np.array([1,1])
    boundaryPolygon[:, 2] = boundCoord[:, 2] + corrMin* np.array([1,-1])
    boundaryPolygon[:, 3] = boundCoord[:, 3] + corrMin* np.array([-1,-1])
    
    # Return extended boundary
    return boundaryPolygon     

def extendRectBoundOnSphere( lon, lat, corrMin ):
    """ Extend the boundary of (an assumed rectangular) domain in lon-lat coordinates on the sphere """
    
    boundCoord = np.empty([3, 4])
    boundCoord[:, 0] = lonlat2Sphere( np.array([lon[0], lat[1]]) )
    boundCoord[:, 1] = lonlat2Sphere( np.array([lon[1], lat[1]]) )
    boundCoord[:, 2] = lonlat2Sphere( np.array([lon[1], lat[0]]) )
    boundCoord[:, 3] = lonlat2Sphere( np.array([lon[0], lat[0]]) )
    
    # Extend boundary
    boundaryPolygon = np.empty( [3, 4] )
    boundaryPolygon[:, 0] = addSphereDistance( boundCoord[:, 0], 45.0, corrMin)    
    boundaryPolygon[:, 1] = addSphereDistance( boundCoord[:, 1], -45.0, corrMin)
    boundaryPolygon[:, 2] = addSphereDistance( boundCoord[:, 2], -135.0, corrMin)    
    boundaryPolygon[:, 3] = addSphereDistance( boundCoord[:, 3], 135.0, corrMin)   
    
    # Return extended boundary
    return boundaryPolygon   

def distanceBetweenPoints( points1, points2 = None ):
    """ Find euclidean distance between points. """
    
    if points2 is None:
        points2 = points1
    # compute distance in each dimensions
    x = np.outer( points1[0, :].reshape( [points1.shape[1], 1] ), np.ones( [1, points2.shape[1]] ) ) - \
        np.outer( np.ones( [points1.shape[1], 1] ), points2[0, :].reshape( [1, points2.shape[1]] ) )
    y = np.outer( points1[1, :].reshape( [points1.shape[1], 1] ), np.ones( [1, points2.shape[1]] ) ) - \
        np.outer( np.ones( [points1.shape[1], 1] ), points2[1, :].reshape( [1, points2.shape[1]] ) )
    # Return Euclidean distance
    return np.sqrt(x*x + y*y)


def distanceBetweenPointsOnSphere( points1, points2 = None ):
    """ Find geodesic distance between points on unit sphere. """
    
    if points2 is None:
        points2 = points1
        
    dist = np.zeros( [ points1.shape[1], points2.shape[1] ] )
    
    for iter1 in range( points1.shape[1] ):
        dist[iter1, :] = distanceOnSphere( points1[:, iter1], points2 )
            
            
    return dist


def smallestDistanceBetweenPointsOnSphere( points1, points2, lib ):
    """ Find smallest geodesic distance between points on unit sphere. """
    
    dims = points1.shape[1]
    if points2.shape[1] != dims:
        raise Exception( "No matching dimensions!" )
    # Get number of points
    numPoints1 = points1.shape[0]
    numPoints2 = points2.shape[0]
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    
    indices = np.zeros( numPoints1, dtype=np.uintc )
    dists = np.zeros( numPoints1, dtype=np.float64 )
    
    points1_p = points1.ctypes.data_as(c_double_p)
    points2_p = points2.ctypes.data_as(c_double_p)
    dists_p = dists.ctypes.data_as(c_double_p)
    indices_p = indices.ctypes.data_as(c_uint_p)
    
    # Setup function
    lib.misc_computeSmallestDistance.restype = ctypes.c_int
    lib.misc_computeSmallestDistance.argtypes = \
        [ ctypes.c_uint, \
         c_double_p, ctypes.c_uint, \
         c_double_p, ctypes.c_uint, \
         c_uint_p, c_double_p, \
         ctypes.c_int ]
            
    status = lib.misc_computeSmallestDistance( ctypes.c_uint( dims ), \
        points1_p, ctypes.c_uint( numPoints1 ), \
        points2_p, ctypes.c_uint( numPoints2 ), \
        indices_p, dists_p, ctypes.c_int(1) )
    if status != 0:
        raise Exception( "Uknown error occured! Error code " + str(status) + " from misc_computeSmallestDistance()" ) 

    return indices, dists

def smallestDistanceBetweenPoints( points1, points2, lib ):
    """ Find smallest Euclidean distance between points in hyperplane. """
    
    dims = points1.shape[1]
    if points2.shape[1] != dims:
        raise Exception( "No matching dimensions!" )
    # Get number of points
    numPoints1 = points1.shape[0]
    numPoints2 = points2.shape[0]
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    
    indices = np.zeros( numPoints1, dtype=np.uintc )
    dists = np.zeros( numPoints1, dtype=np.float64 )
    
    points1_p = points1.ctypes.data_as(c_double_p)
    points2_p = points2.ctypes.data_as(c_double_p)
    dists_p = dists.ctypes.data_as(c_double_p)
    indices_p = indices.ctypes.data_as(c_uint_p)
    
    # Setup function
    lib.misc_computeSmallestDistance.restype = ctypes.c_int
    lib.misc_computeSmallestDistance.argtypes = \
        [ ctypes.c_uint, \
         c_double_p, ctypes.c_uint, \
         c_double_p, ctypes.c_uint, \
         c_uint_p, c_double_p, \
         ctypes.c_int ]
            
    status = lib.misc_computeSmallestDistance( ctypes.c_uint( dims ), \
        points1_p, ctypes.c_uint( numPoints1 ), \
        points2_p, ctypes.c_uint( numPoints2 ), \
        indices_p, dists_p, ctypes.c_int(0) )
    if status != 0:
        raise Exception( "Uknown error occured! Error code " + str(status) + " from misc_computeSmallestDistance()" ) 

    return indices, dists



def mapToHypersphere(x):
    """
    Function that maps points in R^d to the unit hypersphere
    
    :param x: Points to map in n x d format, where 'n' is the number of points and 'd' the dimensionality.
    """
    
    y = x
    if x.ndim == 1:
        y = x.reshape((1, -1))
        
    return y / np.linalg.norm(y, axis=1).reshape((-1,1))