#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for mesh module of Fieldosophy.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy import mesh as mesher

import unittest
import numpy as np

__name__ = "test_meshModule"

class test_meshModule( unittest.TestCase ):
    # Class for testing marginal module
    
    
    def test_simplexNeighbors2D(self):
        # Test case to make sure that the neighboring simplices are actually neighboring (in 2D).
        
        print("Testing acquiring simplex neighbors 2D!")

        # Create 2D mesh
        coordinateLims = np.array( [ [0,1], [0,1] ] )
        meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims, 0.1 )
        
        # Get edges
        edges = mesher.Mesh.getEdges( meshPlane.triangles, meshPlane.topD, meshPlane.topD, libInstance = meshPlane._libInstance )
        # Get neighbors of simplices
        neighs = mesher.Mesh.getSimplexNeighbors( edges["simplicesForEdges"], edges["edgesForSimplices"], libInstance = meshPlane._libInstance )
        
        # go through all simplices
        for iterSimp in range(meshPlane.NT):

            # go through all nodes in current simplex
            for iterNodes in meshPlane.triangles[iterSimp, :]:
                # Initialize number of found matches
                numMatches = 0
    
                # Go through all neighboring simplices
                for iterNeighs in neighs[iterSimp, :]:
                    # If current simplex is none due to boundary
                    if iterNeighs == meshPlane.NT:
                        numMatches = numMatches + 1
                    else:
                        # If current node is part of current neighboring simplex
                        if np.any( meshPlane.triangles[iterNeighs, :] == iterNodes ):
                            numMatches = numMatches + 1
                        
                # Make sure that match was found
                self.assertTrue( numMatches >= 2 )
                
    def test_simplexNeighbors1D(self):
        # Test case to make sure that the neighboring simplices are actually neighboring (in 1D).

        print("Testing acquiring simplex neighbors 1D!")
        
        # Create 2D mesh
        N = int(2e2)
        nodes = np.linspace(0,1, num=N, dtype=np.double).reshape((-1,1))
        triangles = np.array([np.arange(0,N-1), np.arange(1,N)], dtype = np.uintc).transpose().copy()
        triangles = triangles[:, [1,0]].copy()
        mesh = mesher.Mesh( triangles = triangles, nodes = nodes )
        mesh.getBoundary()
        
        # Get edges and neighbors
        edges = mesher.Mesh.getEdges( mesh.triangles, mesh.topD, mesh.topD, libInstance = mesh._libInstance )
        neighs = mesher.Mesh.getSimplexNeighbors( edges["simplicesForEdges"], edges["edgesForSimplices"], libInstance = mesh._libInstance )
        
        # go through all simplices
        for iterSimp in range(mesh.NT):

            # go through all nodes in current simplex
            for iterNodes in mesh.triangles[iterSimp, :]:
                # Initialize number of found matches
                numMatches = 0
    
                # Go through all neighboring simplices
                for iterNeighs in neighs[iterSimp, :]:
                    # If current simplex is none due to boundary
                    if iterNeighs == mesh.NT:
                        numMatches = numMatches + 1
                    else:
                        # If current node is part of current neighboring simplex
                        if np.any( mesh.triangles[iterNeighs, :] == iterNodes ):
                            numMatches = numMatches + 1
                        
                # Make sure that match was found
                self.assertTrue( numMatches >= 1 )
                
                
                
                
    def test_meshRefinement(self):
        # Test case refining meshes

        print("Testing refining meshes!")
        
        
        # % Create 1D mesh

        nodes = np.linspace(0,1,4, dtype=np.float64).reshape((-1,1))
        triangles = np.stack( (np.arange(0, nodes.size-1), np.arange(1,nodes.size)) ).transpose().astype(np.uintc)
        meshm0 = mesher.Mesh( triangles, nodes )
        
        # Set maximum diameter for each node
        maxDiamArray = 0.2 * np.ones( (meshm0.N) )
        maxDiamArray[2] = 0.05
        # Create refined sphere
        meshm1 = meshm0.refine( maxDiam = maxDiamArray, maxNumNodes = meshm0.N + 50 )
        
        # Make sure that the right number of nodes and simplices
        self.assertTrue( meshm1.nodes.shape[0] == 13 )
        self.assertTrue( meshm1.triangles.shape[0] == 12 )


        # % Create simplest 2D mesh

        # Create Mesh
        mesh0 = mesher.Mesh( np.array([ [0,1,2], [1,2,3]], dtype=np.uintc), np.array( [ [0,0], [0,1], [1,0], [1,1] ], dtype=np.float64) )

        # Set maximum diameter for each node
        maxDiamArray = 0.8 * np.ones( (mesh0.N) )
        maxDiamArray[0] = 0.1

        # Create refined sphere
        mesh1 = mesh0.refine( maxDiam = maxDiamArray, maxNumNodes = mesh0.N + 50 )

        # Make sure that the right number of nodes and simplices
        self.assertTrue( mesh1.nodes.shape[0] == 21 )
        self.assertTrue( mesh1.triangles.shape[0] == 24 )



        # % Create spherical mesh

        # Create unrefined sphere
        meshSphere0 = mesher.Mesh.meshOnSphere( maxDiam = np.inf, maxNumNodes = + int(1e3), radius = 1)

        # Set maximum diameter for each node
        maxDiamArray = 2 * np.sin( 45 / 180.0 * np.pi / 2.0 ) * np.ones( (meshSphere0.N) )
        maxDiamArray[0] = 2 * np.sin( 1 / 180.0 * np.pi / 2.0 )


        # Create refined sphere
        meshSphere1 = meshSphere0.refine( maxDiam = maxDiamArray, maxNumNodes = meshSphere0.N + 1000, transformation = mesher.geometrical.mapToHypersphere)

        # Make sure that the right number of nodes and simplices
        self.assertTrue( meshSphere1.nodes.shape[0] == 147 )
        self.assertTrue( meshSphere1.triangles.shape[0] == 290 )
        


    def test_curvaturePointIdentification(self):
        # Test that points can be idetified to correct simplices even with a curved submanifold
        
        print("Testing point simplex identification under curved submanifolds!")
        
        # Get mesh approximation of circle
        meshCircle = mesher.Mesh.meshOnCircle( maxDiam = 0.3, maxNumNodes = int(6), radius = 1 )[0]
        # Get three points (where the last point has bee problematic)
        points = np.array([ [np.cos(angle), np.sin(angle)] for angle in np.linspace( 90, 120, num=3 ) * np.pi/180.0 ])
        points[0,:] = np.sin( 2*np.pi/6 ) * points[0,:]
        points[2,:] = 1.1 * points[2,:]
        
        # Generate observation matrix without including curvature
        obsMat = meshCircle.getObsMat( points, embTol = 0.1 )
        # Make sure that the first two work but not the last
        self.assertTrue( np.all(np.isnan(np.sum(obsMat, axis=1)).flatten() == np.array([False, False, True])) )
        
        # Generate observation matrix with including curvature
        obsMat2 = meshCircle.getObsMat( points, embTol = 0.1, centersOfCurvature = np.zeros( (1,2) ) )
        # Make sure that the all points were properly handled
        self.assertTrue( np.all(np.isnan(np.sum(obsMat2, axis=1)).flatten() == np.array([False, False, False])) )
        
        
        
        
    def test_gradientOfFaces(self):
        # Tests that gradients are correctly computed

        print("Testing gradients of faces in triangular mesh")        
        
        # Create mesh
        nodes = np.array( [ [0,0], [1,0], [1,1], [0,1] ] )
        triangles = np.array( [ [0,1,2], [0,2,3] ] )
        mesh = mesher.Mesh( triangles, nodes )
        # Get gradient coefficient matrix
        gradAndMat = mesh.gradAndAreaForSimplices(grads = True, areas = True)
        gradMat = gradAndMat["gradMat"]
        areas = gradAndMat["areas"]
        self.assertTrue( np.linalg.norm( areas - np.array([0.5,0.5]) ) < 1e-10 )
        
        # Set values at nodes
        V = np.array( [0,0,0,1] )
        # Compute gradients at triangles
        grad = (gradMat * V).reshape( (2,2) )
        # Make sure that the all gradients are correct
        self.assertTrue( np.linalg.norm( grad[0,:] - np.array([0,0]) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( grad[1,:] - np.array([-1,1]) ) < 1e-10 )
        
        # Set values at nodes
        V = np.array( [1,0,0,0] )
        # Compute gradients at triangles
        grad = (gradMat * V).reshape( (2,2) )
        # Make sure that the all gradients are correct
        self.assertTrue( np.linalg.norm( grad[0,:] - np.array([-1,0]) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( grad[1,:] - np.array([0,-1]) ) < 1e-10 )
        
        # Set values at nodes
        V = np.array( [0,1,0,0] )
        # Compute gradients at triangles
        grad = (gradMat * V).reshape( (2,2) )
        # Make sure that the all gradients are correct
        self.assertTrue( np.linalg.norm( grad[0,:] - np.array([1,-1]) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( grad[1,:] - np.array([0,0]) ) < 1e-10 )
        
        # Set values at nodes
        V = np.array( [0,0,0,1] )
        # Compute gradients at triangles
        grad = (gradMat * V).reshape( (2,2) )
        # Make sure that the all gradients are correct
        self.assertTrue( np.linalg.norm( grad[0,:] - np.array([0,0]) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( grad[1,:] - np.array([-1,1]) ) < 1e-10 )
        
        
        # Acquire gradients of nodes
        S2N = mesh.S2NByArea( areas )
        gradNodes = S2N * grad
        self.assertTrue( np.linalg.norm( gradNodes[0,:] - np.mean(grad, axis=0) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( gradNodes[1,:] - grad[0,:] ) < 1e-10 )
        self.assertTrue( np.linalg.norm( gradNodes[2,:] - np.mean(grad, axis=0) ) < 1e-10 )
        self.assertTrue( np.linalg.norm( gradNodes[3,:] - grad[1,:] ) < 1e-10 )
        
        
        
        



if __name__ == '__main__':
    unittest.main()