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



if __name__ == '__main__':
    unittest.main()