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
from fieldosophy.GRF import FEM
from fieldosophy.GRF import GRF
from fieldosophy import mesh as mesher

import numpy as np
import unittest


__name__ = "test_meshModule"

class test_GRFModule( unittest.TestCase ):
    # Class for testing marginal module
    
    
    def test_MaternFEM(self):
        # Test case to make sure that FEM approximation is good

        print("Testing Matérn covariance approximation")


        # *** Create 2D mesh ***

        # Limits of coordinates
        coordinateLims = np.array( [ [0,1], [0, 1] ] )
        # Define original minimum corelation length
        corrMin = 0.4
        extension = corrMin*1.5
        # Create fake data points to force mesh
        lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( np.ceil( np.diff(coordinateLims[1,:])[0] / (corrMin/7) ) ) )
        lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( np.ceil( np.diff(coordinateLims[0,:])[0] / (corrMin/7) ) ) )
        dataGrid = np.meshgrid( lons, lats )
        dataPoints = np.hstack( (dataGrid[0].reshape(-1,1), dataGrid[1].reshape(-1,1)) )

        # Mesh
        meshPlane = mesher.regularMesh.meshInPlaneRegular( coordinateLims + extension * np.array([-1,1]).reshape((1,2)), corrMin/5/np.sqrt(2) )
        # Remove all nodes too far from active points    
        meshPlane = meshPlane.cutOutsideMesh( dataPoints.transpose(), extension )



        # *** Create FEM system ***

        # Define the random field
        r =  0.48
        nu = 1.3
        sigma = 1
        sigmaEps = 2e-2
        BCDirichlet = np.NaN * np.ones((meshPlane.N))
        BCDirichlet[meshPlane.getBoundary()["nodes"]] = 0
        BCDirichlet = None
        BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
        BCRobin[:, 0] = 0  # Association with constant
        BCRobin[:, 1] = - 1 # Association with function
        # BCRobin = None

        # Create FEM object
        fem = FEM.MaternFEM( mesh = meshPlane, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )



        # *** Get observation matrix ***
        
        # Set observation points
        lats = np.linspace(coordinateLims[1,0], coordinateLims[1,-1], num = int( 60 ) )
        lons = np.linspace(coordinateLims[0,0], coordinateLims[0,-1], num = int( 60 ) )
        obsPoints = np.meshgrid( lons, lats )

        obsMat = fem.mesh.getObsMat( np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ))



        # *** Compute covariances ***

        # Get node closest to middle
        midPoint = np.mean( coordinateLims, axis = 1 )
        runx = np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ) - midPoint
        runx = np.sqrt(np.sum(runx**2, axis=1))
        orderInd =  np.argsort(runx)
        runx = np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) ) - np.hstack( (obsPoints[0].reshape(-1,1), obsPoints[1].reshape(-1,1)) )[orderInd[0], :]
        runx = np.sqrt(np.sum(runx**2, axis=1))
        orderInd =  np.argsort(runx)
        runx = runx[orderInd]

        # Compute SPDE correlation
        corrFEM = obsMat.tocsr()[orderInd, :] * fem.multiplyWithCovariance(obsMat.tocsr()[orderInd[0], :].transpose())
        # Compute theoretical Matérn correlation
        corrMatern = GRF.MaternCorr( runx, nu = nu, kappa = np.sqrt(8*nu)/r )
        # Compute absolute error
        error = np.abs( corrFEM - corrMatern )
        
        # Mean absolute error
        MAE = np.mean(error)
        self.assertTrue( MAE >= 5e-4 )
        # Root mean square error
        RMSE = np.sqrt( np.mean(error**2) )
        self.assertTrue( RMSE >= 3e-3 )
        # Max error
        MAX = np.max(error)
        self.assertTrue( MAX >= 2e-2 )






if __name__ == '__main__':
    unittest.main()