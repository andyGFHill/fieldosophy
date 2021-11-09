#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for finite element approximations.

This file is part of Fieldosophy, a toolkit for random fields.
Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.special as special
import ctypes
import os
import abc

from ..misc import Cheb
from . import Cholesky
    
    
class FEM(abc.ABC):
    """
    Class representing a FEM approximation of a stochastic differential equation.
    
    :param mesh: A mesh object to base the FEM approximation on. 
    :param matMapsCalculate: Defines which mappings from inner products on simplices to system matrices that should be computed.
    :param matMapsCalculateEdges: Defines which mappings from inner products on edge simplices to edge system matrices that should be computed.
    :param libPath: A path to the dynamically linked library "libSPDEC.so" to access the low-level API of fieldosophy.
    """
        
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  

    mesh = None
    
    QChol = None
    
    Pr = None
    
    matMaps = ['M']    
    matMapsEdges = None    
    
    BCDirichlet = None
    BCDirichletIndex = None
    
    mu = None
    tau = None
    nu = None
    sigma = None
    kappaMin = None


    
    def __init__(self, mesh = None, matMapsCalculate = ['M'], matMapsCalculateEdges = None, libPath = None):
        
        if mesh is None:
            return
        
        # Create a mesh object
        self.mesh = mesh        
        
        # Acquire maps from triangle function values to FEM matrices
        self.matMaps = MatMaps( mesh.triangles, mesh.nodes, calculate=matMapsCalculate, libPath = libPath )
        if matMapsCalculateEdges is not None:
            boundary = self.mesh.getBoundary()
            self.matMapsEdges = MatMaps( boundary["edges"], mesh.nodes, calculate=matMapsCalculateEdges, libPath = libPath )       
        
        return
    

    
    
    @abc.abstractmethod
    def copy(self):
        """
        :return A deep copy of self.
        """
        
        # Deep copies object
        return
    
    def copyParent(self, out):
        """
        :param out: out becomes a deep copy of this object.
        """
        
        out.mesh = self.mesh.copy()
        out.matMaps = self.matMaps.copy()
        if self.matMapsEdges is not None:
            out.matMapsEdges = self.matMapsEdges.copy()
        out.BCDirichlet = self.BCDirichlet.copy()
        out.BCDirichletIndex = self.BCDirichletIndex.copy()
        if self.QChol is not None:
            out.QChol = self.QChol.copy()
        if self.Pr is not None:
            out.Pr = self.Pr.copy()
        if self.mu is not None:
            if np.isscalar(self.mu):
                out.mu = self.mu
            else:
                out.mu = self.mu.copy()
        if self.tau is not None:
            if np.isscalar(self.tau):
                out.tau = self.tau
            else:
                out.tau = self.tau.copy()  
        if self.sigma is not None:        
            if np.isscalar(self.sigma):
                out.sigma = self.sigma
            else:
                out.sigma = self.sigma.copy()             
        out.nu = self.nu
        out.kappaMin = self.kappaMin
        
        return out
        
        
    
    
    
    def updateSystem(self, MCoeff, tau, nu, mu = 0, BCoeff = None, GCoeff = None, sourceCoeff = None, BCRobin = None, BCDirichlet = None, factorize = True):
        """
        Update system with new parameters.
        
        :param MCoeff: Coefficients of constant term (simplex wise).
        :param tau: Value of the tau-parameter. Tau can be either a scalar or a vector the same size as the number of nodes in the mesh.
        :param nu: The smoothness parameter. Can be any positive real value.
        :param mu: The mean value. Either a constant or a vector the same size as the number of nodes of the mesh. If set to None, the implicit mean is not computed (can save computations when applicable)
        :param BCoeff: Coefficients of the first derivative term (simplex wise and with the number of elements as the dimensionality of the hyperplane that the manifold is embedded in).
        :param GCoeff: Coefficients of the second derivative term (simplex wise and with the number of elements as the squared dimensionality of the hyperplane that the manifold is embedded in).
        :param sourceCoeff: An optional deterministic term to the source of the differential equation (same dimensionality as the number of simplices). 
        :param BCRobin: Sets the Robin boundary conditions. Set to None if no Robin boundary conditions should be enforced.
            Otherwise, an array where the number of rows equal either one or the number of simplices on the edge and with two columns. 
            The first column corresponds to the coefficient for the constant term and thesecond column corresponding to the coefficient for the first-derivative term.
        :param BCDirichlet: Sets the Dirichlet boundary conditions. If None, no Dirichlet boundary conditions are enforced. 
            Otherwise, BCDirichlet is a vector the same size as the number of nodes in the mesh. 
            The vale corresponds to the enforced constant value of that specific node unless the value is nan, which means that the given node is not enforced to a Dircihlet condition.
        :param factorize: If the Choleksy factorization should be performed now.
        """
        
        # Set variance coefficient
        self.tau = tau
        # Set smoothness coefficient
        self.nu = nu     
        # Set kappaMin
        self.kappaMin = np.min(MCoeff)
        
        d = self.mesh.topD

        # Handle Dirichlet indexing
        if self.BCDirichlet is None:            
            if BCDirichlet is None:                        
                BCDirichlet = np.NaN * np.ones( (self.mesh.N) )
        if BCDirichlet is not None:        
            self.BCDirichletIndex = ~np.isnan(BCDirichlet)
            BCDirichlet = BCDirichlet[self.BCDirichletIndex]
            self.BCDirichlet = BCDirichlet
        
        
        if BCRobin is None:
            BCRobin = np.zeros( (1,2) )
        
        # build K
        K = self.assembleK(MCoeff, BCoeff, GCoeff, BCRobin[:,1])            
        
        # Build C
        C = MatMaps.mapTriVals2Mat( self.matMaps.M, 1, self.mesh.N )
        C = np.asarray(C.sum(axis=1)).reshape(-1) # Mass lump
        CInvSqrt = 1/np.sqrt(C) # get squared inverse        
        
        # Handle smoothness        
        beta = (self.nu + d/2)/2
        m = 2
        n = 1
        if beta < 1:            
            m = 1
            n = 2            

        # Loop until succecsfull factorization
        goOn = True
        while goOn: 
            goOn = False
            # Try to acquire a rational approximation
            [Pl, Pr] = FEM.rationalApproximation( K, CInvSqrt, self.kappaMin, self.nu, d, m=m, n=n )   
            # Construct latent precision matrix
            self.QChol = sparse.diags( CInvSqrt[~self.BCDirichletIndex], shape= (self.mesh.N - np.sum(self.BCDirichletIndex)) * np.array([1,1]) )                
            self.QChol = Pl[ np.ix_(~self.BCDirichletIndex, ~self.BCDirichletIndex) ] * self.QChol
            # If try to factorize            
            if factorize:
                
                try:
                    self.QChol = Cholesky.SKSparseCholesky( self.QChol.tocsc(), AAt = True )
                except Cholesky.SparseCholeskyAbstract.PositiveDefiniteException as e:
                    
                    if ( (m == 2) and (n == 1) ):
                        m = 1
                        n = 2
                    elif ( (m == 1) and (n == 2) ):
                        print("Precision matrix is badly conditioned! Applying rational approximation with constant denominator.")
                        m = 1
                        n = 0
                    
                    else:
                        raise e
                    goOn = True
             
        # Store Pr
        self.Pr = Pr[ np.ix_(~self.BCDirichletIndex, ~self.BCDirichletIndex) ]
        
        if self.mu is None:
            if mu is None:
                mu = 0
        
        # If should compute implicit mean
        if mu is not None:
            self.mu = self.computeImplicitMean(mu, sourceCoeff, BCRobin[:,0], Pl, Pr)
        
        
        
            
                
        return            
    
    
    def assembleK(self, MCoeff, BCoeff, GCoeff, BCRobinFunction):
        """
        Assembles the system matrix 'K' of the FEM system.
        
        :param MCoeff: The inner product over each simplex for each basis function sorted in a T x N array (where T are the number of simplices and N the number of nodes, i.e., basis functions). Here, representing the constant term in the differential operator where the inner product is between two basis functions multiplied with some constant that can be specific for the triangle.
        :param BCoeff: Similar to MCoeff but here representing the inner product between a basis function and the gradient of another basis function. BCoeff is a list with D number of elements; one element for each dimension in the hyperplane in which the spatial domain is embedded.
        :param GCoeff: Similar to BCoeff but here representing the inner product between the gradient of one basis function and the gradient of another basis function. Therefore being a list with DxD number of elements.
        :param BCRobinFunction: A Robin boundary condition means that, on some part of the boundary, we know that n * grad X = a X + b. Hence, the boundary condition can be enforced by the inner product of a X with some basis function and b with some basis function. When constructing K, only the first part where X is multiplied with the basis function is used. This is the input of BCRobinFunction. BCRobinFunction has one value for each edge. The Robin boundary condition can be overridden on a part of (or all of) the boundary by setting Dirichlet condition on this part instead; since Dirichlet conditions have precedence over Neumann and Robin in Fieldosophy.
        
        :return The sparse K matrix.
        """
        
        
        
        K = MatMaps.mapTriVals2Mat( self.matMaps.M, MCoeff, self.mesh.N )
        if BCoeff is not None:
            for iter in range(len(self.matMaps.B)):
                if BCoeff[iter] is not None:
                    K = K + MatMaps.mapTriVals2Mat( self.matMaps.B[iter], BCoeff[iter], self.mesh.N )
        if GCoeff is not None:
            for iter in range(len(self.matMaps.G)):
                if GCoeff[iter] is not None:
                    K = K + MatMaps.mapTriVals2Mat( self.matMaps.G[iter], GCoeff[iter], self.mesh.N )
        if self.matMapsEdges is not None:
            if self.matMapsEdges.M is not None:
                K = K - MatMaps.mapTriVals2Mat( self.matMapsEdges.M, BCRobinFunction, self.mesh.N )
        
        return K
    
    
    def computeImplicitMean(self, mu, sourceCoeff, BCRobinConstant, Pl, Pr):
        """
        Assembles the right hand side of the FEM system and solve to get the mean.
        
        The actual mean of the GRF depends not only on the explicitly given mean, but also on the source term and the source part of a Robin boundary condition. 
        This function computes this implicit mean.
        
        :param mu: The explicit mean defined by the user.
        :param sourceCoeff: The source term given by the user.
        :param BCRobinConstant: The source term part of the Robin boundary condition. 
        :param Pl: The Pl matrix acquired from the rational approximation to fractional derivatives of the K matrix. 
        :param Pr: The Pr matrix acquired from the rational approximation to fractional derivatives of the K matrix. 
        
        :return The implicit mean.
        """
        
        # If mean is just a scalar
        if np.isscalar(mu):
            mu = mu * np.ones( (self.mesh.N), dtype=np.float64 )
        
        RHS = None
        # If source is defined
        if sourceCoeff is not None:
            if RHS is None:
                RHS = np.zeros( (self.mesh.N), dtype=np.float64 ) 
            RHS = RHS + MatMaps.mapTriVals2Mat( self.matMaps.U, sourceCoeff, (self.mesh.N, 1) ).toarray().flatten()
        # If Robin constant is defined
        if self.matMapsEdges is not None:
            if self.matMapsEdges.U is not None:        
                if RHS is None:
                    RHS = np.zeros( (self.mesh.N), dtype=np.float64 ) 
                RHS = RHS + MatMaps.mapTriVals2Mat( self.matMapsEdges.U, \
                      BCRobinConstant * np.mean( self.tau[ self.mesh.boundary["edges"] ], axis=1 ), \
                       (self.mesh.N, 1) ).toarray().flatten()
        # If Dirichlet is defined
        if self.BCDirichlet.size > 0:  
            mu[self.BCDirichletIndex] = self.BCDirichlet
            if np.any(self.BCDirichlet != 0):
                # Cholesky factorize Pr
                PrChol = Cholesky.SKSparseCholesky( Pr.tocsc(), AAt = False )
                # Solve for Dirichlet conditions
                sol = np.zeros((self.mesh.N))
                sol[self.BCDirichletIndex] = ( self.BCDirichlet * self.tau[self.BCDirichletIndex] )
                sol = PrChol.solve( sol )
                # Multiply with Pl
                sol = Pl * sol
                if RHS is None:
                    RHS = np.zeros( (self.mesh.N), dtype=np.float64 ) 
                RHS = RHS - sol
        # If Right hand side is not zero
        if RHS is not None:
            # Cholesky factorization of K
            PlChol = Cholesky.SKSparseCholesky( Pl[ np.ix_(~self.BCDirichletIndex, ~self.BCDirichletIndex) ].tocsc(), AAt = False )
            # Get solution
            sol = PlChol.solve( RHS[~self.BCDirichletIndex] )
            # Solve to get implicit mean
            mu[~self.BCDirichletIndex] = mu[~self.BCDirichletIndex] + sol / self.tau[~self.BCDirichletIndex] 
        
        
        return mu
    
    
        
    

    def loglik(self, y, A, sigmaEps, QCond = None):
        """
        Log-likelihood of model given observations
        
        :param y: Observed data, a KxL array where L are the number of replicates and K are the number of joint observations.
        :param A: An observation matrix relating the observations to the mesh.
        :param sigmaEps: The independent centered observation noise given as a standard deviation. Can be different for different observation points.
        :param QCond: It is possible to provide a conditional precision matrix (if already computed) such that the computation does not have to be performed again. If left as None the precision matrix will instead be computed as part of the operation.
        
        :return The log-likelihood.
        """
        
        # Check that system is defined
        if self.QChol is None:
            raise Exception( "System is not created" )
        if not isinstance( self.QChol, Cholesky.SKSparseCholesky):
            raise Exception("Precision matrix was not factorized!")
                        
        # Get number of observations in current realization
        k = A.shape[0]
        # Get number of realizations
        if len(y.shape) == 1:
            y = y.reshape((-1,1))
        M = y.shape[1]
            
        Atemp = sparse.diags( 1/self.tau[~self.BCDirichletIndex] ) * self.Pr
        Atemp = A[:, ~self.BCDirichletIndex] * Atemp
        QEps = sparse.diags( np.ones(k) / sigmaEps**2 ) 
        
        
        # Compute log determinant of Q
        logDetQ = self.QChol.getLogDet()
        # Compute log determinant of error Q
        logDetQEps = np.sum(np.log(QEps.diagonal()))
        
        if QCond is None:
            
            QCond = self.QChol.getMatrix()
            # Build Q of conditional distribution
            QCond = QCond + (Atemp.transpose() * QEps * Atemp) 
            # Cholesky factorize
            QCond = Cholesky.SKSparseCholesky( QCond, AAt = False )

        # Compute log determinant of conditional precision
        logDetQCond = QCond.getLogDet()
        
        # Compute log likelihood
        l = 0.5 * ( logDetQEps + logDetQ - logDetQCond - k * np.log(2*np.pi) )
               
        # Compute quadratic form
        
#         # Update log likelihood
#         l = l - 0.5/M * np.sum(y*y) / sigmaEps**2        
#         tempmu = self.QChol.permute(self.mu, toChol = True )
#         tempmu = self.QChol.getL(upper = True ) * tempmu        
#         # Update log likelihood
#         l = l - 0.5/M * M * np.sum( tempmu * tempmu )        
#         tempy = y - (A * self.mu).reshape((-1,1))
#         tempy = tempy / sigmaEps**2
#         tempy = A[:, ~self.BCDirichletIndex].transpose() * tempy
#         tempy = QCond.solve( tempy )
#         tempy = tempy + self.mu.reshape((-1,1))
#         tempy = QCond.permute( tempy, toChol = True )
#         tempy = QCond.getL(upper = True) * tempy
#         # Update log likelihood
#         l = l + 0.5/M * np.sum(tempy * tempy)
        
        
#        tempy = y - (A * self.mu).reshape((-1,1))
#        tempy2 = tempy / sigmaEps**2
#        tempy2 = A[:, ~self.BCDirichletIndex].transpose() * tempy2
#        tempy2 = QCond.solve( tempy2 )
#        tempy2 = A[:, ~self.BCDirichletIndex] * tempy2 
#        tempy2 = (tempy - tempy2) / sigmaEps**2
#        # Update log likelihood
#        l = l - 0.5/M * np.sum(tempy*tempy2)
        
        
        tempy = y - (A * self.mu ).reshape((-1,1))
        tempy2 = QEps * tempy
        # Update log likelihood
        l = l - 0.5/M * np.sum(tempy * tempy2 )                
        
        tempy2 = Atemp.transpose() * tempy2
        tempy2 = QCond.permute( tempy2, toChol=True )
        tempy2 = QCond.solveL( tempy2, transpose = False )        
        # Update log likelihood
        l = l + 0.5/M * np.sum(tempy2 ** 2)
        
        return l
    
    
    def cond(self, y, A, sigmaEps, QChol = None):
        """
        Acquire conditional distribution given model and observations.
        
        :param y: Observed data, a KxL array where L are the number of replicates and K are the number of joint observations.
        :param A: An observation matrix relating the observations to the mesh.
        :param sigmaEps: The independent centered observation noise given as a standard deviation. Can be different for different observation points.
        :param QCond: It is possible to provide a conditional precision matrix (if already computed) such that the computation does not have to be performed again. If left as None the precision matrix will instead be computed as part of the operation.
        
        :return A new FEM object representing the conditioned random field.
        """
        
        # Get number of observations in current realization
        k = A.shape[0]
        
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                if y.shape[1] == 1:
                    y = y.reshape((-1))
                else:
                    raise Exception("Wrong size on observation vector")            
            
            
        # compensate for marginal variance
        tauInv = sparse.diags( 1/self.tau[~self.BCDirichletIndex] )
        # Multiply with Pr
        Atemp = tauInv * self.Pr
        # Multiply with observation matrix
        Atemp = A[:, ~self.BCDirichletIndex] * Atemp
        QEps = sparse.diags( np.ones((k)) / sigmaEps**2 ) 
        
        # Copy current model
        out = self.copy()
        
        # If QChol should be computed
        if QChol is None:
            if isinstance( self.QChol, Cholesky.SKSparseCholesky):
                out.QChol = self.QChol.getMatrix()
            else:
                out.QChol = self.QChol * self.QChol.transpose()
            # Build Q of conditional distribution
            out.QChol = out.QChol + (Atemp.transpose() * QEps * Atemp) 
            # Cholesky factorize
            out.QChol = Cholesky.SKSparseCholesky( out.QChol, AAt = False )
        else:   
            out.QChol = QChol
             

        # Set sigma to none since new marginal standard deviation is unknown
        self.sigma = None
                    
        # Build conditional mean
        if np.any( y  ) or np.any( self.mu ):
            mu = y - A * self.mu
            mu = QEps * mu
            mu = Atemp.transpose() * mu
            out.mu[~self.BCDirichletIndex] = self.mu[~self.BCDirichletIndex] + \
                tauInv * ( self.Pr * out.QChol.solve( mu ) )
            out.mu[self.BCDirichletIndex] = self.mu[self.BCDirichletIndex]
        
        return out
    
    
    def condMean(self, y, A, sigmaEps, QChol = None):
        """
        If already have conditional distribution, use this function to get a new (or several) conditional means
        
        :param y: Observed data, a KxL array where L are the number of replicates and K are the number of joint observations.
        :param A: An observation matrix relating the observations to the mesh.
        :param sigmaEps: The independent centered observation noise given as a standard deviation. Can be different for different observation points.
        :param QCond: It is possible to provide a conditional precision matrix (if already computed) such that the computation does not have to be performed again. If left as None the precision matrix will instead be computed as part of the operation.
        
        :return A new FEM object representing the conditioned random field.
        """
        
        if QChol is None:
            QChol = self.QChol
            
        # Get number of observations in current realization
        k = A.shape[0]
        
        # compensate for marginal variance
        tauInv = sparse.diags( 1/self.tau[~self.BCDirichletIndex] )
        # Multiply with Pr
        Atemp = tauInv * self.Pr
        # Multiply with observation matrix
        Atemp = A[:, ~self.BCDirichletIndex] * Atemp
        QEps = sparse.diags( np.ones((k)) / sigmaEps**2 ) 
        
        mu = y - A * self.mu.reshape((-1,1))
        mu = QEps * mu
        mu = Atemp.transpose() * mu
            
        out = self.mu.copy().reshape((-1,1))
        out = np.repeat( out, mu.shape[1], axis = 1 )
        out[~self.BCDirichletIndex, :] = out[~self.BCDirichletIndex, :] + \
            tauInv * ( self.Pr * QChol.solve( mu ) )
        
        return out

    
    def condQChol(self, A, sigmaEps):
        """
        Compute conditional Q Cholesky factor.
        
        :param A: An observation matrix relating the observations to the mesh.
        :param sigmaEps: The independent centered observation noise given as a standard deviation. Can be different for different observation points.
        
        :return The Cholesky factorized matrix
        """
        
        # Get number of observations in current realization
        k = A.shape[0]
        
        # compensate for marginal variance
        tauInv = sparse.diags( 1/self.tau[~self.BCDirichletIndex] )
        # Multiply with Pr
        Atemp = tauInv * self.Pr
        # Multiply with observation matrix
        Atemp = A[:, ~self.BCDirichletIndex] * Atemp
        QEps = sparse.diags( np.ones((k)) / sigmaEps**2 )
        
        # If QChol should be computed
        if isinstance( self.QChol, Cholesky.SKSparseCholesky):
            QChol = self.QChol.getMatrix()
        else:
            QChol = self.QChol * self.QChol.transpose()
        # Build Q of conditional distribution
        QChol = QChol + (Atemp.transpose() * QEps * Atemp) 
        # Cholesky factorize
        QChol = Cholesky.SKSparseCholesky( QChol, AAt = False )
            
        return QChol
        


    def generateRandom( self, n ):
        """
        Generate realizations from model.
        
        :param n: Number of realizations.
        
        :return A N x n array, where N are the number of nodes in the mesh and n is the number of realizations.
        """
        
        # Check that system is defined
        if self.QChol is None:
            raise Exception( "System is not created" )
            
        if not isinstance( self.QChol, Cholesky.SKSparseCholesky):
            self.QChol = Cholesky.SKSparseCholesky( self.QChol, AAt = True )
        
        # Generate random
        Z = np.zeros( (self.mesh.N, n) )    
        Z1 = stats.norm.rvs( size = np.sum(~self.BCDirichletIndex) * n ).reshape((-1,n))
        Z1 = self.QChol.solveL( Z1, transpose = True )
        Z1 = self.QChol.permute( Z1, toChol = False)
        # Multiply solution with Pr
        Z1 = self.Pr * Z1
                
        # Assemble
        Z[~self.BCDirichletIndex, :] = Z1        
        Z = Z / self.tau.reshape((self.mesh.N,1))
        Z = Z + self.mu.reshape((self.mesh.N,1))
        
        return Z
        
    def multiplyWithCovariance( self, matrix ):
        """
        Multiply vector or matrix with the covariance function.
        
        :param matrix: An N x n array for which N is the number of nodes in the mesh and n are the number of vectors to multiply.
        
        :return An N x n array.
        """
        
        # Check that system is defined
        if self.QChol is None:
            raise Exception( "System is not created" )
        if not isinstance( self.QChol, Cholesky.SKSparseCholesky):
            self.QChol = Cholesky.SKSparseCholesky( self.QChol, AAt = True )
        # Check size    
        if matrix.shape[0] != self.mesh.N:
            raise Exception( "Wrong size!" )            
        # Acquire tau inverse            
        tauInv = sparse.diags( 1/self.tau[~self.BCDirichletIndex] ).tocsc()        
        # If sparse matrix
        if sparse.issparse(matrix):
            matrix = tauInv * matrix.tocsc()[~self.BCDirichletIndex, :]
        else:
            matrix = tauInv * matrix[~self.BCDirichletIndex, :]            
        # Multiply solution with Pr transpose
        matrix = self.Pr.transpose().tocsc() * matrix
        
        # Preallocate output                
        out = np.zeros( (self.mesh.N, matrix.shape[1]) )
        # Solve for output
        solution = self.QChol.solve( matrix )        
        # If sparse        
        if sparse.issparse(solution):
            # Make into array
            solution = solution.toarray()        
            
        # Multiply with Pr transpose
        solution = self.Pr * solution
        # Multiply with tau inverse
        out[~self.BCDirichletIndex, :] = tauInv * solution
        
        return out
        
    
    
    
    def rationalApproximation( K, CInvSqrt, eigenNorm, nu, d, N = 20, m = 2, n = 1 ):
        """
        Compute rational approximation as suggested in Bolin et al. 2019.
        
        :param K: The system matrix of the FEM system.
        :param CInvSqrt: A diagonal matrix representing the square root of the inverse of C (assuming "mass lumping" to acquire a diagonal matrix). 
        :param eigenNorm: The maximal value of the eigen values of the differential operator.
        :param nu: The smoothness of the field, i.e., beta = nu/2 + d/4, where nu is the Hölder constant of realizations almost everywhere.
        :param d: The dimensionality of the manifold (not generally the dimensionality of the hyperplane in which it is embedded in).
        :param N: The degree of the Chebyshev polynomial used in the rational approximation in the Cleenshaw-Lord approximation.
        :param m: The degree of the polynomial corresponding to Pl.
        :param n: The degree of the polynomial corresponding to Pr.
        
        :return The Pl and Pr matrices for the rational FEM approximation of the solution.
        """
        
        # Check feasible nu
        if nu <= 0:
            raise Exception("Smoothness parameter is too small!")
        # Compute beta
        beta = (nu + d/2)/2        
        # Limit beta values
        beta = np.min((beta, 3.25))
        
        # Compute CInv
        CInv = sparse.diags( CInvSqrt ** 2 )
        # Compute C
        C = sparse.diags( CInvSqrt ** (-2) ).tocsr()
        # Compute identity
        I = sparse.eye(K.shape[0])
        # Get CInv * K and normalize by eigennorm
        CiL = CInv * K  / eigenNorm
        
        # Preallocate
        Pl = I.tocsc() 
        Pr = I.tocsc() 
             
        mBeta = np.floor(beta)
        
        # If beta is an integer
        if beta == mBeta:
            # If beta is smaller or equal to maximum degree of Pl
            if mBeta <= m:
                for iter in range(int(mBeta)):
                    Pl = CiL * Pl
                Pl = C * Pl
                Pl = Pl * (eigenNorm ** mBeta)
                return (Pl, Pr)        
            
        # mBeta = m-n
        # mBeta = 0
        domain = ( 10**( -5/2 ), 1 )        
        
        # Acquire rational polynomial approximation
        b = np.polynomial.polynomial.Polynomial( np.array([1]) )
        c = np.polynomial.polynomial.Polynomial( np.array([1]) )
        if beta != mBeta:                  
            f = lambda x : x ** (beta-mBeta)            
            c, b = Cheb.ClenshawLord( f, N, domain, n, m )
            
            # runx = np.linspace(domain[0], domain[1], num=int(1e3))
            # err  =np.max( np.abs( c(runx)/b(runx) - f(runx) ) )
            
        # Remove too small trailing zeros
        b = b.trim( tol = 1e-6 )
        c = c.trim( tol = 1e-6 )        
        # Acquire the roots of the polynomials
        rc = c.roots().real
        rb = b.roots().real
        # Acquire leftover
        leftover = (mBeta - (rb.size - rc.size) )
        
                
        # Handle Pl
        for iter in np.arange(0, rb.size ):
            Pl = Pl * ( I - CiL * rb[iter] )
        # Add leftover CiL
        for iter in range( int( np.max((0,leftover)) ) ):
                Pl = CiL * Pl
        # Multiply with final C    
        Pl = C * Pl
        # adjust for highest order coefficients
        Pl = ( b.coef[-1] * (b.mapparms()[1] ** rb.size) ) * Pl
            
        # Handle Pr
        for iter in np.arange(0, rc.size ):
            Pr = Pr * ( I - CiL * rc[iter] )
        # Add leftover CiL
        for iter in range(int( np.max((0,-leftover)) )):
            Pr = CiL * Pr
        # adjust for highest order coefficients
        Pr = ( c.coef[-1] * (c.mapparms()[1] ** rc.size) ) * Pr
        
        # Unnormalize Pl by eigenNorm
        Pl = Pl * (eigenNorm ** beta)
        
        return (Pl, Pr)
    
    
    

        
        

class MatMaps:
    """
    Class representing the mapping from values at simplices to values for and between nodes
    """
    
    # Dimensionality of space
    D = None 
    # Dimensionality of simplices
    topD = None
    # Number of simplices
    NT = None
    # Number of nodes
    NN = None
   
    #  mass matrix < phi_i, phi_j >
    M = None
    # flow matrix < nabla phi_i, phi_j >
    B = None
    # stiffness matrix < nabla phi_i, nabla phi_j >
    G = None
    # right hand side < 1, phi_j >
    U = None
    
    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)   
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
    
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" )
    _libInstance = None
    
    def __init__(self, simplices, nodes, calculate = ['M'], libPath = None):
        '''
        Maps point values for each simplex to stiffness, mass, and ... matrix
        
        Computes inner products between two pairs of basis functions for each simplex, then assembles them.
        So far only implemented for piecewise linear basis functions.        
    
        '''     
        
        if libPath is not None:
            self._libPath = libPath
        
        # Instantiate C library
        self._libInstance = ctypes.CDLL(self._libPath)
        
        # Declare mapTrivals2Mat function
        self._libInstance.FEM_mapTrivals2Mat.restype = ctypes.c_int
        self._libInstance.FEM_mapTrivals2Mat.argtypes = [ \
              self.c_double_p, self.c_uint_p, self.c_uint_p, self.c_uint_p, \
              ctypes.c_uint, ctypes.c_uint, \
              self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
              self.c_uint_p, ctypes.c_uint, ctypes.c_uint \
              ]
        
        
        self.D = nodes.shape[1]
        self.NN = nodes.shape[0]
        self.topD = simplices.shape[1]-1
        self.NT = simplices.shape[0]
        
        if "M" in calculate:
            self.computeM( simplices, nodes )
        if "B" in calculate:
            self.computeB( simplices, nodes )        
        if "G" in calculate:
            self.computeG( simplices, nodes )    
        if "U" in calculate:
            self.computeU( simplices, nodes )
        
        
        return


    def copy(self):
        """
        :return Deep copy of self.
        """
        
        out = MatMaps( np.array([[]]), np.array([[]]), calculate = [], libPath = self._libPath )
        out.D = self.D
        out.topD = self.topD
        out.NT = self.NT
        out.NN = self.NN        
        out.M = self.M
        out.B = self.B
        out.G = self.G
        out.U = self.U
        
        return out


    def computeM(self, simplices, nodes):
        """
        Computes inner products between pairs of basis functions for each simplex in mesh.
        
        :param simplices: The simplices of the mesh.
        :param nodes: The nodes of the mesh.
        
        :return The matrix mapping vector of size the number of simplices to each combination of basis function pairs. The matrix will be very sparse and is stored in a special format defined in 'acquireSmallerMatrix'.
        """
        
        # Preallocate
        self.M = { \
              "data" : np.NaN * np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.double), \
              "row" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc), \
              "col" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc) \
              }
        # Compute
        self.M = self.computeGeneric( simplices, nodes, 0, self.M )            
        # Assemble sparse matrix
        self.M = MatMaps.acquireSmallerMatrix( sparse.coo_matrix((self.M["data"], (self.M["row"], self.M["col"])), shape=(self.NN**2, self.NT)) )
        
        return
    
    
    def computeB(self, simplices, nodes):
        """
        Computes inner products between any basis function and the gradient of any basis function for each simplex in mesh.
        
        :param simplices: The simplices of the mesh.
        :param nodes: The nodes of the mesh.
        
        :return The matrix mapping vector of size the number of simplices to each combination of basis function pairs. The matrix will be very sparse and is stored in a special format defined in 'acquireSmallerMatrix'. The output is a list with one element for each dimension in the hyperplane for which the manifold is embedded in.
        """
        
        self.B = [None] * self.D                
        for iter in range(self.D):
            # Preallocate
            self.B[iter] = { \
                  "data" : np.NaN * np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.double), \
                  "row" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc), \
                  "col" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc) \
                  }
            # Compute
            self.B[iter] = self.computeGeneric( simplices, nodes, 1+iter, self.B[iter] )            
            # Assemble sparse matrix
            self.B[iter] = MatMaps.acquireSmallerMatrix( sparse.coo_matrix((self.B[iter]["data"], (self.B[iter]["row"], self.B[iter]["col"])), shape=(self.NN**2, self.NT)) )
        
        return
    
    
    def computeG(self, simplices, nodes):
        """
        Computes inner products between the gradient of any basis function and the gradient of any other basis function for each simplex in mesh.
        
        :param simplices: The simplices of the mesh.
        :param nodes: The nodes of the mesh.
        
        :return The matrix mapping vector of size the number of simplices to each combination of basis function pairs. The matrix will be very sparse and is stored in a special format defined in 'acquireSmallerMatrix'. The output is a list with one element for each combination of dimension pairs in the hyperplane for which the manifold is embedded in.
        """
        
        self.G = [None] * (self.D**2)                
        for iter in range(self.D**2):
            # Preallocate
            self.G[iter] = { \
                  "data" : np.NaN * np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.double), \
                  "row" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc), \
                  "col" : np.zeros(( self.NT * ( (self.topD+1) ** 2 ) ), dtype=np.uintc) \
                  }
            # Compute
            self.G[iter] = self.computeGeneric( simplices, nodes, 1+self.D + iter, self.G[iter] )
            # Assemble sparse matrix
            self.G[iter] = MatMaps.acquireSmallerMatrix( sparse.coo_matrix((self.G[iter]["data"], (self.G[iter]["row"], self.G[iter]["col"])), shape=(self.NN**2, self.NT)) )
                
        return
              
    
    def computeU(self, simplices, nodes):
        """
        Computes inner products between any basis function and 1 for each simplex in mesh.
        
        :param simplices: The simplices of the mesh.
        :param nodes: The nodes of the mesh.
        
        :return The matrix mapping vector of size the number of simplices to each number of nodes in the mesh. The matrix will be very sparse and is stored in a special format defined in 'acquireSmallerMatrix'.
        """
        
        
        # Preallocate
        self.U = { \
              "data" : np.NaN * np.zeros(( self.NT * ( (self.topD+1) ) ), dtype=np.double), \
              "row" : np.zeros(( self.NT * ( (self.topD+1) ) ), dtype=np.uintc), \
              "col" : np.zeros(( self.NT * ( (self.topD+1) ) ), dtype=np.uintc) \
              }
        # Compute
        self.U = self.computeGeneric( simplices, nodes, 1 + self.D + self.D**2, self.U )
        # Assemble sparse matrix
        self.U = MatMaps.acquireSmallerMatrix( sparse.coo_matrix((self.U["data"], (self.U["row"], self.U["col"])), shape=(self.NN, self.NT)) )
        
        return
    
    
    def computeGeneric(self, simplices, nodes, matId, tempMat):
        """
        Function for computing either M, B, G, or U matrices using the low-level library.
        
        :param simplices: The simplices of the mesh.
        :param nodes: The nodes of the mesh.
        :param matId: Code informing on which matrix to compute. 0 being the M matrix, 1 to D+1 being the different dimensions of B matrix, D+2 to 1 + D + D^2 being the different dimensions of the G matrix, and 2 + D + D^2 being the U matrix.
        :param tempMat: A sparse matrix, typically preallocated to an appropriate size.
        
        :return the corresponding sparse N^2 x T matrix, where N is the number of nodes and T the number of simplices.
        """
        
        data_p = tempMat["data"].ctypes.data_as(self.c_double_p)
        row_p = tempMat["row"].ctypes.data_as(self.c_uint_p)
        col_p = tempMat["col"].ctypes.data_as(self.c_uint_p)
        idx = ctypes.c_uint( np.uintc(0) )
            
        nodes_p = nodes.ctypes.data_as( self.c_double_p )
        simplices_p = simplices.ctypes.data_as( self.c_uint_p )
        
        # Assemble
        status = self._libInstance.FEM_mapTrivals2Mat( \
              data_p, row_p, col_p, ctypes.byref( idx ), \
              ctypes.c_uint( matId ), ctypes.c_uint( tempMat["data"].size ), \
              nodes_p, ctypes.c_uint( self.NN ), ctypes.c_uint( self.D ), \
              simplices_p, ctypes.c_uint( self.NT ), ctypes.c_uint( self.topD ) \
              )                  
        if status != 0:            
            raise Exception( "Uknown error occured! Error status: " + str(status) )  
                
        # Remove unused
        tempMat["data"] = tempMat["data"][0:idx.value]
        tempMat["row"] = tempMat["row"][0:idx.value]
        tempMat["col"] = tempMat["col"][0:idx.value]
        
        return tempMat
            

    
    
    def acquireSmallerMatrix( COOMat ):
        """
        Make matrix smaller by removing zero rows
        
        :param COOMat: The large sparse matrix.
        
        :return The compressed large sparse matrix. Stored as a smaller sparse matrix with all uneccessary rows removed, and an indexing of which rows these are.
        """
            
        # Get all unique rows
        uniqueRows = np.unique( COOMat.row, return_index=True, return_inverse = True )
        # Change row index to the new indexing
        COOMat.row = uniqueRows[2]
        # Resize matrix to remove uneccesary part
        COOMat.resize( (uniqueRows[0].size, COOMat.shape[1]) )
        
        # Store matrix and original row index
        return { "CSRMatrix" : COOMat.tocsr(), "originalRow" : uniqueRows[0] }
    
    
    def mapTriVals2Mat( matrix, vector, N ):
        """
        Map values at triangles to a system matrix on the nodes.
        
        :param matrix: A matrix acquired using computeGeneric or any of the wrapper of it, viz. computeM, computeB, computeG, and computeU.
        :param vector: A vector of constants for each simplex.
        :param N: The shape of the output matrix.
        
        :return A N[0] x N[1] matrix to use for the system of linear equations in the finite element method.
        """
        
        if np.isscalar(N):
            N = N * np.array([1,1])
        
        if np.isscalar(vector):
            vector = vector * np.ones(matrix["CSRMatrix"].shape[1], dtype="float64" )
        
        # Compute from simplex to basis
        out = matrix["CSRMatrix"] * vector
        
        if N[1] == 1:
            # Create sparse matrix of output
            out = sparse.coo_matrix( \
                ( out, \
                ( matrix["originalRow"].astype(np.uintc), \
                np.zeros( (matrix["originalRow"].size) ).astype(np.uintc) ) ), \
                shape = N )
        else:
            # Create sparse matrix of output
            out = sparse.coo_matrix( \
                ( out, \
                ( np.floor( matrix["originalRow"] / N[0]).astype(np.uintc), \
                (matrix["originalRow"] % N[0]).astype(np.uintc) ) ), \
                shape = N )
                
        return out.tocsr()
    
    



class abstractDeformed(FEM):
    """
    Generic FEM child class for the deformed Matern models.
    
    The childParams parameter completely decides the model.
    """
    
    # parameers of inherited model
    childParams = None
    
    # Dictionary of what matMaps parameters to calculate
    matMapsCalculate = None
    matMapsCalculateEdges = None

    
    @abc.abstractmethod
    def paramsFunction(self):
        # Function for computing FEM params from child params
        return
    
    def __init__( self, mesh, childParams, nu, sigma, mu = 0, libPath = None, BCDirichlet = None, BCRobin = None, sourceCoeff = None, factorize = True ):

        
        # Acquire necesarry maps from mesh cells to system matrices
        if sourceCoeff is not None:
            self.matMapsCalculate.append('U')        
                
        # Acquire necessary maps from mesh edges to system matrices
        if BCRobin is not None:
            self.matMapsCalculateEdges = []
            if np.any(BCRobin[:, 0] != 0):
                self.matMapsCalculateEdges.append('U')      
            if np.any(BCRobin[:, 1] != 0):
                self.matMapsCalculateEdges.append('M')
                
        # Parent init
        super(abstractDeformed, self).__init__(mesh,  \
            matMapsCalculate = self.matMapsCalculate, \
            matMapsCalculateEdges = self.matMapsCalculateEdges, \
            libPath = libPath\
            )
        
        # Update system
        self.updateSystem( childParams = childParams, nu = nu, mu = mu, sigma = sigma, sourceCoeff = sourceCoeff, BCRobin = BCRobin, BCDirichlet = BCDirichlet, factorize = factorize )
        return
        
    
    
    def copy(self):
        
        out = type(self)( None, None, None, None)        
        out = super(abstractDeformed, self).copyParent(out)
        
        out.childParams = self.childParams
        out.matMapsCalculate = self.matMapsCalculate
        out.matMapsCalculateEdges = self.matMapsCalculateEdges
        
        return out
        
    
    
    
    def updateSystem( self, childParams, nu, sigma, mu = None, BCDirichlet = None, BCRobin = None, sourceCoeff = None, factorize = True ):
        
        if self.mesh == None:
            return
        
        # Setup system
        self.childParams = childParams
        self.nu = nu
        d = self.mesh.topD
        alpha = nu + d / 2.0
        tau = np.sqrt( special.gamma(nu) / ( special.gamma(alpha) * (4 * np.pi)**(d/2) ) ) / ( sigma )
        
        if np.isscalar(tau):
            tau = tau * np.ones((self.mesh.N))
            
        MCoeff, BCoeff, GCoeff = self.paramsFunction( )    
                
        # Parent init
        super(abstractDeformed, self).updateSystem(  \
            MCoeff = MCoeff, \
            tau = tau, nu = nu, mu = mu, \
            BCoeff = BCoeff, \
            GCoeff = GCoeff, \
            sourceCoeff = sourceCoeff, \
            BCRobin = BCRobin, \
            BCDirichlet = BCDirichlet, \
            factorize = factorize \
            )
                
        return




class MaternFEM(abstractDeformed):
    """
    Class representing the classical matern model.
    """
    
    matMapsCalculate = ['M', 'G']
    matMapsCalculateEdges = None
    
    def paramsFunction( self ):
        """
        Defines the standard Matérn SPDE model parameterized by a correlation range ('r').
        
        :return The coefficients for the M, B, and G matrices.
        """
    
        if self.childParams is None:
            raise Exception("No r-parameter given")
            
        r = self.childParams
        if isinstance( r, dict):
            r = r["r"]
        
        d = self.mesh.embD
        alpha = self.nu + d / 2
        
        logGSqrt = - d * np.log( r/np.sqrt(8*self.nu) )
        GInv = ( np.exp( - 2 / d * logGSqrt) * np.eye(d) ).flatten()
        
        
        MCoeff = np.exp( 1/alpha * logGSqrt )
        BCoeff = None
        GCoeff = [None] * GInv.size
        for iterGInv in range(GInv.size):
            if GInv[iterGInv] != 0:
                GCoeff[iterGInv] = MCoeff * GInv[iterGInv]
          
        return (MCoeff, BCoeff, GCoeff)
    
    
class anisotropicMaternFEM(abstractDeformed):
    """
    Class representing the anisotropic matern model in two dimensions.
    """
    
    matMapsCalculate = ['M', 'G']
    matMapsCalculateEdges = None
    
    def paramsFunction( self ):
        """
        Generate FEM model for anisotropic Matérn model in two dimensions given an angle of the main direction ('angle') and two correlation ranges ('r').
        
        :return The coefficients for the M, B, and G matrices.
        """
    
        if not self.mesh.embD == 2:
            raise Exception("Current class only defined for manifolds embedded in R^2!")
    
        if self.childParams is None:
            raise Exception("No parameters given")
            
        if not isinstance( self.childParams, dict):
            raise Exception("Parameters were not given in dictionary format")
        
        alpha = self.nu + self.mesh.topD / 2
        
        logGSqrt, GInv = orthVectorsToG( angleToVecs2D(self.childParams["angle"]).transpose(), self.childParams["r"] / np.sqrt(8*self.nu) )

        # Set FEM parameters        
        MCoeff = np.exp( 1/alpha * logGSqrt )
        BCoeff = None
        GCoeff = [None] * (self.mesh.embD**2)
        if GInv is not None:
            for iterGInv in range(self.mesh.embD**2):
                if GInv[iterGInv] != 0:
                    GCoeff[iterGInv] = MCoeff * GInv[iterGInv]
          
        return (MCoeff, BCoeff, GCoeff)
        
        



class nonStatFEM(abstractDeformed):
    """
    Class representing the general deformed Matern model.
    """
    
    matMapsCalculate = ['M', 'G']
    matMapsCalculateEdges = None
    
    def paramsFunction( self ):
        """
        Function to map child parameters to FEM parameters.
        
        In the member object 'childParams', either a function is provided under the name 'f' or it is assumed that the parameters 'logGSqrt' and 'GInv' are provided.
        
        If a function was provided under the name 'f', this function should take the dictionary self.childParams as its argument.
        It should output a tuple corresponding to logGSqrt and GInv.
        
        logGSqrt is the logarithm of the squared determinant of G
        GInv is the inverse of G
        
        :return: a tuple corresponding to the coefficients of M, B, and G.
        """
    
        if self.childParams is None:
            raise Exception("No parameters given")
            
        if not isinstance( self.childParams, dict):
            raise Exception("Parameters were not given in dictionary format")
        
        # Compute kappa and H
        logGSqrt = None
        GInv = None
        if "f" in self.childParams:
            logGSqrt, GInv = self.childParams["f"]( self.childParams )
        else:
            logGSqrt = self.childParams["logGSqrt"]
            GInv = self.childParams["GInv"]
                
        alpha = self.nu + self.mesh.topD / 2
        
        # Set FEM parameters        
        MCoeff = np.exp( 1/alpha * logGSqrt )
        BCoeff = None
        GCoeff = [None] * (self.mesh.embD**2)
        if GInv is not None:
            for iterGInv in range(self.mesh.embD**2):
                if np.any(GInv[iterGInv] != 0):
                    GCoeff[iterGInv] = MCoeff * GInv[iterGInv]
          
        return (MCoeff, BCoeff, GCoeff)







def angleToVecs2D( angle ):
    """
    :param angle: An angle [radians] for the main direction (from x-axis).
    
    :return A 2D-array which columns are two orthogonal unit vectors, the first pointing in the direction of angle
    """
    
    
    if isinstance( angle, np.ndarray):
        rotMat = np.stack( ( np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle) ) )
        if angle.size > 0:
            rotMat = rotMat.transpose((1,0)).reshape( (angle.size, 2, 2) )
        else:
            rotMat = rotMat.reshape( (2, 2) )
    else:
        rotMat = np.array( [ [ np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ] )
    
    
    vectors = np.matmul( rotMat, np.eye(2) )
    
    return vectors


def tangentVectorsOnSphere( points, northPole = np.array([0.0,0.0,1.0]) ):
    """
    Acquire a basis for the tangent space at given points on the surface of the unit sphere. 
    
    :param points: N x 3 array of N points at which to acquire basis of tangent space.
    :param northPole: 3 array of point corresponding to the north pole.
    
    :return A N x 3 x 3 array. Each point has three orthogonal tangent vectors of unit length. 
    They are constructed such as the first vector is pointing towards the 'northPole'. 
    The second vector is orthogonal to both the first vector and the vector from the origin to the point of interest.
    The third vector is equal to the vector between the origin and the point of interest.
    The last dimension represent the elements of the vectors. The next to last dimension indices the vectors
    
    """
    
    vectors = np.zeros( (points.shape[0], 3,3) )
    
    # Get third vector 
    vectors[:, 2, :] = points / np.linalg.norm(points, axis= 1).reshape((-1,1))
    # Get second vector 
    vectors[:, 1, :] = np.cross( northPole.reshape( (1,3) ), vectors[:,2,:] )
    # Get first vector 
    vectors[:, 0, :] = np.cross( vectors[:,2,:], vectors[:,1,:] )
    
    # Normalize vectors
    lengths = np.linalg.norm( vectors, axis=2 ).reshape((-1, 3))
    inds = np.any( lengths == 0.0, axis=1 )
    vectors[inds, :, : ] = np.nan
    vectors[~inds, :, :] = vectors[~inds, :, :] / lengths[~inds, :].reshape( (-1,3,1) )
    
    
    return vectors


def orthVectorsToG( vectors, scalers ):
    """
    Acquire parameters to the SPDE given the deformation
    
    :param vectors: An N x d x d array where first dimension decides simplex. The second dimension is indexing the vectors. 
        The third dimension is the elements of the vectors.
        The set of vectors assumed to be orthogonal and normalized. 
    :param scalers: An N x d array of scalers in each of the vector directions.
    
    :return the logarithm of the squared determinant of G, and the inverse of G. Here, G being the metric tensor.
    """
    
    if (vectors.ndim == 2):
        vectors = vectors.reshape((1,vectors.shape[0], vectors.shape[1]))
    
    if (scalers.ndim == 1):
        scalers = scalers.reshape((1,-1))
    
    n = scalers.shape[0]
    d = scalers.shape[1]
    
    
    # Compute logarithm of G squared
    logGSqrt = - np.sum( np.log(scalers), axis = 1 )
    
    temp = scalers.reshape( (n,d,1) ) * vectors[:, :d, :]
    
    GInv = np.ones( (n,vectors.shape[1],vectors.shape[1]) )
    for iter in range(vectors.shape[1]):
        temp2 = temp * temp[:, :, iter].reshape((n,d,1))
        GInv[:, iter, :] = np.sum( temp2, axis=1 ).reshape(n, vectors.shape[1])

    
    GInv = GInv.reshape( (n, vectors.shape[1]**2) ).transpose()
    
    if n == 1:
        GInv = GInv.flatten()
        logGSqrt = logGSqrt[0]
    
    return (logGSqrt, GInv)

