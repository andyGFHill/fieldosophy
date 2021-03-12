#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for Cholesky decomposition of a symmetric and positive-definite matrix. 

This file is part of Fieldosophy, a toolkit for random fields.
Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import ctypes
import os
import abc
from sksparse import cholmod




class SparseCholeskyAbstract(abc.ABC):
    # Abstract class for sparse Choleksy factorization
    
    class PositiveDefiniteException(Exception):
        # Raised when not positive definite
        pass
    
    N = None # Number of columns/rows
    
    @abc.abstractmethod
    def copy( self ):
        # Solve linear system with Cholesky triangle
        return
    @abc.abstractmethod
    def solveL( self, inData, transpose = False ):
        # Solve linear system with Cholesky triangle
        return
    @abc.abstractmethod
    def multiply(self, inData):
        # Multiply vector or matrix with original matrix
        return
    @abc.abstractmethod
    def permute( self, inData, toChol ):
        # permute matrix to Cholesky permutation or from
        return
    @abc.abstractmethod
    def solve(self, inData):
        # Solve with original matrix
        return
    @abc.abstractmethod
    def getL(self, upper = False):
        # Get Cholesky triangle
        return
    @abc.abstractmethod
    def getP(self, toChol = False):
        # Get Permutation matrix
        return
    @abc.abstractmethod
    def getLogDet(self):
        # Get log determinant
        return
    @abc.abstractmethod
    def getDet(self):
        # Get determinant
        return    
    @abc.abstractmethod
    def getMatrix(self):
        # Get original matrix
        return
        
       
        
    
    

class InhouseSparseCholesky(SparseCholeskyAbstract):
    """
    Class representing a sparse cholesky factorization based on Fieldosophy-specific C-code, the scipy-sparse package and the Eigen library.
    """

    
    # Declare pointer types
    c_double_p = ctypes.POINTER(ctypes.c_double)       
    c_uint_p = ctypes.POINTER(ctypes.c_uint)  
        
    L = None # The lower triangular Cholesky decomposition
    P = None # the permutation matrix
    
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libCholesky.so" )
    _libInstance = None
    
    def __init__(self, matrix, libPath = None):
        # Initiate a Cholesky factorization
        
        # Set sizes
        self.N = matrix.shape[0]
        
        if libPath is not None:
            self._libPath = libPath
        
        # If not coo matrix
        if not sparse.isspmatrix_coo(matrix):
            matrix = sparse.coo_matrix(matrix)                    
            
        # Instantiate C library
        self._libInstance = ctypes.CDLL(self._libPath) 
        
        # Free memory
        self._libInstance.freeCholesky.restype = ctypes.c_int        
        self._libInstance.freeSparse.restype = ctypes.c_int
        self._freeStorage()
        
            
        # Get data    
        data = matrix.data
        if matrix.data.dtype is not np.dtype("float64"):
            data = data.astype(np.float64)
        row = matrix.row.astype(np.uintc)
        col = matrix.col.astype(np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)  
        
        # Perform Cholesky factorization
        self._libInstance.choleskyPrepare.restype = ctypes.c_int
        self._libInstance.choleskyPrepare.argtypes = [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint, ctypes.c_uint ]  
        status = self._libInstance.choleskyPrepare( data_p, row_p, col_p, ctypes.c_uint(self.N), ctypes.c_uint(len(data)))    
        if status != 0:
            if status == 1:
                raise Exception( "Internal Cholesky storage was not empty!" ) 
            if status == 2:
                raise Exception( "Failed to analyze pattern!" ) 
            if status == 3:
                raise Exception( "Failed to factorize!" ) 
            raise Exception( "Uknown error occured!" ) 

            
        # Get permutation matrix since the matrix is Cholesky factorized as LL^T = P A P^T
        self._libInstance.getPermutation.restype = ctypes.c_int
        self._libInstance.getPermutation.argtypes = [ self.c_uint_p, ctypes.c_uint ]  
        perm = np.zeros(self.N, dtype=np.uintc)
        perm_p = perm.ctypes.data_as(self.c_uint_p)
        status = self._libInstance.getPermutation( perm_p, ctypes.c_uint(self.N) )
        if status == 0:
            self.P = sparse.coo_matrix( (np.ones(self.N), ( perm, np.arange(0,self.N))), shape = (self.N, self.N) ).tocsr()
        elif status == 2: # If permutation size was wrong
            # Assume no permutation            
            self.P = sparse.diags( np.ones(self.N), offsets=0, shape = (self.N, self.N) )    
        else: 
            raise Exception( "Uknown error occured!" ) 
            
        
        # Get Lower Cholesky triangle
        self.L = sparse.identity(self.N, dtype='float64', format='coo')
        self.L = self._multiplySparseL( self.L )
        
        # Free internal storage
        self._freeStorage()
            
        
        
    def solveL( self, inData, transpose = False ):
        """
        Solve linear system using lower Cholesky triangle.
        
        :param inData: Vector (or collection of vectors) to be solved.
        :param transpose: Specifies if the Cholesky triangle should be transposed or not before solving.
        
        :return: An output of the same shape as inData.
        """
    
        out = None
        
        # If is sparse matrix
        if sparse.issparse(inData):
            if transpose:
                out = splinalg.spsolve(self.L.transpose().tocsr(), inData, permc_spec="NATURAL")
            else:
                out = splinalg.spsolve(self.L.tocsr(), inData, permc_spec="NATURAL")
            # Take care that dimensions are correct
            if len(out.shape) < len(inData.shape):
                    out = out.reshape(inData.shape)
        else: # If dense matrix
            if transpose:
                out = splinalg.spsolve_triangular(self.L.transpose().tocsr(), inData, lower=False)
            else:
                out = splinalg.spsolve_triangular(self.L.tocsr(), inData, lower=True)
        
        return out

    def multiply(self, inData):
        """
        Multiplies input data with the matrix (which is represented by a Cholesky decomposition internally).
        
        :param inData: Vector (or collection of vectors) to be multiplied.        
        :return: An output of the same shape as inData.
        """
        
        out = inData.copy()
        out = self.permute( out, toChol = True )
        out = self.L.transpose() * out
        out = self.L * out
        out = self.permute( out, toChol = False )
    
    def permute( self, inData, toChol ):
        """
        Permutes input data to sparse Cholesky permutation, or from.
        
        :param inData: Vector (or collection of vectors) to be permuted.
        :param toChol: Specifies whether the permutation should be to sparse Cholesky or from.
        :return: An output of the same shape as inData.
        """

        return self.getP(toChol = toChol) * inData
        
        
    def solve(self, inData):
        """
        Solve linear system using Cholesky decomposition of matrix.
        
        :param inData: Vector (or collection of vectors) to be solved.        
        :return: An output of the same shape as inData.
        """
        
        out = inData.copy()
        out = self.permute( out, toChol = True )
        out = self.solveL( out, transpose = False )
        out = self.solveL( out, transpose = True )
        out = self.permute( out, toChol = False )
        
        return out
    
    def getL(self, upper = False):
        """
        Get lower Cholesky triangle
        
        :param upper: True if lower triangular Cholesky triangle should be transposed before it is returned.
        :return: A sparse Cholesky triangle.
        """
        
        # Get Cholesky triangle
        if upper:
            return self.L.transpose()
        else:
            return self.L

    def getP(self, toChol = False):
        """
        Get permutation matrix.
        
        :param toChol: If the permutation matrix should map to sparse Cholesky representation, or from it.
        :return: Permutation matrix.
        """
        
        if toChol:
            return self.P
        else:
            return self.P.transpose()
        
        return
    
    def getLogDet(self):
        """
        :return: The log-determinant of original matrix
        """
        
        # Get log determinant
        return 2 * np.sum( np.log( self.L.diagonal() ) )
    
    def getDet(self):
        """
        :return: The determinant of original matrix
        """
        
        return self.L.diagonal()**2
        
        
    def _solveDense( self, inData ):
        # Solve system for a dense matrix
        
        # Make sure that sizes are correct
        if (inData.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(inData.shape[0]) + "." )
        
        outData = inData.copy()
        if outData.dtype is not np.dtype("float64"):
            outData = outData.astype(np.float64)
        
        if (len(inData.shape) == 1):
            outData = outData.reshape((-1,1))            
            
        # Define solve function
        self._libInstance.solveByCholesky.restype = ctypes.c_int
        self._libInstance.solveByCholesky.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_uint ]
            
        data_p = outData.ctypes.data_as(self.c_double_p)
        status = self._libInstance.solveByCholesky( data_p, ctypes.c_uint(outData.shape[0]), ctypes.c_uint(outData.shape[1]) )
        if status != 0:
            raise Exception( "Cholesky factorization not performed first!" )
            
        return outData
    
    
    def _solveDenseL( self, inData, transpose = False ):
        # Solve system for the lower-triangular Cholesky factor
        
        # Make sure that sizes are correct
        if (inData.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(inData.shape[0]) + "." )
        
        outData = inData.copy()
        if outData.dtype is not np.dtype("float64"):
            outData = outData.astype(np.float64)
        
        if (len(inData.shape) == 1):
            outData = outData.reshape((-1,1))            
            
        # Define solve function
        self._libInstance.solveByCholeskyL.restype = ctypes.c_int
        self._libInstance.solveByCholeskyL.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool ]
            
        data_p = outData.ctypes.data_as(self.c_double_p)
        status = self._libInstance.solveByCholeskyL( data_p, ctypes.c_uint(outData.shape[0]), ctypes.c_uint(outData.shape[1]), ctypes.c_bool(transpose) )
        if status != 0:
            raise Exception( "Cholesky factorization not performed first!" )
            
        return outData    
        
    def _multiplyDenseL( self, inData, transpose = False ):
        # Solve system for a dense matrix
        
        # Make sure that sizes are correct
        if (inData.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(inData.shape[0]) + "." )
        
        outData = inData.copy()
        
        if outData.dtype is not np.dtype("float64"):
            outData = outData.astype(np.float64)            
            
        # Define solve function
        self._libInstance.multiplyByCholeskyL.restype = ctypes.c_int
        self._libInstance.multiplyByCholeskyL.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool ]
            
        data_p = outData.ctypes.data_as(self.c_double_p)
        status = self._libInstance.multiplyByCholeskyL( data_p, ctypes.c_uint(outData.shape[0]), ctypes.c_uint(outData.shape[1]), ctypes.c_bool(transpose) )
        if status != 0:
            raise Exception( "Cholesky factorization not performed first!" )
            
        return outData     
        
    
    
    def _multiplySparseL( self, inData, transpose = False  ):
        # Solve system for a sparse matrix
        
        # Make sure that sizes are correct
        if (inData.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(inData.shape[0]) + "." )
        
        # If not coo matrix
        if not sparse.isspmatrix_coo(inData):
            inData = sparse.coo_matrix(inData)
        
        # Get data    
        data = inData.data
        if data.dtype is not np.dtype("float64"):
            data = data.astype(np.float64)
        row = inData.row.astype(np.uintc)
        col = inData.col.astype(np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)  
        numNonZeros = np.array(data.size)
        numNonZeros_p = numNonZeros.ctypes.data_as(self.c_uint_p)
        
        
        # Perform sparse multiplication
        self._libInstance.sparseMultiplyByCholeskyL.restype = ctypes.c_int
        self._libInstance.sparseMultiplyByCholeskyL.argtypes = [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint, \
                ctypes.c_uint, self.c_uint_p, ctypes.c_bool ]          
        status = self._libInstance.sparseMultiplyByCholeskyL( data_p, row_p, col_p, ctypes.c_uint(inData.shape[0]), \
                 ctypes.c_uint(inData.shape[1]), numNonZeros_p, ctypes.c_bool(transpose) )    
        if status != 0:
            if status == 1:
                raise Exception( "Internal Cholesky storage was empty!" ) 
            if status == 2:
                raise Exception( "Internal sparse matrix was not empty!" ) 
            raise Exception( "Uknown error occured!" ) 
        
        
        data = np.zeros(numNonZeros, dtype=np.double)
        row = np.zeros(numNonZeros, dtype=np.uintc)
        col = np.zeros(numNonZeros, dtype=np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)
        
        # Perform retrieval
        self._libInstance.getTripletsOfSparseMatrix.restype = ctypes.c_int
        self._libInstance.getTripletsOfSparseMatrix.argtypes = [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint ]          
        status = self._libInstance.getTripletsOfSparseMatrix( data_p, row_p, col_p, ctypes.c_uint(numNonZeros))    
        if status != 0:
            if status == 1:
                raise Exception( "Internal sparse matrix was empty!" ) 
            if status == 2:
                raise Exception( "Numbers of non-zeros did not match!" )
                
        # Free storage 
        self._libInstance.freeSparse.restype = ctypes.c_int
        status = self._libInstance.freeSparse() 
                
        # Create output matrix
        outData = sparse.coo_matrix( ( data, (row, col)), shape = inData.shape )
        return outData
    
    
    
    def _solveSparse( self, inData ):
        # Solve system for a sparse matrix
        
        # Make sure that sizes are correct
        if (inData.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(inData.shape[0]) + "." )
        
        # If not coo matrix
        if not sparse.isspmatrix_coo(inData):
            inData = sparse.coo_matrix(inData)
        
        # Get data    
        data = inData.data
        if data.dtype is not np.dtype("float64"):
            data = data.astype(np.float64)
        row = inData.row.astype(np.uintc)
        col = inData.col.astype(np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)  
        numNonZeros = np.array(data.size)
        numNonZeros_p = numNonZeros.ctypes.data_as(self.c_uint_p)
        
        # Perform sparse solve
        self._libInstance.sparseSolveByCholesky.restype = ctypes.c_int
        self._libInstance.sparseSolveByCholesky.argtypes = [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint, ctypes.c_uint, self.c_uint_p ]          
        status = self._libInstance.sparseSolveByCholesky( data_p, row_p, col_p, ctypes.c_uint(inData.shape[0]), ctypes.c_uint(inData.shape[1]), numNonZeros_p )    
        if status != 0:
            if status == 1:
                raise Exception( "Internal Cholesky storage was empty!" ) 
            if status == 2:
                raise Exception( "Internal sparse matrix was not empty!" ) 
            raise Exception( "Uknown error occured!" ) 
        
        
        data = np.zeros(numNonZeros, dtype=np.double)
        row = np.zeros(numNonZeros, dtype=np.uintc)
        col = np.zeros(numNonZeros, dtype=np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)
        
        # Perform retrieval
        self._libInstance.getTripletsOfSparseMatrix.restype = ctypes.c_int
        self._libInstance.getTripletsOfSparseMatrix.argtypes = [ self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint ]          
        status = self._libInstance.getTripletsOfSparseMatrix( data_p, row_p, col_p, ctypes.c_uint(numNonZeros))    
        if status != 0:
            if status == 1:
                raise Exception( "Internal sparse matrix was empty!" ) 
            if status == 2:
                raise Exception( "Numbers of non-zeros did not match!" )
                
        # Free storage 
        self._libInstance.freeSparse.restype = ctypes.c_int
        status = self._libInstance.freeSparse() 
                
        # Create output matrix
        outData = sparse.coo_matrix( ( data, (row, col)), shape = inData.shape )
        return outData
    
    
    

    # # Acquire lower triangular Cholesky matrix
    # FEMC.getTripletsOfSparseMatrix.restype = ctypes.c_int
    # FEMC.getTripletsOfSparseMatrix.argtypes = [ c_double_p, c_uint_p, c_uint_p, ctypes.c_uint, ctypes.c_uint ]  
    # status = FEMC.getTripletsOfSparseMatrix( data_p, row_p, col_p, lN, lNumNonZeros )    
    # if status != 0:
    #     return None
    # L = sparse.coo_matrix((data, (row, col)), shape=(lN.value, lN.value)).tocsr()
    
    # return { "P" : P, "L" : L }    
        
        

    def _freeStorage(self):
        # Declare free storage function and call it
        
        # self._libInstance.freeCholesky.restype = ctypes.c_int
        status = self._libInstance.freeCholesky()
                
        # self._libInstance.freeSparse.restype = ctypes.c_int
        status = self._libInstance.freeSparse()   
        

    def __del__(self):
        # Destructor
        
        # Free storage
        self._freeStorage()
        
        
    def _CGSolve( self, A, b ):
        # Solve linear system using conjugate gradient
        
        # Make sure that sizes are correct
        if (b.shape[0] != self.N):
            raise Exception( "Wrong size of vector! Expected " + str(self.N) + " but got " + str(b.shape[0]) + "." )
        if (A.shape[0] != self.N) or (A.shape[1] != self.N):
            raise Exception( "Wrong size of Matrix! Expected " + str(self.N) + " x " + str(self.N) + " but got " + str(A.shape) + "." )
        
        if A.dtype is not np.dtype("float64"):
            A = A.astype(np.float64)            
        
        # If not coo matrix
        if not sparse.isspmatrix_coo(A):
            A = sparse.coo_matrix(A)  
        
        # Get data for A
        data = A.data
        if A.data.dtype is not np.dtype("float64"):
            A = A.astype(np.float64)
        row = A.row.astype(np.uintc)
        col = A.col.astype(np.uintc)
        data_p = data.ctypes.data_as(self.c_double_p)
        row_p = row.ctypes.data_as(self.c_uint_p)
        col_p = col.ctypes.data_as(self.c_uint_p)  
        
        # Get data for b
        outData = b.copy()        
        if outData.dtype is not np.dtype("float64"):
            outData = outData.astype(np.float64)            
        outData_p = outData.ctypes.data_as(self.c_double_p)

        # Perform CG solve
        self._libInstance.CGSolve.restype = ctypes.c_int
        self._libInstance.CGSolve.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_uint, \
                  self.c_double_p, self.c_uint_p, self.c_uint_p, ctypes.c_uint, ctypes.c_uint ]  
        status = self._libInstance.CGSolve( \
               outData_p, ctypes.c_uint( outData.shape[0] ), ctypes.c_uint( outData.shape[1] ), \
               data_p, row_p, col_p, ctypes.c_uint(self.N), ctypes.c_uint(len(data)))    
        if status != 0:            
            raise Exception( "Uknown error occured!" )  
        
        return outData    
        
      
    
    
    
class SKSparseCholesky(SparseCholeskyAbstract):
    """
    Wrapper class representing a sparse cholesky factorization using Scikit.sparse.
    """
    
    
    _chol = None
    
    def __init__(self, matrix, AAt = False, ordering = "AMD"):
        # Initiate a Cholesky factorization
        
        # Set sizes
        self.N = matrix.shape[0]
        
        if AAt:
            # Analyze matrix
            self._chol = cholmod.analyze_AAt( matrix, mode = "auto", ordering_method = "best", use_long=False )
            # Factor matrix
            self._chol = self._chol.cholesky_AAt(matrix)
        else:
            # Analyze matrix
            self._chol = cholmod.analyze( matrix, mode = "auto", ordering_method = "best", use_long=False )
            # Factor matrix
            self._chol = self._chol.cholesky(matrix)
            
        if np.any( self._chol.D() <= 0 ):
            raise SparseCholeskyAbstract.PositiveDefiniteException("Matrix is not positive definite")
        
    def copy(self):
        out = SKSparseCholesky( sparse.diags(np.array([1])).tocsc() )
        out._chol = self._chol.copy()
        out.N = self.N
        
        return out
        
        
    def solveL( self, inData, transpose = False ):
        """
        Solve linear system using lower Cholesky triangle.
        
        :param inData: Vector (or collection of vectors) to be solved.
        :param transpose: Specifies if the Cholesky triangle should be transposed or not before solving.
        
        :return: An output of the same shape as inData.
        """
        D = self._chol.D()
        out = sparse.diags( 1/np.sqrt(D) )
        if transpose:
            out = out * inData
            out = self._chol.solve_Lt( out, use_LDLt_decomposition = True )
        else:            
            out = out * self._chol.solve_L( inData, use_LDLt_decomposition = True )
            
        return out
        
    def multiply(self, inData):
        """
        Multiplies input data with the matrix (which is represented by a Cholesky decomposition internally).
        
        :param inData: Vector (or collection of vectors) to be multiplied.        
        :return: An output of the same shape as inData.
        """
        out = self.permute(inData, toChol = True)
        out = self.getL(upper = True) * out
        out = self.getL(upper = False) * out
        out = self.permute(inData, toChol = False)
        
        return out
    
    def permute( self, inData, toChol ):
        """
        Permutes input data to sparse Cholesky permutation, or from.
        
        :param inData: Vector (or collection of vectors) to be permuted.
        :param toChol: Specifies whether the permutation should be to sparse Cholesky or from.
        :return: An output of the same shape as inData.
        """
        if toChol:
            return self._chol.apply_P(inData)
        else:
            return self._chol.apply_Pt(inData)
        
    def solve(self, inData):
        """
        Solve linear system using Cholesky decomposition of matrix.
        
        :param inData: Vector (or collection of vectors) to be solved.        
        :return: An output of the same shape as inData.
        """
        return self._chol.solve_A(inData)
        
    def getL(self, upper = False):
        """
        Get lower Cholesky triangle
        
        :param upper: True if lower triangular Cholesky triangle should be transposed before it is returned.
        :return: A sparse Cholesky triangle.
        """
        L, D = self._chol.L_D()        
        out = np.sqrt( D )
        if upper:
            out = out * L.transpose()
        else:
            out = L * out
        
        return out.tocsc()
    
    def getP(self, toChol = False):
        """
        Get permutation matrix.
        
        :param toChol: If the permutation matrix should map to sparse Cholesky representation, or from it.
        :return: Permutation matrix.
        """
        mat = sparse.identity(self.N)        
        
        if toChol:
            return self._chol.apply_P( mat )
        else:
            return self._chol.apply_Pt( mat )
        
    def getPIndex(self, toChol = False):
        # Get Permutation index
        if toChol:
            return self._chol.P()
        else:
            return self._chol.P().transpose()
        
    def getLogDet(self):
        """
        :return: The log-determinant of original matrix
        """
        try:
            return self._chol.logdet()
        except:
            raise SparseCholeskyAbstract.PosDef
    
    def getDet(self):
        """
        :return: The determinant of original matrix
        """
        return self._chol.det()
    
    def getMatrix(self):
        """
        :return: The original matrix that has been factorized.
        """
        mat = self.getL()
        mat = self._chol.apply_Pt(mat)
        mat = mat * mat.transpose()
        
        return mat    
    