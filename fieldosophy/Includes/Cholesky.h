/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef CHOLESKY_H
#define CHOLESKY_H

extern "C"
{
        
    // Define types
    typedef Eigen::SparseMatrix<double> SpMat; // Define sparse matrix type
    typedef Eigen::Triplet<double, unsigned int> Triplet; // Define triplet of value row and col
    typedef Eigen::SimplicialLLT< SpMat, Eigen::Lower, Eigen::AMDOrdering<int> > CholeskyType;
    
    // Global variables
    SpMat * gSpMat = NULL; // Storage of matrix
    CholeskyType * gCholesky = NULL;    // Storage of cholesky decomposition
    
    // Free storage
    int freeCholesky();
    
    // Free storage
    int freeSparse();
        
    // Get current sparse sparse matrix as triplets of double arrays
    int getTripletsOfSparseMatrix( double * const pData, unsigned int * const pRow, 
        unsigned int * const pCol, const unsigned int pNumNonZeros );
        
    // Create sparse matrix from double arrays of triplets
    SpMat createSparseMatrixFromTriplets( const double * const pData, const unsigned int * const pRow, 
        const unsigned int * const pCol, const unsigned int pN, const unsigned int pM, const unsigned int pNumNonZeros );
        
    // Populate interior sparse matrix from triplets
    int populateInteriorSparse( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, 
        const unsigned int pN, const unsigned int pM, const unsigned int pNumNonZeros );
    
    // Prepare for Cholesky factorization
    unsigned int choleskyPrepare( const double * const pData, const unsigned int * const pRow, 
        const unsigned int * const pCol, const unsigned int pN, const unsigned int pNumNonZeros );
    
    // Acquire info on size of Cholesky factor
    int infoOnCholesky( unsigned int * const pN );
    
    // Acquire permutation
    int getPermutation( unsigned int * const pPermutation, const unsigned int pN );
       
    // Solve using Cholesky
    int solveByCholesky( double * pData, const unsigned int pN, const unsigned int pM );
    
    // Solve using Cholesky
    int sparseSolveByCholesky( const double * const pData, const unsigned int * const pRow, 
        const unsigned int * const pCol, const unsigned int pN, const unsigned int pM, 
        unsigned int * const pNumNonZeros  );
    
    // Solve lower triangular Cholesky
    int solveByCholeskyL( double * pData, const unsigned int pN, const unsigned int pM, const bool pTranspose = 0 );
    
    // Multiply lower triangular Cholesky
    int multiplyByCholeskyL( double * pData, const unsigned int pN, const unsigned int pM, const bool pTranspose = 0 );
    
    // Solve using Cholesky
    int sparseMultiplyByCholeskyL( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, 
        const unsigned int pM, unsigned int * const pNumNonZeros, const bool pTranspose = 0  );
    
    // Conjugate gradient
    unsigned int CGSolve( double * pDataB, const unsigned int pNB, const unsigned int pMB,
        const double * const pDataMat, const unsigned int * const pRow, const unsigned int * const pCol, 
        const unsigned int pN, const unsigned int pNumNonZeros );
    

}



#endif // CHOLESKY_H


