/* 
* C/C++ functions for Cholesky decomposition.
*
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*
* Author: Anders Gunnar Felix Hildeman
* Date: 2020-04
*/


#include "Eigen/SparseCholesky"
#include "Eigen/IterativeLinearSolvers"
#include <vector>
#include "Cholesky.h"



extern "C"
{
        
        
    // Free storage
    int freeCholesky()
    {
        if (gCholesky != NULL)
            delete gCholesky;
            gCholesky = NULL;
    
        return 0;
    }
    
    // Free storage
    int freeSparse()
    {
        if (gSpMat != NULL)
            delete gSpMat;
            gSpMat = NULL;
    
        return 0;
    }
    
    // Get current sparse sparse matrix as triplets of double arrays
    int getTripletsOfSparseMatrix( double * const pData, unsigned int * const pRow, unsigned int * const pCol, const unsigned int pNumNonZeros )
    {
        // Check sizes
        if (gSpMat == NULL)
            return 1;
        if ( pNumNonZeros != gSpMat->nonZeros() )
            return 2;
            
        // Loop over outer dimension
        unsigned int lIterTriplets = 0;
        for( unsigned int lIter = 0; lIter < gSpMat->outerSize(); lIter++)
            // Loop over inner dimension
            for( SpMat::InnerIterator lIterInner(*gSpMat,lIter); lIterInner; ++lIterInner)
            {
                // Set current value
                pRow[lIterTriplets] = lIterInner.row();
                pCol[lIterTriplets] = lIterInner.col();
                pData[lIterTriplets] = lIterInner.value();
                // Increase iterator
                lIterTriplets++;
            }
                
        return 0;
    
    }
    
    // Create sparse matrix from double arrays of triplets
    SpMat createSparseMatrixFromTriplets( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, const unsigned int pM, const unsigned int pNumNonZeros )
    {    
        // Create sparse matrix from input
        std::vector<Triplet> lTriplets;
        lTriplets.reserve(pNumNonZeros);
        for( unsigned int lIter = 0; lIter < pNumNonZeros; lIter++ )
        {
          // Insert triplet
          lTriplets.push_back(Triplet(pRow[lIter],pCol[lIter],pData[lIter]));
          //printf("Row: %u, Vol: %u, Value %f\n", pRow[lIter], pCol[lIter], pData[lIter]);
        }                
        SpMat lOrigMat = SpMat( pN, pM );
        lOrigMat.setFromTriplets(lTriplets.begin(), lTriplets.end());

        return lOrigMat;
    }        
    
    // Populate interior sparse matrix from triplets
    int populateInteriorSparse( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, const unsigned int pM, const unsigned int pNumNonZeros )
    {
        if (gSpMat != NULL)
            return 1;
    
        // Create sparse matrix
        SpMat lOrigMat = createSparseMatrixFromTriplets( pData,  pRow,  pCol, pN, pM, pNumNonZeros );
        // Get deep copy of matrix    
        gSpMat = new SpMat( lOrigMat );
        
        return 0;
    }   
    
    
    // Prepare for Cholesky factorization
    unsigned int choleskyPrepare( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, const unsigned int pNumNonZeros )
    {
        if (gCholesky != NULL)
            return 1;
    
        // Create original sparse matrix in Eigen format
        SpMat lOrigMat = createSparseMatrixFromTriplets( pData,  pRow,  pCol, pN, pN, pNumNonZeros );

        // Create cholesky object
        gCholesky = new CholeskyType();
        gCholesky->analyzePattern( lOrigMat );
        if( gCholesky->info() != Eigen::Success ) 
            return 2;
        gCholesky->factorize( lOrigMat );
        if( gCholesky->info() != Eigen::Success ) 
            return 3;
        
        return 0;
    }    
    
    // Acquire info on size of Cholesky factor
    int infoOnCholesky( unsigned int * const pN )
    {
        // If no Cholesky factorization exists
        if (gCholesky == NULL)
            return 1;
            
        // Get permutation matrix
        *pN = gCholesky->permutationP().indices().size();
        
        return 0;
    }
    
    // Acquire permutation
    int getPermutation( unsigned int * const pPermutation, const unsigned int pN )
    {
        // If no Cholesky factorization exists
        if (gCholesky == NULL)
            return 1;
        if ( pN != gCholesky->permutationP().indices().size() )
            return 2;
            
        // Get permutation matrix
        for (unsigned int lIter = 0; lIter < pN; lIter++)
            pPermutation[lIter] = gCholesky->permutationP().indices()[lIter];        
        
        return 0;
    }
    
        
    // Solve using Cholesky
    int solveByCholesky( double * pData, const unsigned int pN, const unsigned int pM )
    {   
        // Make sure that gCholesky is defined
        if (gCholesky == NULL)
            return 1;         
        // Map vector to Eigen vector
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lb( pData, pN, pM );
        // Solve system
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lResults = gCholesky->solve(lb);    
        // Copy solution to input array
        std::copy( lResults.data(), lResults.data() + pN*pM, pData );
        
        return 0;
    }      
    
    // Solve using Cholesky
    int sparseSolveByCholesky( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, const unsigned int pM, unsigned int * const pNumNonZeros  )
    {   
        // Make sure that gCholesky is defined
        if (gCholesky == NULL)
            return 1;         
        // Make sure that sparse matrix is not defined
        if (gSpMat != NULL)
            return 2;    
            
        // Create sparse matrix
        SpMat lb = createSparseMatrixFromTriplets( pData,  pRow,  pCol, pN, pM, *pNumNonZeros );                                        
        // Solve system
        gSpMat = new SpMat( gCholesky->solve(lb) );
        // Get number of nonzeros for the result
        *pNumNonZeros = gSpMat->nonZeros();
        
        return 0;        
    }
    
    // Solve lower triangular Cholesky
    int solveByCholeskyL( double * pData, const unsigned int pN, const unsigned int pM, const bool pTranspose)
    {   
        // Make sure that gCholesky is defined
        if (gCholesky == NULL)
            return 1;         
        // Map vector to Eigen vector
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lb( pData, pN, pM );
        // Solve system
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lResults;
        if ( pTranspose )
            lResults = gCholesky->matrixL().transpose().solve(lb);
        else
            lResults = gCholesky->matrixL().solve(lb);
        // Copy solution to input array
        std::copy( lResults.data(), lResults.data() + pN*pM, pData );
        
        return 0;
    }      

    // Multiply lower triangular Cholesky
    int multiplyByCholeskyL( double * pData, const unsigned int pN, const unsigned int pM, const bool pTranspose)
    {   
        // Make sure that gCholesky is defined
        if (gCholesky == NULL)
            return 1;         
        // Map vector to Eigen vector
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lb( pData, pN, pM );
        // multiply 
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lResults;
        if ( pTranspose )
            lResults = gCholesky->matrixL().transpose() * lb;
        else
            lResults = gCholesky->matrixL() * lb;            
        // Copy solution to input array
        std::copy( lResults.data(), lResults.data() + pN*pM, pData );
        
        return 0;
    }        
    
    
    // Solve using Cholesky
    int sparseMultiplyByCholeskyL( const double * const pData, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, 
        const unsigned int pM, unsigned int * const pNumNonZeros, const bool pTranspose )
    {   
        // Make sure that gCholesky is defined
        if (gCholesky == NULL)
            return 1;         
        // Make sure that sparse matrix is not defined
        if (gSpMat != NULL)
            return 2;    
            
        // Create sparse matrix
        SpMat lb = createSparseMatrixFromTriplets( pData,  pRow,  pCol, pN, pM, *pNumNonZeros );                                        
        // multiply 
        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lResults;
        SpMat lResults;
        if ( pTranspose )
            gSpMat = new SpMat( gCholesky->matrixL().transpose() * lb );
        else
            gSpMat = new SpMat( gCholesky->matrixL() * lb );
        // Get number of nonzeros for the result
        *pNumNonZeros = gSpMat->nonZeros();
        
        return 0;        
    }
    
    
    // Conjugate gradient
    unsigned int CGSolve( double * pDataB, const unsigned int pNB, const unsigned int pMB,
        const double * const pDataMat, const unsigned int * const pRow, const unsigned int * const pCol, const unsigned int pN, const unsigned int pNumNonZeros )
    {
        // Create original sparse matrix in Eigen format
        SpMat lOrigMat = createSparseMatrixFromTriplets( pDataMat,  pRow,  pCol, pN, pN, pNumNonZeros );    
        
        Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> lCG;
        
        lCG.compute( lOrigMat );
        // TODO Check
        
        // Map vector to Eigen vector
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lb( pDataB, pNB, pMB );
        // Solve
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lResult = lCG.solve(lb);
        // Copy solution to input array
        std::copy( lResult.data(), lResult.data() + pNB*pMB, pDataB );
        
        return 0;
    }   
    

    

}





