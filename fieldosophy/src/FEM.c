/* 
* C functions for solving differential equations using the finite element method.
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


#include "Eigen/Dense"
#include <vector>
#include "FEM.h"
#include "mesh.hxx"




extern "C"
{

    
    // Computes inner products over a standard triangle
    int FEM_compInProdOverStandTri( const unsigned int pD, double * const pVolume, double * const pOneInner, 
        double * const pMDiag, double * const pMOffDiag )
    {    
        // Compute volume factorial pD!
        double lFactorial = 1.0d;
        for (unsigned int lIterDims = 1; lIterDims <= pD; lIterDims++)
        {
            lFactorial *= (double)lIterDims;
        }
        
        // Get volume <1, 1>
        *pVolume = 1.0d / lFactorial;
        
        // Get inner product of <1, phi_i>
        *pOneInner = 1.0d;
        *pOneInner /= (double)pD + 1.0d;
        *pOneInner /= lFactorial;
        
        // M Diagonal <phi_i, phi_j>
        *pMDiag = 2.0d;
        *pMDiag /= (double)pD + 2.0d;
        *pMDiag /= (double)pD + 1.0d;
        *pMDiag /= lFactorial;
        
        // M off-diagonal <phi_i, phi_j>
        if ( pD == 0 )
            *pMOffDiag = 0.0d;
        else
            *pMOffDiag = 1.0d;
        if ( pD > 1 )
        {
            *pMOffDiag = (double)pD - 1.0d;
            *pMOffDiag *= (double)pD;
            *pMOffDiag /= 2.0d;
        }
        *pMOffDiag /= (double)pD + 2.0d;
        *pMOffDiag /= (double)pD + 1.0d;
        *pMOffDiag /= lFactorial;
    
        return 0;
    }
    
    
    
    
    inline void updateAndAdvanceSparseMatrix( 
        double * const pData, unsigned int * const pRow, unsigned int * const pCol, unsigned int * const pDataIndex,
        const double & pInData, const unsigned int & pInRow, const unsigned int & pInCol )
    {
        pRow[*pDataIndex] = pInRow; // Update row
        pCol[*pDataIndex] = pInCol; // Update col
        pData[*pDataIndex] = pInData; // Update data
        (*pDataIndex)++; // Increase index
    
        return;
    }
    
    
    
    
    // Maps simplex values to matrices
    int FEM_mapTrivals2Mat(
        double * const pData, unsigned int * const pRow, unsigned int * const pCol, unsigned int * const pDataIndex,
        const unsigned int pMatType, const unsigned int pNumNonZeros,
        const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD
    )
    {
    
        // If dimensionality is too big
        if (pTopD > pD)
        {
            // Error
            return 1;
        }
        // If matrix type is incorrect
        if ( pMatType >= 2 + pD + pD*pD )
            // Error
            return 4;
        
        // Acquire important constants
        double lStandardVolume;
        double lOneInner;
        double lMDiag;
        double lMOffDiag;
        const int lStandStatus = FEM_compInProdOverStandTri( pTopD, &lStandardVolume, &lOneInner, &lMDiag, &lMOffDiag );
        if (lStandStatus)
            // Error
            return 2;
        
        // Loop through each simplex
        for (unsigned int lIterSimpl = 0; lIterSimpl < pNumSimplices; lIterSimpl++)
        {
            // Get current node indices
            Eigen::Map<const Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>> lCurNodeInds( &pSimplices[(pTopD+1) * lIterSimpl], pTopD+1, 1 );                                 
            // Get current nodal points
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lCurNodePoints( pTopD+1, pD );
            for (unsigned int lIterPoint = 0; lIterPoint < pTopD+1; lIterPoint++)
            {
                // Copy point to matrix
                std::copy( &pNodes[ pD * lCurNodeInds.coeff(lIterPoint) ], 
                    &pNodes[ pD * lCurNodeInds.coeff(lIterPoint) + pD ], 
                    lCurNodePoints.data() + (pD*lIterPoint) );
            }
                        
            // Acquire transformation to standard simplex
            MapToSimp lF( lCurNodePoints.data(), pD, pTopD );
            
            // If M matrix
            if (pMatType == 0)
            {
                // Loop through all nodes
                for (unsigned int lIterNode1 = 0; lIterNode1 < pTopD+1; lIterNode1++)
                {
                    // Get current node index
                    const unsigned int lCurNode1 = lCurNodeInds.coeff(lIterNode1);
                    // Insert diagonal
                    updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                        lMDiag * lF.getAbsDeterminant(),
                        lCurNode1*pNumNodes + lCurNode1,
                        lIterSimpl );
                    
                    // Loop through all nodes of higher value
                    for (unsigned int lIterNode2 = lIterNode1+1; lIterNode2 < pTopD+1; lIterNode2++)
                    {
                        // Get current node
                        const unsigned int lCurNode2 = lCurNodeInds.coeff(lIterNode2);
                        // Insert off-diagonal
                        updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                            lMOffDiag * lF.getAbsDeterminant(),
                            lCurNode1*pNumNodes + lCurNode2,
                            lIterSimpl );
                        updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                            lMOffDiag * lF.getAbsDeterminant(),
                            lCurNode2*pNumNodes + lCurNode1,
                            lIterSimpl );
                    }
                }
            }
            // If B matrix
            else if ( pMatType < 1 + pD )
            {
                // Get current dimension of interest
                const unsigned int lDimOfInterest = pMatType - 1;
            
                // Loop through all nodes
                for (unsigned int lIterNode1 = 0; lIterNode1 < pTopD+1; lIterNode1++)
                {
                    // Get current node index
                    const unsigned int lCurNode1 = lCurNodeInds.coeff(lIterNode1);
                    // get gradient of current node  
                    Eigen::VectorXd lGrad1(pTopD);
                    if (lIterNode1 > 0) // If not first nodal points
                    {
                        lGrad1.setZero();
                        lGrad1(lIterNode1-1) = 1.0d;
                    }
                    else // If first nodal point
                    {
                        lGrad1.setOnes();
                        lGrad1 = -lGrad1;
                    }
                    // Multiply gradient with inverse transform
                    lGrad1 = lF.solveTransposed(lGrad1);                    
                    // Insert into matrix if not zero 
                    if (lGrad1.coeff(lDimOfInterest) != 0.0d)
                    {
                        // Loop through all nodes once more
                        for (unsigned int lIterNode2 = 0; lIterNode2 < pTopD+1; lIterNode2++)
                        {
                            // Get current node
                            const unsigned int lCurNode2 = lCurNodeInds.coeff(lIterNode2);
                            // Insert
                            updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                                lOneInner * lF.getAbsDeterminant() * lGrad1.coeff(lDimOfInterest),
                                lCurNode1*pNumNodes + lCurNode2,
                                lIterSimpl );
                        }
                    }
                }
            }
            // If G matrix
            else if ( pMatType < 1 + pD + pD*pD )
            {
                // Get current dimensions of interest
                const unsigned int lDimOfInterest1 = (pMatType - 1 - pD)/pD;
                const unsigned int lDimOfInterest2 = (pMatType - 1 - pD)%pD;
            
                // Loop through all nodes
                for (unsigned int lIterNode1 = 0; lIterNode1 < pTopD+1; lIterNode1++)
                {
                    // Get current node index
                    const unsigned int lCurNode1 = lCurNodeInds.coeff(lIterNode1);
                    // get gradient of current node  
                    Eigen::VectorXd lGrad1(pTopD);
                    if (lIterNode1 > 0) // If not first nodal points
                    {
                        lGrad1.setZero();
                        lGrad1(lIterNode1-1) = 1.0d;
                    }
                    else // If first nodal point
                    {
                        lGrad1.setOnes();
                        lGrad1 = -lGrad1;
                    }
                    
                    // Multiply gradient with inverse transform
                    lGrad1 = lF.solveTransposed(lGrad1); 
                    
                    // If lGrad1 is not zero
                    if ( (lGrad1.coeff(lDimOfInterest1) != 0.0d) || (lGrad1.coeff(lDimOfInterest2) != 0.0d) )
                    {
                        if ( (lGrad1.coeff(lDimOfInterest1) != 0.0d)  && (lGrad1.coeff(lDimOfInterest2) != 0.0d))
                            // Insert diagonals
                            updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                                lStandardVolume * lF.getAbsDeterminant() * lGrad1.coeff(lDimOfInterest1) * lGrad1.coeff(lDimOfInterest2),
                                lCurNode1*pNumNodes + lCurNode1,
                                lIterSimpl );
                    
                        // Loop through all nodes above
                        for (unsigned int lIterNode2 = lIterNode1+1; lIterNode2 < pTopD+1; lIterNode2++)
                        {
                            // Get current node
                            const unsigned int lCurNode2 = lCurNodeInds.coeff(lIterNode2);
                            // get gradient of current node  
                            Eigen::VectorXd lGrad2(pTopD);
                            lGrad2.setZero();
                            lGrad2(lIterNode2-1) = 1.0d;
                            // Multiply gradient with inverse transform
                            lGrad2 = lF.solveTransposed(lGrad2);                             
                            
                            // Insert off-diagonals
                            if ( (lGrad1.coeff(lDimOfInterest1) != 0.0d) && (lGrad2.coeff(lDimOfInterest2) != 0.0d) )
                                updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                                    lStandardVolume * lF.getAbsDeterminant() * lGrad1.coeff(lDimOfInterest1) * lGrad2.coeff(lDimOfInterest2),
                                    lCurNode1*pNumNodes + lCurNode2,
                                    lIterSimpl );
                            if ( (lGrad2.coeff(lDimOfInterest1) != 0.0d) && (lGrad1.coeff(lDimOfInterest2) != 0.0d) )
                                updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                                    lStandardVolume * lF.getAbsDeterminant() * lGrad2.coeff(lDimOfInterest1) * lGrad1.coeff(lDimOfInterest2),
                                    lCurNode2*pNumNodes + lCurNode1,
                                    lIterSimpl );
                        }
                    }
                }
            }
            // If U matrix
            else
            {            
                // Loop through all nodes
                for (unsigned int lIterNode1 = 0; lIterNode1 < pTopD+1; lIterNode1++)
                    // Insert
                    updateAndAdvanceSparseMatrix( pData, pRow, pCol, pDataIndex,
                        lOneInner * lF.getAbsDeterminant(),
                        lCurNodeInds.coeff(lIterNode1),
                        lIterSimpl );
            } // End of U
                
        }   // Stop looping through simplices
    
    
        return 0;
    }
    
    
    
    
        
}





