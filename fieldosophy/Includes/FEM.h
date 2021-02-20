/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code Form is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef FEM_H
#define FEM_H


extern "C"
{       
    // Computes inner products over a standard triangle
    int FEM_compInProdOverStandTri( const unsigned int pD, double * const pVolume, double * const pOneInner, 
        double * const pMDiag, double * const pMOffDiag );
    
    
    // Maps triangle values to matrices
    int FEM_mapTrivals2Mat(
        double * const pData, unsigned int * const pRow, unsigned int * const pCol, unsigned int * const pDataIndex,
        const unsigned int pMatType, const unsigned int pNumNonZeros,
        const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD
        );            
}





#endif // FEM_H