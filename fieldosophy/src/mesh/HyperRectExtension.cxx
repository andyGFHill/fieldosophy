/* 
* C/C++ functions for the hyper rectangular mesh class.
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
* Date: 2021-01
*/



#include "hyperRectExtension.hxx"
#include "misc.hxx"









// Define a hyper-rectangular extension of the mesh 
int HyperRectExtension::defineHyperRectExtension( const unsigned int pNumNewDims, const double * const pOffset, 
    const double * const pStepLengths, const unsigned int * const pNumSteps )
{
    // Populate variables
    mOffset.clear();
    mOffset.insert(mOffset.end(), pOffset, &pOffset[pNumNewDims]);
    mStepLengths.clear(); 
    mStepLengths.insert( mStepLengths.end(), pStepLengths, &pStepLengths[pNumNewDims]);
    mNumSteps.clear();
    mNumSteps.insert( mNumSteps.end(), pNumSteps, &pNumSteps[pNumNewDims]);
    
    // Update number of nodes
    for (std::vector<unsigned int>::const_iterator lIter = getNumSteps().begin(); lIter != getNumSteps().end(); ++lIter)
        mNN *= *lIter+1;
    // Update number of simplices
    mNT *= getNumCopiesPerLevel(mOffset.size());

    return 0;
}



// Get neighbors for simplex
int HyperRectExtension::getNeighborsFromSimplex( const unsigned int pSimplexInd, std::set<unsigned int> & pNeighs, std::set<unsigned int> & pTemp ) const
{
    // Get simplex projected onto implicit mesh
    const unsigned int lProjectedSimp = projectSimplex(pSimplexInd);
    // Get neighbors in implicit mesh
    int lStatus = getMesh().getNeighborsFromSimplex(lProjectedSimp, pNeighs);
    if (lStatus)
        return lStatus;
        
    // Go through all extended dimensions
    for (unsigned int lIterDims = 0; lIterDims < mOffset.size(); lIterDims++)
    {
        pTemp.clear();
        pTemp.insert( pNeighs.begin(), pNeighs.end() );
        pNeighs.clear();
        
        // Get multiplier of current level
        const unsigned int lMultiplier = getNumCopiesPerLevel( lIterDims ) * mMesh.getNT();
        // Get current dimensions level
        const unsigned int lCurLevel = pSimplexInd / lMultiplier;
        // Go through neighbors and add needed neighbor indices
        for (std::set<unsigned int>::iterator lIterNeighs = pTemp.begin(); lIterNeighs != pTemp.end(); ++lIterNeighs)
            pNeighs.insert( *lIterNeighs + lCurLevel * lMultiplier );
    }
    
    // Go through all extended dimensions
    std::vector<unsigned int>::const_iterator lIterNumSteps = getNumSteps().begin();
    for (unsigned int lIterDims = 0; lIterDims < mOffset.size(); lIterDims++)
    {
        // Get multiplier of current level
        const unsigned int lMultiplier = getNumCopiesPerLevel( lIterDims );
        // Get current dimensions level
        const unsigned int lCurLevel = pSimplexInd / (lMultiplier * mMesh.getNT());
        // If current level is not zero
        if (lCurLevel > 0)
            // Add neighbor below
            pNeighs.insert( pSimplexInd - mMesh.getNT() * lMultiplier );
        // If current level is not maximum
        if (lCurLevel+1 < *lIterNumSteps)
            // Add neighbor above
            pNeighs.insert( pSimplexInd + mMesh.getNT() * lMultiplier );
        
        // Increment iterator
        ++lIterNumSteps;
    }

    
    return 0;
}

// Get a simplex index for a simplex where points are part
int HyperRectExtension::getASimplexForPoint( double * const pPoints, const unsigned int pNumPoints,
    unsigned int * const pSimplexIds, double * const pBarycentricCoords,
    const double pEmbTol, const double * const pCenterOfCurvature, const unsigned int pNumCentersOfCurvature ) const
{
    // --- TODO ---
    // Handle when extended
    
    // So far only works for no extension of an implicit mesh
    int lStatus = getMesh().getASimplexForPoint( pPoints, pNumPoints, pSimplexIds, pBarycentricCoords, pEmbTol, pCenterOfCurvature, pNumCentersOfCurvature );
    if (lStatus)
        return lStatus;
        
    return 0;
}

// Get a set of all simplices for which the given point is a member.
int HyperRectExtension::getAllSimplicesForPoint( double * const pPoint, std::set<unsigned int> & pSimplexId,
    std::set<unsigned int> & pOutput,
    const double pEmbTol, const double * const pCenterOfCurvature ) const
{
    // --- TODO ---
    // So far only works for no extension of an implicit mesh
    int lStatus = getMesh().getAllSimplicesForPoint( pPoint, pSimplexId, pOutput, pEmbTol, pCenterOfCurvature );
    if (lStatus)
        return lStatus;
        
    return 0;
}


// Get a simplex index for a simplex where node is a part
int HyperRectExtension::getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const
{
    // --- TODO ---
    // So far only works for no extension of an implicit mesh
    int lStatus = getMesh().getASimplexForNode( pNodes, pNumNodes, pSimplexIds);
    if (lStatus)
        return lStatus;
        
    return 0;    
}


// Get a simplex index for a simplex where set is a part
int HyperRectExtension::getASimplexForSet( const std::set<unsigned int> & pSet, unsigned int & pSimplexId) const
{
    // --- TODO ---
    // So far only works for no extension of an implicit mesh
    int lStatus = getMesh().getASimplexForSet( pSet, pSimplexId);
    if (lStatus)
        return lStatus;
        
    return 0;    
}


int HyperRectExtension::getAllSimplicesForSet( const std::set<unsigned int> & pSet, std::set<unsigned int> & pSimplexId, std::set<unsigned int> & pOutput ) const
{
    // --- TODO ---
    // So far only works for no extension of an implicit mesh
    
    int lStatus = getMesh().getAllSimplicesForSet( pSet, pSimplexId, pOutput );
    if (lStatus)
        return lStatus;
        
    return 0;    
}

// Get a vector of chosen node    
int HyperRectExtension::getNode( const unsigned int pNodeInd, std::vector<double> & pOutput ) const
{
    // Clear output
    pOutput.clear();
    
    return getMesh().getNode(pNodeInd, pOutput);

    return 0;
}

// Get a set of chosen simplex
int HyperRectExtension::getSimplex( const unsigned int pSimplexInd, std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp ) const
{
    // --- TODO ---
    // So far not included actual hyper rect. Only implicitMesh functionality.

    return getMesh().getSimplex(pSimplexInd, pOutput, pTemp);
}

// Get diameter of simplex
double HyperRectExtension::getDiameter( const unsigned int pSimplexInd ) const
{
    // --- TODO ---
    // So far not included actual hyper rect. Only implicitMesh functionality.
    
    return getMesh().getDiameter( pSimplexInd );
}

// Is node part of simplex
bool HyperRectExtension::isNodePartOfSimplex(const unsigned int pNodeInd, const unsigned int pSimplexInd, std::set<unsigned int> & pSimplexVec, std::set<unsigned int> & pTemp1) const
{
    getSimplex( pSimplexInd, pSimplexVec, pTemp1 );
    for (std::set<unsigned int>::const_iterator lIter = pSimplexVec.begin(); lIter != pSimplexVec.end(); ++lIter)
        if (pNodeInd == *lIter)
            return true;
    return false;
}

// Is point part of simplex
int HyperRectExtension::isPointPartOfSimplex( double * const pPoint, const unsigned int pSimplexInd, double * const pStandardCoords, double * const pBarycentricCoords, 
    const double pEmbTol, const double * const pCenterOfCurvature, int * const pStatus ) const
{
    return mMesh.isPointPartOfSimplex( pPoint, pSimplexInd, pStandardCoords, pBarycentricCoords, pEmbTol, pCenterOfCurvature, pStatus );
}



// Global list storing HyperRectExtension
GlobalVariablesList<HyperRectExtension> gHyperRectExtension;




// Load internally stored mesh
const HyperRectExtension * hyperRectExtension_load( const unsigned int pId) { return gHyperRectExtension.load(pId); }


extern "C"
{

    // Create HyperRectExtension mesh
    int hyperRectExtension_createMesh( const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        const double* const pOffsetImplicit, const unsigned int * const pCopiesPerDimension,
        const double* const pOffsetHyper, const double * const pStepLengths, const unsigned int * const pNumSteps, const unsigned int pExtendDims,
        unsigned int * const pId, unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices, const unsigned int * const pNeighs )
    {
        // If submanifold
        if (pTopD != pD)
            // Return failure
            return 1;
    
        // Create internal representation of mesh
        ConstMesh lConstMesh( pNodes, pD, pNumNodes, pSimplices, pNumSimplices, pTopD, pNeighs );
        // Create implicit mesh from ConstMesh
        ImplicitMesh lImplicitMesh(lConstMesh);
        if (pOffsetImplicit != NULL && pCopiesPerDimension != NULL)
        {
            // Extend implicit mesh
            int lImplicitStatus = lImplicitMesh.defineExtensionInCurrentDims( pOffsetImplicit, pCopiesPerDimension );
            // If failed
            if (lImplicitStatus)
                // Return error
                return 2;
        }
        
    
        // Create HyperRectExtension and store it
        *pId = gHyperRectExtension.store( HyperRectExtension(lImplicitMesh) );
        HyperRectExtension * const lMesh =  gHyperRectExtension.load( *pId );
        if (lMesh == NULL)
            return 3;
        
        if (pExtendDims > 0)
        {
            // Define a hyper-rectangular extension of the mesh 
            int lStatus = lMesh->defineHyperRectExtension( pExtendDims, pOffsetHyper, pStepLengths, pNumSteps );
            if (lStatus)
                return 3 + lStatus;
        }
            
        // Get number of nodes and simplices
        *pNewNumNodes = lMesh->getNN();
        *pNewNumSimplices = lMesh->getNT();
    
        return 0;
    }

    // Remove mesh
    int hyperRectExtension_eraseMesh( const unsigned int pId)
    {
        // Load implicit mesh from storage
        return gHyperRectExtension.erase(pId);
    }
    
    
}   // End of extern "C"






