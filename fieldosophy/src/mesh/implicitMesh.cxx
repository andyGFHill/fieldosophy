/* 
* C/C++ functions for the implicit mesh class.
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


#include <omp.h>
#include "implicitMesh.hxx"
#include "misc.hxx"







// Define an extension of a mesh occupying a hyperrectangular region (implicitly connecting extreme points in each dimension)
int ImplicitMesh::defineExtensionInCurrentDims( const double * const pOffset, const unsigned int * const pNumCopiesPerDim )
{

    // Clear former information
    mCopiesPerDimension.clear();
    mOffset.clear();
    
    // Make sure that not a submanifold
    if (mMesh.getTopD() != mMesh.getD())
        return 1;
    // Make sure that explicit mesh is extendable
    if (mPairing.size() == 0)
        return 2;
    
    // Loop through each dimension
    for (unsigned int lIterDims = 0; lIterDims < mMesh.getD(); lIterDims++ )
    {
        // Insert current dimensions copies in vector
        mCopiesPerDimension.push_back( pNumCopiesPerDim[lIterDims] );
        // Set offset in current dimension
        mOffset.push_back(pOffset[lIterDims]); // offset in each dimension
    }

    // Start by assuming the number of simplices and nodes of explicit mesh
    mNT = mMesh.getNT();
    // Loop through each dimension
    for (std::vector<unsigned int>::const_iterator lIterCopies = mCopiesPerDimension.begin();
        lIterCopies != mCopiesPerDimension.end(); ++lIterCopies )
        // Multiply with current dimensions number of multiplications
        mNT *= *lIterCopies;

    // Initialize number of nodes to zero
    mNN = mMesh.getNN() * getNumSectors();
    
    return 0;
}



// Analyzes ConstMesh and compute bounding box and boundingNodes
int ImplicitMesh::populateBoundingBox()
{
    // Clear bounding box
    mBoundingBox.clear();
    // Clear bounding nodes
    mBoundingNodes.clear();
    // Clear bounding simplices
    mBoundingSimplices.clear();
    // Clear pairing
    mPairing.clear();
    // Clear pairing simplices
    mPairingSimplices.clear();
    
    // Loop through all nodes in constMesh to fill up bounding box
    const double * lIterNodes = mMesh.getNodes();
    for (unsigned int lIter = 0; lIter < mMesh.getNN(); lIter++)
    {
        // Loop through all dimensions
        for (unsigned int lIterDim = 0; lIterDim < mMesh.getD(); lIterDim++)
        {
            // Get current node value
            const double lCurNodeVal = lIterNodes[lIterDim];
            
            // If bounding box is not preallocated
            if (mBoundingBox.size() <= lIterDim)
                // Insert current node in both back and front side
                mBoundingBox.push_back( std::pair<double, double>( lCurNodeVal, lCurNodeVal) );
            else
            {
                if ( mBoundingBox.at(lIterDim).first > lCurNodeVal )
                    mBoundingBox.at(lIterDim).first = lCurNodeVal;
                if ( mBoundingBox.at(lIterDim).second < lCurNodeVal )
                    mBoundingBox.at(lIterDim).second = lCurNodeVal;
            }
            
            // If bounding nodes is not preallocated yet
            if (mBoundingNodes.size() <= lIterDim)
                // Preallocate
                mBoundingNodes.push_back( std::pair<std::set<unsigned int>, std::set<unsigned int> >(std::set<unsigned int>(), std::set<unsigned int>()) );
            
        }
        // Advance node counter
        lIterNodes = &lIterNodes[mMesh.getD()];
    }
    
    // Loop through all nodes again to fill up bounding nodes
    lIterNodes = mMesh.getNodes();
    for (unsigned int lIter = 0; lIter < mMesh.getNN(); lIter++)
    {
        // Loop through all dimensions
        for (unsigned int lIterDim = 0; lIterDim < mMesh.getD(); lIterDim++)
        {
            // Get current node value
            const double lCurNodeVal = lIterNodes[lIterDim];
            
            // If current node is part of front side
            if ( mBoundingBox[lIterDim].first == lCurNodeVal )
                // Add its index to bounding nodes for current dimensions front side
                mBoundingNodes[lIterDim].first.insert(lIter);
            // If current node is part of back side
            if ( mBoundingBox[lIterDim].second == lCurNodeVal )
                // Add its index to bounding nodes for current dimensions back side
                mBoundingNodes[lIterDim].second.insert(lIter);
        }
        // Advance node counter
        lIterNodes = &lIterNodes[mMesh.getD()];
    }
    
    // Loop through all simplices to find bounding simplices
    const unsigned int * lIterSimplices = mMesh.getSimplices();
    for (unsigned int lIter = 0; lIter < mMesh.getNT(); lIter++)
    {
        // Loop through all dimensions
        for (unsigned int lIterDim = 0; lIterDim < mMesh.getD(); lIterDim++)
        {
            // Initialize number of nodes in front
            unsigned int lNumInFront = 0;
            // Initialize number of nodes in back
            unsigned int lNumInBack = 0;
            // Loop through all nodes in current simplex
            for (unsigned int lIterNode = 0; lIterNode < mMesh.getTopD()+1; lIterNode++)
            {
                // Get current node value
                const unsigned int lCurNodeInd = lIterSimplices[lIterNode];
                // If current node is part of front side
                if ( mBoundingNodes[lIterDim].first.count(lCurNodeInd) )
                    // Add number of nodes in front
                    lNumInFront++;
                // If current node is part of back side
                if ( mBoundingNodes[lIterDim].second.count(lCurNodeInd) )
                    // Add number of nodes in back
                    lNumInBack++;
            }
            
            // If current dimension does not exist yet for mBoundingSimplices
            if (mBoundingSimplices.size() <= lIterDim)
                // Create it
                mBoundingSimplices.push_back( std::pair<std::set<unsigned int>, std::set<unsigned int> >(std::set<unsigned int>(), std::set<unsigned int>()) );
            
            // If current simplex has an edge on the front boundary
            if (lNumInFront >= mMesh.getTopD())
                mBoundingSimplices.at(lIterDim).first.insert(lIter);
            // If current simplex has an edge on the back boundary
            if (lNumInBack >= mMesh.getTopD())
                mBoundingSimplices.at(lIterDim).second.insert(lIter);
        }

        // Advance simplex counter
        lIterSimplices = &lIterSimplices[mMesh.getTopD()+1];
    }
    
    
    // If on embedded manifold
    if ( mMesh.getD() != mMesh.getTopD() )
    {
        // Clear all pairings to mark that explicit mesh is not extendable
        mPairing.clear();
        // Return from function signalling success
        return 0;
    }
    
    // Loop through all dimensions to find pairings
    std::vector<double> lNode;
    lNode.reserve(mMesh.getD());
    for (unsigned int lIterDim = 0; lIterDim < mMesh.getD(); lIterDim++)
    {
        // Initialize current dimension
        mPairing.push_back( std::vector<std::pair<unsigned int, unsigned int>>() );
        // Acquire set of all nodes in front in current dimension
        std::set<unsigned int> lCurFront = getBoundingNodes().at(lIterDim).first;
        // Acquire set of all nodes in back in current dimension
        std::set<unsigned int> lCurBack = getBoundingNodes().at(lIterDim).second;
        
        // Loop as long as the front set is not empty
        while ( lCurFront.size() > 0 )
        {
            // Get current front node
            const unsigned int lCurFrontNodeIndex = *lCurFront.begin();
            // pop front node
            lCurFront.erase(lCurFront.begin());
            // Get node coordinate of current front node if it would have been a back node
            int lStatus = getNode(lCurFrontNodeIndex, lNode);
            if (lStatus)
                return 1;
            lNode[lIterDim] = getBoundingBox()[lIterDim].second;
            // Declare flag of if pairing has been found
            bool lFoundPairing;
            // Loop through all back nodes
            for ( std::set<unsigned int>::const_iterator lIterBack = lCurBack.begin(); lIterBack != lCurBack.end(); ++lIterBack)
            {                
                // Assume pairing will be found
                lFoundPairing = true;
                
                // Compare real back node with theoretical
                for (unsigned int lIterDim2 = 0; lIterDim2 < mMesh.getD(); lIterDim2++)
                    // If did not find pairing
                    if ( mMesh.getNodes()[ *lIterBack * getD() + lIterDim2 ] != lNode[lIterDim2])
                    {
                        // Set that pairing was not found
                        lFoundPairing = false;
                        // Get out of loop
                        break;
                    }
                // If did find pairing
                if (lFoundPairing)
                {
                    // Insert pairing
                    mPairing.back().push_back( std::pair<unsigned int, unsigned int>( lCurFrontNodeIndex, *lIterBack ) );
                    // Pop current back node
                    lCurBack.erase(lIterBack);
                    // Get out of loop
                    break;
                }
            }
            // If did not find pairing
            if (!lFoundPairing)
            {
                // Clear all pairings to mark that explicit mesh is not extendable
                mPairing.clear();
                // Return from function signalling success
                return 0;
            }
        }   // End of loop through all in current front
    }   // End of loop through all dimensions
    
    // Loop through all dimensions to find pairings of simplices
    for (unsigned int lIterDim = 0; lIterDim < mMesh.getD(); lIterDim++)
    {
        // Initialize current dimension
        mPairingSimplices.push_back( std::vector<std::pair<unsigned int, unsigned int>>() );
        // Acquire set of all simplices in front in current dimension
        std::set<unsigned int> lCurFront = getBoundingSimplices().at(lIterDim).first;
        // Acquire set of all simplices in back in current dimension
        std::set<unsigned int> lCurBack = getBoundingSimplices().at(lIterDim).second;
        
        // Loop as long as the front set is not empty
        while ( lCurFront.size() > 0 )
        {
            // Get current front node
            const unsigned int lCurFrontSimplexIndex = *lCurFront.begin();
            // pop front simplex
            lCurFront.erase(lCurFront.begin());
            // Initialize found pairing as false
            bool lFoundPairing = false;
            
            // Loop through all back simplices
            for ( std::set<unsigned int>::const_iterator lIterBack = lCurBack.begin(); lIterBack != lCurBack.end(); ++lIterBack)
            {                
                // Initialize number of nodes incommon
                unsigned int lNumNodesPaired = 0;
                // Go through each front node
                for (unsigned int lIterNodeFront = 0; lIterNodeFront < mMesh.getTopD()+1; lIterNodeFront++)
                {
                    // Get current front node index
                    const unsigned int lCurFrontNodeInd = mMesh.getSimplices()[ lCurFrontSimplexIndex * (mMesh.getTopD()+1) + lIterNodeFront ];
                    // Loop through all node pairings
                    for ( std::vector<std::pair<unsigned int, unsigned int>>::const_iterator lIterPairings = mPairing.at(lIterDim).begin(); 
                        lIterPairings != mPairing.at(lIterDim).end(); ++lIterPairings)
                        // See if current front node index is in current pairing
                        if ( lIterPairings->first == lCurFrontNodeInd )
                        {
                            // Go through each back node index
                            for (unsigned int lIterNodeBack = 0; lIterNodeBack < mMesh.getTopD()+1; lIterNodeBack++)
                            {
                                // Get current front node index
                                const unsigned int lCurBackNodeInd = mMesh.getSimplices()[ *lIterBack * (mMesh.getTopD()+1) + lIterNodeBack ];
                                // See if current back node index is in current pairing
                                if ( lIterPairings->second == lCurBackNodeInd )
                                {
                                    // Increment number of pairings
                                    lNumNodesPaired++;
                                    break;
                                }
                            }
                            break;
                        }
                }
                // If number of pairings are enough to share edge
                if (lNumNodesPaired >= mMesh.getTopD())
                {
                    // Insert pairing of simplices
                    mPairingSimplices.back().push_back( std::pair<unsigned int, unsigned int>( lCurFrontSimplexIndex, *lIterBack ) );
                    // Pop current back node
                    lCurBack.erase(lIterBack);
                    // Mark that pairing was found
                    lFoundPairing = true;
                    // Get out of loop
                    break;
                }
            }   // End of loop over all node pairings
            // If did not find pairing
            if (!lFoundPairing)
            {
                // Clear all pairings to mark that explicit mesh is not extendable
                mPairing.clear();
                mPairingSimplices.clear();
                // Return from function signalling success
                return 0;
            }
        }   // End of loop through all in current front
    }   // End of loop through all dimensions

    return 0;
}

int ImplicitMesh::getNode(const unsigned int pNodeIndex, std::vector<double> & pOutput) const
{
    // Clear output
    pOutput.clear();
    
    // If out of bounds
    if (pNodeIndex >= getNN())
        // Return empty
        return 1;
        
    // Get sector and explicit index of node
    unsigned int lSector;
    unsigned int lExplicitInd;    
    int lStatus = nodeInd2SectorAndExplicitInd( pNodeIndex, lSector, lExplicitInd );
    if (lStatus)
        // Return empty
        return 1 + lStatus;
        
    // Initialize output as explicit node
    pOutput.insert( pOutput.end(), &mMesh.getNodes()[lExplicitInd * getD()], &mMesh.getNodes()[(lExplicitInd+1) * getD()] );
    
    // If only explicit
    if (!isExtended())
        return 0;
    
    // Go through all dimensions
    std::vector<double>::const_iterator lIterOffset = mOffset.begin();
    std::vector<double>::iterator lIterOut = pOutput.begin();
    std::vector<unsigned int>::const_iterator lIterCopies = mCopiesPerDimension.begin();
    std::vector< std::pair<double, double> >::const_iterator lIterBoundingBox = mBoundingBox.begin();
    for (unsigned int lIterDims = 0; lIterDims < getD(); lIterDims++)
    {
        // Get length of a sector in current dimension
        const double lLength = lIterBoundingBox->second - lIterBoundingBox->first;
        // Get which interval in current dimension
        const unsigned int lSectorCurDim = lSector % *lIterCopies;
        const double lCurOffset = *lIterOffset + lSectorCurDim * lLength;
        // Adjust output
        *lIterOut += lCurOffset;
        
        // Update sector for next dimension
        lSector = lSector / *lIterCopies;
        
        // Update iterators
        ++lIterOffset;
        ++lIterOut;
        ++lIterCopies;
    }
    
    return 0;
}

int ImplicitMesh::getSimplex(const unsigned int pSimplexIndex, std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp) const
{
    // Clear output
    pOutput.clear();
    pTemp.clear();
    
    // If out of bounds
    if (pSimplexIndex >= getNT())
        // Return empty
        return 1;

    // Get sector number and explicit simplex index
    unsigned int lSector = simplexInd2Sector(pSimplexIndex);
    unsigned int lExplicitSimplex = simplexInd2ExplicitInd(pSimplexIndex);
    // If out of bounds
    if ( (lSector > getNN() ) || (lExplicitSimplex > mMesh.getNT()) )
        // Return empty vector to flag error
        return 2;
    
    // Initialize output from explicit simplex
    pTemp.insert( &mMesh.getSimplices()[ lExplicitSimplex * (mMesh.getTopD()+1) ], &mMesh.getSimplices()[ (lExplicitSimplex+1) * (mMesh.getTopD()+1) ] );
    // Go through vector elements
    for ( std::set<unsigned int >::iterator lIter = pTemp.begin(); lIter != pTemp.end(); ++lIter  )
    {
        // Get current node index
        unsigned int lCurNodeIndex;
        int lStatus = sectorAndExplicit2NodeInd( lSector, *lIter, lCurNodeIndex );
        if (lStatus)
            // Return empty vector to flag error
            return 3;
        // Update element in output
        pOutput.insert(lCurNodeIndex);
    }
    if (pOutput.size() != mMesh.getTopD()+1)
        return 4;
    
    return 0;
}

// Get diameter of simplex
double ImplicitMesh::getDiameter( const unsigned int pSimplexInd ) const
{
    if ( pSimplexInd >= getNT() )
        return -1.0d;
    // Get explicit simplex
    const unsigned int lExplicitSimplex = simplexInd2ExplicitInd(pSimplexInd);
    // Get diameter of explicit simplex    
    return getConstMesh().getDiameter( lExplicitSimplex );
}





int ImplicitMesh::nodeInd2SectorAndExplicitInd( const unsigned int pNodeInd, unsigned int & pSector, unsigned int & pExplicitInd ) const
{
    if (pNodeInd >= getNN())
        return 2;
    pSector = pNodeInd / mMesh.getNN();
    pExplicitInd = pNodeInd % mMesh.getNN();

    return 0;
}

int ImplicitMesh::sectorAndExplicit2NodeInd( unsigned int pSector, const unsigned int pExplicitInd, unsigned int & pNodeInd ) const
{
    // Initialize
    pNodeInd = pExplicitInd;
    
    // If no extension
    if (!isExtended())
    {
        if (pSector > 0)
            return 1;
        else
            return 0;
    }

    // Handle errors
    if (pExplicitInd >= mMesh.getNN())
        return 1;
    if (pSector >= getNumCopiesPerSector(getD()) )
        return 1;
        
    // Loop through dimensions
    std::vector< std::pair< std::set<unsigned int>, std::set<unsigned int> > >::const_iterator lIterBoundingNodes = getBoundingNodes().begin();
    std::vector< std::vector<std::pair<unsigned int, unsigned int>> >::const_iterator lIterPairing = mPairing.begin();
    for (unsigned int lIterDims = 0; lIterDims < getD(); lIterDims++)
    {
        // Get current dimensions sector
        unsigned int lCurDimSector = getCurDimSector( pSector, lIterDims );
    
        // If sector is not the first in current dimension
        if ( lCurDimSector > 0 )
            // If current explicitInd is in front
            if (lIterBoundingNodes->first.count(pNodeInd))
            {
                // Move back to former sector
                lCurDimSector -= 1;
                pSector -= getNumCopiesPerSector( lIterDims );                                
                // Find pairing
                for ( std::vector<std::pair<unsigned int, unsigned int>>::const_iterator lIterPairing2 = lIterPairing->begin(); 
                    lIterPairing2 != lIterPairing->end(); ++lIterPairing2)
                    if (lIterPairing2->first == pNodeInd)
                    {
                        // Switch explicit ind to back
                        pNodeInd = lIterPairing2->second;
                        break;
                    }
            }
            
        // Increment
        ++lIterBoundingNodes;
        ++lIterPairing;
    }
    
    pNodeInd = pSector * mMesh.getNN() + pNodeInd;
    

    return 0;
}



unsigned int ImplicitMesh::getNumInteriorPoints() const
{
    unsigned int lInteriorPoints = mMesh.getNN();
    std::set<unsigned int> lOldSet, lSet;
    for (unsigned int lIterDims = 0; lIterDims < getD(); lIterDims++)
    {
        const std::set<unsigned int> & lFirstNodes = getBoundingNodes()[lIterDims].first;
        misc_setIntersection<unsigned int>( lFirstNodes, lOldSet, lSet);                
        const unsigned int lCurNumNodes = lFirstNodes.size() - lSet.size();
        lInteriorPoints -= lCurNumNodes;
        lOldSet.insert( lFirstNodes.begin(), lFirstNodes.end() );
    }
    return lInteriorPoints;
}



// Get sector of point
unsigned int ImplicitMesh::point2Sector( const double * const pPoint ) const
{
    // If no extension
    if (!isExtended())
    {
        return 0;
    }

    // Initialize sector
    unsigned int lSector = 0;
    // Go through dimensions
    for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
    {
        // Get current point element
        const double lCurVal = pPoint[lIterDim];
        // Get length of bounding box in current dimension
        const double lLength = getBoundingBoxLength(lIterDim);
        // Get in which sector of current dimension the point is in
        double lTemp = lCurVal - mOffset[lIterDim];
        if (lTemp < 0)
            return getNT();
        lTemp /= lLength;
        // If no implicit extension 
        if (!isExtended()) 
        {
            if (lTemp == 1.0)
                return 0;
            else
                return getNT();
        }
        // Get truncated value
        unsigned int lCurDimSector = lTemp;
        // If just out of bounds                
        if ( (lCurDimSector == mCopiesPerDimension[lIterDim]) && (lTemp == (double)lCurDimSector) )
            // Decrement current dimension
            lCurDimSector--;

        lSector += lCurDimSector * getNumCopiesPerSector(lIterDim);
    }
    return lSector;
}

// Get neighbors for simplex
int ImplicitMesh::getNeighborsFromSimplex(const unsigned int pSimplexInd, std::set<unsigned int> & pNeighs) const
{
    // If simplex is out of bounds
    if ( pSimplexInd >= getNT() )
        // Return failure
        return 1;
    // If no neighbors exist
    if (mMesh.getNeighs() == NULL)
        // Return no neighbor
        return 2;
 
    // Clear output
    pNeighs.clear();
    // Get current sector
    const unsigned int lSector = simplexInd2Sector( pSimplexInd );
    // Get explicit simplex index
    const unsigned int lExplicitSimp = simplexInd2ExplicitInd( pSimplexInd );
    // Get explicit neghbors of explicit simplex
    const unsigned int * const lExplicitNeighs = &mMesh.getNeighs()[lExplicitSimp * (getTopD()+1)];
    
    // Flag that no simplices outside of sector has been added
    unsigned int lSide = 0;
    
    // Go through all neighbors
    for (unsigned int lIterNeighs = 0; lIterNeighs < getTopD()+1; lIterNeighs++)
    {
        // Get current explicit neighbor
        const unsigned int lExplicitNeigh = lExplicitNeighs[lIterNeighs];
        // If neighbor exist in explicit mesh
        if (lExplicitNeigh < mMesh.getNT())    
        {
            // Get implicit simplex
            const unsigned int lImplicitSimp = sectorAndExplicit2SimplexInd( lSector, lExplicitNeigh );
            // If error
            if (lImplicitSimp >= getNT())
                return 3;
            // Insert simplex
            pNeighs.insert( lImplicitSimp );
        }
        else // If neighbor does not exist in explicit mesh
            // If mesh is extended
            if ( isExtended() )
                // Get which side to start investigate (from front of lowest dimension and forward)
                for (unsigned int lIterSide = lSide; lIterSide < 2 * getD(); lIterSide++)
                {
                    // Get current dimension
                    const unsigned int lCurDim = lIterSide / 2;
                    // Get if front or in back
                    const bool lInFront = (lIterSide % 2 == 0);
                    // Get sector index in current dimension
                    const unsigned int lCurSector = getCurDimSector( lSector, lCurDim );
                    // If the sector should have a neighboring sector in this dimension
                    if ( ( (lCurSector > 0) && lInFront ) || ( (lCurSector+1 < mCopiesPerDimension[lCurDim]) && !lInFront ) )
                    {
                        // Get sector of potential neighbor
                        unsigned int lNeighSector = lSector;
                        if (lInFront)
                            lNeighSector -=  getNumCopiesPerSector( lCurDim );
                        else
                            lNeighSector +=  getNumCopiesPerSector( lCurDim );
                        
                        // Loop through paired simplices for current border
                        for ( std::vector<std::pair<unsigned int, unsigned int>>::const_iterator lIterPairings = mPairingSimplices[lCurDim].begin(); 
                            lIterPairings != mPairingSimplices[lCurDim].end(); ++lIterPairings)
                        {
                            // Get potential matching of explicit simplex
                            const unsigned int lPotentialExplicitSimp = lInFront ? lIterPairings->first : lIterPairings->second;
                            
                            // If current pairing includes the current explicit simplex
                            if ( lPotentialExplicitSimp == lExplicitSimp )
                            {
                                // Get explicit neighbor index
                                const unsigned int lExplicitNeigh = lInFront ? lIterPairings->second : lIterPairings->first;
                                // Get implicit simplex
                                const unsigned int lImplicitSimp = sectorAndExplicit2SimplexInd( lNeighSector, lExplicitNeigh );
                                // If error
                                if (lImplicitSimp >= getNT())
                                    return 3;
                                // Insert simplex
                                pNeighs.insert( lImplicitSimp );
                                // Insert that current side has been handled
                                lSide = lIterSide+1;
                                // Break out of loop
                                break;
                            }
                        }
                    }
                }
            

    }   // End of loop through all neighbors
    
    return 0;
}

// Get a simplex index for a simplex where points are part
int ImplicitMesh::getASimplexForPoint( double * const pPoints, const unsigned int pNumPoints, 
    unsigned int * const pSimplexIds, double * const pBarycentricCoords, 
    const double pEmbTol, const double * const pCenterOfCurvature, const unsigned int pNumCentersOfCurvature) const
{    
    int lStatus = 0;

    // If no implicit extension
    if (!isExtended())
        // run through ConstMesh equivalent to acquire explicit simplices for all points
        return mMesh.getASimplexForPoint( pPoints, pNumPoints, pSimplexIds, pBarycentricCoords, pEmbTol, pCenterOfCurvature, pNumCentersOfCurvature );

    // Loop through all points to translate them to explicit sectors
    #pragma omp parallel for reduction(|:lStatus)
    for (unsigned int lIterPoints = 0; lIterPoints < pNumPoints; lIterPoints++)
    {
        // If error
        if (lStatus != 0)
            continue;
    
        double * const lPointPtr = &pPoints[lIterPoints * getD()];
        unsigned int * const lSimpPtr = &pSimplexIds[lIterPoints];
        double * const lBarycentricPtr = &pBarycentricCoords[ lIterPoints * (getTopD()+1) ];
    
        // Get sector of current point
        const unsigned int lSector = point2Sector( lPointPtr );
        if (lSector == getNT())
            lStatus = 1;
        // Go through each dimension and translate to sector 0
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
            lPointPtr[lIterDim] -= mOffset[lIterDim] + ((double)getCurDimSector( lSector, lIterDim )) * getBoundingBoxLength(lIterDim);
        // run through ConstMesh equivalent to acquire explicit simplices for all points
        lStatus = mMesh.getASimplexForPoint( lPointPtr, 1, lSimpPtr, lBarycentricPtr, pEmbTol, pCenterOfCurvature, pNumCentersOfCurvature );
        // Go through each dimension and translate to original sector again
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
            lPointPtr[lIterDim] += mOffset[lIterDim] + ((double)getCurDimSector( lSector, lIterDim )) * getBoundingBoxLength(lIterDim);
        // Handle errors
        if (lStatus)
            lStatus = 1+lStatus;
        // Update from explicit simplex to implicit simplex
        *lSimpPtr = sectorAndExplicit2SimplexInd( lSector, *lSimpPtr );
        if (lStatus)
            *lSimpPtr = getNT();
    }
    
    return 0;
}

// Get a set of all simplices for which the given point is a member.
int ImplicitMesh::getAllSimplicesForPoint( double * const pPoint, std::set<unsigned int> & pSimplexId,
    std::set<unsigned int> & pOutput,
    const double pEmbTol, const double * const pCenterOfCurvature ) const
{
    // Clear output
    pOutput.clear();
    std::set<unsigned int> lExplicitSimp, lExcludeSimp;
    int lStatus = 0;

    // If no implicit extension
    if (!isExtended())
        // Get all explicit simplices for point
        return mMesh.getAllSimplicesForPoint( pPoint, *pSimplexId.begin(), pOutput, pEmbTol, pCenterOfCurvature );
    
    while ( pSimplexId.size() > 0 )
    {
        // Pop front
        const unsigned int lSimplexId = *pSimplexId.begin();
        pSimplexId.erase(pSimplexId.begin());
        if (lSimplexId >= getNT())
            return 2;
            
        // If simplex should be excluded
        if ( lExcludeSimp.count( lSimplexId ) )
            // Continue
            continue;
        lExcludeSimp.insert( lSimplexId );
            
        // Acquire current simplex sector
        const unsigned int lSector = simplexInd2Sector(lSimplexId);    
        // Acquire explicit simplex
        const unsigned int lExplicitSimplex = simplexInd2ExplicitInd( lSimplexId );
        // Go through each dimension and translate to sector 0
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
            pPoint[lIterDim] -= mOffset[lIterDim] + ((double)getCurDimSector( lSector, lIterDim )) * getBoundingBoxLength(lIterDim);
        // Get all explicit simplices for point
        int lStatus = mMesh.getAllSimplicesForPoint( pPoint, lExplicitSimplex, lExplicitSimp, pEmbTol, pCenterOfCurvature  );
        if (lStatus)
            return 3;
        
        // Go through each dimension 
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
        {
            // Get current dimension sector
            const unsigned int lCurDimSector = getCurDimSector( lSector, lIterDim );
            // Get current bounding box
            const std::pair<double, double> & lCurBoundingBox = getBoundingBox()[lIterDim];
            // Get if in front or back
            const bool lIsFront = (pPoint[lIterDim] <= lCurBoundingBox.first);
            const bool lIsBack = (pPoint[lIterDim] >= lCurBoundingBox.second);
            
            // If (not the first and part of front) or (if not the last and part of back)
            if ( ( (lCurDimSector > 0) && lIsFront ) || ( (lCurDimSector + 1 < mCopiesPerDimension[lIterDim]) && lIsBack ) )
            {
                // Acquire neighboring sector
                const unsigned int lNeighborSector = lIsFront ? lSector - getNumCopiesPerSector(lIterDim) : lSector + getNumCopiesPerSector(lIterDim);
                // Go through all explicit simplices
                for ( std::set<unsigned int>::const_iterator lIterSimplices = lExplicitSimp.begin(); lIterSimplices != lExplicitSimp.end(); ++lIterSimplices )
                {
                    // Get paired simplex    
                    const unsigned int lPairedExplicitSimplex = findExplicitPairedSimplex(*lIterSimplices, lIterDim, lIsFront);
                    // If found a pairing
                    if ( lPairedExplicitSimplex < mMesh.getNT() )
                    {
                        // Get potential new index
                        const unsigned int lPotentialNewIndex = sectorAndExplicit2SimplexInd( lNeighborSector, lPairedExplicitSimplex );
                        // If not already present in output
                        if (pOutput.count(lPotentialNewIndex) == 0)
                            // Add to simplex set
                            pSimplexId.insert(lPotentialNewIndex);
                        // break out of loop
                        break;
                    }
                }
                // If no explicit simplices, use explicit simplex instead
                if ( lExplicitSimp.size() == 0 )
                {
                    // Get paired simplex    
                    const unsigned int lPairedExplicitSimplex = findExplicitPairedSimplex( lExplicitSimplex, lIterDim, lIsFront);
                    // If found a pairing
                    if ( lPairedExplicitSimplex < mMesh.getNT() )
                    {
                        // Get potential new index
                        const unsigned int lPotentialNewIndex = sectorAndExplicit2SimplexInd( lNeighborSector, lPairedExplicitSimplex );
                        // If not already present in output
                        if (pOutput.count(lPotentialNewIndex) == 0)
                            // Add to simplex set
                            pSimplexId.insert(lPotentialNewIndex);
                    }
                }
            }   // end of if front or back
            
            // translate point back to original sector    
            pPoint[lIterDim] += mOffset[lIterDim] + ((double)lCurDimSector) * getBoundingBoxLength(lIterDim);        
        }   // End of loop through dimensions
        
        // Go through explicit simplices and transform from explicit to implicit simplices
        for ( std::set<unsigned int>::const_iterator lIterSimplices = lExplicitSimp.begin(); lIterSimplices != lExplicitSimp.end(); ++lIterSimplices )
            pOutput.insert( sectorAndExplicit2SimplexInd( lSector, *lIterSimplices ) );
    }   // End of loop while pSimplexId is not empty
    
    
    return 0;
}

// Is point part of simplex
int ImplicitMesh::isPointPartOfSimplex( double * const pPoint, const unsigned int pSimplexInd, 
    double * const pStandardCoords, double * const pBarycentricCoords, 
    const double pEmbTol, const double * const pCenterOfCurvature, int * const pStatus ) const
{
    // Get sector of current point
    const unsigned int lSector = point2Sector( pPoint );
    if (lSector == getNT())
        return 1;
    // Get explicit simplex
    const unsigned int lSimplexInd = simplexInd2ExplicitInd( pSimplexInd );
        
    // Go through each dimension and translate to sector 0
    if (lSector != 0)
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
            pPoint[lIterDim] -= mOffset[lIterDim] + ((double)getCurDimSector( lSector, lIterDim )) * getBoundingBoxLength(lIterDim);        
    // Get standard- and/or barycentric coordinates for point given simplex
    const unsigned int lNumCentersOfCurvature = (pCenterOfCurvature == NULL) ? 0 : 1;
    double lBaryStatus;
    int lErrorStatus = mMesh.getCoordinatesGivenSimplex( pPoint, 1, lSimplexInd,
        pStandardCoords, pBarycentricCoords, pEmbTol, pCenterOfCurvature, lNumCentersOfCurvature, &lBaryStatus);
    if (pStatus != NULL)
    {
        *pStatus = 1;
    
        if (lBaryStatus < 0)
            *pStatus = -1;
        else if (lBaryStatus == 0)
            *pStatus = 0;
    }
    
        
    // Go through each dimension and translate to original sector again
    if (lSector != 0)
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
            pPoint[lIterDim] += mOffset[lIterDim] + ((double)getCurDimSector( lSector, lIterDim )) * getBoundingBoxLength(lIterDim);
    if (lErrorStatus)
        return lErrorStatus;
        
    return 0;
}

// Get a simplex index for a simplex where node is a part
int ImplicitMesh::getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const
{   
    int lStatus = 0;
 
    // If no implicit extension
    if (!isExtended())
        // run through ConstMesh equivalent to acquire explicit simplices for all points
        return mMesh.getASimplexForNode( pNodes, pNumNodes, pSimplexIds );

    // Loop through all nodes
    #pragma omp parallel for reduction(|:lStatus)
    for (unsigned int lIterNodes = 0; lIterNodes < pNumNodes; lIterNodes++)
    {
        // If error
        if (lStatus != 0)
            continue;
    
        // Get current node index
        const unsigned int lCurNodeInd = pNodes[lIterNodes];
        // Get explicit node and sector
        unsigned int lExplicitInd;
        unsigned int lSector;
        lStatus = nodeInd2SectorAndExplicitInd( lCurNodeInd, lSector, lExplicitInd );
        if (lStatus)
        {
            lStatus = 1;
            continue;
        }
        // run through ConstMesh equivalent to acquire explicit simplices for node
        lStatus = mMesh.getASimplexForNode( &lExplicitInd, 1, &pSimplexIds[lIterNodes] );
        if (lStatus)
        {
            lStatus = 1+lStatus;
            continue;
        }
        // Acquire implicit simplex    
        pSimplexIds[lIterNodes] = sectorAndExplicit2SimplexInd( lSector, pSimplexIds[lIterNodes] );
    }
    
    return 0;
}

// Get a simplex index for a simplex where set is a part
int ImplicitMesh::getASimplexForSet( const std::set<unsigned int> & pSet, unsigned int & pSimplexId) const
{   
    int lStatus = 0;
 
    // If no implicit extension
    if (!isExtended())
        // run through ConstMesh equivalent to acquire explicit simplices for all points
        return mMesh.getASimplexForSet( pSet, pSimplexId );
    
    std::set<unsigned int> lExplicitSet;
    unsigned int lSector;
    bool lSectorSet = false;
    unsigned int lFlipAround = 0;
    
    while ( lFlipAround < pSet.size() )
    {
    
        lExplicitSet.clear();
        for ( std::set<unsigned int>::const_iterator lIter = pSet.begin(); lIter != pSet.end(); ++lIter )
        {
            // Acquire explicit node
            unsigned int lNodeSector;
            unsigned int lExplicitNode;
            lStatus = nodeInd2SectorAndExplicitInd( *lIter, lNodeSector, lExplicitNode );
            if (lStatus)
                return 1;
                
            if (!lSectorSet)
            {
                lSector = lNodeSector;
                lSectorSet = true;
            }
                
            // If sectors of node and other nodes are not the same
            if (lNodeSector != lSector)
            {
                // Try to convert explicit node to suiting sector
                lStatus = getExplicitIndInSector( lExplicitNode, lNodeSector, lSector );
                // If failed
                if (lStatus)
                {
                    // Set current sector to the right one
                    lSector = lNodeSector;
                    // increment flip around
                    ++lFlipAround;
                    // break out
                    break;
                }
                
            }
            // Insert into set of explicit nodes            
            lExplicitSet.insert(lExplicitNode);
        }
        
        // run through ConstMesh equivalent to acquire explicit simplices for node
        lStatus = mMesh.getASimplexForSet( lExplicitSet, pSimplexId );
        if (lStatus)
        {
            lStatus = 1+lStatus;
            continue;
        }
        // Acquire implicit simplex    
        pSimplexId = sectorAndExplicit2SimplexInd( lSector, pSimplexId );
        
        return 0;
    }
    
    return 4;
    
}

// Get a set of all simplices for which the given set is a member.
int ImplicitMesh::getAllSimplicesForSet( const std::set<unsigned int> & pSet, std::set<unsigned int> & pSimplexId,
    std::set<unsigned int> & pOutput) const
{
    // Clear output
    pOutput.clear();
    int lStatus = 0;
    
    std::set<unsigned int> lExplicitSimp;
    
    // If no implicit extension
    if (!isExtended())
        return mMesh.getAllSimplicesForSet( pSet, *pSimplexId.begin(), pOutput );
        
    // Get copy of curent set
    std::set<unsigned int> lExplicitSet;
    
    while ( pSimplexId.size() > 0 )
    {
        // Pop front
        const unsigned int lSimplexId = *pSimplexId.begin();
        pSimplexId.erase(pSimplexId.begin());
        if (lSimplexId >= getNT())
            return 2;
    
        // Acquire current simplex sector
        const unsigned int lSimplexSector = simplexInd2Sector(lSimplexId);    
        // Acquire explicit simplex
        const unsigned int lExplicitSimplex = simplexInd2ExplicitInd( lSimplexId );
        
        // Get set of nodes in corresponding sector
        lExplicitSet.clear();
        for ( std::set<unsigned int>::const_iterator lIter = pSet.begin(); lIter != pSet.end(); ++lIter )
        {
            // Acquire explicit node
            unsigned int lNodeSector;
            unsigned int lExplicitNode;
            lStatus = nodeInd2SectorAndExplicitInd( *lIter, lNodeSector, lExplicitNode );
            if (lStatus)
                return 1;
            // If sectors of node and simplex are not the same
            if (lSimplexSector != lNodeSector)
            {
                // Try to convert explicit node to suiting sector
                lStatus = getExplicitIndInSector( lExplicitNode, lNodeSector, lSimplexSector );
                if (lStatus)
                    // Flag error
                    return 4;
            }
            // Insert into set of explicit nodes            
            lExplicitSet.insert(lExplicitNode);
        }
        
        // Get all explicit simplices for set
        lStatus = mMesh.getAllSimplicesForSet( lExplicitSet, lExplicitSimplex, lExplicitSimp  );
        if (lStatus)
            return 3;    
    
        // Go through each dimension 
        for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
        {        
            // Get current dimension sector
            const unsigned int lCurDimSector = getCurDimSector( lSimplexSector, lIterDim );
            // Get if in front or back
            bool lIsFront = true;
            bool lIsBack = true;
            for ( std::set<unsigned int>::const_iterator lIterExplicitSet = lExplicitSet.begin(); lIterExplicitSet != lExplicitSet.end(); ++lIterExplicitSet )
            {
                lIsFront &= getBoundingNodes()[lIterDim].first.count( *lIterExplicitSet );
                lIsBack &= getBoundingNodes()[lIterDim].second.count( *lIterExplicitSet );
            }
            
            // If not the first and part of front or if not the last and part of back
            if ( ( (lCurDimSector > 0) && lIsFront ) || ( (lCurDimSector + 1 < mCopiesPerDimension[lIterDim]) && lIsBack) )
            {
                // Acquire neighboring sector
                const unsigned int lNeighborSector = lIsFront ? lSimplexSector - getNumCopiesPerSector(lIterDim) : lSimplexSector + getNumCopiesPerSector(lIterDim);
                // Go through all explicit simplices involving lExplicitNode
                for ( std::set<unsigned int>::const_iterator lIterSimplices = lExplicitSimp.begin(); lIterSimplices != lExplicitSimp.end(); ++lIterSimplices )
                {
                    // Get paired simplex    
                    const unsigned int lPairedExplicitSimplex = findExplicitPairedSimplex(*lIterSimplices, lIterDim, lIsFront);
                    // If found a pairing
                    if ( lPairedExplicitSimplex < mMesh.getNT() )
                    {
                        // Get potential new index
                        const unsigned int lPotentialNewIndex = sectorAndExplicit2SimplexInd( lNeighborSector, lPairedExplicitSimplex );

                        // If not already present in output
                        if (pOutput.count(lPotentialNewIndex) == 0)
                        {
                            // Add to simplex set
                            pSimplexId.insert(lPotentialNewIndex);
                        }
                        // break out of loop
                        break;
                    }
                }   // End of going through all explicit simplices 
            }
        }   // End of looping through dimension    
        
        // Go through explicit simplices and transform from explicit to implicit simplices and insert in output
        for ( std::set<unsigned int>::const_iterator lIterSimplices = lExplicitSimp.begin(); lIterSimplices != lExplicitSimp.end(); ++lIterSimplices )
            pOutput.insert( sectorAndExplicit2SimplexInd( lSimplexSector, *lIterSimplices ) );
    
    }   // End of while loop through pSimplexId
    
    return 0;
}


int ImplicitMesh::getExplicitIndInSector( unsigned int & pExplicitInd, 
    unsigned int & pFromSector, const unsigned int pToSector ) const
{
    // Is sectors are the same
    if ( pToSector == pFromSector)
        return 0;
    if (!isExtended())
        return 1;
    // if sectors are out of bounds
    if ( ( pFromSector >= getNumCopiesPerSector( getD()) ) || ( pToSector >= getNumCopiesPerSector( getD()) ) )
        return 1;
    // If explicit ind is out of bounds
    if (pExplicitInd >= mMesh.getNN())
        return 2;
        
    bool lMatched = false;
    // Go through each dimension 
    for (unsigned int lIterDim = 0; lIterDim < getD(); lIterDim++)
    {
        // Get current dimension sectors
        const unsigned int lCurFromSector = getCurDimSector( pFromSector, lIterDim );
        const unsigned int lCurToSector = getCurDimSector( pToSector, lIterDim );
        // Get if node is in front or back
        const bool lIsFront = getBoundingNodes()[lIterDim].first.count(pExplicitInd);
        const bool lIsBack = getBoundingNodes()[lIterDim].second.count(pExplicitInd);
        // If node is in front and from sector is one higher than to sector
        if ( lIsFront && (lCurToSector+1 == lCurFromSector) )
        {
            // Adjust from sector
            pFromSector -= getNumCopiesPerSector( lIterDim );
            // set explicit node to paired explicit node
            pExplicitInd = findExplicitPairedNode(pExplicitInd, lIterDim, lIsFront);
        }
        // If node is in back and to sector is one higher than from sector
        else if ( lIsBack && (lCurToSector == lCurFromSector+1) )
        {
            // Adjust from sector
            pFromSector += getNumCopiesPerSector( lIterDim );
            // set explicit node to paired explicit node
            pExplicitInd = findExplicitPairedNode(pExplicitInd, lIterDim, lIsFront);
        }
        if (pFromSector == pToSector)
            lMatched = true;
    }
    // If did not find cause of mismatch
    if (!lMatched)
        // Flag error
        return 4;
        
    return 0;
}










// Populate arrays with explicit mesh from implicit mesh
int ImplicitMesh::populateArraysFromFullMesh( double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
    unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
    unsigned int * const pNeighs ) const
{

    int lStatus = 0;

    // If nodes should be retrieved
    if (pNodes != NULL)
    {
        // Handle errors
        if (pNumNodes != getNN())
            return 1;
            
        #pragma omp parallel reduction( | : lStatus )
        {
            
            // Preallocate nodes vector
            std::vector<double> lNode;
            lNode.reserve( getD() );
            
            // Loop through all nodes
            #pragma omp for
            for ( unsigned int lIterNodes = 0; lIterNodes < getNN(); lIterNodes++ )
            {
                if (lStatus)
                    continue;
            
                // Get pointer to current output for node
                double * lNodePtr = &pNodes[ getD() * lIterNodes ];
            
                // Get current node
                lStatus = getNode(lIterNodes, lNode);
                if (lStatus)
                    continue;
                
                // Loop through all dimensions
                for ( std::vector<double>::const_iterator lIter = lNode.begin(); lIter != lNode.end(); ++lIter )
                {
                    // Move current double to output node
                    *lNodePtr = *lIter;
                    // Increment iterator
                    ++lNodePtr;
                }
            }
        }   // end of parallel section
        // Handle error
        if (lStatus)
            return 2;
        
    }   // End of if nodes should be used
    
    // If simplices should be retrieved
    if (pSimplices != NULL)
    {
        // Handle errors
        if (pNumSimplices != getNT())
            return 1;
            
        #pragma omp parallel reduction( | : lStatus )
        {
            
            // Preallocate simplices vector
            std::set<unsigned int> lSimplex;
            std::set<unsigned int> lTemp1;
            
            // Loop through all simplices
            #pragma omp for
            for ( unsigned int lIterSimplices = 0; lIterSimplices < getNT(); lIterSimplices++ )
            {
                if (lStatus)
                    continue;
                unsigned int * lSimplexPtr = &pSimplices[ (getTopD()+1) * lIterSimplices ];
            
                // Get current simplex
                lStatus = getSimplex(lIterSimplices, lSimplex, lTemp1);
                // Handle error
                if (lStatus)
                    continue;
                
                // Loop through all dimensions
                for ( std::set<unsigned int>::const_iterator lIter = lSimplex.begin(); lIter != lSimplex.end(); ++lIter )
                {
                    // Move current element to output simplex
                    *lSimplexPtr = *lIter;
                    // Increment iterator
                    ++lSimplexPtr;
                }
            }
        }   // end of parallel section
        // Handle error
        if (lStatus)
            return 3;
    }   // end of if simplices should be retrieved

    // If neighbors should be retrieved
    if (pNeighs != NULL)
    {
        // Handle errors
        if (pNumSimplices != getNT())
            return 1;
            
        #pragma omp parallel reduction( | : lStatus )
        {
            
            std::set<unsigned int> lCurNeighs;
            # pragma omp for
            for ( unsigned int lIterSimplices = 0; lIterSimplices < getNT(); lIterSimplices++ )
            {
                if (lStatus)
                    continue;
            
                unsigned int * lNeighsPtr = &pNeighs[ (getTopD()+1) * lIterSimplices ];
            
                // Get current neighbors
                lStatus = getNeighborsFromSimplex(lIterSimplices, lCurNeighs);
                if (lStatus)
                    continue;
                // Loop through all elements of vector
                for ( std::set<unsigned int>::const_iterator lIter = lCurNeighs.begin(); lIter != lCurNeighs.end(); ++lIter )
                {
                    // Move current element to output neighbor
                    *lNeighsPtr = *lIter;
                    // Increment iterator
                    lNeighsPtr++;
                }
                // Fill in the surplus
                for (unsigned int lIter = lCurNeighs.size(); lIter < getTopD()+1; lIter++)
                {
                    // Fill upp with dummies
                    *lNeighsPtr = getNT();
                    // Increment iterator
                    ++lNeighsPtr;
                }
            }
        
        }   // End of parallel section
        // Handle error
        if (lStatus)
            return 4;
            
    }   // End of if neighbors should be retrieved
    
    
    return 0;
}






// Global list storing implicit meshes
GlobalVariablesList<ImplicitMesh> gImplicitMeshes;


extern "C"
{

    
    
    
    // Create full mesh from implicit mesh
    int implicitMesh_createImplicitMesh(const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        const double* const pOffset, const unsigned int * const pCopiesPerDimension,
        unsigned int * const pId, unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices,
        const unsigned int * pNeighs)
    {
    
        // If submanifold
        if (pTopD != pD)
            // Return failure
            return 1;
    
        // Create internal representation of mesh
        ConstMesh lConstMesh( pNodes, pD, pNumNodes, pSimplices, pNumSimplices, pTopD, pNeighs );
        
        // Create implicit mesh from ConstMesh
        ImplicitMesh lImplicitMesh(lConstMesh);
        // If mesh should be extended
        if ( pOffset != NULL && pCopiesPerDimension != NULL )
        {
            // Extend implicit mesh
            int lImplicitStatus = lImplicitMesh.defineExtensionInCurrentDims( pOffset, pCopiesPerDimension );
            // If failed
            if (lImplicitStatus)
                // Return error
                return lImplicitStatus + 1;
        }
        
        
        // Get extensions number of nodes
        *pNewNumNodes = lImplicitMesh.getNN();
        // Get extensions number of simplices
        *pNewNumSimplices = lImplicitMesh.getNT();
        
        // Store implicit mesh
        *pId = gImplicitMeshes.store( lImplicitMesh );
        
        // Return success
        return 0;
    }
    
    // Get full mesh from implicit mesh
    int implicitMesh_retrieveFullMeshFromImplicitMesh( const unsigned int pId,
        double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD)
    {
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;
        
        int lStatus = lImplicitMesh->populateArraysFromFullMesh( pNodes, pNumNodes, pD,
            pSimplices, pNumSimplices, pTopD, NULL );
        // Handle error
        if (lStatus)
            // Return failure
            return lStatus+1;
        
        return 0;
    }
    
    // Get full neighborhood from implicit mesh
    int implicitMesh_retrieveFullNeighsFromImplicitMesh( const unsigned int pId,
        unsigned int * const pNeighs, const unsigned int pNumSimplices, const unsigned int pTopD)
    {
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;
        
        int lStatus = lImplicitMesh->populateArraysFromFullMesh( NULL, 0, 0,
            NULL, pNumSimplices, pTopD, pNeighs );
        // Handle error
        if (lStatus)
            // Return failure
            return lStatus+1;
        
        return 0;
    }
    
    // Remove implicit mesh
    int implicitMesh_eraseImplicitMesh( const unsigned int pId)
    {
        // Load implicit mesh from storage
        return gImplicitMeshes.erase(pId);
    }
    
    int implicitMesh_nodeInd2SectorAndExplicit( const unsigned int pId, const unsigned int pNodeInd, unsigned int * const pSector, unsigned int * const pExplicitInd)
    {
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;            
        int lStatus = lImplicitMesh->nodeInd2SectorAndExplicitInd( pNodeInd, *pSector, *pExplicitInd );
        if (lStatus)
            return 1 + lStatus;
        return 0;
    }
    
    int implicitMesh_nodeSectorAndExplicit2Ind( const unsigned int pId, 
        const unsigned int pSector, const unsigned int pExplicitInd, unsigned int * const pNodeInd)
    {
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;
            
        int lStatus = lImplicitMesh->sectorAndExplicit2NodeInd( pSector, pExplicitInd, *pNodeInd );
        if (lStatus)
            return 1 + lStatus;            
        return 0;
    }
    
    // Checks whether a certain point is in a certain simplex
    int implicitMesh_pointInSimplex( const unsigned int pId, const double * const pPoint, const unsigned int pDim, const unsigned int pSimplex, bool * const pOut,
        const double pEmbTol, const double * const pCenterOfCurvature )
    {
        int lStatus = 0;
        
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;
        // If wrong dimensionality
        if (pDim != lImplicitMesh->getD())
            return 2;
        if (pSimplex >= lImplicitMesh->getNT())
            return 2;
    
        // Copy point into a new vector
        std::vector<double> lPoint( pPoint, &pPoint[pDim] );
        // Get one simplex index for a simplex where point is a part
        unsigned int lOneSimplex;
        lStatus = lImplicitMesh->getASimplexForPoint( lPoint.data(), 1, &lOneSimplex, NULL, 0, NULL, 0);
        if (lStatus)
            return 3;
        
        // Get a set of all simplices for which the given node is a member.
        std::set<unsigned int> lSimplexIds;
        lSimplexIds.insert(lOneSimplex);
        std::set<unsigned int> lAllSimplices;
        lStatus = lImplicitMesh->getAllSimplicesForPoint( lPoint.data(), lSimplexIds, lAllSimplices, pEmbTol, pCenterOfCurvature );
        if (lStatus)
            return 4;
            
        // Get if pSimplex is part of lAllSimplices
        *pOut = lAllSimplices.count(pSimplex);
    
        return 0;
    }    
    
    // Checks whether a certain node is in a certain simplex
    int implicitMesh_nodeInSimplex( const unsigned int pId, const unsigned int pNode, const unsigned int pSimplex, bool * const pOut )
    {
        int lStatus = 0;
        
        // Load implicit mesh from storage
        const ImplicitMesh * const lImplicitMesh = gImplicitMeshes.load(pId);
        // Check if successfull
        if (lImplicitMesh == NULL)
            return 1;
    
        // Get one simplex index for a simplex where node is a part
        unsigned int lOneSimplex = lImplicitMesh->getConstMesh().getNT();
        lStatus = lImplicitMesh->getASimplexForNode( &pNode, 1, &lOneSimplex);
        if (lStatus)
            return 2;
        
        // Get a set of all simplices for which the given node is a member.
        std::set<unsigned int> lSimplexIds;
        lSimplexIds.insert(lOneSimplex);
        std::set<unsigned int> lAllSimplices;
        std::set<unsigned int> lTempSet1;
        std::set<unsigned int> lTempSet2;
        lStatus = lImplicitMesh->getAllSimplicesForNode( pNode, lSimplexIds, lAllSimplices );
        if (lStatus)
            return 2 + lStatus;
    
        // Get if pSimplex is part of lAllSimplices
        unsigned int lNumCounts = lAllSimplices.count(pSimplex);
        *pOut = (lNumCounts > 0);
    
        return 0;
    }
    
        
}   // End of extern "C"






