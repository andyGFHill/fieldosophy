/* 
* C/C++ functions for mesh related operations.
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



#include <math.h>
#include <omp.h>

#include "Eigen/Dense"

#include "mesh.hxx"
#include "meshGraph.hxx"
#include "misc.hxx"



Eigen::VectorXd ConstMesh::getSimplexStandardCoords( const double * const pPoint, const unsigned int pSimplexId, double * const pDivergence  ) const
{
    // Get current point
    Eigen::Map<const Eigen::VectorXd> lCurCoords( pPoint, getD() );
    
    // Get coordinates of nodes in current simplex
    std::vector<int> lCurNodes( &getSimplices()[(getTopD()+1) * pSimplexId], &getSimplices()[(getTopD()+1) * (pSimplexId + 1)] );
    std::vector< double > lCurNodeCoords;
    lCurNodeCoords.reserve( (getTopD() + 1) * getD() );
    
    for (unsigned int lIterNodes = 0; lIterNodes < getTopD()+1; lIterNodes++)
        lCurNodeCoords.insert (lCurNodeCoords.end(), 
            &getNodes()[lCurNodes[lIterNodes]*getD()], 
            &getNodes()[lCurNodes[lIterNodes]*getD() + getD()]);
                            
    // Acquire standard coordinates transform
    MapToSimp lF( lCurNodeCoords.data(), getD(), getTopD() );    
    // check if on submanifold
    if (pDivergence != NULL)
        // Compute the distance off-submanifold
        *pDivergence = lF.getDivergence( lCurCoords );
    
    // Return standard coordinates
    return lF.getStandardCoord(lCurCoords);
}


// Get which simplex points belong to
int ConstMesh::getASimplexForPoint( const double * const pPoints, const unsigned int pNumPoints, 
            unsigned int * const pSimplexIds, double * const pBarycentricCoords, const double pEmbTol ) const
{
    // Assume first point belong to node 0
    unsigned int lCurGraphNode = 0;
    int lStatus = 0;    
    // Loop through each point
    #pragma omp parallel for firstprivate(lCurGraphNode) reduction(|:lStatus)
    for ( unsigned int lIterPoints = 0; lIterPoints < pNumPoints; lIterPoints++ )
    {         
        // If error
        if (lStatus != 0)
            continue;
    
        // Get if mesh graph has been computed
        bool lUseMeshGraph = (mMeshGraph != NULL);
        // Initiate that simplex is not found
        bool lSimplexFound = false;
        
        // If mesh graph exist
        if (lUseMeshGraph)
        {
            // Get node number of current point
            int lOtherStatus = mMeshGraph->getNodeOfPoint( &pPoints[lIterPoints*getD()], getD(), &lCurGraphNode );
            // Get node object of current point
            const MeshGraph::Node * lGraphNode = (lOtherStatus == 0) ? mMeshGraph->getNode(lCurGraphNode) : NULL;
            // If cannot find graph node
            if (lGraphNode == NULL)
            {
                if (lOtherStatus == 0)
                    // flag error
                    lStatus |= 1;
                else
                    // flag error
                    lStatus |= lOtherStatus + 2;
                    
                // Look without mesh graph
                lUseMeshGraph = false;
            }
            else
            {
                // Loop through all triangles in node
                std::vector< unsigned int >::const_iterator lIterSimp = std::begin( lGraphNode->mTriangles );
                while ( lIterSimp != std::end( lGraphNode->mTriangles ) )
                {
                    // Get current point in standard coordinates of chosen simplex
                    double lDivergence = 0.0;
                    double * const lDivergencePtr = (getD() > getTopD()) ? &lDivergence : NULL;
                    const Eigen::VectorXd lStandCoords = getSimplexStandardCoords( &pPoints[lIterPoints*getD()], *lIterSimp, lDivergencePtr );
                    // Get barycentric coordinates of current point
                    const Eigen::VectorXd lBarycentricCoords = mesh_getBarycentricFromStandard(lStandCoords);
                    // If all barycentric coordinates are in between 0 and 1 we found the correct simplex
                    bool lInsideSimplex = true;
                    for (unsigned int lIterDims = 0; lIterDims < lBarycentricCoords.size(); lIterDims++)
                        // If not inside in current simplex
                        if ( ( lBarycentricCoords.coeff(lIterDims) < 0.0d ) || ( lBarycentricCoords.coeff(lIterDims) > 1.0d ) )
                        {
                            lInsideSimplex = false;
                            break;
                        }
                    // If divergence was too large
                    if (lDivergence > pEmbTol)
                        lInsideSimplex = false;
                        
                    // If found the correct simplex
                    if (lInsideSimplex)
                    {
                        // Set that simplex found
                        lSimplexFound = true;
                        // Set membership
                        pSimplexIds[lIterPoints] = *lIterSimp;
                        // If barycentric coordinates should be given
                        if (pBarycentricCoords != NULL)
                            // Copy coordinates
                            memcpy( &pBarycentricCoords[lIterPoints*(getTopD()+1)], lBarycentricCoords.data(), (getTopD()+1)*sizeof(double) );
                        // Break since no more simplex have to be searched for this particular point
                        break;
                    }
                
                    // Iterate
                    ++lIterSimp;
                }
                // If did not find simplex
                if ( !lSimplexFound & ( getD() != getTopD() ) )
                    // Look without mesh graph
                    lUseMeshGraph = false;
            }
                
        }   // End of handling of MeshGraph
        
        // If mesh graph has not been computed (or failed)
        if ( !lUseMeshGraph)
        {
            // Loop through each simplex
            for (unsigned int lIterSimp = 0; lIterSimp < getNT(); lIterSimp++)
            {                    
                // Get current point in standard coordinates of chosen simplex
                double lDivergence = 0.0;
                double * const lDivergencePtr = (getD() > getTopD()) ? &lDivergence : NULL;
                const Eigen::VectorXd lStandCoords = getSimplexStandardCoords( &pPoints[lIterPoints*getD()], lIterSimp, lDivergencePtr );
                // Get barycentric coordinates of current point
                const Eigen::VectorXd lBarycentricCoords = mesh_getBarycentricFromStandard(lStandCoords);
                // If all barycentric coordinates are in between 0 and 1 we found the correct simplex
                bool lInsideSimplex = true;
                for (unsigned int lIterDims = 0; lIterDims < lBarycentricCoords.size(); lIterDims++)
                    // If not inside in current simplex
                    if ( ( lBarycentricCoords.coeff(lIterDims) < 0.0d ) || ( lBarycentricCoords.coeff(lIterDims) > 1.0d ) )
                    {
                        lInsideSimplex = false;
                        break;
                    }
                // If divergence was too large
                if (lDivergence > pEmbTol)
                    lInsideSimplex = false;
                    
                // If found the correct simplex
                if (lInsideSimplex)
                {
                    // Set that simplex found
                    lSimplexFound = true;
                    // Set membership
                    pSimplexIds[lIterPoints] = lIterSimp;
                    // If barycentric coordinates should be given
                    if (pBarycentricCoords != NULL)
                        // Copy coordinates
                        memcpy( &pBarycentricCoords[lIterPoints*(getTopD()+1)], lBarycentricCoords.data(), (getTopD()+1)*sizeof(double) );
                    // Break since no more simplex have to be searched for this particular point
                    break;
                }                
            }   // End loop over simplices
        }
            
        // If no simplex was found
        if (!lSimplexFound)
        {
            // Set membership to one number larger than maximum
            pSimplexIds[lIterPoints] = getNT();
            // If barycentric coordinates should be given
            if (pBarycentricCoords != NULL)
                // Insert zeros
                for (unsigned int lIterDims = 0; lIterDims < (getTopD()+1); lIterDims++)
                    pBarycentricCoords[lIterPoints*(getTopD()+1) + lIterDims] = 0.0;
        }
        
        
    }   // end loop over points


    return lStatus;
}

// Get a simplex index for a simplex where node is a part
int ConstMesh::getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const
{

    // Assume first node belong to graph node 0
    unsigned int lCurGraphNode = 0;
    int lStatus = 0;    
    // Loop through each node
    #pragma omp parallel for firstprivate(lCurGraphNode) reduction(|:lStatus)
    for ( unsigned int lIterNodeInds = 0; lIterNodeInds < pNumNodes; lIterNodeInds++ )
    {         
        // If error
        if (lStatus != 0)
            continue;
            
        // Get current node index
        const unsigned int lCurNodeInd = pNodes[lIterNodeInds];
        if (lCurNodeInd >= getNN())
        {
            lStatus = 1;
            continue;
        }
        
        
        // Get current node array
        const double * const lCurNode = &getNodes()[lCurNodeInd*getD()];
    
        // Get if mesh graph has been computed
        bool lUseMeshGraph = (mMeshGraph != NULL);
        // Initiate that simplex is not found
        bool lSimplexFound = false;
        
        // If mesh graph exist
        if (lUseMeshGraph)
        {
            // Get graph node number of current node
            int lOtherStatus = mMeshGraph->getNodeOfPoint( lCurNode, getD(), &lCurGraphNode );
            // Get node object of current point
            const MeshGraph::Node * lGraphNode = (lOtherStatus == 0) ? mMeshGraph->getNode(lCurGraphNode) : NULL;
            // If cannot find graph node
            if (lGraphNode == NULL)
            {
                // getNode failed
                if (lOtherStatus == 0)
                    // flag error
                    lStatus |= 1;
                else    // If getNodePoint failed
                    // flag error
                    lStatus |= lOtherStatus + 2;
                // do not use mesh graph
                lUseMeshGraph = false;
            }
            else // If mesh graph was found
            {
                // Loop through all simplices in graph node
                for ( std::vector< unsigned int >::const_iterator lIterSimp = std::begin( lGraphNode->mTriangles );
                    lIterSimp != std::end( lGraphNode->mTriangles ); ++lIterSimp )
                {
                    lSimplexFound = isNodePartOfSimplex( lCurNodeInd, *lIterSimp );
                    // If found simplex
                    if (lSimplexFound)
                    {
                        // Set membership
                        pSimplexIds[lIterNodeInds] = *lIterSimp;
                        // Break out of loop
                        break;
                    }
                }
                // If did not find simplex
                if ( !lSimplexFound )
                    // Look without mesh graph
                    lUseMeshGraph = false;
            }
        }   // End of handling of MeshGraph
        
        // If mesh graph has not been computed (or failed)
        if ( !lUseMeshGraph)
        {
            // Loop through each simplex
            for (unsigned int lIterSimp = 0; lIterSimp < getNT(); lIterSimp++)
            {
                lSimplexFound = isNodePartOfSimplex( lCurNodeInd, lIterSimp );
                // If found simplex
                if (lSimplexFound)
                {
                    // Set membership
                    pSimplexIds[lIterNodeInds] = lIterSimp;
                    // Break out of loop
                    break;
                }
            }   // End loop over simplices
        }   // End of if not using mesh graph
            
        // If no simplex was found
        if (!lSimplexFound)
            // Set membership to one number larger than maximum
            pSimplexIds[lIterNodeInds] = getNT();        
        
    }   // end loop over nodes


    return lStatus;
}


// Get a set of all simplices for which the given point is a member.
int ConstMesh::getAllSimplicesForPoint( const double * const pPoint, unsigned int pSimplexId, 
    std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp ) const
{   
    // Clear output
    pOutput.clear();
    pTemp.clear();
     
    // If out of bounds
    if (pSimplexId > getNT())
        return 1;
    // If no neighborhood exists
    if (getNeighs() == NULL)
        return 2;
        
    // If have no suggestions of simplexId
    if (pSimplexId == getNT())
    {
        // Acquire one
        int lStatus = getASimplexForPoint( pPoint, 1, &pSimplexId);
        // Handle error
        if (lStatus)
            return 3;
    }
       
    // Insert current simplex
    pTemp.insert( pSimplexId );
    
    // Go through all simplices chosen
    while ( pTemp.size() > 0)
    {
        // Pop first value
        const unsigned int lCurSimplex = *pTemp.begin();
        pTemp.erase( pTemp.begin() ); 
        // If current simplex is already present in output
        if ( pOutput.count( lCurSimplex ) )
            // skip ahead
            continue;   
        
        // Get if lcurSimplex is outside (-1), on border (0), or in the middle of (1) simplex
        int lBaryStatus;        
        int lStatus = getCoordinatesGivenSimplex( pPoint, 1, lCurSimplex, NULL, NULL, NULL, &lBaryStatus);
        // If bad status
        if (lStatus)
            // Continue without taking notice
            continue;
        // If outside
        if (lBaryStatus < 0)
            // skip ahead
            continue;
        // Insert simplex into output
        pOutput.insert( lCurSimplex );
        // Investigate all neighboring simplices
        pTemp.insert( 
            &getNeighs()[lCurSimplex*(getTopD()+1)], 
            &getNeighs()[lCurSimplex*(getTopD()+1) + (getTopD()+1)] 
            );
    }     

    return 0;
}

int ConstMesh::getAllSimplicesForNode( const unsigned int pNode, unsigned int pSimplexId,
    std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp ) const
{
    // Clear output
    pOutput.clear();
    pTemp.clear();
    
    // If simplex if is out of bounds
    if (pSimplexId > getNT())
        return 1;
    // If no neighborhood exists
    if (getNeighs() == NULL)
        return 2;
        
    // If have no suggestions of simplexId
    if (pSimplexId == getNT())
    {
        // Acquire one
        int lStatus = getASimplexForNode( &pNode, 1, &pSimplexId);
        // Handle error
        if (lStatus)
            return 3;
    }

    // Insert current simplex
    pTemp.insert( pSimplexId );
    // Go through all simplices chosen
    while ( pTemp.size() > 0)
    {
        // Pop first value
        const unsigned int lCurSimplex = *pTemp.begin();
        pTemp.erase( pTemp.begin() );    
        // If current simplex is already present in output
        if ( pOutput.count( lCurSimplex ) )
            continue;
        // if current simplex is not associated with node    
        if ( !isNodePartOfSimplex( pNode, lCurSimplex ) )
            continue;
        
        // Insert simplex into output
        pOutput.insert( lCurSimplex );
        // Investigate all neighboring simplices
        pTemp.insert( 
            &getNeighs()[lCurSimplex*(getTopD()+1)], 
            &getNeighs()[lCurSimplex*(getTopD()+1) + (getTopD()+1)] 
            );
    }     

    return 0;
}

// Get standard- and/or barycentric coordinates for points given simplex
int ConstMesh::getCoordinatesGivenSimplex( const double * const pPoints, const unsigned int pNumPoints, const unsigned int pSimplexId,
    double * const pStandardCoords, double * const pBarycentricCoords, double * const pDivergence, int * const pStatus) const
{
    // If given simplex does not exist
    if (pSimplexId >= getNT())
        // Flag error
        return 1;

    // Loop through each point
    for (unsigned int lIterPoints = 0; lIterPoints < pNumPoints; lIterPoints++)
    {
        // Get current point in standard coordinates of chosen simplex
        double * const lDivergencePtr = (pDivergence == NULL) ? NULL : &pDivergence[lIterPoints];
        const Eigen::VectorXd lStandCoords = getSimplexStandardCoords( &pPoints[lIterPoints*getD()], pSimplexId, lDivergencePtr );
        // If should return standard coordinates
        if (pStandardCoords != NULL)
            // Copy coordinates
            memcpy( &pStandardCoords[lIterPoints*getTopD()], lStandCoords.data(), getTopD()*sizeof(double) );
            
        // If should return barycentric coordinates or status
        if ( (pBarycentricCoords != NULL) || (pStatus != NULL) )
        {    
            // Get barycentric coordinates of current point
            const Eigen::VectorXd lBarycentricCoords = mesh_getBarycentricFromStandard(lStandCoords);        
            // If barycentric coordinates should be returned
            if (pBarycentricCoords != NULL)
                // Copy coordinates
                memcpy( &pBarycentricCoords[lIterPoints*(getTopD()+1)], lBarycentricCoords.data(), (getTopD()+1)*sizeof(double) );
            // If status should be returned
            if (pStatus != NULL)
            {
                int lBaryStatus = 0;
                for (unsigned int lIterBary = 0; lIterBary < lBarycentricCoords.size(); lIterBary++)
                {
                    const double lCurBaryCoord = lBarycentricCoords.coeff(lIterBary);
                     // If current barycentric coordinate is below 0 or above 1   
                    if ( (lCurBaryCoord < 0.0) || (lCurBaryCoord > 1.0) )
                        // Set as outside
                        lBaryStatus = -1;
                    // If status is not outside
                    if ( lBaryStatus >= 0 )
                        // If current barycentric coordinate is not on an edge
                        if ( (lCurBaryCoord != 0.0) || (lCurBaryCoord != 1.0) )
                            // Set as inside
                            lBaryStatus = 1;
                }
            
                // Return value for this point
                pStatus[lIterPoints] = lBaryStatus;
            }
        }
    }

    return 0;
}


std::set<unsigned int> ConstMesh::getUniqueNodesOfSimplexCollection( const unsigned int * const pSimplices, const unsigned int pNumSimplices ) const
{
    std::set<unsigned int> lUniqueNodes;

    // Loop through all simplices
    for (unsigned int lIterSimps = 0; lIterSimps < pNumSimplices; lIterSimps++)
    {
        // Get current simplex index
        const unsigned int lSimplexId = pSimplices[lIterSimps];
        if (lSimplexId >= getNT())
            continue;
        // Insert all nodes from current simplex
        lUniqueNodes.insert( &getSimplices()[lSimplexId * (getTopD()+1)], &getSimplices()[ (lSimplexId+1) * (getTopD()+1)] );
    }
    return lUniqueNodes;
}


ConstMesh::~ConstMesh()
{
    if (mMeshGraph != NULL)
        delete mMeshGraph;
}

// Populate arrays with corresponding mesh
int ConstMesh::populateArrays( double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, 
        unsigned int * const pNeighs ) const
{
    if (pD != getD())
        return 1;
    if (pTopD != getTopD())
        return 2;
    if (pNumNodes != getNN())
        return 3;
    if (pNumSimplices != getNT())
        return 4;
        
    // Copy nodes
    memcpy( pNodes, getNodes(), getNN()*getD()*sizeof(double) );
    // Copy simplices
    memcpy( pSimplices, getSimplices(), getNT()*(getTopD()+1)*sizeof(unsigned int) );
    // If neighs are defined
    if ( (pNeighs != NULL) && (getNeighs() != NULL) )
        memcpy( pNeighs, getNeighs(), getNT()*(getTopD()+1)*sizeof(unsigned int) );

    return 0;
}

int ConstMesh::computeMeshGraph( const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles )
{
    if (mMeshGraph == NULL)
        // Create mesh graph
        mMeshGraph = new MeshGraph( *this, pMaxNumNodes, pMinDiam, pMinNumTriangles );

    return 0;
}

bool ConstMesh::areSimplicesNeighbors( const unsigned int pSimpInd1, const unsigned int pSimpInd2 ) const
{
    // If out of bounds
    if (pSimpInd1 >= getNT())
        return false;
    if (pSimpInd2 >= getNT())
        return false;
        
    unsigned int lNumMatches = 0;
        
    // Loop through all node indices in simplex 1
    for (unsigned int lIter1 = 0; lIter1 < getTopD()+1; lIter1++)
    {
        // Get current node index
        const unsigned int lCurNodeIndex1 = getSimplices()[ pSimpInd1 * (getTopD()+1) + lIter1 ];
        // If current node index is dummy
        if (lCurNodeIndex1 >= getNN())
            continue;
        
        // Loop through all node indices in simplex 1    
        for (unsigned int lIter2 = 0; lIter2 < getTopD()+1; lIter2++)
        {
            // Get current node index
            const unsigned int lCurNodeIndex2 = getSimplices()[ pSimpInd2 * (getTopD()+1) + lIter2 ];
            // If current node index is dummy
            if (lCurNodeIndex2 >= getNN())
                continue;
            // See if match
            if ( lCurNodeIndex1 == lCurNodeIndex2 )
            {
                lNumMatches++;
                break;
            }
        }
    }

    if (lNumMatches == getTopD() )
        return true;
        
    return false;
}








FullMesh::FullMesh( const double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
    const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
    const unsigned int * const pNeighs ) :
    // Call base class constructor to set const values
    ConstMesh( NULL, pD, pNumNodes, NULL, pNumSimplices, pTopD, NULL )
{
    // Copy nodes
    mFullNodes.assign( pNodes, pNodes + pNumNodes*pD );
    // Copy simplices
    mFullSimplices.assign( pSimplices, pSimplices + pNumSimplices*(pTopD+1) );
    // If neighbors were given
    if (pNeighs != NULL)
        mFullNeighs.assign( pNeighs, pNeighs + pNumSimplices*(pTopD+1) );
    
    // Update pointers
    updateConstMeshPointers();
}
    
    
    
// Method refining all nodes accordingly
int FullMesh::refine( const unsigned int pMaxNumNodes, std::vector<double> & pMaxDiam, int (* transformationPtr)(double *, unsigned int) )
{
    // See if allready reached maximum number of nodes
    if (getNN() >= pMaxNumNodes)
        return 0;
    // See if allready reached maximum number of nodes
    if ( (pMaxDiam.size() != 1) && (pMaxDiam.size() != getNN() ) )
        return 1;
    // If no neigbors given
    if (!hasNeighs())
        // Return error since those are necessary
        return 1;
        
    // Get index of maximum number of simplices
    const unsigned int lMaxNumSimplices = -1;
    // Preallocate status return variable
    int lStatus = 0;
    
    // Preallocate vector for storing all edges indices
    std::vector<std::set<unsigned int>> lEdgeIndices;
    // Compute edges indices
    {
        if (getTopD() == 1)
        {    
            lEdgeIndices.push_back(std::set<unsigned int>() );
            lEdgeIndices.at(0).insert(0);
            lEdgeIndices.at(0).insert(1);
        }
        else
        {
            // Compute number of combinations of chosing 2 elements among (pTopologicalD+1) elements with order
            unsigned int lNumCombinations = 1;        
            for (unsigned int lIter = getTopD()+1; lIter > 2; lIter--)
            {
                lNumCombinations *= lIter;
            }
            // Compute number of edges for a simplex ( (pTopologicalD + 1) choose 2 )
            unsigned int lNChooseK = 1;
            for (unsigned int lIter = 2; lIter <= (getTopD() + 1 - 2); lIter++)
            {
                lNChooseK *= lIter;
            }
            lNChooseK = lNumCombinations / lNChooseK;
            
            // Get vector of indices
            std::vector<unsigned int> lIndicesOfSimplex;
            for (unsigned int lIterNumNodesInSimplex = 0; lIterNumNodesInSimplex < getTopD()+1; lIterNumNodesInSimplex++)
                lIndicesOfSimplex.push_back(lIterNumNodesInSimplex);
            // Compute edges
            lEdgeIndices.reserve(lNChooseK);
            const int lStatus = mesh_getEdgesOfSimplex( lEdgeIndices, lNumCombinations,
                2, getTopD(), lIndicesOfSimplex.data() );
            if (lStatus != 0)
                // Flag error
                return 2;
        }
    }
    
    // Preallocate
    std::set<unsigned int> lNeighboringSimplices1;
    std::set<unsigned int> lNeighboringSimplices2;
    std::set<unsigned int> lSharingSimplices;
    std::set<unsigned int> lTempSet;
    std::vector<unsigned int> lSimplexAccounting;
    lSimplexAccounting.reserve(getNT());
    std::vector<double> lSimplexMaxEdge;
    lSimplexMaxEdge.reserve(getNT());
    
    // Loop through all simplices to rename dummy variables
    #pragma omp parallell for
    for ( unsigned int lIterSimplex = 0; lIterSimplex < getNT(); lIterSimplex++ )
        // Set dummy variables of current simplex and neighborhood to be maximal
        for (unsigned int lIterNodeInds = 0; lIterNodeInds < getTopD()+1; lIterNodeInds++)
        {
            unsigned int * const lCurNodeIndsPtr = &mFullSimplices.data()[ lIterSimplex * (getTopD()+1) + lIterNodeInds ];
            if ( *lCurNodeIndsPtr >= getNN() )
                *lCurNodeIndsPtr = pMaxNumNodes;                
            unsigned int * const lCurSimplexIndsPtr = &mFullNeighs.data()[ lIterSimplex * (getTopD()+1) + lIterNodeInds ];
            if ( *lCurSimplexIndsPtr >= getNT() )
                *lCurSimplexIndsPtr = lMaxNumSimplices;
        }
    
    // While the number of nodes are not too great and there are edges to split
    bool lContinueToLoop = true;
    while ( lContinueToLoop )
    {
        // Get number of nodes before updating this round
        const unsigned int lLastEpochNN = getNN();
        // Assume no edges to split anymore
        unsigned int lNumSimplicesToUpdate = 0;

        // Initialize simplex accounting to no updated simplices
        lSimplexAccounting.assign( getNT(), 0 );
        // Initialize simplex max edge to no distance
        lSimplexMaxEdge.assign( getNT(), 0.0d );
        
        // Loop through all simplices
        #pragma omp parallell for reduction( +:lNumSimplicesToUpdate )
        for ( unsigned int lIterSimplex = 0; lIterSimplex < getNT(); lIterSimplex++ )
        {            
            // Loop through all edges to find the largest edge
            for ( unsigned int lIterEdges = 0; lIterEdges < lEdgeIndices.size(); lIterEdges++)
            {
                // Get the indices of the two points in edge
                const unsigned int lNode1EdgeIndex = *lEdgeIndices[lIterEdges].begin();
                const unsigned int lNode2EdgeIndex = *std::next(lEdgeIndices[lIterEdges].begin());
                // Get node indices of the two points in edge
                const unsigned int lNode1Index = getSimplices()[ lIterSimplex * (getTopD()+1) + lNode1EdgeIndex ];
                const unsigned int lNode2Index = getSimplices()[ lIterSimplex * (getTopD()+1) + lNode2EdgeIndex ];
                // Get maximum distance of current edge
                double lMaxDiam = pMaxDiam[0];
                // If more than one value
                if ( pMaxDiam.size() > 1 )
                    // Use the smallest of them
                    lMaxDiam = ( pMaxDiam[ lNode1Index ] < pMaxDiam[ lNode2Index ] ) ? pMaxDiam[ lNode1Index ] : pMaxDiam[ lNode2Index ];
                
                // Map the two nodes of current edge
                Eigen::Map<const Eigen::VectorXd> lNode1( &getNodes()[getD() * lNode1Index], getD() );
                Eigen::Map<const Eigen::VectorXd> lNode2( &getNodes()[getD() * lNode2Index], getD() );
                // Compute length in between points
                const double lDistance = (lNode1-lNode2).norm();
                
                // If current distance is larger than maximum allowed distance
                if ( lDistance > lMaxDiam )
                    // If current distance if larger than maximum
                    if (lDistance > lSimplexMaxEdge[ lIterSimplex ])
                    {
                        lSimplexMaxEdge[ lIterSimplex ] = lDistance;
                        lSimplexAccounting[ lIterSimplex ] = lIterEdges;
                    }                
            }
            
            // If should update simplex
            if (lSimplexMaxEdge[ lIterSimplex ] > 0)
                // Increment number of edges to split
                lNumSimplicesToUpdate++;            
        }
        
        if (lNumSimplicesToUpdate == 0)
            lContinueToLoop = false;
        else
            // Loop through all simplices
            for ( unsigned int lIterSimplex = 0; lIterSimplex < lSimplexMaxEdge.size(); lIterSimplex++ )
                // If the current edge should be split
                if ( lSimplexMaxEdge.at( lIterSimplex ) > 0 )
                {
                    // If maximum number of nodes already
                    if (pMaxNumNodes <= getNN())
                        // Break out 
                        break;
                
                    // Check that number of simplices to update is not zero
                    if (lNumSimplicesToUpdate == 0)
                        return 3;
                    // Check that number of edges are correct
                    if (lEdgeIndices.at( lSimplexAccounting[ lIterSimplex ] ).size() != 2)
                        return 4;
                        
                    // Get the indices of the edges of the two points in edge
                    const unsigned int lNode1EdgeIndex = *lEdgeIndices[ lSimplexAccounting[ lIterSimplex ] ].begin();
                    const unsigned int lNode2EdgeIndex = *std::next(lEdgeIndices[ lSimplexAccounting[ lIterSimplex ] ].begin());
                    // Get the indices of the two points in edge
                    const unsigned int lNode1Index = mFullSimplices[ lIterSimplex * (getTopD()+1) + lNode1EdgeIndex ];
                    const unsigned int lNode2Index = mFullSimplices[  lIterSimplex * (getTopD()+1) + lNode2EdgeIndex ];
                    
                    // Get all simplices sharing node1
                    lStatus = getAllSimplicesForNode( lNode1Index, lIterSimplex, lNeighboringSimplices1, lTempSet );
                    if (lStatus)
                        return 5;
                    // Get all simplices sharing node2
                    lStatus = getAllSimplicesForNode( lNode2Index, lIterSimplex, lNeighboringSimplices2, lTempSet );
                    if (lStatus)
                        return 5;
                    // Get all simplices sharing both
                    misc_setIntersection( lNeighboringSimplices1, lNeighboringSimplices2, lSharingSimplices);
                    if ( lSharingSimplices.size() == 0 )
                        return 5;
                    // Assume no other simplices sharing edge has a different value
                    bool lSharingSimplicesAgree = true;
                    // Loop through all these simplices
                    for ( std::set<unsigned int>::const_iterator lIterIntersection = lSharingSimplices.begin(); lIterIntersection != lSharingSimplices.end(); lIterIntersection++ )
                    {
                        // If other simplex has already been updated
                        if ( *lIterIntersection >= lSimplexMaxEdge.size() )
                            lSharingSimplicesAgree = false;
                        // If new simplex has a max edge with a different value
                        else if (lSimplexMaxEdge[ *lIterIntersection ] != lSimplexMaxEdge[ lIterSimplex ] )
                            // Flag that old simplex should not be updated
                            lSharingSimplicesAgree = false;
                            
                        if (!lSharingSimplicesAgree)
                            break;
                    }
                        
                    // If all sharing simplices do not agree
                    if (!lSharingSimplicesAgree)
                    {
                        // Mark current simplex as not to update
                        lSimplexMaxEdge[ lIterSimplex ] = 0.0d;
                        // Remove number of simplices to update since one has been split now
                        lNumSimplicesToUpdate--;
                    }
                    else
                    {
                    
                        // Map the two nodes of current edge
                        Eigen::Map<const Eigen::VectorXd> lNode1( &mFullNodes.data()[getD() * lNode1Index], getD() );
                        Eigen::Map<const Eigen::VectorXd> lNode2( &mFullNodes.data()[getD() * lNode2Index], getD() );
                        // Get node in between
                        const Eigen::VectorXd lNode3 = 0.5 * (lNode1 + lNode2);
                        // Push the second point into the back of nodes and replace its values with node 3 instead
                        for (unsigned int lIterD = 0; lIterD < getD(); lIterD++)
                        {
                            mFullNodes.push_back( lNode3[lIterD] );
                        }
                        // Increase the number of nodes
                        mNumNodes++;
                        // Update pointers
                        updateConstMeshPointers();
                        
                        // If one maximum diameter for each node
                        if ( pMaxDiam.size() > 1 )
                            // Add maximum allowed diameter for new node
                            pMaxDiam.push_back( 0.5 * (pMaxDiam[lNode1Index] + pMaxDiam[lNode2Index]) );
                        
                        // Loop through all simplices sharing edge
                        for ( std::set<unsigned int>::const_iterator lIterIntersection = lSharingSimplices.begin(); lIterIntersection != lSharingSimplices.end(); lIterIntersection++ )
                        {                        
                            // Add new shadow simplex and update old simplex
                            for (unsigned int lIterSimpDim = 0; lIterSimpDim < getTopD()+1; lIterSimpDim++)
                            {
                                // Get current simplex index
                                const unsigned int lCurNodeIndex = mFullSimplices[  *lIterIntersection * (getTopD()+1) + lIterSimpDim ];
                                // If equal to node 1 index
                                if (lCurNodeIndex == lNode1Index)
                                {
                                    // Let original simplex keep node 1 while new simplex use node 3 instead
                                    mFullSimplices.push_back( getNN() - 1 );
                                }
                                else
                                {
                                    // If equal to node 2 index
                                    if (lCurNodeIndex == lNode2Index)
                                        // Let new simplex keep node 2 while original simplex use node 3 instead
                                        mFullSimplices[ *lIterIntersection * (getTopD()+1) + lIterSimpDim ] = getNN()-1;
                                    
                                    // Insert the same index
                                    mFullSimplices.push_back( lCurNodeIndex );
                                }
                                    
                                // Add a copy of old neighborhood
                                mFullNeighs.push_back( mFullNeighs[ *lIterIntersection * (getTopD()+1) + lIterSimpDim ] );
                            }
                            // Increase the number of simplices
                            mNumSimplices++;   
                            // Update pointers
                            updateConstMeshPointers();
                            
                            // Mark current simplex as updated
                            lSimplexMaxEdge[ *lIterIntersection ] = 0.0d;
                            // Remove number of simplices to update since one has been split now
                            lNumSimplicesToUpdate--;
                            
                        } // End of loop through sharing simplices        
                        
                        // Handle neighborhoods
                        {                                
                            // Loop through all simplices sharing edge to update their neighborhood
                            std::set<unsigned int>::const_iterator lIterIntersection = lSharingSimplices.begin();
                            for ( unsigned int lIterChangedSimplices = 0; lIterChangedSimplices < lSharingSimplices.size(); lIterChangedSimplices++ )
                            {
                                // Get complementary shadow
                                const unsigned int lComplementaryShadow = (lIterChangedSimplices + getNT()) - lSharingSimplices.size();
                                
                                // Go through neighborhood
                                for ( unsigned int lIterNeighIndices = 0; lIterNeighIndices < getTopD() + 1; lIterNeighIndices++)
                                {
                                    // Get index of supposed neighbor
                                    const unsigned int lCurNeighIndex = getNeighs()[*lIterIntersection * (getTopD()+1) + lIterNeighIndices];
                                    
                                    // If index is not dummy
                                    if (lCurNeighIndex < getNT() )
                                    {
                                        // Make sure that neighbor is considered neighbor with this
                                        unsigned int lNeighborsThisIndex = getTopD()+1;
                                        for ( unsigned int lIterNeighIndices2 = 0; lIterNeighIndices2 < getTopD() + 1; lIterNeighIndices2++)
                                            // If the same as the original simplex
                                            if ( mFullNeighs.at( lCurNeighIndex * (getTopD()+1) + lIterNeighIndices2 ) == *lIterIntersection )
                                            {
                                                lNeighborsThisIndex = lIterNeighIndices2;
                                                // Break out
                                                break;
                                            }

                                        // If not present in neighbors neighbors
                                        if (lNeighborsThisIndex == getTopD()+1 )
                                            // Signal error
                                            return 10;
                                    
                                        // If not actual neighbors
                                        if ( !areSimplicesNeighbors( *lIterIntersection, lCurNeighIndex ) )
                                        {
                                            // Is the former neigbor part of updated simplices?
                                            if ( lSharingSimplices.count( lCurNeighIndex ) )
                                            {
                                                // This case should not be possible
                                                // Signal error
                                                return 11;
                                            }
                                            else
                                            {
                                                // The shadow should be switched to complementary shadow
                                                mFullNeighs[ lComplementaryShadow * (getTopD()+1) + lIterNeighIndices ] = lCurNeighIndex;
                                                // In the other neighbor, the neighbor should be complementary shadow
                                                mFullNeighs[ lCurNeighIndex * (getTopD()+1) + lNeighborsThisIndex ] = lComplementaryShadow;
                                                // Set old link to dummy
                                                mFullNeighs[ *lIterIntersection * (getTopD()+1) + lIterNeighIndices ] = lMaxNumSimplices;
                                            }
                                        }
                                    }
                                }   // End of going through neighborhhods
                                
                                // Go through neighborhood
                                for ( unsigned int lIterNeighIndices = 0; lIterNeighIndices < getTopD() + 1; lIterNeighIndices++)
                                {
                                    // Get index of supposed neighbor
                                    const unsigned int lCurNeighIndex = getNeighs()[lComplementaryShadow * (getTopD()+1) + lIterNeighIndices];
                                    
                                    // If index is not a dummy
                                    if (lCurNeighIndex < lMaxNumSimplices)
                                    {
                                        // Make sure that neighbor is considered neighbor with this
                                        unsigned int lNeighborsThisIndex = getTopD()+1;
                                        for ( unsigned int lIterNeighIndices2 = 0; lIterNeighIndices2 < getTopD() + 1; lIterNeighIndices2++)
                                            // If the same as the original simplex
                                            if ( 
                                                (getNeighs()[lCurNeighIndex * (getTopD()+1) + lIterNeighIndices2] == *lIterIntersection) || 
                                                (getNeighs()[lCurNeighIndex * (getTopD()+1) + lIterNeighIndices2] == lComplementaryShadow) 
                                                )
                                            {
                                                lNeighborsThisIndex = lIterNeighIndices2;
                                                // Break out
                                                break;
                                            }

                                        // If not present in neighbors neighbors
                                        if (lNeighborsThisIndex == getTopD()+1 )
                                            // Signal error
                                            return 13;
                                    
                                        // If not actual neighbors
                                        if ( !areSimplicesNeighbors( lComplementaryShadow, lCurNeighIndex ) )
                                        {
                                            // Is the former neigbor part of updated simplices?
                                            if ( lSharingSimplices.count( lCurNeighIndex ) )
                                            {
                                                // find its order
                                                unsigned int lIterChangedSimplices2 = 0;
                                                for ( ; lIterChangedSimplices2 <= lSharingSimplices.size(); lIterChangedSimplices2++)
                                                {
                                                    if ( *std::next(lSharingSimplices.begin(), lIterChangedSimplices2 ) == lCurNeighIndex )
                                                        break;
                                                }
                                                if (lIterChangedSimplices2 == lSharingSimplices.size())
                                                    return 14;
                                            
                                                // Get its complementary
                                                const unsigned int lComplementaryShadow2 = (lIterChangedSimplices2 + getNT()) - lSharingSimplices.size();                                            
                                                // The old neighbor should be switched to complementary shadow
                                                mFullNeighs[ lComplementaryShadow * (getTopD()+1) + lIterNeighIndices ] = lComplementaryShadow2;
                                                // In the other neighbor, the neighbor should be complementary shadow
                                                mFullNeighs[ lComplementaryShadow2 * (getTopD()+1) + lNeighborsThisIndex ] = lComplementaryShadow;

                                            }
                                            else
                                            {
                                                // The old neighbor should be switched to original index
                                                mFullNeighs[ *lIterIntersection * (getTopD()+1) + lIterNeighIndices ] = lCurNeighIndex;
                                                // In the other neighbor, the neighbor should be original or complementary shadow
                                                mFullNeighs[ lCurNeighIndex * (getTopD()+1) + lNeighborsThisIndex ] = *lIterIntersection;
                                                // Set old link to dummy
                                                mFullNeighs[ lComplementaryShadow * (getTopD()+1) + lIterNeighIndices ] = lMaxNumSimplices;
                                            }
                                        }    
                                        // If they are neigbors and it is written that the original is
                                        else if (lNeighborsThisIndex == *lIterIntersection)
                                        {
                                            // The old neighbor should be switched to original index
                                            mFullNeighs[ lComplementaryShadow * (getTopD()+1) + lIterNeighIndices ] = lCurNeighIndex;
                                            // In the other neighbor, the neighbor should be original or complementary shadow
                                            mFullNeighs[ lCurNeighIndex * (getTopD()+1) + lNeighborsThisIndex ] = lComplementaryShadow;
                                        }
                                    }
                                }   // End of going through neighborhhods
                                
                                // If original and complementary are not already paired
                                {
                                    bool lIsPart = false;
                                    unsigned int lFirstDummy = getTopD()+1;
                                    for ( unsigned int lIterNeighIndices = 0; lIterNeighIndices < getTopD() + 1; lIterNeighIndices++)
                                    {
                                        const unsigned int lCurVal = getNeighs()[ *lIterIntersection * ( getTopD() + 1 ) + lIterNeighIndices ];
                                        // If current new simplex is already part of neighborhood
                                        if ( lCurVal == lComplementaryShadow )
                                            lIsPart = true;
                                        // If dummy
                                        else if ( (lCurVal >= getNT() ) && (lFirstDummy > getTopD()) )
                                            lFirstDummy = lIterNeighIndices;
                                    }
                                    // If current new simplex is not part and a dummy exists
                                    if ( (!lIsPart) && lFirstDummy <= getTopD() )
                                    {
                                        mFullNeighs[ *lIterIntersection * ( getTopD() + 1 ) + lFirstDummy ] = lComplementaryShadow;
                                    }
                                    
                                    lIsPart = false;
                                    lFirstDummy = getTopD()+1;
                                    for ( unsigned int lIterNeighIndices = 0; lIterNeighIndices < getTopD() + 1; lIterNeighIndices++)
                                    {
                                        const unsigned int lCurVal = getNeighs()[ lComplementaryShadow * ( getTopD() + 1 ) + lIterNeighIndices ];
                                        // If current new simplex is already part of neighborhood
                                        if ( lCurVal == *lIterIntersection )
                                            lIsPart = true;
                                        // If dummy
                                        else if ( ( lCurVal >= getNT() ) && (lFirstDummy > getTopD()) )
                                            lFirstDummy = lIterNeighIndices;
                                    }
                                    // If current new simplex is not part and a dummy exists
                                    if ( (!lIsPart) && lFirstDummy <= getTopD() )
                                    {
                                        mFullNeighs[ lComplementaryShadow * ( getTopD() + 1 ) + lFirstDummy ] = *lIterIntersection;
                                    }
                                }
                                
                                // Go through all neighbors and make sure that they are truly neighbors
                                for ( unsigned int lIterNeighIndices = 0; lIterNeighIndices < getTopD() + 1; lIterNeighIndices++)
                                {
                                    unsigned int lCurVal = getNeighs()[ *lIterIntersection * ( getTopD() + 1 ) + lIterNeighIndices ];
                                    if (lCurVal < getNT())
                                      // Make sure that neighbors are neighbors
                                      if (!areSimplicesNeighbors( *lIterIntersection, lCurVal ))
                                          return 16;
                                          
                                    lCurVal = getNeighs()[ lComplementaryShadow * ( getTopD() + 1 ) + lIterNeighIndices ];
                                    if (lCurVal < getNT())
                                        // Make sure that neighbors are neighbors
                                        if (!areSimplicesNeighbors( lComplementaryShadow, lCurVal ))
                                            return 17;
                                }
                            
                                // Increment pointer
                                lIterIntersection++;
                            }   // End of loop over updated simplices
                        
                        }   // End of handling neighborhood scope 
                        
                    }   // End of if other simplices did not have higher                
                    
                    
                }   // End of split of current simplex            
                
        // If reached maximum number of nodes
        if (pMaxNumNodes <= getNN())
        {
            // Set that no more should be updated
            lContinueToLoop = false;
            break;
        }
        
        // If has been updating and transformation is defined
        if ( (getNN() > lLastEpochNN) && (transformationPtr != NULL) )
        {
            // Transform updated points
            lStatus = (*transformationPtr)(&mFullNodes.data()[lLastEpochNN * getD()], getNN()-lLastEpochNN );
            if (lStatus)
                return 6;
        }

    }   // End of while loop while edges to split
    
    // Loop through all simplices
    #pragma omp parallell for
    for ( unsigned int lIterSimplex = 0; lIterSimplex < getNT(); lIterSimplex++ )
    {
        // Sort neighbors
        std::sort( &mFullNeighs.data()[ lIterSimplex * (getTopD()+1) ], &mFullNeighs.data()[ lIterSimplex * (getTopD()+1) ] + (getTopD()+1) );

        // Go through all elements    
        for (unsigned int lIterNodeInds = 0; lIterNodeInds < getTopD()+1; lIterNodeInds++)
        {
            // Set dummy variables to highest node number
            unsigned int * const lCurNodeIndsPtr = &mFullSimplices.data()[ lIterSimplex * (getTopD()+1) + lIterNodeInds ];
            if ( *lCurNodeIndsPtr > getNN() )
                *lCurNodeIndsPtr = getNN();
            // change value to dummy variables to highest simplex number
            unsigned int * const lCurSimplexIndsPtr = &mFullNeighs.data()[ lIterSimplex * (getTopD()+1) + lIterNodeInds ];
            if ( *lCurSimplexIndsPtr >= getNT() )
                *lCurSimplexIndsPtr = getNT();
        }
    }
    
    
        
    
    return 0;
}



// Method refining a simplex
int FullMesh::refineSimplex( const unsigned int pChosenSimplex, const double pMaxDiam , const unsigned int pMaxNewSimplices )
{

    // If max new simplices is zero
    if (pMaxNewSimplices == 0)
        // Stop refining
        return 0;

    // Get nodes of current simplex
    std::vector< Eigen::Map<const Eigen::VectorXd> > lNodes;
    for (unsigned int lIterNodes = 0; lIterNodes < getTopD()+1; lIterNodes++)
    {
        // Get node id
        const unsigned int lNodeId = mFullSimplices[pChosenSimplex * (getTopD()+1) + lIterNodes];
        // Insert node
        lNodes.push_back( Eigen::Map<const Eigen::VectorXd>( &mFullNodes.at( lNodeId*getD() ), getD() ) );
    }

    // Only dimensions 1 and 2 incorporated so far
    switch( getTopD() )
    {
        case 1:
        {
            // Create a node in between the current two
            Eigen::VectorXd lNewNode = 0.5d * ( lNodes.at(0) + lNodes.at(1) );  
            
            // If diameter is too small
            if (pMaxDiam >= ( lNodes.at(0) + lNodes.at(1) ).norm() )
                // Stop refining
                return 0;    
            
            // Introduce the new node to the list of nodes
            for (unsigned int lIterDims = 0; lIterDims < getD(); lIterDims++)
                mFullNodes.push_back( lNewNode(lIterDims) );
            mNumNodes++;
            
            // modify old simplex and create new simplex
            const unsigned int lSecondNodeId = mFullSimplices[pChosenSimplex*(getTopD()+1)+1];
            mFullSimplices[pChosenSimplex*(getTopD()+1)+1] = getNN()-1;
            mFullSimplices.push_back(getNN()-1);
            mFullSimplices.push_back(lSecondNodeId);
            mNumSimplices++;
            
            // Update pointers
            updateConstMeshPointers();
            
            // Call refine on the new simplices
            if (pMaxNewSimplices > 2)
            {
                refineSimplex( pChosenSimplex, pMaxDiam , (pMaxNewSimplices-1)/2 );
                refineSimplex( getNT()-1, pMaxDiam , (pMaxNewSimplices-1)/2 );
            }
        
            break;
        }    
        case 2:
        {
            
            // Get all edges of simplex
            const unsigned int lEdges[] = { 0, 1, 1, 2, 0, 2 };
            // find largest diameter
            double lLargestVal = 0.0;
            unsigned int lLargestInd = 0;            
            // Loop through all edges and compute length
            for (unsigned int lIterEdges = 0; lIterEdges < 3; lIterEdges++)
            {
                // Compute length
                const double lCurLength = ( lNodes.at( lEdges[lIterEdges*2] ) + lNodes.at( lEdges[lIterEdges*2+1] ) ).norm();
                // If largest so far
                if (lCurLength > lLargestVal)
                {
                    lLargestVal = lCurLength;
                    lLargestInd = lIterEdges;
                }
            }
            
            // If diameter is too small
            if (pMaxDiam >= lLargestVal )
                // Stop refining
                return 0;    
            
            
            // Split longest edge to get new node
            const Eigen::VectorXd lNewNode = 0.5d * ( lNodes.at( lEdges[lLargestInd*2] ) + lNodes.at( lEdges[lLargestInd*2+1] ) );
            
            // Introduce the new node to the list of nodes
            for (unsigned int lIterDims = 0; lIterDims < getD(); lIterDims++)
                mFullNodes.push_back( lNewNode(lIterDims) );
            mNumNodes++;
            
            // modify old simplex and create new simplex
            const unsigned int lSecondNodeId =  mFullSimplices[pChosenSimplex * (getTopD()+1) + lEdges[lLargestInd*2 + 1] ]; // Get node to exclude from first triangle
            const unsigned int lThirdNodeId = 3 - ( lEdges[lLargestInd*2] + lEdges[lLargestInd*2 + 1] ); // Get node not part of the splitted edge
            mFullSimplices[pChosenSimplex * (getTopD()+1) + lEdges[lLargestInd*2 + 1] ] = getNN()-1; // Instead insert new node
            mFullSimplices.push_back(getNN()-1); // Insert new node in new triangle
            mFullSimplices.push_back(lSecondNodeId); // Insert node excluded from old triangle
            mFullSimplices.push_back(lThirdNodeId); // Insert node not part of the splitted edge
            mNumSimplices++; // Advance simplex counter
            
            // Update pointers
            updateConstMeshPointers();
            
            // Call refine on the new simplices
            if (pMaxNewSimplices > 2)
            {
                refineSimplex( pChosenSimplex, pMaxDiam , (pMaxNewSimplices-1)/2 );
                refineSimplex( getNT()-1, pMaxDiam , (pMaxNewSimplices-1)/2 );
            }
        
        
            break;
        }    
        default:
            // Flag error
            return 1;
    }

    return 0;
}


// Populate arrays with corresponding mesh
int FullMesh::populateArrays( double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNeighs )
{
    // Update pointers of ConstMesh
    updateConstMeshPointers();
    // Call ConstMesh function
    return ConstMesh::populateArrays( pNodes, pD, pNumNodes, pSimplices, pNumSimplices, pTopD, pNeighs );    
}







// Compute all simplex boundaries and which simplices they are boundaries of.
int SimplexEdges::computeEdges( 
        const unsigned int * const pSimplices, const unsigned int pTopD, 
        const unsigned int pNumSimplices, const unsigned int pEdgeDim )
{

    // Make sure that mEdges is empty
    if ( mEdges.size() > 0)
        return 1;

    // Compute number of combinations of chosing pEdgeDim elements among (pTopologicalD+1) elements with order
    unsigned int lNumCombinations = 1;        
    for (unsigned int lIter = pTopD+1; lIter > pEdgeDim; lIter--)
    {
        lNumCombinations *= lIter;
    }
    // Compute number of edges for a simplex ( (pTopologicalD + 1) choose pEdgeDim )
    unsigned int lNChooseK = 1;
    for (unsigned int lIter = 2; lIter <= (pTopD + 1 - pEdgeDim); lIter++)
    {
        lNChooseK *= lIter;
    }
    lNChooseK = lNumCombinations / lNChooseK;
    mNumEdgesPerSimplex = lNChooseK;
    
    // Get vector of indices
    std::vector<unsigned int> lIndicesOfSimplex;
    for (unsigned int lIterNumNodesInSimplex = 0; lIterNumNodesInSimplex < pTopD+1; lIterNumNodesInSimplex++)
        lIndicesOfSimplex.push_back(lIterNumNodesInSimplex);
    // Preallocate vector for storing all edges indices
    std::vector<std::set<unsigned int>> lEdgeIndices;
    lEdgeIndices.reserve(lNChooseK);
    // Compute edges
    {
        const int lStatus = mesh_getEdgesOfSimplex( lEdgeIndices, lNumCombinations,
            pEdgeDim, pTopD, lIndicesOfSimplex.data() );
        if (lStatus != 0)
            // Flag error
            return 2;
    }
    
    // Loop through all simplices
    for (unsigned int lIterSimplex = 0; lIterSimplex < pNumSimplices; lIterSimplex++)
    {    
        // Loop through all current edges
        for ( std::vector<std::set<unsigned int>>::const_iterator lIterEdgeIndices = lEdgeIndices.begin(); 
            lIterEdgeIndices != lEdgeIndices.end(); ++lIterEdgeIndices )
        {
            // Get current edge from edge indices
            std::set<unsigned int> lCurEdge;
            for (std::set<unsigned int>::const_iterator lIterEdgeIndex = lIterEdgeIndices->begin();
                lIterEdgeIndex != lIterEdgeIndices->end(); ++lIterEdgeIndex )
            {
                // Insert current node from current edge into lCurEdge
                lCurEdge.insert( pSimplices[ lIterSimplex * (pTopD+1) + *lIterEdgeIndex ] );
            }
        
            // Create corresponding edgeElement
            SimplexEdges::EdgeElement lCurEdgeElement( lCurEdge, std::set<unsigned int>() );
            // See if edge exists in set
            std::set< SimplexEdges::EdgeElement >::iterator lFoundEdge =  mEdges.find( lCurEdgeElement );
            // If existed
            if ( lFoundEdge != mEdges.end() )
            {
                // Get edge element and insert new simplex
                lCurEdgeElement = *lFoundEdge;
                lCurEdgeElement.second.insert( lIterSimplex );
                // Update maximum simplices per edge
                if (lCurEdgeElement.second.size() > mMaxSimplicesPerEdge)
                    mMaxSimplicesPerEdge = lCurEdgeElement.second.size();
                // Insert the edge element by removing and inserting a new                
                mEdges.erase(lFoundEdge);
                ++lFoundEdge;
                mEdges.insert( lCurEdgeElement );
            }
            else
            {
                // Add current simplex to edge element
                lCurEdgeElement.second.insert(lIterSimplex);
                // Insert current edge element
                mEdges.insert( lCurEdgeElement );
            }
        }
    }
    
    // Store the dimensionality of edges
    mEdgeDim = pEdgeDim;
    
    return 0;
}


// Populate edges array
int SimplexEdges::populateEdges( unsigned int * pEdges, const unsigned int pNumEdges, const unsigned pNumNodes ) const
{
    if (pNumEdges != mEdges.size())
        return 1;
    if (pNumNodes != mEdgeDim)
        return 2;
    
    // Loop through all edges
    unsigned int lIter = 0;
    for ( std::set< EdgeElement, CompareEdges >::const_iterator lIterEdges = mEdges.begin();
        lIterEdges != mEdges.end(); ++lIterEdges )
        // Loop through all nodes in current edge
        for ( std::set<unsigned int>::const_iterator lIterNodes = lIterEdges->first.begin(); 
            lIterNodes != lIterEdges->first.end(); ++lIterNodes )
        {
            // Insert current node in output
            pEdges[lIter] = *lIterNodes;
            // Increase counter
            lIter++;
        }

    return 0;
}

// Populate simplices for each edge array
int SimplexEdges::populateEdgesSimplexList( unsigned int * const pSimplexList, const unsigned int pNumEdges, 
        const unsigned pMaxNumSimplicesPerEdge, const unsigned int pNumSimplices ) const
{
    if (pNumEdges != mEdges.size())
        return 1;
    if (pMaxNumSimplicesPerEdge != mMaxSimplicesPerEdge)
        return 2;
    
    // Loop through all edges
    unsigned int lIterEdgesIndex = 0;
    for ( std::set< EdgeElement, CompareEdges >::const_iterator lIterEdges = mEdges.begin();
        lIterEdges != mEdges.end(); ++lIterEdges )
    {
        unsigned int lIterSimplexIndex = 0;
        // Loop through all simplices in current edge
        for ( std::set<unsigned int>::const_iterator lIterSimplices = lIterEdges->second.begin(); 
            lIterSimplices != lIterEdges->second.end(); ++lIterSimplices )
        {
            // Insert current simplex in output
            pSimplexList[lIterEdgesIndex * pMaxNumSimplicesPerEdge + lIterSimplexIndex] = *lIterSimplices;
            // Increase counter
            lIterSimplexIndex++;
        }
        // Add nan for the rest of the elements if not associated with maximal number of simplices 
        while (lIterSimplexIndex < pMaxNumSimplicesPerEdge)
        {
            // Add dummy for the edges that do not have full coverage of simplices
            pSimplexList[lIterEdgesIndex * pMaxNumSimplicesPerEdge + lIterSimplexIndex] = pNumSimplices;
            // Increase counter
            lIterSimplexIndex++;
        }
        
        // Increase counter
        lIterEdgesIndex++;
    }

    return 0;
}

// Populate map of all edges to each simplex
int SimplexEdges::populateSimplexEdgesList( unsigned int * const pEdgeList, 
    const unsigned int pNumSimplices, const unsigned int pNumEdgesPerSimplex ) const
{
    if ( pNumEdgesPerSimplex != getNumEdgesPerSimplex() )
        return 1;


    // Loop through edge list and set to nan
    for (unsigned int lIter = 0; lIter < pNumSimplices*pNumEdgesPerSimplex; lIter++)
        // A value higher than any edge index is considered as nan
        pEdgeList[lIter] = mEdges.size();

    // Loop through all edges
    unsigned int lIterEdgesIndex = 0;
    for ( std::set< EdgeElement, CompareEdges >::const_iterator lIterEdges = mEdges.begin();
        lIterEdges != mEdges.end(); ++lIterEdges )
    {
        // Loop through all simplices
        for ( std::set<unsigned int>::const_iterator lIterSimplices = lIterEdges->second.begin(); 
            lIterSimplices != lIterEdges->second.end(); ++lIterSimplices )
        {
            // Get pointer to current simplex list of edges
            unsigned int * const lCurSimplexList = &pEdgeList[ *lIterSimplices * pNumEdgesPerSimplex ];
            
            // find first non-written edge for this simplex
            unsigned int lIterSimplexElement = 0;
            while ( lCurSimplexList[lIterSimplexElement] < mEdges.size() )
            {
                if ( lIterSimplexElement >= pNumEdgesPerSimplex )
                    // Flag error
                    return 1;
                // Increase counter
                lIterSimplexElement++;
            }
            // Insert 
            lCurSimplexList[lIterSimplexElement] = lIterEdgesIndex;
        }
        // Increase counter
        lIterEdgesIndex++;
    }
    
    return 0;
}

// Acquire index of given edge
unsigned int SimplexEdges::findEdgeIndexGivenEdge( const unsigned int * const pEdge,
    const unsigned int * pEdges, const unsigned int pNumEdges, const unsigned int pEdgeDim )
{

    if (pNumEdges == 0)
        // Flag error
        return 1;

    // Create compare class
    CompareNodeSets lCompareNodeSets( pEdgeDim );
    
    unsigned int lLeftIndex = 0;
    unsigned int lRightIndex = pNumEdges-1;
    unsigned int lDifference = lRightIndex - lLeftIndex;
    
    // Sanity check
    {
        // Is edge to the left of left index?
        if ( lCompareNodeSets( pEdge, &pEdges[ lLeftIndex * pEdgeDim ] ) )
            // Flag error
            return pNumEdges;
        // Is edge to the right of right index?
        if ( lCompareNodeSets( &pEdges[ lRightIndex * pEdgeDim ], pEdge ) )
            // Flag error
            return pNumEdges;
    }
    
    // Loop while tightening the left right indices
    while ( lDifference > 5 )
    {
        // Get middle index 
        const unsigned int lMiddleIndex = lDifference / 2 + lLeftIndex;
        // Is the edge to the right of middle index?
        if ( lCompareNodeSets( &pEdges[ lMiddleIndex * pEdgeDim ], pEdge ) )
            // Set left index to middle index
            lLeftIndex = lMiddleIndex;
        else
        {
            // Set right index to middle index
            lRightIndex = lMiddleIndex;
            // Is the edge not to the left of middle index?
            if ( !lCompareNodeSets( pEdge, &pEdges[ lMiddleIndex * pEdgeDim ] ) )
                // The middle index is the correct index
                lLeftIndex = lMiddleIndex;
        }
        // Update difference
        lDifference = lRightIndex - lLeftIndex;
    }
    // The last steps can be performed in linear order
    while (lDifference > 0 )
    {
        // Move left index forward
        lLeftIndex++;
        
        // Is the edge not to the right of left index?
        if ( !lCompareNodeSets( &pEdges[ lLeftIndex * pEdgeDim ], pEdge ) )
            // We have found the right index
            lRightIndex = lLeftIndex;
        // If some error (can happen if input is not in order as it should be)
        else if (lLeftIndex >= lRightIndex)
            // Flag error
            return pNumEdges+1;
            
        // Update difference
        lDifference = lRightIndex - lLeftIndex;
    }
    // The correct index should be the same as the left index (or right index since they should be equal)
    return lLeftIndex;
}







MapToSimp::MapToSimp( const double * const pPoints, const unsigned int pD, const unsigned int pTopD ) : mD(pD), mTopD(pTopD)
{
        
    // Get vector of first point
    Eigen::Map<const Eigen::VectorXd> lPoint0( pPoints, mD );
    mPoint0 = Eigen::VectorXd( lPoint0 ); 
    
    if (mTopD > 0)
    {
        // Get matrix of consecutive points 
        Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > lPoints( &pPoints[mD], mD, mTopD ); 
        // Get matrix of point 0 subtracted from consecutive points ( the matrix of row vectors {point_j - _point_0}_j )
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> lF = lPoints.colwise() - mPoint0;
        // Perform a QR facotorization of lF
        mQR = lF.colPivHouseholderQr();
        
        // Get the R1 matrix
        Eigen::MatrixXd lR1 = mQR.matrixR().topLeftCorner(mTopD, mTopD).triangularView<Eigen::Upper>();
        // Get the determinant
        mDeterminant = lR1.diagonal().prod();
    }
    else
        mDeterminant = 1.0d;
}

Eigen::VectorXd MapToSimp::solve( const Eigen::VectorXd & pVector ) const
{
    // If of full rank
    if (mD == mTopD)
        return mQR.solve(pVector);

    // Get the R1 matrix
    Eigen::MatrixXd lR1 = mQR.matrixR().topLeftCorner(mTopD, mTopD).triangularView<Eigen::Upper>();
    // Get the permutation matrix P
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lP = mQR.colsPermutation(); 
    // Get Q1
    const Eigen::MatrixXd lQ = mQR.householderQ();
    const Eigen::MatrixXd lQ1 = lQ.topLeftCorner(mD, mTopD);
    
    // Get solution
    Eigen::VectorXd lTemp = lQ1.transpose() * pVector;
    lTemp = lR1.lu().solve( lTemp );
    lTemp = lP * lTemp;
    
    return lTemp;
}

Eigen::VectorXd MapToSimp::solveTransposed( const Eigen::VectorXd & pVector ) const
{
    // Get the R1^T matrix
    Eigen::MatrixXd lR1T = mQR.matrixR().topLeftCorner(mTopD, mTopD).transpose().triangularView<Eigen::Lower>();
    // Get the permutation matrix P
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lP = mQR.colsPermutation(); 
    // Get Q1
    const Eigen::MatrixXd lQ = mQR.householderQ();
    const Eigen::MatrixXd lQ1 = lQ.topLeftCorner(mD, mTopD);    
    
     // Get solution
    Eigen::VectorXd lTemp = lP.inverse() * pVector;
    lTemp = lR1T.lu().solve( lTemp );
    lTemp = lQ1 * lTemp;

    return lTemp;
}

// Get divergence length from subspace spanned by simplex
double MapToSimp::getDivergence( const Eigen::VectorXd & pVector ) const
{
    if (mD == mTopD)
        return 0.0d;

    // Get vector from point 0
    const Eigen::VectorXd lVector = pVector - mPoint0;
    // Get Q2
    const Eigen::MatrixXd lQ = mQR.householderQ();
    const Eigen::MatrixXd lQ2 = lQ.block(0, mTopD, mD, mD - mTopD);
    // Get projection on complement to simplex subspace
    const Eigen::VectorXd lCompProj = lQ2.transpose() * lVector;
    // return l2 distance of lCompProj
    return lCompProj.norm();
}

// Map from simplex space to original space
Eigen::VectorXd MapToSimp::multiplyWithF( const Eigen::VectorXd & pVector ) const
{
    // Get the R1 matrix
    Eigen::MatrixXd lR1 = mQR.matrixR().topLeftCorner(mTopD, mTopD).triangularView<Eigen::Upper>();
    // Get the permutation matrix P
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lP = mQR.colsPermutation(); 
    // Get Q1
    const Eigen::MatrixXd lQ = mQR.householderQ();
    const Eigen::MatrixXd lQ1 = lQ.topLeftCorner(mD, mTopD);    
    
    Eigen::VectorXd lTemp = (lP.inverse() * pVector);
    lTemp = lR1 * lTemp;
    lTemp = lQ1 * lTemp;
    
    return lTemp;
}






std::vector< std::pair< unsigned int, unsigned int> > ExtendMesh::neighSimps( 
    const unsigned int pSimplexInd, const char pPlaced) const
{
    // Get all edge indices of current simplex
    const std::vector<unsigned int> lEdges = edgesOfOldSimplex( pSimplexInd );

    // Get all simplices sharing these edges
    std::vector< std::pair< unsigned int, unsigned int> > lCurNeighSimplices;
    for ( std::vector<unsigned int>::const_iterator lIterEdges = lEdges.begin(); 
        lIterEdges != lEdges.end(); ++lIterEdges )
    {
        // Get first simplex of edge
        unsigned int lCurNeigh = mEdgeIdentity->at( *lIterEdges * 2 );
        // If happens to be current simplex
        if ( lCurNeigh == pSimplexInd) 
            // Get second simplex of edge
            lCurNeigh = mEdgeIdentity->at( *lIterEdges * 2 + 1);
        // If current neighbor is an actual simplex
        if (lCurNeigh < mNumSimplices)
        {
            // Se if placed
            const bool lIsPlaced = mPlacedSimplices.at(lCurNeigh);
            // If placed and we want placed, or if not placed and want not placed or if we want anything
            if ( (pPlaced == 0) || ( (pPlaced == 1) && lIsPlaced ) || ( (pPlaced == 2) && !lIsPlaced )  )
                // Insert neighbor in list
                lCurNeighSimplices.push_back( std::pair<unsigned int, unsigned int>(lCurNeigh, *lIterEdges) );
        }
    }

    return lCurNeighSimplices;
}

std::list< std::set<unsigned int> > ExtendMesh::computeComplyEdgesFromPlaced( 
    const unsigned int pSimplexInd, const std::vector<std::pair<unsigned int, unsigned int>> & pNeighs, 
            const unsigned int * const pNewSimplices ) const
{
    // Initialize list of all edges to comply to
    std::list< std::set<unsigned int> > lComplyEdges;
    // Go through all neighbouring simplices
    for ( std::vector< std::pair<unsigned int, unsigned int> >::const_iterator lIterNeighs = pNeighs.begin(); 
        lIterNeighs != pNeighs.end(); ++lIterNeighs )
    {
        // Get set of the edge connecting the two simplices
        std::set<unsigned int> lOrigEdge;
        lOrigEdge.insert( mEdges->begin() + mTopD * lIterNeighs->second, mEdges->begin() + mTopD * (lIterNeighs->second + 1) );
              
        // Go through all new sub-simplices of old neighbor
        for (unsigned int lIterSubSimp = 0; lIterSubSimp < (mTopD+1); lIterSubSimp++)
        {
            // Initialize set of matching nodes
            std::set<unsigned int> lMatches;
            // Loop through all node indices of sub simp
            for (unsigned int lIterNode = 0; lIterNode < (mTopD+2); lIterNode++)
            {
                // Get current node index
                const unsigned int lCurNodeIndex = pNewSimplices[ lIterNeighs->first * (mTopD+1) * (mTopD+2) + lIterSubSimp * (mTopD+2) + lIterNode ];
                // Get corresponding value in old mesh
                const unsigned int lCurNodeOldIndex = lCurNodeIndex % mNumNodes;
                // See if old value is in lOrigEdge
                if ( lOrigEdge.count( lCurNodeOldIndex ) > 0 )
                    // If so, add index
                    lMatches.insert( lCurNodeIndex );
            }
            // If lMatches has the right number of matching indices
            if (lMatches.size() == mTopD+1)
                // Add to comply edges
                lComplyEdges.push_back( lMatches );
        }
    }
    return lComplyEdges;
}

std::set<unsigned int> ExtendMesh::projectNewOntoOldSimplex( std::set<unsigned int> & pNewSimplex ) const
{
    std::set<unsigned int> lOldSimplex;
    // Go through all elements of current simplex
    for ( std::set<unsigned int>::const_iterator lIterSimplex = pNewSimplex.begin(); 
        lIterSimplex != pNewSimplex.end(); ++lIterSimplex )
    {
        lOldSimplex.insert( (*lIterSimplex) % mNumNodes );
    }
    return lOldSimplex;
}

bool ExtendMesh::onPrismEdge( std::set<unsigned int> & pNewSimplex ) const
{
    // Get projection onto old simplex
    std::set<unsigned int> lProjected = projectNewOntoOldSimplex( pNewSimplex );
    // If too few elements
    if (lProjected.size() <= mTopD)
        return true;
    
    // Initialize as zero    
    char lSides = 0;
    // Go through all elements
    for ( std::set<unsigned int>::const_iterator lIter = pNewSimplex.begin(); lIter != pNewSimplex.end(); ++lIter )
    {
        // If on lower side
        if ( (*lIter) / mNumNodes == 0 )
            lSides |= 1;
        else // If on higher side
            lSides |= 2;
    }
    // If has not been on both sides
    if (lSides != 3)
        return true;
        
    return false;
}

bool ExtendMesh::isSimplexOnBorder( const unsigned int pSimplexId ) const
{
    // If simplex id is not viable
    if (pSimplexId > mNumSimplices)
        // Return on border
        return true;
    // Go through all edges of simplex
    for ( std::vector<unsigned int>::const_iterator lIter = std::next( mSimplexIdentity->begin(), pSimplexId * mNumEdgesSimp );
        lIter != std::next( mSimplexIdentity->begin(), (pSimplexId+1) * mNumEdgesSimp ); ++lIter )
    {
        // Get first simplex of edge
        const unsigned int lFirstSimplex = mEdgeIdentity->at( 2 * (*lIter) );
        // Get second simplex of edge
        const unsigned int lSecondSimplex = mEdgeIdentity->at( 2 * (*lIter) + 1 );
        
        // If any simplex is null
        if ( (lFirstSimplex >= mNumSimplices) || (lSecondSimplex >= mNumSimplices) )
            // Mark that on border
            return true;
    }
    
    return false;
}

unsigned int ExtendMesh::shortestPathToFreedom( const unsigned int pSimplexId, const unsigned int pPrevSimplexId, 
    std::list<unsigned int> & pPath ) const
{
    // Get all placed neighboring old simplices
    const std::vector< std::pair< unsigned int, unsigned int> > lNeighs = neighSimps( pSimplexId, 1 );
    // If number of placed simplices is not (mTopD+1) 
    if ( lNeighs.size() < (mTopD+1) )
    {
        // Push in current simplex
        pPath.push_front(pSimplexId);
        // one of the neighbors needs to be either the border or non-placed simplex
        return 0;
    }

    // Initialize minimum distance
    unsigned int lMinDistance = std::numeric_limits<unsigned int>::max();
    // Loop over neighbors
    for ( std::vector< std::pair< unsigned int, unsigned int> >::const_iterator lIterNeighs = lNeighs.begin(); 
        lIterNeighs != lNeighs.end(); ++lIterNeighs )
    {
        // If neighbor is previous step on path
        if ( lIterNeighs->first == pPrevSimplexId)
            // Continue loop
            continue;
            
        // Call shortest path on current neighbor
        std::list<unsigned int> lTempPath;
        const unsigned int lCurDistance = shortestPathToFreedom( lIterNeighs->first, pSimplexId, lTempPath );
        // If distance is smaller than current minimum
        if (lCurDistance < lMinDistance)
        {
            // Replace current minimum
            lMinDistance = lCurDistance;
            // Use current path
            pPath = lTempPath;
        }
    }
    // Push in current simplex
    pPath.push_front( pSimplexId );
    // return minimum distance plus one
    return lMinDistance + 1;
}

int ExtendMesh::computeNewSubSimplices( const unsigned int pSimplexInd, const unsigned int * const pNewSimplices, std::vector< std::set<unsigned int> > & pOut ) const
{
    // reserve output
    pOut.reserve( mTopD+1 );

    // Get all possible node indices in current original simplex
    const std::set<unsigned int> lPossibleNodeInds = allPossibleNodeInds( pSimplexInd );
    
    // Get all neighbors of current simplex which are placed
    std::vector< std::pair< unsigned int, unsigned int> >  lCurNeighs = neighSimps( pSimplexInd, 1 );
    // Compute list of all edge faces to comply to
    std::list< std::set<unsigned int> > lComplyFaces = computeComplyEdgesFromPlaced( pSimplexInd, lCurNeighs, pNewSimplices );
    // Initialize list of all edges to avoid
    std::list< std::set<unsigned int> > lBannedEdges;
    std::set<unsigned int> lIntersection;
    
    
    // Walk through all new sub-simplices from old simplex
    for (unsigned int lIterSubSimp = 0; lIterSubSimp < (mTopD+1); lIterSubSimp++)
    {
        // Initialize list of all edges to claim
        std::vector< std::set<unsigned int> > lClaimedFaces;
    
        // Initialize current sub-simplex
        std::set<unsigned int> lCurSubSimp;
        
        // Loop through comply edges
        for ( std::list< std::set<unsigned int> >::iterator lIterComplyEdges1 = lComplyFaces.begin(); 
            lIterComplyEdges1 != lComplyFaces.end(); ++lIterComplyEdges1 )
        {
            // Set current sub-simplex as curretn comply edge
            lCurSubSimp = *lIterComplyEdges1;
        
            // Loop through remaining comply edges
            for ( std::list< std::set<unsigned int> >::iterator lIterComplyEdges2 = std::next( lIterComplyEdges1, 1 ); 
            lIterComplyEdges2 != lComplyFaces.end();  )
            {
                // Get intersection of current edge and current sub-simplex
                misc_setIntersection( lCurSubSimp, *lIterComplyEdges2, lIntersection);
                    
                // Assume shouldn't claim face            
                bool lClaimFace = false;            
                // If current simplex is already fully defined 
                if (lCurSubSimp.size() == mTopD+2)
                {
                    // if edge is fully covered
                    if (lIntersection.size() == mTopD+1)
                        // Claim face
                        lClaimFace = true;
                }
                else if (lIntersection.size() == mTopD) // Else if face is covered enough
                {
                    // Acquire union of current edge and current sub-simplex
                    std::set<unsigned int> lUnion = lCurSubSimp;
                    lUnion.insert( lIterComplyEdges2->begin(), lIterComplyEdges2->end() );
                    // If not on prism edge
                    if ( !onPrismEdge( lUnion ) )
                    {
                        // Set to current sub simp
                        lCurSubSimp = lUnion;
                        // Claim face
                        lClaimFace = true;
                    }
                }
                   
                // If should claim face
                if (lClaimFace)                
                {
                    // Claim edge
                    lClaimedFaces.push_back( *lIterComplyEdges2 );
                    // Remove comply edge
                    lIterComplyEdges2 = lComplyFaces.erase( lIterComplyEdges2 );
                }
                else
                    // Increment
                    ++lIterComplyEdges2;
            }
            
            // If sub-simplex is fully defined or reached end
            if ( (lCurSubSimp.size() >= mTopD + 2) || ( lIterComplyEdges1 ==  std::prev(lComplyFaces.end(),1) ) )
            {
                // Claim edge
                lClaimedFaces.push_back( *lIterComplyEdges1 );
                // Remove comply edge
                lIterComplyEdges1 = lComplyFaces.erase( lIterComplyEdges1 );
                // Break out of loop
                break;
            }
        }
        
        // If sub simplex is yet not fully defined
        if ( lCurSubSimp.size() < mTopD + 2 )
        {
            // If no sub simp
            if (lCurSubSimp.size() == 0)
                // Walk through all nodes in new simplex
                for (unsigned int lIterNode = 0; lIterNode < (mTopD+2); lIterNode++)
                {
                    // Get matching node in original simplex
                    const unsigned int lCurNodeInd = (lIterNode + lIterSubSimp) % (mTopD+1);
                    // Get node in original simplex
                    unsigned int lCurNode = mOldSimplices[ pSimplexInd * (mTopD+1) + lCurNodeInd];
                    // If needed, raise it
                    if ( lIterNode + lIterSubSimp >= mTopD+1 )
                        lCurNode += mNumNodes;
                    // Add node
                    lCurSubSimp.insert( lCurNode );
                }
            else // If lCurSubSimp is partially defined (then has to be lacking only one)
            {
                // Acquire set of possible nodes to choose from 
                std::set<unsigned int> lNodesToChooseFrom;
                std::set_difference( lPossibleNodeInds.begin(), lPossibleNodeInds.end(), 
                    lCurSubSimp.begin(), lCurSubSimp.end(),
                    std::inserter( lNodesToChooseFrom, lNodesToChooseFrom.begin() ) );
                // Go through each of them separately
                for ( std::set<unsigned int>::const_iterator lIterChooseFrom = lNodesToChooseFrom.begin(); 
                    lIterChooseFrom != lNodesToChooseFrom.end(); ++lIterChooseFrom)
                {
                    // Get set with added current node
                    std::set<unsigned int> lTestSet = lCurSubSimp;
                    lTestSet.insert( *lIterChooseFrom );
                    
                    // If on prism edge
                    if ( onPrismEdge( lTestSet ) )
                        // This is not an allowed sub-simplex
                        continue;
                    
                    // Assume okay
                    bool lAllowed = true;
                    // See if current test set is sharing faces with any of the banned faces
                    for ( std::list< std::set<unsigned int> >::const_iterator lIterBanned = lBannedEdges.begin(); 
                        lIterBanned != lBannedEdges.end(); ++lIterBanned )
                        // If the face is shared
                        if ( misc_numSharedElements(lTestSet, *lIterBanned, lIntersection) == (mTopD+1) )
                        {
                            // Forbid current simplex
                            lAllowed = false;
                            break;
                        }
                    // If test set is allowed
                    if (lAllowed)
                    {
                        // Set current sub-simplex to test set
                        lCurSubSimp = lTestSet;
                        break;
                    }
                }                
            }
            // Make sure that simplex has the right number of nodes     
            assert( lCurSubSimp.size() == (mTopD+2) );
        }
        
        // Insert current sub-simplex into output
        pOut.push_back( lCurSubSimp );
        
        // Copy claimed to banned
        for ( std::vector<std::set<unsigned int>>::const_iterator lIterClaimed = lClaimedFaces.begin(); lIterClaimed != lClaimedFaces.end(); ++lIterClaimed)
            // Push into banned edges
            lBannedEdges.push_back( *lIterClaimed );
        
        // Go through all elements of current simplex
        for ( std::set<unsigned int>::const_iterator lIterCurSimp = lCurSubSimp.begin(); 
            lIterCurSimp != lCurSubSimp.end(); ++lIterCurSimp )
        {
            // Acquire an edge by excluding current element
            std::set<unsigned int> lCurEdge;
            std::set_difference( lCurSubSimp.begin(), lCurSubSimp.end(),
                lIterCurSimp, std::next(lIterCurSimp, 1),
                std::inserter(lCurEdge, lCurEdge.begin()) );

            // Assume edge is not banned
            bool lBanned = false;
            // Goo through all claimed
            for ( std::list< std::set<unsigned int> >::const_iterator lIterBanned = lBannedEdges.begin(); 
                lIterBanned != lBannedEdges.end(); ++lIterBanned )
                // See if current edge is banned
                if ( lCurEdge == *lIterBanned )
                {
                    // Mark as claimed and exit loop
                    lBanned = true;
                    break;
                }
                
            // If not banned
            if (!lBanned)
            {
                // If on prism edge
                if ( onPrismEdge( lCurEdge ) )
                    // Add to banned list since on edge
                    lBannedEdges.push_back(lCurEdge);
                else
                    // Add edge to comply list
                    lComplyFaces.push_back( lCurEdge );        
            }
        }
        
    } // End of loop over sub-simplices
    
    // if faces has not been complied
    if( lComplyFaces.size() != 0 )
        // Return no comply error
        return 1;
        
    return 0;
}









int mesh_getEdgesAndRelations( std::vector<unsigned int> &pEdges, std::vector<unsigned int> &pSimplexIdentity, std::vector<unsigned int> &pEdgeIdentity,
    const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, const unsigned int pNumEdgesSimp )
{    
    // Get number of edges for each simplex
    const unsigned int lNumEdgesSimp = (pNumEdgesSimp == 0) ? mesh_nchoosek( pTopD+1, pTopD ) : pNumEdgesSimp;
    
    // Initialize vector of all edges
    pEdges = std::vector<unsigned int>( lNumEdgesSimp * pNumSimplices * pTopD, 0 );
    // Initialize simplex identities
    pSimplexIdentity = std::vector<unsigned int>( lNumEdgesSimp * pNumSimplices, 0 );
    // Initialize edge index counter
    unsigned int lEdgeIndex = 0;
    // Get edges and simplex identity    
    {
        int lStatus = mesh_getEdgesOfSimplices( pEdges.data(), &lEdgeIndex, pSimplexIdentity.data(), pTopD, 
            pTopD, pSimplices, pNumSimplices );
        if (lStatus)
            return 1 + lStatus;
    }
    // Remove unnecessary pEdges space
    pEdges.erase( pEdges.begin() + pTopD * lEdgeIndex, pEdges.end() );
    
    // Initialize simplex memberships of all edges
    pEdgeIdentity = std::vector<unsigned int>( 2*lEdgeIndex, pNumSimplices );
    
    // Loop through all simplices        
    for (unsigned int lIterSimp = 0; lIterSimp < pNumSimplices; lIterSimp++)
    {
        // Loop through all edges
        for (unsigned int lIterEdge = 0; lIterEdge < lNumEdgesSimp; lIterEdge++ )
        {   // Get current edge
            const unsigned int lCurEdge = pSimplexIdentity.at( lNumEdgesSimp * lIterSimp + lIterEdge );
            // If no simplex inserted for current edge
            if ( pEdgeIdentity.at( lCurEdge * 2 ) == pNumSimplices )
                // Do insert
                pEdgeIdentity.at( lCurEdge * 2 ) = lIterSimp;
            // If no second simplex inserted for current edge
            else if ( pEdgeIdentity.at( lCurEdge * 2 + 1 ) == pNumSimplices )
                // Do insert
                pEdgeIdentity.at( lCurEdge * 2 + 1) = lIterSimp;
            // If both were inserted already
            else
                // Return error
                return 1;
        }
    }
        
    return 0;
}




// Get barycentric coordinates from standard simplex coordinates
inline Eigen::VectorXd mesh_getBarycentricFromStandard( const Eigen::VectorXd & lStand )
{
    Eigen::VectorXd lBary( lStand.size()+1 );
    
    double lFirstBary = 0.0d;
    for (unsigned int lIter = 0; lIter < lStand.size(); lIter++)
    {
        lBary(lIter+1) = lStand.coeff(lIter);
        lFirstBary += lBary.coeff(lIter+1);
    }
    lBary(0) = 1.0d - lFirstBary;
    
    return lBary;
}



// Acquire a vector of all unique edges of a simplex
int mesh_getEdgesOfSimplex( std::vector< std::set<unsigned int> > & pOut, const unsigned int pNumCombinations,
    const unsigned int pEdgeDim, const unsigned int pTopologicalD, const unsigned int * const pSimplex )
{
    // If dimensioanlity is not correct
    if (pTopologicalD < pEdgeDim)
        return 1;
    
    const unsigned int lAllocation = pNumCombinations * pEdgeDim;

    // Create vector for storing all allocated edges
    std::vector< unsigned int> lEdges(lAllocation);
    // Acquire current simplex edges
    int lStatus = mesh_recurrentEdgeFinder( lEdges.data(), lAllocation, 
        pEdgeDim, pSimplex, pTopologicalD+1);
    // If error
    if (lStatus)
        return 2;
        
    // Convert from a vector to a vector of sets
    std::vector< unsigned int >::const_iterator lIterEdge = lEdges.begin();
    // Loop through all edges
    for (unsigned int lIter = 0; lIter < pNumCombinations; lIter++)
    {   
        // Insert edge in pOut
        pOut.push_back( std::set<unsigned int>(lIterEdge, lIterEdge + pEdgeDim) );
        // Advance iterator
        lIterEdge = lIterEdge + pEdgeDim;
    }
        
    // Go through the vector of sets    
    for (std::vector<std::set<unsigned int> >::iterator lIterEdge = pOut.begin(); lIterEdge != pOut.end(); ++lIterEdge ) 
    {
        // Go through all sets behind it
        for (std::vector<std::set<unsigned int> >::iterator lIterEdge2 = lIterEdge+1; lIterEdge2 != pOut.end();  ) 
        {
            // If equal
            if ( *lIterEdge == *lIterEdge2 )
                // Remove lIterEdge2
                lIterEdge2 = pOut.erase(lIterEdge2);
            else
                ++lIterEdge2;
        }
    }
    
    // Return vector of sets
    return 0;
}




// Global list storing full meshes
GlobalVariablesList<FullMesh> gMeshes;
// Global list storing boundaries of simplices
GlobalVariablesList<SimplexEdges> gEdgesOfSimplices;




extern "C"
{

    


    // Find which simplex the points belong to (only works in R^d, and kind of on spheres, no embedded manifolds)
    int mesh_getObservationMatrix( double * const pData, unsigned int * const pRow, unsigned int * const pCol, const unsigned int pNumNonZeros,
        const double * const pPoints, const unsigned int pNumPoints,
        const double * const pNodes, const unsigned int pNumNodes,
        const unsigned int * const pMesh, const unsigned int pNumSimplices,
        const unsigned int pD, const unsigned int pTopD, const double pEmbTol)
    {
        // Create internal representation of mesh
        ConstMesh lConstMesh( pNodes, pD, pNumNodes, pMesh, pNumSimplices, pTopD );
        
        // If number of points are many 
        if (pNumPoints > 10)
            // Compute mesh graph to speed up the process
            lConstMesh.computeMeshGraph( 1000, 0.0d, 20 );
        
        // Get which triangle each point is a member of
        std::vector<unsigned int> lMemberList;
        lMemberList.reserve(pNumPoints);
        std::vector<double> lBaryList;
        lBaryList.reserve(pNumPoints*(pTopD+1));
        int lStatus = lConstMesh.getASimplexForPoint( pPoints, pNumPoints, lMemberList.data(), lBaryList.data(), pEmbTol );
        if (lStatus != 0)
            // Flag error
            return lStatus;
        
        // Keeps track of which data index the next observation element should be pushed into
        unsigned int lDataIndex = 0;
    
        // Loop through each point
        for ( unsigned int lIterPoint = 0; lIterPoint < pNumPoints; lIterPoint++ )
        {            
            // Loop through each node in simplex
            for (unsigned int lIterNodes = 0; lIterNodes < (pTopD+1); lIterNodes++)
            {
                const unsigned int lSimplexId = lMemberList[lIterPoint];
                // If simplex was found
                if ( lSimplexId  < lConstMesh.getNT() )
                {
                    // Get lIterNodes node from the simplex of the current point
                    const unsigned int lCurNode = pMesh[ lSimplexId * (pTopD+1) + lIterNodes ];
                
                    pData[lDataIndex + lIterNodes] = lBaryList[lIterPoint*(pTopD+1) + lIterNodes];
                    pRow[lDataIndex + lIterNodes] = lIterPoint;
                    pCol[lDataIndex + lIterNodes] = lCurNode;
                }
                else
                {
                    pData[lDataIndex + lIterNodes] = 0.0;
                    pRow[lDataIndex + lIterNodes] = lIterPoint;
                    pCol[lDataIndex + lIterNodes] = 0;
                }
    
            }        
            // Iterate data index
            lDataIndex += pTopD+1;

        }   // Stop looping over points    
    
        return 0;
    }
    
    
    
    // Recurrent investigation of all edges (or sub-edges) [thread safe]
    int mesh_recurrentEdgeFinder( unsigned int * const pEdgeList, const unsigned int pAllocateSpace,
        const unsigned int pD, const unsigned int * const pCurPointConfig, const unsigned int pNumPointConfig )
    {
    
        // If not enough points are removed yet
        if ( pNumPointConfig > pD )
        {
            // Loop through all combinations of removing one of them
            for (unsigned int lIterRemove = 0; lIterRemove < pNumPointConfig; lIterRemove++)
            {
                // Create vector of current point config
                std::vector< unsigned int > lCurPointConfig;
                lCurPointConfig.reserve(pNumPointConfig-1);        
                // Populate current combination
                for ( unsigned int lIterPoint = 0; lIterPoint < pNumPointConfig; lIterPoint++ )
                    // If not the one that should be removed
                    if (lIterPoint != lIterRemove)
                        // Assign to current point configuration
                        lCurPointConfig.push_back( pCurPointConfig[lIterPoint] );
                // Call function recursively
                int lStatus = mesh_recurrentEdgeFinder( &pEdgeList[lIterRemove * (pAllocateSpace / pNumPointConfig)], pAllocateSpace / pNumPointConfig, pD, 
                    lCurPointConfig.data(), pNumPointConfig-1);
                    
                if (lStatus)
                    return lStatus;
            }
        }
        else // If an edge is attained
        {   
            // Assert that enough space is available
            if ( pAllocateSpace < pD )
                return 1;        
            // Loop through all points 
            for ( unsigned int lIterPoint = 0; lIterPoint < pD; lIterPoint++ )
            {
                // Populate edge
                pEdgeList[lIterPoint] = pCurPointConfig[lIterPoint];
            }
        }
       
       return 0;
    }   
    
    
    
    
    
    
    
    
    // Get edges of simplices
    int mesh_getEdgesOfSimplices( unsigned int * const pEdges, unsigned int * pEdgeIndex, unsigned int * const pSimplexIdentity, const unsigned int pEdgeDim, 
        const unsigned int pTopologicalD, const unsigned int * const pSimplex, const unsigned int pNumSimplices )
    {
        // If dimensioanlity is not correct
        if (pTopologicalD < pEdgeDim)
            return 1;
            
        // Compute number of combinations of chosing pEdgeDim elements among (pTopologicalD+1) elements with order
        unsigned int lNumCombinations = 1;        
        for (unsigned int lIter = pTopologicalD+1; lIter > pEdgeDim; lIter--)
        {
            lNumCombinations *= lIter;
        }
        // Compute number of edges for a simplex ( (pTopologicalD + 1) choose pEdgeDim )
        unsigned int lNChooseK = 1;
        for (unsigned int lIter = 2; lIter <= (pTopologicalD + 1 - pEdgeDim); lIter++)
        {
            lNChooseK *= lIter;
        }
        lNChooseK = lNumCombinations / lNChooseK;
        
        // Get vector of indices
        std::vector<unsigned int> lIndicesOfSimplex;
        for (unsigned int lIterNumNodesInSimplex = 0; lIterNumNodesInSimplex < pTopologicalD+1; lIterNumNodesInSimplex++)
            lIndicesOfSimplex.push_back(lIterNumNodesInSimplex);
        // Preallocate vector for storing all edges indices
        std::vector<std::set<unsigned int>> lEdgeIndices;
        lEdgeIndices.reserve(lNChooseK);
        // Compute edges
        {
            const int lStatus = mesh_getEdgesOfSimplex( lEdgeIndices, lNumCombinations,
                pEdgeDim, pTopologicalD, lIndicesOfSimplex.data() );
            if (lStatus != 0)
                // Flag error
                return 2;
        }        
        
        // Get private copy of number of placed edges
        unsigned int lEdgeIndex = *pEdgeIndex;
        
        // Create parallel section for open mp
        #pragma omp parallel firstprivate(lEdgeIndex)
        {
            // Preallocate vector for temporary storing all edges for each simplex
            std::vector<std::set<unsigned int>> lNewEdges;
            lNewEdges.reserve(lNChooseK);
            // Preallocate vector for storing old edges associated with current simplex
            std::vector< unsigned int > lOldEdges;
            lOldEdges.reserve(lNChooseK);
            // Preallocate set for outer edges
            std::set<unsigned int> lOuterEdgeSet;

            // Loop through each simplex
            #pragma omp parallell for 
            for (unsigned int lIterSimp = 0; lIterSimp < pNumSimplices; lIterSimp++)
            {                    

                // Populate all new edges
                for ( std::vector<std::set<unsigned int>>::const_iterator lIterEdgeIndices = lEdgeIndices.begin(); 
                    lIterEdgeIndices != lEdgeIndices.end(); ++lIterEdgeIndices )
                {
                    // Get current edge from edge indices
                    std::set<unsigned int> lCurEdge;
                    for (std::set<unsigned int>::const_iterator lIterEdgeIndex = lIterEdgeIndices->begin();
                        lIterEdgeIndex != lIterEdgeIndices->end(); ++lIterEdgeIndex )
                        // Insert current node from current edge into lCurEdge
                        lCurEdge.insert( pSimplex[ lIterSimp * (pTopologicalD+1) + *lIterEdgeIndex ] );
                    // Add current edge to new edges
                    lNewEdges.push_back( lCurEdge );
                }
            
                
                // Loop through all prior edges from beginning to end, i.e., outer edges
                for (unsigned int lIterOuterEdge = 0; lIterOuterEdge < lEdgeIndex; lIterOuterEdge++)
                {
                    // Get set of nodes corresponding to current outer edge
                    lOuterEdgeSet.clear();
                    for (unsigned int lIterDim = 0; lIterDim < pEdgeDim; lIterDim++)
                        lOuterEdgeSet.insert( pEdges[ lIterOuterEdge * pEdgeDim + lIterDim ] );
                        
                    // Loop through all edges of current simplex, i.e., inner edges
                    for ( std::vector<std::set<unsigned int>>::iterator lIterInnerEdgeSet = lNewEdges.begin(); lIterInnerEdgeSet != lNewEdges.end(); )
                    {
                        // If inner edge is the same as outer edge
                        if ( lOuterEdgeSet == *lIterInnerEdgeSet )
                        {
                            // Add old edge
                            lOldEdges.push_back( lIterOuterEdge );
                            // Remove current inner edge
                            lIterInnerEdgeSet = lNewEdges.erase(lIterInnerEdgeSet);
                        }
                        else
                            ++lIterInnerEdgeSet;
                    }
                }   // Stop looping through prior edges
                
                #pragma omp critical (mesh_getEdgesOfSimplices_AccessAndWrite)
                {
                    // Loop through all prior edges not already seen
                    for (unsigned int lIterOuterEdge = lEdgeIndex; lIterOuterEdge < *pEdgeIndex; lIterOuterEdge++)
                    {
                        // Get set of nodes corresponding to current outer edge
                        lOuterEdgeSet.clear();
                        for (unsigned int lIterDim = 0; lIterDim < pEdgeDim; lIterDim++)
                            lOuterEdgeSet.insert( pEdges[ lIterOuterEdge * pEdgeDim + lIterDim ] );
                            
                        // Loop through all edges of current simplex, i.e., inner edges
                        for ( std::vector<std::set<unsigned int>>::iterator lIterInnerEdgeSet = lNewEdges.begin(); lIterInnerEdgeSet != lNewEdges.end(); )
                        {
                            // If inner edge is the same as outer edge
                            if ( lOuterEdgeSet == *lIterInnerEdgeSet )
                            {
                                // Add old edge
                                lOldEdges.push_back( lIterOuterEdge );
                                // Remove current inner edge
                                lIterInnerEdgeSet = lNewEdges.erase(lIterInnerEdgeSet);
                            }
                            else
                                ++lIterInnerEdgeSet;
                        }
                    }   // Stop looping through prior edges
                    
                    // Loop through all remaining edges of current simplex
                    for ( std::vector<std::set<unsigned int>>::const_iterator lIterInnerEdgeSet = lNewEdges.begin(); lIterInnerEdgeSet != lNewEdges.end(); ++lIterInnerEdgeSet)
                    {
                        // Get iterator 
                        std::set<unsigned int>::const_iterator lCurDim = lIterInnerEdgeSet->begin();                                
                        // go through all dimensions of edge
                        for (unsigned int lIterDim = 0; lIterDim < pEdgeDim; lIterDim++)
                        {
                            // Add dimension
                            pEdges[pEdgeDim * (*pEdgeIndex) + lIterDim] = *lCurDim;
                            ++lCurDim;
                        }
                        // Add to simplex mapping
                        lOldEdges.push_back( *pEdgeIndex );
                        // Advance counter
                        (*pEdgeIndex)++;
                    }                    
                    
                    // Update private copy of number of edges
                    lEdgeIndex = *pEdgeIndex;
                    
                }   // End of critical section
                
                // Initiate index for simplex mapping of edges
                unsigned int lCurSimplexMapInd = 0;
                // Loop through all edges already shared with prior simplices
                for ( std::vector< unsigned int >::iterator lIterOldEdges = lOldEdges.begin(); lIterOldEdges != lOldEdges.end(); ++lIterOldEdges )
                {
                    // Mark current simplex to old edge
                    pSimplexIdentity[ lIterSimp * lNChooseK + lCurSimplexMapInd++ ] = *lIterOldEdges;
                }
                
                // Clear vectors
                lNewEdges.clear();
                lOldEdges.clear();
            
            }   // Stop looping over simplices
        } // Stop parallel section    

        return 0;
    } 
    
    
    
    int mesh_computeEdges( const unsigned int pEdgeDim, const unsigned int * const pSimplices,
        const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNumEdges, unsigned int * const pEdgeId, unsigned int * const pMaxSimplicesPerEdge )
    {
        // Store edge
        *pEdgeId = gEdgesOfSimplices.store( SimplexEdges() );
            
        // Get reference to target edges
        SimplexEdges * const lEdges = gEdgesOfSimplices.load(*pEdgeId);
        
        // Compute edges
        int lStatus = lEdges->computeEdges( pSimplices, pTopD, pNumSimplices, pEdgeDim );
        // If error
        if (lStatus != 0)
            // Flag error
            return lStatus;
            
        // Get maximum simplices per edge
        *pMaxSimplicesPerEdge = lEdges->getMaxSimplicesPerEdge();
        // Get number of edges
        *pNumEdges = lEdges->getNumEdges();
    
        return 0;
    }
    
    // Populate arrays of edges, associated simplices, and associated edges to each simplex
    int mesh_populateEdges( unsigned int * const pEdges, const unsigned int pEdgeDim, const unsigned int pNumEdges, const unsigned int pEdgeId,
        unsigned int * const pSimplicesForEdges, const unsigned int pMaxSimplicesPerEdge, const unsigned int pNumSimplices, 
        unsigned int * const pEdgesForSimplices, const unsigned int pNumEdgesPerSimplex )
    {
        // Loop through computed edges until finding the right one
        const SimplexEdges * const lEdges = gEdgesOfSimplices.load( pEdgeId );
        
        // If edges should be retrieved
        if (pEdges != NULL)
        {
            // Retrieve edges
            int lStatus = lEdges->populateEdges( pEdges, pNumEdges, pEdgeDim );
            // Handle error
            if (lStatus != 0)
                return 1 + lStatus;
        }                
        // If simplices for edges should be retrieved
        if (pSimplicesForEdges != NULL)
        {
            // Retrieve edges
            int lStatus = lEdges->populateEdgesSimplexList( pSimplicesForEdges, pNumEdges, 
                pMaxSimplicesPerEdge, pNumSimplices );
            // Handle error
            if (lStatus != 0)
                return 3 + lStatus;
        }        
        // If simplices for edges should be retrieved
        if (pEdgesForSimplices != NULL)
        {
            // Retrieve edges
            int lStatus = lEdges->populateSimplexEdgesList( pEdgesForSimplices, 
                pNumSimplices, pNumEdgesPerSimplex );
            // Handle error
            if (lStatus != 0)
                return 5 + lStatus;
        }        
    
        return 0;
    }
    
    // Clear saved edges
    int mesh_clearEdges( const unsigned int pEdgeId )
    {
        // Remove edges
        return gEdgesOfSimplices.erase( pEdgeId );
    }
    
    // Get neighborhood of each simplex
    int mesh_getSimplexNeighborhood( const unsigned int pNumEdges, const unsigned int pNumSimplices,
        const unsigned int * const pSimplicesForEdges, const unsigned int pMaxSimplicesPerEdge,
        const unsigned int * const pEdgesForSimplices, const unsigned int pNumEdgesPerSimplex,
        unsigned int * const pNeighborhood )
    {
        // Make sure that there are no more than two simplices per edge
        if ( pMaxSimplicesPerEdge != 2 )
            // Flag error
            return 1;
    
        // Loop through each simplex
        for (unsigned int lIterSimplices = 0; lIterSimplices < pNumSimplices; lIterSimplices++)
        {
            // Initialize iterator of where to insert current found neighbor
            unsigned int lIterNeighs = 0;
            // Loop through each edge for current simplex
            for (unsigned int lIterEdges = 0; lIterEdges < pNumEdgesPerSimplex; lIterEdges++)
            {
                // Get current edge index
                const unsigned int lCurEdge = pEdgesForSimplices[ lIterSimplices * pNumEdgesPerSimplex + lIterEdges ];
                // Loop through simplices of current edge
                for (unsigned int lIterOverEdgesSimps = 0; lIterOverEdgesSimps < 2; lIterOverEdgesSimps++)
                {
                    // Get current simplex of edge
                    const unsigned int lCurSimplexOfEdge = pSimplicesForEdges[lCurEdge * 2 + lIterOverEdgesSimps];
                    // If current potential neighbor is not current simplex nor nan
                    if ( ( lCurSimplexOfEdge != lIterSimplices ) && ( lCurSimplexOfEdge != pNumSimplices ) )
                    {
                        // Set current neighbor to be the potential neighbor
                        pNeighborhood[lIterSimplices * pNumEdgesPerSimplex + lIterNeighs ] = lCurSimplexOfEdge;
                        // Update number if neighbors found
                        lIterNeighs++;
                        // Break out of loop
                        break;
                    }
                }
            }
            // Loop through all potential neighbors not found
            for ( ; lIterNeighs < pNumEdgesPerSimplex; lIterNeighs++)
                // Add dummy
                pNeighborhood[lIterSimplices * pNumEdgesPerSimplex + lIterNeighs ] = pNumSimplices;
        }
        return 0;  
    }
    
    
    
    
    
    // Refines chosen simplices
    int mesh_refineMesh( const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        const unsigned int * const pNeighs,
        unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices, unsigned int * const pId,
        const unsigned int pMaxNumNodes, const double * const pMaxDiam, const unsigned int pNumMaxDiam,
        int (* transformationPtr)(double *, unsigned int) )
    {
    
        if ( (pNumMaxDiam > 1) && (pNumMaxDiam != pNumNodes) )
            return 1;
        // Create max diameter vector
        std::vector<double> lMaxDiam( pMaxDiam, pMaxDiam + pNumMaxDiam );
        
        // Create internal mesh
        FullMesh lMesh( pNodes, pD, pNumNodes, pSimplices, pNumSimplices, pTopD, pNeighs );
        // Refine
        int lStatus = lMesh.refine( pMaxNumNodes, lMaxDiam, transformationPtr );
        if (lStatus)
            return 1 + lStatus;
            
        // Store mesh
        *pId = gMeshes.store(lMesh);        
        // Update output
        *pNewNumNodes = lMesh.getNN();
        *pNewNumSimplices = lMesh.getNT();
        
        return 0;
    }
    
        
    // Acquire new mesh
    int mesh_acquireMesh( const unsigned int pId, 
        double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNeighs )
    {
        // Load stored mesh
        FullMesh * const lMesh = gMeshes.load( pId );
    
        // If mesh did not exist
        if (lMesh == NULL)
            // Return error
            return 5;

        // Populate arrays
        lMesh->populateArrays( pNodes, pD, pNumNodes, pSimplices, pNumSimplices, pTopD, pNeighs );
        // Remove mesh
        gMeshes.erase(pId);
        // Return success
        return 0;
    }


    // Binomial coefficient
    inline unsigned int mesh_nchoosek( unsigned int n, unsigned int k )
    {
        if (n < k)
            return 1;
    
        // Get Number of possible edges for simplices (pTopD+1 choose pTopD)
        unsigned int lNumCombinationsNom = 1;        
        for (unsigned int lIter = n; lIter > n-k; lIter--)
        {
            lNumCombinationsNom *= lIter;
        }
        unsigned int lNumCombinationsDenom = 1;        
        for (unsigned int lIter = 1; lIter <= k; lIter++)
        {
            lNumCombinationsDenom *= lIter;
        }
        return ( lNumCombinationsNom / lNumCombinationsDenom );
    }    



    
    // Extend mesh to new dimension
    int mesh_extendMesh( unsigned int * const pNewSimplices, const unsigned int pNewNumSimplices,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, const unsigned int pNumNodes )
    {
        // Check if new number of simplices matches what it should be
        if (pNewNumSimplices != pNumSimplices * (pTopD+1) )
            // Flag error
            return 1;
            
        // Get number of edges for each simplex
        const unsigned int lNumEdgesSimp = mesh_nchoosek( pTopD+1, pTopD ); 
        // Get edges and which edges belong to each simplex in original mesh and vice versa
        std::vector<unsigned int> lEdges;
        std::vector<unsigned int> lSimplexIdentity;
        std::vector<unsigned int> lEdgeIdentity;
        int lStatus = mesh_getEdgesAndRelations( lEdges, lSimplexIdentity, lEdgeIdentity, 
            pSimplices, pNumSimplices, pTopD, lNumEdgesSimp ); 
        if (lStatus)
            return lStatus + 3;
            
        // Initialize extend mesh object
        ExtendMesh extendMesh( &lEdges, &lSimplexIdentity, &lEdgeIdentity, 
            pSimplices, pNumSimplices, pTopD, pNumNodes, lNumEdgesSimp );
        
        
        // Walk through all original simplices 
        for (unsigned int lIterSimp = 0; lIterSimp < pNumSimplices; lIterSimp++)
        {
            // Preallocate vector of nonplaced neighbors
            std::list<unsigned int> lNonPlacedList;
            // Populate with current
            lNonPlacedList.push_back( lIterSimp );
            
            // do loop until lNonPlacedList is empty
            do
            {     
                // Get current simplex
                const unsigned int lCurSimplexID = lNonPlacedList.front();
                // Remove from list
                lNonPlacedList.pop_front();
                // Fill up list with all non-placed neighbors of current simplex
                {
                    // Get all neighbors of current simplex which are not placed
                    const std::vector< std::pair<unsigned int, unsigned int> > lCurNeighs = extendMesh.neighSimps( lCurSimplexID, 2 );
                    // Insert them into lNonPlacedNeighs
                    for ( std::vector< std::pair<unsigned int, unsigned int> >::const_iterator lIterNeighs = lCurNeighs.begin();
                        lIterNeighs != lCurNeighs.end(); ++lIterNeighs )
                        // Make sure it is not already present in list
                        if ( std::find( std::begin(lNonPlacedList), std::end(lNonPlacedList), lIterNeighs->first ) == std::end(lNonPlacedList) )
                            lNonPlacedList.push_back( lIterNeighs->first );
                }
                
                // If current simplex is already placed
                if (extendMesh.isSimplexPlaced( lCurSimplexID ))
                    // Continue loop
                    continue;
                
                // Acquire all new sub-simplices and update neighbor list
                std::vector< std::set<unsigned int> > lSubSimplices;
                int lComputeStatusStatus = extendMesh.computeNewSubSimplices( lCurSimplexID, pNewSimplices, lSubSimplices );
                // If did not comply
                if ( lComputeStatusStatus == 1 )
                {
                    // Find shortest route from current simplex to open territory
                    std::list<unsigned int> lShortestPath;
                    unsigned int lRouteDistance = extendMesh.shortestPathToFreedom( lCurSimplexID, lCurSimplexID, lShortestPath );
                    // Go through path
                    for ( std::list<unsigned int>::const_iterator lIterPath = lShortestPath.begin(); 
                        lIterPath != lShortestPath.end(); ++lIterPath)
                    {
                        // Cut down all simplices through this path
                        extendMesh.markPlacement( *lIterPath, false );                        
                        // Add them, a new, to list from non-compliant simplex out to open territory
                        lNonPlacedList.push_front( *lIterPath );
                    }                    
                    // Try once more to create sub-simplices for non-compliant simplex
                    lSubSimplices.clear();
                    lComputeStatusStatus = extendMesh.computeNewSubSimplices( lCurSimplexID, pNewSimplices, lSubSimplices );

                    // Handle possible error
                    if ( lComputeStatusStatus == 1 )
                        return 3;
                }
                
                // Make sure is of right size
                if ( lSubSimplices.size() != (pTopD+1) )
                    return 2;
                // copy to new simplices
                {
                    unsigned int lIterNode = 0;
                    for ( std::vector< std::set<unsigned int> >::const_iterator lIterSubSimp = lSubSimplices.begin();
                        lIterSubSimp != lSubSimplices.end(); ++lIterSubSimp )
                    {
                        assert( lIterSubSimp->size() == (pTopD + 2) );
                        for ( std::set<unsigned int>::const_iterator lIterSet = lIterSubSimp->begin();
                            lIterSet != lIterSubSimp->end(); ++lIterSet )
                        {
                            // Insert current node ind from current sub-simplex
                            pNewSimplices[ lCurSimplexID * (pTopD+1) * (pTopD+2) + lIterNode ] = *lIterSet;
                            // Increment counter
                            lIterNode++;
                        }
                    }
                }
                // Mark current simplex as placed
                extendMesh.markPlacement( lCurSimplexID );
                
            } while (lNonPlacedList.size() > 0);
        }
    
        return 0;
    }
    
    
    
    
    
        
}   // End of extern "C"






