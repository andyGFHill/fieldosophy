/* 
* C/C++ functions for the MeshGraph class.
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


#include "meshGraph.hxx"
#include "mesh.hxx"
#include <vector>
#include <limits>



namespace MeshGraphGlobals
{
    // Remove graph
    int removeGraph( const unsigned int pGraphIndex );
    // Get graph
    const MeshGraph * getGraph( const unsigned int pGraphIndex );
    // Add graph to list
    unsigned int addGraphToList( MeshGraph & pGraph );
        
    // vector of graphs created
    std::vector< std::pair<MeshGraph, unsigned int > > gGraphList;
}



// Constructor
MeshGraph::MeshGraph( const unsigned int pD, const double * const pBoundaries )
{
    // Get dimensionality of cuboid
    mD = pD;
    // Preallocate boundaries    
    mBoundaries.reserve(pD*2);    
    // Fill up boundaries
    for (unsigned int lIter = 0; lIter < pD * 2; lIter++)
        mBoundaries.push_back( pBoundaries[lIter] );
    
    // Create mother node
    mNodeList.push_back( Node( mBoundaries ) );
    
    return;
}

// Constructor
MeshGraph::MeshGraph( const ConstMesh & pMesh, const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles )
{
    // Get dimensionality of cuboid
    mD = pMesh.getD();
    // Preallocate bounding box    
    mBoundaries.reserve(mD*2);    
    for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
    {
        mBoundaries.push_back( std::numeric_limits<double>::infinity() );
        mBoundaries.push_back( -std::numeric_limits<double>::infinity() );
    }        
    // Loop through all nodes of mesh
    for (unsigned int lIterNodes = 0; lIterNodes < pMesh.getNN(); lIterNodes++)
    {
        // Loop through all dimensions
        for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
        {
            // Get value of current node
            const double lCurNodeVal = pMesh.getNodes()[ lIterNodes*mD + lIterDim ];
            // Is the current lower boundary too high
            if ( mBoundaries.at( lIterDim*2 ) > lCurNodeVal )
                mBoundaries.at( lIterDim*2 ) = lCurNodeVal;
            // Is the current higher boundary too low
            if ( mBoundaries.at( lIterDim*2+1 ) < lCurNodeVal )
                mBoundaries.at( lIterDim*2+1 ) = lCurNodeVal;
        }
        
    }
    
    // Create mother node
    mNodeList.push_back( Node( mBoundaries ) );
    
    // Populate graph
    populate( pMesh.getSimplices(), pMesh.getNodes(), pMesh.getNT(), pMesh.getD(), 
        pMesh.getNN(), pMesh.getTopD(), pMaxNumNodes, pMinDiam, pMinNumTriangles );
    
    return;
}
    
    
// Populate graph from mesh
int MeshGraph::populate( const unsigned int * pTriangles, const double * pPoints, const unsigned int pNumTriangles, 
        const unsigned int pD, const unsigned int pNumPoints, const unsigned int pTopD,
        const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles )
{

    // Ensure that only one mother node exists
    if (mNodeList.size() > 1)
        // Flag error
        return 1;
        
    if (pD != mD)
        // flag error
        return 2;

    // Insert all triangles into the mother node
    for (unsigned int lIter = 0; lIter < pNumTriangles; lIter++)
    {
        mNodeList.back().mTriangles.push_back( lIter );
    }
    

    // Assume that we shuld continue to split nodes
    bool lContinueSplitting = true;
    // Continue go through and split as long as no condition is met for stopping
    while (lContinueSplitting)
    {
        // Get current number of nodes
        const unsigned int lCurNumNodes = mNodeList.size();
        // Loop through all dimensions
        for (unsigned int lIterDims = 0; lIterDims < mD; lIterDims++)
        {
            // Get current number of nodes
            const unsigned int lCurNumNodes2 = mNodeList.size();
            
            // Loop through all current nodes
            for (unsigned int lIterNode = 0; lIterNode < lCurNumNodes2; lIterNode++)
            {
                // If should continue splitting 
                if (lContinueSplitting)
                    // If number of triangles is not too small
                    if ( getNode(lIterNode)->mTriangles.size() > pMinNumTriangles )
                    {
                        // Get diameter of current node in current dimension
                        const double lCurDiam = getNode(lIterNode)->mBoundaries.at(lIterDims*2 + 1) - getNode(lIterNode)->mBoundaries.at(lIterDims*2);
                        // If current diameter is not too small
                        if ( lCurDiam > pMinDiam * 2.0d )
                        {
                            // If not reached maximum number of nodes
                            if (pMaxNumNodes > mNodeList.size())
                                // Split current node
                                splitNode( lIterNode, lIterDims, pTriangles, pPoints, pTopD );        
                        }
                    }
            }    // End of loop through nodes
        }   // End of loop through dimensions
        
        // If number of nodes has not increased
        if ( mNodeList.size() <= lCurNumNodes )
            // Flag to stop splitting
            lContinueSplitting = false;
        
    }   // End of while loop for refining graph
    
    return 0;
}


// split node by dimension
int MeshGraph::splitNode( const unsigned int pNodeInd, const unsigned int pDim, 
        const unsigned int * const pTriangles, const double * const pPoints, const unsigned int pTopD )
{
    // Get current boundaries
    std::vector<double> pBoundaries1 = mNodeList.at(pNodeInd).mBoundaries;
    // Set higher boundary to half the length of the cuboid
    pBoundaries1[ pDim * 2 + 1 ] = ( pBoundaries1.at( pDim * 2 ) + pBoundaries1.at( pDim * 2 + 1 ) ) / 2.0d;
    // Get current boundaries again
    std::vector<double> pBoundaries2 = mNodeList.at(pNodeInd).mBoundaries;
    // Set lower boundary to half the length of the original cuboid
    pBoundaries2.at( pDim * 2 ) = pBoundaries1.at( pDim * 2 + 1 );
    // Insert first new boundary in chosen node
    mNodeList.at(pNodeInd).mBoundaries = pBoundaries1;    
    // Create new node from second boundary
    mNodeList.push_back( Node( pBoundaries2 ) );
    // Go through all neighbors of original node
    for ( std::vector< std::pair<unsigned int, unsigned int> >::iterator lIter = std::begin( mNodeList.at(pNodeInd).mNeighNodes ); 
        lIter != std::end( mNodeList.at(pNodeInd).mNeighNodes); 
        ++lIter)
    {
        // If not the direction neighboring the chosen node 
        if (lIter->second != pDim*2 )
            // Add connection between new node and the neighbor
            addConnection( mNodeList.size()-1, lIter->first, lIter->second );
    }    
    // Remove all neighbors on the appropriate side of the chosen node
    breakSideConnection( pNodeInd, pDim*2+1 );
    // Add connection between the chosen and new nodes
    addConnection( pNodeInd, mNodeList.size()-1, pDim*2+1 );
    
    
    // Loop through all triangles in chosen node
    for ( auto lIterTriangles = std::begin( mNodeList.at(pNodeInd).mTriangles ); lIterTriangles != std::end( mNodeList.at(pNodeInd).mTriangles); )
    {
        // Preallocate vector of nodal points of current triangle
        std::vector< std::vector<double> > lCurNodalPoints;
        lCurNodalPoints.reserve( pTopD+1 );
        // Acquire pointer to current triangle nodal indices
        const unsigned int * const lNodalIndices = &pTriangles[ *lIterTriangles * (pTopD+1) ];        
        // loop through all nodes in triangle
        for (unsigned int lIterNode = 0; lIterNode < pTopD+1; lIterNode++ )
        {
            // Acquire pointer to current nodal point
            const double * const lNodalPointIndex = &pPoints[ lNodalIndices[lIterNode] * mD ];
            // Get current nodal point
            lCurNodalPoints.push_back( std::vector<double>( lNodalPointIndex, lNodalPointIndex + mD  ) );
        }
        
        // See if current triangle is inside new cuboid
        if ( mNodeList.back().triangleInside( lCurNodalPoints ) )
        {
            // Add to new cuboid
            mNodeList.back().mTriangles.push_back( *lIterTriangles );
        }
        
        // See if current triangle is not inside chosen cuboid
        if ( !mNodeList.at(pNodeInd).triangleInside( lCurNodalPoints ) )
            // Remove from chosen cuboid
            lIterTriangles = mNodeList.at(pNodeInd).mTriangles.erase( lIterTriangles );
        else
            // Increment iterator
            ++lIterTriangles;
    }
    
    
    return 0;
}



int MeshGraph::breakSideConnection( const unsigned int pNodeInd, const unsigned int pDirection )
{

    // Preallocate vector of nodes to break connection with
    std::vector<unsigned int> lBreakUps;
    
    // Loop through all connections for node
    for ( auto lIter = std::begin( mNodeList.at(pNodeInd).mNeighNodes); lIter != std::end( mNodeList.at(pNodeInd).mNeighNodes); ++lIter)
    {
        // If the direction to be removed
        if (lIter->second == pDirection )
        {
            // List link to be broken
            lBreakUps.push_back( lIter->first );
        }            
    }
    
    // Loop through all nodes to break connection with
    for ( auto lIter = std::begin( lBreakUps ); lIter != std::end( lBreakUps ); ++lIter)
    {
        // Break current connection
        breakConnection( pNodeInd, *lIter );
    }
    
    return 0;
}

int MeshGraph::breakConnection( const unsigned int pNodeInd1, const unsigned int pNodeInd2 )
{
    // Loop through all connections for node 1
    for ( auto lIter = std::begin( mNodeList.at(pNodeInd1).mNeighNodes); lIter != std::end( mNodeList.at(pNodeInd1).mNeighNodes); )
    {
        // If  found current connection
        if (lIter->first == pNodeInd2 )
            // Remove
            lIter = mNodeList.at(pNodeInd1).mNeighNodes.erase( lIter );
        else
            // Increment iterator
            ++lIter;
    }
    // Loop through all connections for node 2
    for ( auto lIter = std::begin( mNodeList.at(pNodeInd2).mNeighNodes); lIter != std::end( mNodeList.at(pNodeInd2).mNeighNodes); )
    {
        // If  found current connection
        if (lIter->first == pNodeInd1 )
            // Remove
            lIter = mNodeList.at(pNodeInd2).mNeighNodes.erase( lIter );
        else
            ++lIter;
    }
    
    return 0;
}

// add connection between two formerly non-neighboring nodes
int MeshGraph::addConnection( const unsigned int pNodeInd1, const unsigned int pNodeInd2, const unsigned int pNode12Direction )
{

    const unsigned int lNode21Direction = ( pNode12Direction % 2 == 0 ) ?  pNode12Direction + 1 : pNode12Direction - 1;

    // Loop through all connections for node 1
    for ( auto lIter = std::begin( mNodeList.at(pNodeInd1).mNeighNodes); lIter != std::end( mNodeList.at(pNodeInd1).mNeighNodes); )
    {
        // If  found current connection already
        if (lIter->first == pNodeInd2 )
            // Remove
            lIter = mNodeList.at(pNodeInd1).mNeighNodes.erase( lIter );
        else
            ++lIter;
    }
    // Loop through all connections for node 2
    for ( auto lIter = std::begin( mNodeList.at(pNodeInd2).mNeighNodes); lIter != std::end( mNodeList.at(pNodeInd2).mNeighNodes); )
    {
        // If  found current connection
        if (lIter->first == pNodeInd1 )
            // Remove
            lIter = mNodeList.at(pNodeInd2).mNeighNodes.erase( lIter );
        else
            ++lIter;
    }
    
    // Add connection 2 to 1
    mNodeList.at(pNodeInd1).mNeighNodes.push_back( std::pair<unsigned int, unsigned int>( pNodeInd2, pNode12Direction ) );
    // Add connection 1 to 2
    mNodeList.at(pNodeInd2).mNeighNodes.push_back( std::pair<unsigned int, unsigned int>( pNodeInd1, lNode21Direction ) );
    
    return 0;
}

const MeshGraph::Node * MeshGraph::getNode( const unsigned int pNodeIndex ) const
{
    // if node does not exist
    if ( pNodeIndex >= getNumNodes() )
        // Flag error
        return NULL;
        
    return &mNodeList.at(pNodeIndex);
}

std::pair< std::vector< MeshGraph::Node >::const_iterator, std::vector< MeshGraph::Node >::const_iterator > 
    MeshGraph::getNodeListIterator() const
{
    return std::pair< std::vector< MeshGraph::Node >::const_iterator, std::vector< MeshGraph::Node >::const_iterator >
        ( std::begin(mNodeList), std::end(mNodeList) );
}



int MeshGraph::getNodeOfPoint( const double * const pPoint, const unsigned int pD, unsigned int * const pNode ) const
{
    if (pD != mD)
        // Flag error
        return 4;

    // Assume that correct node is not found
    bool lFound = false;
    // Keep check of number of iterations
    unsigned int lIterOuter = 0;
    // Loop until found 
    while (!lFound )
    {
        // Assume found
        lFound = true;
        // Loop through all dimensions
        for (unsigned int lIterDim = 0; lIterDim < pD; lIterDim++)
        {
            // Get pointer to current node 
            const MeshGraph::Node * lNode = getNode(*pNode);
            if (lNode == NULL)
                // Flag error
                return 1;
                
            // See if point is too high for current node
            const bool lTooHigh = ( pPoint[lIterDim] > lNode->mBoundaries.at(lIterDim*2+1) );
            // See if point is too low for current node
            const bool lTooLow = ( pPoint[lIterDim] < lNode->mBoundaries.at(lIterDim*2) );
        
            // If point is higher than higher value of graph node or lower than lower
            if ( lTooHigh || lTooLow )
            {
                // Denounce candidate for being the right one
                lFound = false;
                        
                // Get target dimension
                const unsigned int lTargetDim = (lTooHigh) ? lIterDim*2+1 : lIterDim*2;
                // Go through neighbor nodes until finding the first neighbor on the target side
                std::vector< std::pair<unsigned int, unsigned int> >::const_iterator lIterNeighs = std::begin( lNode->mNeighNodes );
                while ( lIterNeighs != std::end( lNode->mNeighNodes ) )
                {
                    // If current is target dimension
                    if ( lIterNeighs->second == lTargetDim )
                    {
                        // Set current node to be the newly found one
                        *pNode = lIterNeighs->first;
                        // break out of loop
                        break;
                    }  
                    
                    ++lIterNeighs;
                }
                // If no suiting neighbor was found, i.e., reached end
                if ( lIterNeighs == std::end( lNode->mNeighNodes ) )
                    // Flag error
                    return 2;
            }
        }   // end of loop through dimensions
        
        if (lIterOuter > mNodeList.size())
            // Flag error
            return 3;
        // Increase counter
        lIterOuter++;
    }   // end of loop until found
    
    return 0;   
}





 
 
 
MeshGraph::Node::Node( const std::vector<double> &pBoundaries )
{
    // Get dimensionality
    mD = pBoundaries.size() / 2;
    // Preallocate size
    mBoundaries = pBoundaries;
}        
MeshGraph::Node::Node( const Node & pNode )
{
    mD = pNode.mD;
    mTriangles = pNode.mTriangles;
    mBoundaries = pNode.mBoundaries;
    mNeighNodes = pNode.mNeighNodes;
}

bool MeshGraph::Node::triangleInside( const std::vector< std::vector<double> > & pPoints ) const
{

    // Get bounding box
    std::vector<double> lBoundingBox;
    lBoundingBox.reserve(mD*2);
    for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
    {
        const double lCurNodeVal = pPoints[0].at(lIterDim);
        lBoundingBox[lIterDim * 2] = lCurNodeVal;
        lBoundingBox[lIterDim * 2+1] = lCurNodeVal;
    }
    // Loop through all nodes of triangle
    for ( std::vector< std::vector<double> >::const_iterator lIterNode = std::begin( pPoints )+1; lIterNode != std::end( pPoints ); ++lIterNode)
    {
        for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
        {
            const double lCurNodeVal = lIterNode->at(lIterDim);
            if (lCurNodeVal < lBoundingBox[lIterDim * 2])
                lBoundingBox[lIterDim * 2] = lCurNodeVal;
            if (lCurNodeVal > lBoundingBox[lIterDim * 2+1])
                lBoundingBox[lIterDim * 2+1] = lCurNodeVal;
        }
    }

    // Assume bounding box is inside
    bool lIsInside = true;
    
    // Loop through all dimensions
    for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
    {
        // See if lower value of triangle is higher than higher value of graph node
        const bool lLowTooHigh = lBoundingBox[lIterDim*2] > mBoundaries.at(lIterDim*2+1);
        // See if higher value of triangle is lower than lower value of graph node
        const bool lHighTooLow = lBoundingBox[lIterDim*2+1] < mBoundaries.at(lIterDim*2);
        
        // If any is true, the triangle is not inside
        if ( lLowTooHigh || lHighTooLow )
            lIsInside = false;
    
    }
    
    return lIsInside;
}

// See if point is inside
bool MeshGraph::Node::pointInside( const std::vector<double> & pPoints ) const
{
    // Assume point is inside
    bool lIsInside = true;    
    // Loop through all dimensions
    for (unsigned int lIterDim = 0; lIterDim < mD; lIterDim++)
    {
        // See if lower value of triangle is higher than higher value of graph node
        const bool lTooHigh = pPoints.at(lIterDim) > mBoundaries.at(lIterDim*2+1);
        // See if higher value of triangle is lower than lower value of graph node
        const bool lTooLow = pPoints.at(lIterDim) < mBoundaries.at(lIterDim*2);
        
        // If any is true, the triangle is not inside
        if ( lTooHigh || lTooLow )
            lIsInside = false;
    }
    
    return lIsInside;
}







// Add graph 
unsigned int MeshGraphGlobals::addGraphToList( MeshGraph & pGraph )
{
    // Get id of new graph
    const unsigned int lId = (gGraphList.size() > 0) ? gGraphList.back().second + 1 : 0;
    // Push to back of graph list
    gGraphList.push_back( std::pair<MeshGraph, unsigned int >( pGraph, lId) );
    
    return lId;
}

// Remove graph
int MeshGraphGlobals::removeGraph( const unsigned int pGraphIndex )
{
    // Loop through available graphs to see if any matches graph index
    for ( auto lIter = std::begin( gGraphList ); lIter != std::end( gGraphList );  )
    {
        // If current matches
        if ( lIter->second == pGraphIndex )
        {
            // Remove graph
            lIter = gGraphList.erase( lIter );    
            // flag as succesfull
            return 0;
        }
        else
            ++lIter;
    }
        
    // If reached bottom, no graph was found, flag error
    return 1;
}

// Get graph
const MeshGraph * MeshGraphGlobals::getGraph( const unsigned int pGraphIndex )
{
    // Loop through available graphs to see if any matches graph index
    for ( auto lIter = std::begin( gGraphList ); lIter != std::end( gGraphList ); ++lIter )
    {
        if ( lIter->second == pGraphIndex )
            return &lIter->first;
    }
    
    // If reaches here, no graph with given number existed
    return NULL;
}






extern "C"
{

    // Function for creating graph (and storing it) given mesh
    int MeshGraph_createGraph( const double * pPoints, const unsigned int pNumPoints, const unsigned int pD, 
        const unsigned int * pTriangles, const unsigned int pNumTriangles, const unsigned int pTopD,
        const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles,
        unsigned int * const pNumNodes, unsigned int * const pGraphIndex )
    {
    
        // Create ConstMesh object
        ConstMesh lMesh( pPoints, pD, pNumPoints, pTriangles, pNumTriangles, pTopD );
        
        // Create graph object from mesh
        MeshGraph lGraph( lMesh, pMaxNumNodes, pMinDiam, pMinNumTriangles );
        
        // Store number of nodes
        *pNumNodes = lGraph.getNumNodes();
        
        // Store graph object in list
        *pGraphIndex = MeshGraphGlobals::addGraphToList( lGraph );
        
        return 0;
    }
        
    // Get number of nodes 
    unsigned int MeshGraph_getNumNodes( const unsigned int pGraphIndex )
    {        
        // Get pointer to graph
        const MeshGraph * const lGraph = MeshGraphGlobals::getGraph( pGraphIndex );
        // If graph does not exist
        if ( lGraph == NULL )
            // Flag error
            return 0;

        // Get number of nodes in graph            
        return lGraph->getNumNodes();
    }
    
    // Function for acquiring boundary boxes for all nodes in graph
    int MeshGraph_getNodeBoundaries( const unsigned int pGraphIndex, double * const pBoundaries, const unsigned int pNumNodes, const unsigned int pD )
    {
    
        // Get pointer to graph
        const MeshGraph * const lGraph = MeshGraphGlobals::getGraph( pGraphIndex );
        // If graph does not exist
        if ( lGraph == NULL )
            // Flag error
            return 1;
        // If dimension is wrong
        if ( pD != lGraph->getD() )    
            // Flag error
            return 2;
        // If number of nodes are wrong
        if ( pNumNodes != lGraph->getNumNodes() )    
            // Flag error
            return 3;
            
        // Loop through all nodes
        for (unsigned int lIterNodes = 0; lIterNodes < pNumNodes; lIterNodes++)
        {
            // Loop through all dimensions
            for (unsigned int lIterDims = 0; lIterDims < pD; lIterDims++)
            {
                // Populate array
                pBoundaries[lIterNodes * (pD*2) + lIterDims*2] = lGraph->getNode(lIterNodes)->mBoundaries.at(lIterDims*2);
                pBoundaries[lIterNodes * (pD*2) + lIterDims*2 + 1] = lGraph->getNode(lIterNodes)->mBoundaries.at(lIterDims*2+1);
            }
        }
        
        return 0;
    }
    
    
    // Free up saved graph
    int MeshGraph_freeGraph( const unsigned int pGraphIndex ) 
    { 
        return MeshGraphGlobals::removeGraph(pGraphIndex); 
    }
    
    // Get number of graphs
    unsigned int MeshGraph_getNumGraphs( ) 
    { 
        return MeshGraphGlobals::gGraphList.size(); 
    }  
    
    // Function for acquiring number of triangles for specific node
    int MeshGraph_getNodeNumTriangles( const unsigned int pGraphIndex, 
        const unsigned int pNodeIndex, unsigned int * const pNumTriangles )
    {
        // Get pointer to graph
        const MeshGraph * const lGraph = MeshGraphGlobals::getGraph( pGraphIndex );
        // If graph does not exist
        if ( lGraph == NULL )
            // Flag error
            return 1;
            
        // Get node
        const MeshGraph::Node * lNode = lGraph->getNode( pNodeIndex );
        // If node does not exist
        if ( lNode == NULL )
            // Flag error
            return 2;
            
        // Inser the number of triangles in current node
        *pNumTriangles = lNode->mTriangles.size();
        
        return 0;
    }
    
    
    // Function for aquiring triangles for specific node
    int MeshGraph_getNodeTriangles( const unsigned int pGraphIndex, const unsigned int pNodeIndex, const unsigned int pD,
        const unsigned int pNumTriangles, unsigned int * const pTriangleIds )
    {
    
        // Get pointer to graph
        const MeshGraph * const lGraph = MeshGraphGlobals::getGraph( pGraphIndex );
        // If graph does not exist
        if ( lGraph == NULL )
            // Flag error
            return 1;
        // If dimension is wrong
        if ( pD != lGraph->getD() )    
            // Flag error
            return 2;
        // Get node
        const MeshGraph::Node * const lNode = lGraph->getNode( pNodeIndex );
        // If node does not exist
        if ( lNode == NULL )
            // Flag error
            return 3;
        // If number of triangles are wrong
        if ( pNumTriangles != lNode->mTriangles.size() )    
            // Flag error
            return 4;
        
        // Populate triangle indices
        for (unsigned int lIter = 0; lIter < pNumTriangles; lIter++)
        {
            pTriangleIds[lIter] = lNode->mTriangles.at(lIter);
        }
        
        return 0;
    }
    
    
    int MeshGraph_getNodesOfPoints( const unsigned int pGraphIndex, 
        const double * const pPoints, const unsigned int pD, 
        unsigned int * const pPointIds, const unsigned int pNumPoints )
    {
        // Get pointer to graph
        const MeshGraph * const lGraph = MeshGraphGlobals::getGraph( pGraphIndex );
        // If graph does not exist
        if ( lGraph == NULL )
            // Flag error
            return 1;
        // If dimension is wrong
        if ( pD != lGraph->getD() )    
            // Flag error
            return 2;
            
        // Assume first point belong to node 0
        unsigned int lCurNode = 0;

        // Loop through all points
        for ( unsigned int lIterPoints = 0; lIterPoints < pNumPoints; lIterPoints++ )
        {
            // Get node of current point
            int lStatus = lGraph->getNodeOfPoint( &pPoints[lIterPoints*pD], pD, &lCurNode );
            if (lStatus != 0)
                // flag error
                return lStatus + 2;
                
            // Set current points nodal membership
            pPointIds[lIterPoints] = lCurNode;
        }
        
        return 0;
    }
    
    
}