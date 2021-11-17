/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/


#ifndef MESHGRAPH_HXX
#define MESHGRAPH_HXX

#include <vector>

// Forward declaration
class ConstMesh;


// Class representing the graph of regular hyper rectangles and their triangles
class MeshGraph
{
    public:
        // Class for each node in graph
        class Node
        {
            public:
            
            Node( const std::vector<double> &pBoundaries );
            Node( const Node & pNode);
            
            // See if any node of triangle is inside node
            bool triangleInside( const std::vector< std::vector<double> > & pPoints ) const;
            // See if point is inside
            bool pointInside( const std::vector<double> & pPoints ) const;
            
            // Stores neighbor nodes (which node and which direction)
            std::vector< std::pair<unsigned int, unsigned int> > mNeighNodes;
            // vector of all triangles in node
            std::vector< unsigned int > mTriangles;
            // Dimensionality
            unsigned int mD;
            // vector of coordinates for cuboid
            std::vector<double> mBoundaries;
        };
        
        // Constructor
        MeshGraph( const ConstMesh & pMesh, const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles,
            const double * const pPoint = NULL, const unsigned int * const pNumPoints = NULL );
        MeshGraph( const unsigned int pD, const double * const pBoundaries );  
        
        // Populate graph from mesh
        int populate( const unsigned int * pTriangles, const double * pPoints, const unsigned int pNumTriangles, 
            const unsigned int pManD, const unsigned int pNumPoints, const unsigned int pTopD,
            const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles );    
        // Get number of nodes
        unsigned int getNumNodes() const { return mNodeList.size(); }
        // Get dimensionality
        unsigned int getD() const { return mD; }
        // Acquire graph node 
        const Node * getNode( const unsigned int pNodeIndex ) const;
        // Acquire node list iterator
        std::pair< std::vector< Node >::const_iterator, std::vector< Node >::const_iterator > getNodeListIterator() const;
        // Acquire which node a point belongs to
        int getNodeOfPoint( const double * const pPoint, const unsigned int pD, unsigned int * const pNode ) const;
    
    
    private:
    
        // split node by dimension
        int splitNode( const unsigned int pNodeInd, const unsigned int pDim, 
            const unsigned int * const pTriangles, const double * const pPoints, const unsigned int pTopD );
        // break connection between node and all nodes at one side
        int breakSideConnection( const unsigned int pNodeInd, const unsigned int pDirection );
        // break connection between two neighboring nodes
        int breakConnection( const unsigned int pNodeInd1, const unsigned int pNodeInd2 );
        // break connection between two formerly non-neighboring nodes
        int addConnection( const unsigned int pNodeInd1, const unsigned int pNodeInd2, const unsigned int pNode21Direction );
        
        // vector of all nodes
        std::vector<Node> mNodeList;
        // vector of coordinates for cuboid
        std::vector<double> mBoundaries;
        // dimensionality
        unsigned int mD;
};


extern "C"
{
        // Function for creating graph (and storing it) given mesh
        int MeshGraph_createGraph( const double * pPoints, const unsigned int pNumPoints, const unsigned int pD, 
            const unsigned int * pTriangles, const unsigned int pNumTriangles, const unsigned int pTopD,
            const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles,
            unsigned int * const pNumNodes, unsigned int * const pGraphIndex );
            
        // Free up saved graph
        int MeshGraph_freeGraph( const unsigned int pGraphIndex );
        
        // Get number of graphs
        unsigned int MeshGraph_getNumGraphs( );
        
        // Get number of nodes 
        unsigned int MeshGraph_getNumNodes( const unsigned int pGraphIndex );
        
        // Function for acquiring boundary boxes for all nodes in graph
        int MeshGraph_getNodeBoundaries( const unsigned int pGraphIndex, double * const pBoundaries, const unsigned int pNumNodes, const unsigned int pD );
        
        // Function for acquiring number of triangles for specific node
        int MeshGraph_getNodeNumTriangles( const unsigned int pGraphIndex, const unsigned int pNodeIndex, unsigned int * const pNumTriangles );
        
        // Function for aquiring triangles for specific node
        int MeshGraph_getNodeTriangles( const unsigned int pGraphIndex, const unsigned int pNodeIndex, const unsigned int pD,
            const unsigned int pNumTriangles, unsigned int * const pTriangleIds );
            
        // Function for aquiring nodal identity of given points
        int MeshGraph_getNodesOfPoints( const unsigned int pGraphIndex, 
            const double * const pPoints, const unsigned int pD, 
            unsigned int * const pPointIds, const unsigned int pNumPoints );
       
}


#endif // MESHGRAPH_HXX