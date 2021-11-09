/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/


#ifndef MESH_HXX
#define MESH_HXX

#include <Eigen/Dense>

#include <set>
#include <vector>
#include <list>


// Forward declaration
class MeshGraph;

// Class representing a mesh initiated from pointers to triangles and nodes
class ConstMesh
{
    public:

        ConstMesh( const double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
            const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, 
            const unsigned int * const pNeighs = NULL ) : 
            mNodes(pNodes), mD(pD), mNumNodes(pNumNodes), mSimplices(pSimplices), mNumSimplices(pNumSimplices), mTopD(pTopD), mNeighs(pNeighs) {}
            
        ConstMesh( const ConstMesh & pMesh ) 
        {
            mNodes = pMesh.getNodes();
            mD = pMesh.getD();
            mNumNodes = pMesh.getNN();
            mSimplices = pMesh.getSimplices();
            mNumSimplices = pMesh.getNT();
            mTopD = pMesh.getTopD();
            mNeighs = pMesh.mNeighs;
        }
        
        ConstMesh & operator=(const ConstMesh & pMesh)
        {
            if (&pMesh != this)
                *this = ConstMesh(pMesh);
            
            return *this;
        }
        
            
        ~ConstMesh();
            
        // Get variables
        inline const double * getNodes() const {return mNodes;}
        inline const unsigned int * getSimplices() const {return mSimplices;}
        inline const unsigned int * getNeighs() const {return mNeighs;}
        inline const unsigned int getNN() const {return mNumNodes;}
        inline const unsigned int getNT() const {return mNumSimplices;}
        inline const unsigned int getD() const {return mD;}
        inline const unsigned int getTopD() const {return mTopD;}
        inline const bool hasNeighs() const {return (mNeighs != NULL);}
        
        
        // Get diameter of simplex
        double getDiameter( const unsigned int pSimplexInd ) const;
        // Get a simplex index for a simplex where points are part
        int getASimplexForPoint( const double * const pPoints, const unsigned int pNumPoints, 
            unsigned int * const pSimplexIds, double * const pBarycentricCoords, 
            const double pEmbTol, const double * const pCenterOfCurvature, const unsigned int pNumCentersOfCurvature) const;
        // Get a set of all simplices for which the given point is a member.
        int getAllSimplicesForPoint( const double * const pPoint, unsigned int pSimplexId, 
            std::set<unsigned int> & pOutput,
            const double pEmbTol, const double * const pCenterOfCurvature ) const;
        // Get a simplex index for a simplex where node is a part
        int getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const;
        // Get a simplex index for a simplex where set is a part
        int getASimplexForSet( const std::set<unsigned int> & pSet, unsigned int & pSimplexId) const;
        // Get a set of all simplices for which the given node is a member.
        int getAllSimplicesForNode( const unsigned int pNode, unsigned int pSimplexId,
            std::set<unsigned int> & pOutput ) const
        {
            std::set<unsigned int> lSet;
            lSet.insert(pNode);
            return getAllSimplicesForSet( lSet, pSimplexId, pOutput );
        }
        // Get a set of all simplices for which the given set is a member.
        int getAllSimplicesForSet( const std::set<unsigned int> & pSet, unsigned int pSimplexId,
            std::set<unsigned int> & pOutput ) const;
        
        // Get standard- and/or barycentric coordinates for points given simplex
        int getCoordinatesGivenSimplex( const double * const pPoints, const unsigned int pNumPoints, const unsigned int pSimplexId,
            double * const pStandardCoords, double * const pBarycentricCoords, 
            const double pEmbTol, const double * const pCenterOfCurvature, const unsigned int pNumCentersOfCurvature, double * const pBaryOutsidedness) const;
        // Get a set of all node indices part of a collection of simplices
        std::set<unsigned int> getUniqueNodesOfSimplexCollection( const unsigned int * const pSimplices, const unsigned int pNumSimplices ) const;
        // Populate arrays with corresponding mesh
        int populateArrays( double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
            unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
            unsigned int * const pNeighs = NULL ) const;
        // Computes a mesh graph of mesh
        int computeMeshGraph( const unsigned int pMaxNumNodes, const double pMinDiam, const unsigned int pMinNumTriangles );
        // See if two simplices are neighbors
        bool areSimplicesNeighbors( const unsigned int pSimpInd1, const unsigned int pSimpInd2 ) const;
        // See if node is part of simplex
        inline bool isNodePartOfSimplex( const unsigned int pNode, const unsigned int pSimplex ) const
        {
            if (pSimplex > getNT())
                return false;
            for (unsigned int lIter = 0; lIter < (getTopD()+1); lIter++ )
            {
                if ( pNode == getSimplices()[pSimplex * (getTopD()+1) + lIter] )
                    return true;
            }
            return false;
        }
        
        // See if node is part of simplex
        inline bool isSetPartOfSimplex( const std::set<unsigned int> & pSet, const unsigned int pSimplex ) const
        {
            if (pSimplex > getNT())
                return false;                
            unsigned int lNumMatches = 0;
            // Loop through simplex
            for (unsigned int lIter = 0; lIter < (getTopD()+1); lIter++ )
            {
                const unsigned int lCurNode = getSimplices()[pSimplex * (getTopD()+1) + lIter];
                if ( pSet.count(lCurNode) > 0 )
                    ++lNumMatches;
            }
            if (lNumMatches == pSet.size() )
                return true;
            
            return false;
        }
        
        
    
        
        
    protected:
    
        const double * mNodes;    
        unsigned int mD;
        unsigned int mNumNodes;
        
        const unsigned int * mSimplices;
        unsigned int mNumSimplices;
        unsigned int mTopD;
        
        const unsigned int * mNeighs;
        
        // Pointer to storage of mesh graph for node
        MeshGraph * mMeshGraph = NULL;
        // Compute the standard simplex coordinates of chosen point
        Eigen::VectorXd getSimplexStandardCoords( const double * const pPoint, const unsigned int pSimplexId, std::vector< double > & pCurNodeCoords,
            double * const pDivergence = NULL, const double * const pCenterOfCurvature = NULL ) const;

};

// Class representing a full mesh (saving nodes and triangles inside)
class FullMesh : public ConstMesh
{
    public:

        FullMesh( const double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
                const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
                const unsigned int * const pNeighs = NULL );
        
        // Method refining all simplices accordingly
        int refine( const unsigned int pMaxNumNodes, std::vector<double> & pMaxDiam, int (* transformationPtr)(double *, unsigned int) );
        // Method refining a simplex
        int refineSimplex( const unsigned int pChosenSimplex, const double pMaxDiam, const unsigned int pMaxNewSimplices = 1);
        // Overloaded member function for populating arrays from mesh
        int populateArrays( double * const pNodes, const unsigned int pD, const unsigned int pNumNodes, 
            unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
            unsigned int * const pNeighs = NULL );    

    private:
    
        // Function for updating ConstMesh pointers when FullMesh is changing
        inline void updateConstMeshPointers()
        {
            // Set pointers to data
            mNodes = (mFullNodes.size() == 0) ? NULL : mFullNodes.data();
            mSimplices = (mFullSimplices.size() == 0) ? NULL : mFullSimplices.data();
            mNeighs = (mFullNeighs.size() == 0) ? NULL : mFullNeighs.data();
            return;
        }
    
        // Stores simplices
        std::vector< unsigned int > mFullSimplices;
        // Store nodes
        std::vector< double > mFullNodes;
        // Store neigbors
        std::vector< unsigned int > mFullNeighs;

};






// Functor for comparison of sets of node indices to give a weak ordering
class CompareNodeSets
{
    public:
        CompareNodeSets( const unsigned int pD = 0 ) : mD(pD) {}
        
        bool operator() ( const std::set<unsigned int> pLhs, const std::set<unsigned int> pRhs )
        {
            // Initialize comparison as less than
            bool lLhsLessThanRhs = false;
            // Loop through set
            std::set<unsigned int>::const_iterator lIterRhs = pRhs.begin();
            for ( std::set<unsigned int>::const_iterator lIterLhs = pLhs.begin(); lIterLhs !=  pLhs.end();  )
            {
                // If current 
                if ( *lIterLhs < *lIterRhs )
                {
                    // Set that Lhs is smaller than Rhs
                    lLhsLessThanRhs = true;
                    // Stop looping
                    break;
                }
                else 
                    if ( *lIterLhs > *lIterRhs )
                        // Stop looping since rhs is apparently smaller
                        break;
                
                // Advance iterators
                ++lIterLhs;
                ++lIterRhs;
            }
            // Return result
            return lLhsLessThanRhs;
        }
        bool operator() ( const unsigned int * const pLhs, const unsigned int * const pRhs )
        {
            // Initialize comparison as less than
            bool lLhsLessThanRhs = false;
            // Loop through arrays
            for ( unsigned int lIter = 0; lIter < mD; lIter++  )
            {
                // If current 
                if ( pLhs[lIter] < pRhs[lIter] )
                {
                    // Set that Lhs is smaller than Rhs
                    lLhsLessThanRhs = true;
                    // Stop looping
                    break;
                }
                else 
                    if ( pLhs[lIter] > pRhs[lIter] )
                        // Stop looping since rhs is apparently smaller
                        break;
            }
            // Return result
            return lLhsLessThanRhs;
        }
    
    private:
        const unsigned int mD;
};


// Class representing boundaries of all simplices in a mesh
class SimplexEdges
{
    public:

        SimplexEdges() {}
    
        // Compute all simplex boundaries and which simplices they are boundaries of.
        int computeEdges( 
            const unsigned int * const pSimplices, const unsigned int pTopD, 
            const unsigned int pNumSimplices, const unsigned int pEdgeDim );
        // Populate edges array
        int populateEdges( unsigned int * pEdges, const unsigned int pNumEdges, const unsigned pNumNodes ) const;
        // Populate simplex array associated with edges array
        int populateEdgesSimplexList( unsigned int * const pSimplexList, const unsigned int pNumEdges, 
            const unsigned pMaxNumSimplicesPerEdge, const unsigned int pNumSimplices ) const;
        // Populate map of all edges to each simplex
        int populateSimplexEdgesList( unsigned int * const pEdgeList, const unsigned int pNumSimplices, 
            const unsigned int pNumEdgesPerSimplex ) const;
            
        // Acquire number of maximum simplices per edge
        inline const unsigned int & getMaxSimplicesPerEdge() const { return mMaxSimplicesPerEdge; }
        // Get number of edges
        inline unsigned int getNumEdges() const { return mEdges.size(); }
        // Get number of edges for each simplex
        inline const unsigned int & getNumEdgesPerSimplex() const { return mNumEdgesPerSimplex; }
        
        // Function for finding edge index from edge definition in logarithmic time
        static unsigned int findEdgeIndexGivenEdge( const unsigned int * const pEdge,
            const unsigned int * pEdges, const unsigned int pNumEdges, const unsigned int pEdgeDim );
    
        // Typedef of the pair of boundary node indices and simplices associated with boundary
        typedef std::pair< std::set<unsigned int>, std::set<unsigned int> > EdgeElement;

    private:

        // Functor for comparison of EdgeElements
        class CompareEdges
        {
            public:
            bool operator() ( const EdgeElement & pLhs, const EdgeElement & pRhs )
            { 
                CompareNodeSets lTemp;
                return lTemp(pLhs.first, pRhs.first); 
            }
        };
        
        // Store the maximum number of simplices for any edge
        unsigned int mMaxSimplicesPerEdge = 1;
        // Store the number of nodes in edges
        unsigned int mEdgeDim = 0;
        // Store number of edges per simplex
        unsigned int mNumEdgesPerSimplex = 0;
        // Store simplex boundaries
        std::set< EdgeElement, CompareEdges > mEdges;
};


// Class representing a mapping from original simplices on a manifold (embedded or not) to standard simplex coordinates
class MapToSimp
{
    public:
        // Constructor
        MapToSimp( const double * const pPoints, const unsigned int pD, const unsigned int pTopD );
        // Acquire determinant (of R1 if pTopD < pD)
        double getAbsDeterminant() const {return std::abs(mDeterminant);}
        // Solve 
        Eigen::VectorXd solve( const Eigen::VectorXd & pVector ) const;
        // Solve transposed
        Eigen::VectorXd solveTransposed( const Eigen::VectorXd & pVector ) const;
        // Get length between hyperplane of simplex and vector
        double getOrthogonalLength( const Eigen::VectorXd & pVector ) const;
        // Get parameter value 't' of line parameterized as 'pLinePoint' + t*'pLineVector', for the point where line cuts hyperplane of simplex.
        double getLineIntersection( const Eigen::VectorXd & pLinePoint, const Eigen::VectorXd & pLineVector ) const;
        // Get standard coordinates of vector
        Eigen::VectorXd getStandardCoord( const Eigen::VectorXd & pVector ) const {return solve(pVector - mPoint0);}
        // right multiplication of F with vector
        Eigen::VectorXd multiplyWithF( const Eigen::VectorXd & pVector ) const;        
        // Map point from standard simplex space to original space
        Eigen::VectorXd getOriginalCoord( const Eigen::VectorXd & pVector ) const { return ( multiplyWithF( pVector ) + mPoint0 );  }
    
    private:
    
        const unsigned int mD;
        const unsigned int mTopD;
        double mDeterminant;
        Eigen::VectorXd mPoint0;
        Eigen::ColPivHouseholderQR< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > mQR;

};


// Class aiding in extending mesh to new dimension
class ExtendMesh
{

    public:
        
        ExtendMesh( const std::vector<unsigned int> * const pEdges,
            const std::vector<unsigned int> * const pSimplexIdentity,
            const std::vector<unsigned int> * const pEdgeIdentity,
            const unsigned int * const pSimplices, 
            const unsigned int pNumSimplices, const unsigned int pTopD, 
            const unsigned int pNumNodes, const unsigned int pNumEdgesSimp ) : 
            mEdges(pEdges), mSimplexIdentity(pSimplexIdentity), mEdgeIdentity(pEdgeIdentity),
            mOldSimplices(pSimplices), mNumSimplices(pNumSimplices), mTopD(pTopD), mNumNodes(pNumNodes), mNumEdgesSimp(pNumEdgesSimp) 
            {
                // Create vector of which simplices that have been placed
                mPlacedSimplices = std::vector<bool>(mNumSimplices, false);
            }
            
        // Mark that simplices has been placed covering old simplex
        inline int markPlacement( const unsigned int pSimplexId, const bool pMarker = true )
        {
            if ( pSimplexId > mNumSimplices )
                return 1;
            mPlacedSimplices.at(pSimplexId) = pMarker;
            return 0;
        }
        // Get if simplex is placed
        inline bool isSimplexPlaced( const unsigned int pSimplexId ) const
        {
            if ( pSimplexId > mNumSimplices )
                return true;
            return mPlacedSimplices.at(pSimplexId);
        }
        // get all node index values to choose from for extensions of specific simplex
        inline std::set<unsigned int> allPossibleNodeInds( const unsigned int pSimplexInd ) const
        {
            // Get all nodes possible for sub simplices
            std::set<unsigned int> lPossibleNodeInds;
            for ( unsigned int lIterNodes = 0; lIterNodes < ( mTopD + 1 ); lIterNodes++ )
            {
                lPossibleNodeInds.insert( mOldSimplices[ pSimplexInd * (mTopD+1) + lIterNodes ] );
                lPossibleNodeInds.insert( mOldSimplices[ pSimplexInd * (mTopD+1) + lIterNodes ] + mNumNodes );
            }
        
            return lPossibleNodeInds;
        }
        // Get all edge indices of current simplex
        inline std::vector<unsigned int> edgesOfOldSimplex( const unsigned int pSimplexInd ) const
        {
            return std::vector< unsigned int >(
                mSimplexIdentity->begin() + pSimplexInd * mNumEdgesSimp, 
                mSimplexIdentity->begin() + (pSimplexInd + 1) * mNumEdgesSimp );
        }
        // Is simplex on the border
        bool isSimplexOnBorder( const unsigned int pSimplexId ) const;
        // Get all old simplices sharing an edge with current simplex, as well as which edge are shared between them
        std::vector< std::pair< unsigned int, unsigned int> > neighSimps( 
            const unsigned int pSimplexInd, const char pPlaced = 0 ) const;
        // Computes all new edges of old simplex that are fixed due to already placed new simplices
        std::list< std::set<unsigned int> > computeComplyEdgesFromPlaced( 
            const unsigned int pSimplexInd, const std::vector<std::pair<unsigned int, unsigned int>> & pNeighs, 
            const unsigned int * const pNewSimplices ) const;
        // Computes the simplex when projected onto old dimensionality
        std::set<unsigned int> projectNewOntoOldSimplex( std::set<unsigned int> & pNewSimplex ) const;
        // See if simplex is on edge of prism
        bool onPrismEdge( std::set<unsigned int> & pNewSimplex ) const;
            
        // Acquire vector of new simplices from an old simplex (taking into account placed neighbors)
        int computeNewSubSimplices( 
            const unsigned int pSimplexInd, const unsigned int * const pNewSimplices, std::vector< std::set<unsigned int> > & pOut  ) const;
        // Acquire list of simplices which mark the path from given simplex to closest simplex that is not placed (or is an edge)
        unsigned int shortestPathToFreedom( const unsigned int pSimplexId, const unsigned int pPrevSimplexId, std::list<unsigned int> & pPath ) const;
        
        

    protected:
    
        const std::vector<unsigned int> * const mEdges;
        const std::vector<unsigned int> * const mSimplexIdentity;
        const std::vector<unsigned int> * const mEdgeIdentity;
        std::vector<bool> mPlacedSimplices;
        const unsigned int mTopD;
        const unsigned int mNumNodes;
        const unsigned int mNumSimplices;
        const unsigned int mNumEdgesSimp;
        const unsigned int * const mOldSimplices;

};


// Get edges of simplex
int mesh_getEdgesOfSimplex( std::vector< std::set<unsigned int> > & pOut, const unsigned int pNumCombinations,
    const unsigned int pEdgeDim, const unsigned int pTopologicalD, const unsigned int * const pSimplex );
// Acquire edges and edge identities from simplex mapping    
int mesh_getEdgesAndRelations( std::vector<unsigned int> &pEdges, std::vector<unsigned int> &pSimplexIdentity, std::vector<unsigned int> &pEdgeIdentity,
    const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, const unsigned int pNumEdgesSimp = 0 );    
        

extern "C"
{    

    // Get observation matrix (only works in R^d, no embedded manifolds)
    int mesh_getObservationMatrix( double * const pData, unsigned int * const pRow, unsigned int * const pCol, const unsigned int pNumNonZeros,
        const double * const pPoints, const unsigned int pNumPoints,
        const double * const pNodes, const unsigned int pNumNodes,
        const unsigned int * const pMesh, const unsigned int pNumSimplices,
        const unsigned int pD, const unsigned int pTopD, const double pEmbTol = 0.0d,
        const double * const pCenterOfCurvature = NULL, const unsigned int pNumCentersOfCurvature = 0);
    
    // Recurrent investigation of all edges (or sub-edges) [thread safe]
    int mesh_recurrentEdgeFinder( unsigned int * const pEdgeList, const unsigned int pAllocateSpace,
        const unsigned int pD, const unsigned int * const pCurPointConfig, const unsigned int pNumPointConfig );
    
    // Get edges of simplices
    int mesh_getEdgesOfSimplices( unsigned int * const pEdges, unsigned int * pEdgeIndex, unsigned int * const pSimplexIdentity, const unsigned int pEdgeDim, 
        const unsigned int pTopologicalD, const unsigned int * const pSimplex, const unsigned int pNumSimplices ); 
    // Compute edges of chosen dimensionality and which simplices that are associated to them
    int mesh_computeEdges( const unsigned int pEdgeDim, const unsigned int * const pSimplices,
        const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNumEdges, unsigned int * const pEdgeId, unsigned int * const pMaxSimplicesPerEdge );
    // Populate arrays of edges, associated simplices, and associated edges to each simplex
    int mesh_populateEdges( unsigned int * const pEdges, const unsigned int pEdgeDim, const unsigned int pNumEdges, 
        const unsigned int pEdgeId,
        unsigned int * const pSimplicesForEdges, const unsigned int pMaxSimplicesPerEdge, const unsigned int pNumSimplices, 
        unsigned int * const pEdgesForSimplices, const unsigned int pNumEdgesPerSimplex );
    // Clear saved edges
    int mesh_clearEdges( const unsigned int pEdgeId );
    
    // Get neighborhood of each simplex
    int mesh_getSimplexNeighborhood( const unsigned int pNumEdges, const unsigned int pNumSimplices,
        const unsigned int * const pSimplicesForEdges, const unsigned int pMaxSimplicesPerEdge,
        const unsigned int * const pEdgesForSimplices, const unsigned int pNumEdgesPerSimplex,
        unsigned int * const pNeighborhood
      );
    
    // Refines chosen simplices
    int mesh_refineMesh( const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices, unsigned int * const pId,
        const unsigned int pMaxNumNodes, const double * const pMaxDiam, const unsigned int pNumMaxDiam,
        int (* transformationPtr)(double *, unsigned int) );
        
    // Populate new mesh
    int mesh_acquireMesh( const unsigned int pId, 
        double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        unsigned int * const pNeighs = NULL );
        
    // Binomial coefficient
    inline unsigned int mesh_nchoosek( unsigned int n, unsigned int k );
    
    // Extend mesh
    int mesh_extendMesh( unsigned int * const pNewSimplices, const unsigned int pNewNumSimplices,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD, const unsigned int pNumNodes );
        
        
        
}





#endif // MESH_HXX