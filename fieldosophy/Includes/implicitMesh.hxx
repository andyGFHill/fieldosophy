/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef IMPLICIT_MESH_HXX
#define IMPLICIT_MESH_HXX

#include <set>
#include <vector>
#include <list>

#include "mesh.hxx"



/**
* Class representing a mesh which can be implicitly extended
*/
class ImplicitMesh
{

    public:
    
        ImplicitMesh( const ConstMesh & pConstMesh ) : mMesh(pConstMesh)
        {
            mNN = mMesh.getNN();
            mNT = mMesh.getNT();
            
            populateBoundingBox();
        }
        
        ImplicitMesh( const ImplicitMesh & pMesh ) : mMesh(pMesh.mMesh)
        {
            mNN = pMesh.mNN;
            mNT = pMesh.mNT;        
            mBoundingBox = pMesh.mBoundingBox;
            mBoundingNodes = pMesh.mBoundingNodes;
            mBoundingSimplices = pMesh.mBoundingSimplices;
                
            mCopiesPerDimension = pMesh.mCopiesPerDimension;
            mOffset = pMesh.mOffset;
            mPairing = pMesh.mPairing;
            mPairingSimplices = pMesh.mPairingSimplices;
            
        }
        
        ImplicitMesh & operator=(const ImplicitMesh & pMesh)
        {
            if (&pMesh != this)
                *this = ImplicitMesh(pMesh);
            
            return *this;
        }
        
        inline const ConstMesh & getConstMesh() const {return mMesh;}
        inline unsigned int getD() const { return mMesh.getD(); }
        inline unsigned int getTopD() const { return mMesh.getTopD(); }    
        inline unsigned int getNN() const {return mNN;}
        inline unsigned int getNT() const {return mNT;}
        inline unsigned int getNumSectors() const {return getNT() / mMesh.getNT();}
        
        const std::vector< std::pair<double, double> > & getBoundingBox() const {return mBoundingBox;}
        const std::vector< std::pair< std::set<unsigned int>, std::set<unsigned int> > > & getBoundingNodes() const {return mBoundingNodes;}
        const std::vector< std::pair< std::set<unsigned int>, std::set<unsigned int> > > & getBoundingSimplices() const {return mBoundingSimplices;}
        int getNode(const unsigned int pNodeIndex, std::vector<double> & pOutput) const;
        int getSimplex(const unsigned int pSimplexIndex, std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp) const;
        
        // Get sector index and explicit index given actual node index
        int nodeInd2SectorAndExplicitInd( const unsigned int pNodeInd, unsigned int & pSector, unsigned int & pExplicitInd ) const;
        // Get actual node index given sector index and explicit index
        int sectorAndExplicit2NodeInd( unsigned int pSector, const unsigned int pExplicitInd, unsigned int & pNodeInd) const;
        // Get sector of point
        unsigned int point2Sector( const double * const pPoint ) const;
        // Get actual simplex index given sector index and explicit index
        inline unsigned int sectorAndExplicit2SimplexInd( const unsigned int pSector, const unsigned int pExplicitInd ) const
        {
            unsigned int lIndex = pExplicitInd;
            lIndex += pSector * mMesh.getNT();            
            return lIndex;
        }
        // Get sector index given simplex
        inline unsigned int simplexInd2Sector(const unsigned int pSimplexInd) const { return pSimplexInd / mMesh.getNT(); }
        // Get explicit simplex index given actual simplex index
        inline unsigned int simplexInd2ExplicitInd( const unsigned int pSimplexInd ) const { return pSimplexInd % mMesh.getNT(); }
        // Get neighbors for simplex
        int getNeighborsFromSimplex(const unsigned int pSimplexInd, std::set<unsigned int> & pNeighs) const;
        // Get a simplex index for a simplex where points are part
        int getASimplexForPoint( double * const pPoints, const unsigned int pNumPoints,
            unsigned int * const pSimplexIds, double * const pBarycentricCoords = NULL ) const;
        // Get a set of all simplices for which the given point is a member.
        int getAllSimplicesForPoint( double * const pPoint, std::set<unsigned int> & pSimplexId,
            std::set<unsigned int> & pOutput, std::set<unsigned int> & pExplicitSimp, std::set<unsigned int> & pTemp ) const;
        // Get a simplex index for a simplex where node is a part
        int getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const;
        // Get a set of all simplices for which the given node is a member.
        int getAllSimplicesForNode( const unsigned int pNode, std::set<unsigned int> & pSimplexId,
            std::set<unsigned int> & pOutput, std::set<unsigned int> & pExplicitSimp, std::set<unsigned int> & pTemp ) const;
        // Is point part of simplex
        int isPointPartOfSimplex( double * const pPoint, const unsigned int pSimplexInd, double * const pStandardCoords, double * const pBarycentricCoords, double * const pDivergence, int * const pStatus) const;
        
        // Populate arrays with explicit mesh from implicit mesh
        int populateArraysFromFullMesh( double * const pNodes, const unsigned int pNumNodes, const unsigned int pD, 
            unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
            unsigned int * const pNeighs) const;
        
        
        /**
        * Define an extension of a mesh occupying a hyperrectangular region (implicitly connecting extreme points in each dimension)
        *
        * @param pOffset A d-dimensional array defining the offset of the ConstMesh nodes in each dimension.
        * @param pNumCopiesPerDim An d array with the number of copies of ConstMesh in the current dimension.
        *
        * @return 0 if everything is okay.
        */
        int defineExtensionInCurrentDims( const double * const pOffset, const unsigned int * const pNumCopiesPerDim );
        
        
    private:
    
        // Analyzes ConstMesh and compute bounding box and boundingNodes
        int populateBoundingBox();
        
        // Help functions
        inline unsigned int getNumCopiesPerSector( const unsigned int pDim ) const
        {
            // Get number of copies up until target dimension
            unsigned int lTotalCopies = 1;
            std::vector<unsigned int>::const_iterator lIterNumCopies = mCopiesPerDimension.begin();
            for (unsigned int lIterDims = 0; lIterDims < pDim; lIterDims++)
            {
                lTotalCopies *= *lIterNumCopies;
                ++lIterNumCopies;
            }
            
            return lTotalCopies;
        }
        inline unsigned int getCurDimSector( const unsigned int pSector, const unsigned int pDim ) const
        {                
            return ( (pSector / getNumCopiesPerSector(pDim)) % mCopiesPerDimension[pDim] );
        }
        inline unsigned int getSectorFromDimSector(const unsigned int pDimSector, const unsigned int pDim) const
        {
            return ( pDimSector * getNumCopiesPerSector( pDim ) );
        }
        unsigned int sector2ExplicitIndexSize( const unsigned int pSector ) const;
        std::vector<unsigned int> sector2ExplicitIndexing( const unsigned int pSector ) const;
        unsigned int sector2ExplicitIndexReverse( const unsigned int pSector, const unsigned int pIndex ) const;
        unsigned int sector2ExplicitIndex( const unsigned int pSector, const unsigned int pIndex ) const;
        inline double getBoundingBoxLength( const unsigned int pDim ) const 
        {
            if (pDim >= getD())
                return 0.0;
            const std::pair<unsigned int, unsigned int> & lCurDimBox = getBoundingBox()[pDim];
            return (lCurDimBox.second - lCurDimBox.first);
        }
        inline unsigned int findExplicitPairedNode(const unsigned int pNodeInd, const unsigned int pDim, const bool pIsFront) const
        {
            if (mPairing.size() > pDim)
                for ( std::vector< std::pair<unsigned int, unsigned int> >::const_iterator lIter = mPairing[pDim].begin(); lIter != mPairing[pDim].end(); ++lIter)
                {
                    const unsigned int lCurNode = pIsFront ? lIter->first : lIter->second;
                    if (pNodeInd == lCurNode)
                        return pIsFront ? lIter->second : lIter->first;
                }
            return mMesh.getNN();
        }
        inline unsigned int findExplicitPairedSimplex(const unsigned int pSimpInd, const unsigned int pDim, const bool pIsFront) const
        {
            if (mPairingSimplices.size() > pDim)
                for ( std::vector< std::pair<unsigned int, unsigned int> >::const_iterator lIter = mPairingSimplices[pDim].begin(); lIter != mPairingSimplices[pDim].end(); ++lIter)
                {
                    const unsigned int lCurSimp = pIsFront ? lIter->first : lIter->second;
                    if (pSimpInd == lCurSimp)
                        return pIsFront ? lIter->second : lIter->first;
                }
            return mMesh.getNT();
        }
        int getExplicitIndInSector( unsigned int & pExplicitInd, unsigned int & pFromSector, const unsigned int pToSector ) const;
        
        
        
    
        ConstMesh mMesh;                                        // The mesh
        unsigned  int mNN;                                      // Number (implicit) nodes
        unsigned int mNT;                                       // Number (implicit) simplices
        std::vector< std::pair<double, double> > mBoundingBox;  // Bounding box of ConstMesh
        std::vector< std::pair< std::set<unsigned int>, std::set<unsigned int> > > mBoundingNodes;  // Bounding nodes on each rectangular boundary
        std::vector< std::pair< std::set<unsigned int>, std::set<unsigned int> > > mBoundingSimplices;  // Bounding simplices on each rectangular boundary
        
        
        // Original dimensions extension
        std::vector<unsigned int> mCopiesPerDimension; // Defines number of copies in each dimension of given explicit mesh
        std::vector<double> mOffset; // offset in each dimension
        std::vector< std::vector< std::pair<unsigned int, unsigned int> > > mPairing; // Pairing of front and back nodes for implicit extention
        std::vector< std::vector< std::pair<unsigned int, unsigned int> > > mPairingSimplices; // Pairing of front and back simplices for implicit extention
            



};




extern "C"
{    

    
    // Create implicit mesh
    int implicitMesh_createImplicitMesh(const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        const double* const pOffset, const unsigned int * const pCopiesPerDimension,
        unsigned int * const pId, unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices, 
        const unsigned int * const pNeighs = NULL);    
        
    // Get full mesh from implicit mesh
    int implicitMesh_retrieveFullMeshFromImplicitMesh( const unsigned int pId,
        double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD);
        
    // Get full neighborhood from implicit mesh
    int implicitMesh_retrieveFullNeighsFromImplicitMesh( const unsigned int pId,
        unsigned int * const pNeighs, const unsigned int pNumSimplices, const unsigned int pTopD);
        
    // Remove implicit mesh
    int implicitMesh_eraseImplicitMesh( const unsigned int pId);
    
    // Debug functions
    int implicitMesh_nodeInd2SectorAndExplicit( const unsigned int pId, const unsigned int pNodeInd, unsigned int * const pSector, unsigned int * const pExplicitInd);
    int implicitMesh_nodeSectorAndExplicit2Ind( const unsigned int pId, 
        const unsigned int pSector, const unsigned int pExplicitInd, unsigned int * const pNodeInd);
        
    // Checks whether a certain point is in a certain simplex
    int implicitMesh_pointInSimplex( const unsigned int pId, const double * const pPoint, const unsigned int pDim, 
        const unsigned int pSimplex, bool * const pOut );
    // Checks whether a certain node is in a certain simplex
    int implicitMesh_nodeInSimplex( const unsigned int pId, const unsigned int pNode, 
        const unsigned int pSimplex, bool * const pOut );
        
        
}





#endif // IMPLICIT_MESH_HXX