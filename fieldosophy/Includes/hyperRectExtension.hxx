/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef HYPERRECTEXTENSION_HXX
#define HYPERRECTEXTENSION_HXX


#include "implicitMesh.hxx"



/**
* Class representing a mesh that can be extended hyper rectangularly
*/
class HyperRectExtension
{
    public:
    
        HyperRectExtension( const ImplicitMesh & pMesh) : mMesh(pMesh)
        {
            mNN = mMesh.getNN();
            mNT = mMesh.getNT();
        }
        
        HyperRectExtension( const HyperRectExtension & pMesh ) : mMesh(pMesh.mMesh)
        {
            mNN = pMesh.mNN;
            mNT = pMesh.mNT;        
            
            mNumSteps = pMesh.mNumSteps;
            mOffset = pMesh.mOffset;
            mStepLengths = pMesh.mStepLengths;
        }
        
        HyperRectExtension & operator=(const HyperRectExtension & pMesh)
        {
            if (&pMesh != this)
                *this = HyperRectExtension(pMesh);
            
            return *this;
        }
        
        // Define a hyper-rectangular extension of the mesh 
        int defineHyperRectExtension( const unsigned int pNumNewDims, const double * const pOffset, 
            const double * const pStepLengths, const unsigned int * const pNumSteps );
    
        // Get functions
        inline const ImplicitMesh & getMesh() const {return mMesh;}
        inline unsigned int getExtendedD() const { return mOffset.size();}
        inline unsigned int getD() const { return mMesh.getD() + getExtendedD(); }
        inline unsigned int getTopD() const { return mMesh.getTopD() + getExtendedD(); }    
        inline unsigned int getNN() const {return mNN;}
        inline unsigned int getNT() const {return mNT;}
        inline const std::vector<unsigned int> & getNumSteps() const {return mNumSteps;}
        inline unsigned int getNodesPerSimplex() const 
        { 
            unsigned int lOutput = mMesh.getTopD()+1;
            for (unsigned int lIter = 0; lIter < getExtendedD(); lIter++)
                lOutput *= 2;
            return lOutput;
        }
        
        
        // Get diameter of simplex
        double getDiameter( const unsigned int pSimplexInd ) const;
        // Get neighbors for simplex
        int getNeighborsFromSimplex(const unsigned int pSimplexInd, std::set<unsigned int> & pNeighs, std::set<unsigned int> & pTemp) const;
        // Get a simplex index for a simplex where points are part
        int getASimplexForPoint( double * const pPoints, const unsigned int pNumPoints,
            unsigned int * const pSimplexIds, double * const pBarycentricCoords,
            const double pEmbTol, const double * const pCenterOfCurvature, const unsigned int pNumCentersOfCurvature) const;
        // Get a set of all simplices for which the given point is a member.
        int getAllSimplicesForPoint( double * const pPoint, std::set<unsigned int> & pSimplexId, std::set<unsigned int> & pOutput,
            const double pEmbTol, const double * const pCenterOfCurvature ) const;
        // Get a simplex index for a simplex where node is a part
        int getASimplexForNode( const unsigned int * const pNodes, const unsigned int pNumNodes, unsigned int * const pSimplexIds) const;
        // Get a simplex index for a simplex where set is a part
        int getASimplexForSet( const std::set<unsigned int> & pSet, unsigned int & pSimplexId) const;
        // Get a set of all simplices for which the given node is a member.
        int getAllSimplicesForNode( const unsigned int pNode, std::set<unsigned int> & pSimplexId,
            std::set<unsigned int> & pOutput ) const
        {
            std::set<unsigned int> lSet;
            lSet.insert(pNode);
            return getAllSimplicesForSet( lSet, pSimplexId, pOutput );
        }
        // Get a set of all simplices for which the given set is a member.
        int getAllSimplicesForSet( const std::set<unsigned int> & pSet, std::set<unsigned int> & pSimplexId, std::set<unsigned int> & pOutput ) const;
        
        // Get a vector of chosen node    
        int getNode( const unsigned int pNodeInd, std::vector<double> & pOutput ) const;
        // Get a set of chosen simplex
        int getSimplex( const unsigned int pSimplexInd, std::set<unsigned int> & pOutput, std::set<unsigned int> & pTemp ) const;
        // Is node part of simplex
        bool isNodePartOfSimplex(const unsigned int pNodeInd, const unsigned int pSimplexInd, std::set<unsigned int> & pSimplexVec, std::set<unsigned int> & pTemp2) const;
        // Is point part of simplex
        int isPointPartOfSimplex( double * const pPoint, const unsigned int pSimplexInd, double * const pStandardCoords, double * const pBarycentricCoords, 
            const double pEmbTol, const double * const pCenterOfCurvature, int * const pStatus) const;

        
    
    private:
    
        // Get number of simplices for moving one level in current dimension
        inline unsigned int getNumCopiesPerLevel( const unsigned int pDim ) const
        {
            const unsigned int lDim = (pDim <= getD()) ? pDim : getD();
            // Get number of copies up until target dimension
            unsigned int lTotalCopies = 1;
            std::vector<unsigned int>::const_iterator lIterNumCopies = getNumSteps().begin();
            for (unsigned int lIterDims = 0; lIterDims < lDim; lIterDims++)
            {
                lTotalCopies *= *lIterNumCopies;
                ++lIterNumCopies;
            }
            return lTotalCopies;
        }
        // Project simplex to implicit mesh
        inline unsigned int projectSimplex(const unsigned int pSimplexInd) const { return pSimplexInd % mMesh.getNT();}
        // Get level of simplex in all dimensions
        inline unsigned int getLevelOfSimplex(const unsigned int pSimplexInd) const {return pSimplexInd / mMesh.getNT();}
        // Get level of simplex in chosen dimension
        inline unsigned int getLevelOfSimplexInDim(const unsigned int pSimplexInd, const unsigned int pDim) const {return pSimplexInd / ( mMesh.getNT() * getNumCopiesPerLevel(pDim) );}
        // Project node to implicit mesh
        inline unsigned int projectNode( const unsigned int pNodeInd ) const { return pNodeInd % mMesh.getNN(); }
    
    
        ImplicitMesh mMesh; // Implicit mesh to base extension on
        unsigned int mNN;
        unsigned int mNT;
        
        // Extension into new dimensions
        std::vector<double> mOffset;  // Offset for each new dimension
        std::vector<double> mStepLengths;  // Step lengths for each new dimension
        std::vector<unsigned int> mNumSteps;  // Number of steps for each new dimension
        
};


// Load internally stored mesh
const HyperRectExtension * hyperRectExtension_load( const unsigned int pId);


extern "C"
{    

    
    // Create mesh
    int hyperRectExtension_createMesh( const double * const pNodes, const unsigned int pNumNodes, const unsigned int pD,
        const unsigned int * const pSimplices, const unsigned int pNumSimplices, const unsigned int pTopD,
        const double* const pOffsetImplicit, const unsigned int * const pCopiesPerDimension,
        const double* const pOffsetHyper, const double * const pStepLengths, const unsigned int * const pNumSteps, const unsigned int pExtendDims,
        unsigned int * const pId, unsigned int * const pNewNumNodes, unsigned int * const pNewNumSimplices,
        const unsigned int * const pNeighs = NULL );

    // Remove mesh
    int hyperRectExtension_eraseMesh( const unsigned int pId);
        
        
}

    
#endif // HYPERRECTEXTENSION_HXX