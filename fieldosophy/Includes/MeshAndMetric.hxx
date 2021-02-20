/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

#ifndef MESHANDMETRIC_HXX
#define MESHANDMETRIC_HXX


#include "hyperRectExtension.hxx"



/**
* Class representing a metric for a (possibly partially implicit ) mesh (possibly extended on hyperrectangles) 
*/
class MeshAndMetric
{
    public:

        // Class representing a metric on a mesh (possibly implicit and hyper rectangularly extended)
        class Metric
        {
            public:
            
                enum class TensorMode { Stationary, PerSimplex, PerNode };
                 
                Metric( const double * const pMetricTensor, const unsigned int pNumElements, const unsigned int pNumTensors, TensorMode pTensorMode, bool pScalar, 
                    const unsigned int * const pSectors = NULL, const unsigned int * const pNumSectorDimensions = NULL ) : 
                    mMetricTensor(pMetricTensor), mNumElements(pNumElements), mNumTensors(pNumTensors), mTensorMode(pTensorMode), mScalar(pScalar) 
                {
                    // Populate mSectors
                    if (pNumSectorDimensions != NULL && pSectors != NULL)
                        for (unsigned int lIter = 0; lIter < *pNumSectorDimensions; lIter++)
                            mSectors.push_back( pSectors[lIter] );
                }
                
                Metric( const Metric & pMetric ) : mMetricTensor(pMetric.mMetricTensor), mNumElements(pMetric.mNumElements), 
                    mNumTensors(pMetric.mNumTensors), mTensorMode(pMetric.mTensorMode), mScalar(pMetric.mScalar) 
                {
                }
                
                Metric & operator=(const Metric & pMetric)
                {
                    if (&pMetric != this)
                        *this = Metric(pMetric);
                    return *this;
                }
                
                inline const double * getMetricTensor() const {return mMetricTensor;}
                inline unsigned int getNumElements() const {return mNumElements;}
                inline unsigned int getNumTensors() const {return mNumTensors;}
                inline TensorMode getTensorMode() const {return mTensorMode;}
                inline bool isScalar() const {return mScalar;}
                inline const std::vector<unsigned int> & getSectors() const {return mSectors;}
        
            private:
            
                const double * mMetricTensor = NULL;     // Pointer to the actual tensors
                unsigned int mNumElements;              // Number of elements in one tensor
                unsigned int mNumTensors;               // Number of tensors
                
                TensorMode mTensorMode;                 // Defines the mode of the metric                
                bool mScalar = true;                    // Is the tensor a scalar (isotropic) or a D-dimensional tensor
                std::vector<unsigned int> mSectors;     // A vector defining number of steps in each sector grid (only works with the last (extended) dimensions of mesh)
        };    

        
        // Constructor
        MeshAndMetric( const HyperRectExtension pMesh, const Metric pMetric ) : mMesh(pMesh), mMetric(pMetric) {}        
        MeshAndMetric( const MeshAndMetric & pMeshAndMetric ) : mMesh(pMeshAndMetric.mMesh), mMetric(pMeshAndMetric.mMetric) {}
        
        MeshAndMetric & operator=(const MeshAndMetric & pMeshAndMetric)
        {
            if (&pMeshAndMetric != this)
                *this = MeshAndMetric(pMeshAndMetric);
            return *this;
        }
        
        inline const HyperRectExtension & getMesh() const {return mMesh;}
        inline const Metric & getMetric() const {return mMetric;}
        
        // Get metric tensor of node
        int getMetricOfNode(const unsigned int pNodeIndex, std::vector<double> & pOutput) const;
        // Get metric tensor of simplex
        int getMetricOfSimplex(const unsigned int pSimplexIndex, std::vector<double> & pOutput ) const;
        // Get metric tensor of point
        int getMetricOfPoint(const double * const pPoint, std::vector<double> & pOutput ) const;
        // Compute convex sum distance between two points using their respective 
        int computeConvexDistanceBetweenPoints(const double * const pPoint1, const double * const pPoint2, double & pOutput) const;
    
    
    private:
    
        // Constant member variables
        HyperRectExtension mMesh;     // The mesh
        Metric mMetric;                     // The metric of the mesh
    
};



// Load internally stored mesh
const MeshAndMetric * meshAndMetric_load( const unsigned int pId);



// Functions with pure C communication points
extern "C"
{    

    // Creates a MeshAndMetric
    int meshAndMetric_create( const unsigned int pMeshId, unsigned int * const pId,
        const double * const pMetricTensor, const unsigned int pNumElements, const unsigned int pNumTensors, 
        const int pTensorMode,
        unsigned int * const pNumNodes, unsigned int * const pNumSimplices, 
        const unsigned int * const pSectors = NULL, const unsigned int * const pNumSectorDimensions = NULL );
    
    // check whether meshAndMetric is already stored
    int meshAndMetric_check( const unsigned int pId );
    
    // Removes a MeshAndMetric if available
    int meshAndMetric_erase( const unsigned int pId );
    
}









#endif // MESHANDMETRIC_HXX