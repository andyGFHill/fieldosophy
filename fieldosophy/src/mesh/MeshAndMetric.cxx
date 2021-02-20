/* 
* C/C++ functions for the MeshAndMetric class.
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


#include <cmath>
#include "Eigen/Dense"

#include "MeshAndMetric.hxx"
#include "misc.hxx"










// Get metric tensor of node
int MeshAndMetric::getMetricOfNode(const unsigned int pNodeIndex, std::vector<double> & pOutput) const
{
    // Clear output
    pOutput.clear();

    // Make sure that stationary
    if (getMetric().getTensorMode() != Metric::TensorMode::Stationary)
        // Flag error
        return 1;
    
    // Get Metric
    const MeshAndMetric::Metric & lMetric = getMetric();
    // Insert output
    pOutput.insert( pOutput.end(), lMetric.getMetricTensor(), &lMetric.getMetricTensor()[lMetric.getNumElements()] );
    
    return 0;
}

// Get metric tensor of simplex
int MeshAndMetric::getMetricOfSimplex(const unsigned int pSimplexIndex, std::vector<double> & pOutput ) const
{
    // Clear output
    pOutput.clear();

    // Make sure that stationary
    if (getMetric().getTensorMode() != Metric::TensorMode::Stationary)
        // Flag error
        return 1;
    
    // Get Metric
    const MeshAndMetric::Metric & lMetric = getMetric();
    // Insert output
    pOutput.insert( pOutput.end(), lMetric.getMetricTensor(), &lMetric.getMetricTensor()[lMetric.getNumElements()] );
    
    return 0;
}

// Get metric tensor of point
int MeshAndMetric::getMetricOfPoint(const double * const pPoint, std::vector<double> & pOutput ) const
{
    // Clear output
    pOutput.clear();

    // Make sure that stationary
    if (getMetric().getTensorMode() != Metric::TensorMode::Stationary)
        // Flag error
        return 1;
    
    // Get Metric
    const MeshAndMetric::Metric & lMetric = getMetric();
    // Insert output
    pOutput.insert( pOutput.end(), lMetric.getMetricTensor(), &lMetric.getMetricTensor()[lMetric.getNumElements()] );
    
    return 0;
}

// Compute convex sum distance between two points using their respective 
int MeshAndMetric::computeConvexDistanceBetweenPoints(const double * const pPoint1, const double * const pPoint2, double & pOutput) const
{
    int lStatus = 0;

    // Get metric of point 1
    std::vector<double> lMetric1;
    lStatus = getMetricOfPoint(pPoint1, lMetric1);
    if (lStatus)
        return 1;
    // Get metric of point 2
    std::vector<double> lMetric2;
    lStatus = getMetricOfPoint(pPoint2, lMetric2);
    if (lStatus)
        return 1;
    
    // Get an Eigen::vector representation of the points
    Eigen::Map<const Eigen::VectorXd> lPoint1( pPoint1, getMesh().getD() );
    Eigen::Map<const Eigen::VectorXd> lPoint2( pPoint2, getMesh().getD() );
    // Get vector of difference between points
    const Eigen::VectorXd lDiff = lPoint2 - lPoint1;
    // Get length of vector
    const double lDiffNorm = lDiff.norm();
    // If length is zero
    if (lDiffNorm == 0.0)
    {
        pOutput = 0.0d;
        return 0;
    }
    
    // Get metric tensor
    Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > lMetric1Map( lMetric1.data(), getMesh().getD(), getMesh().getD() );
    // Get innet product
    double lContrib1 = lDiff.transpose() * ( lMetric1Map * lDiff );
    
    // Get metric tensor
    Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > lMetric2Map( lMetric2.data(), getMesh().getD(), getMesh().getD() );
    // Get innet product
    double lContrib2 = lDiff.transpose() * ( lMetric2Map * lDiff );
    
    // Compute distance and return
    if ( lContrib1 == lContrib2 )
    {
        pOutput = std::sqrt(lContrib1);
    }
    else
    {
        pOutput = ( lContrib2 * std::sqrt(lContrib2) ) - ( lContrib1 * std::sqrt(lContrib1) );
        pOutput /= lContrib2 - lContrib1;
        pOutput *= 2.0d / 3.0d;
    }
    
    return 0;
}









// Global list storing MeshAndMetrics
GlobalVariablesList<MeshAndMetric> gMeshesAndMetrics;




const MeshAndMetric * meshAndMetric_load( const unsigned int pId) { return gMeshesAndMetrics.load(pId); }



// Functions with pure C communication points
extern "C"
{    
    
    // Creates a MeshAndMetric
    int meshAndMetric_create( const unsigned int pMeshId, unsigned int * const pId,
        const double * const pMetricTensor, const unsigned int pNumElements, const unsigned int pNumTensors, 
        const int pTensorMode,
        unsigned int * const pNumNodes, unsigned int * const pNumSimplices, 
        const unsigned int * const pSectors, const unsigned int * const pNumSectorDimensions )
    {
        // Try to load mesh
        const HyperRectExtension * const lMesh = hyperRectExtension_load(pMeshId);
        if (lMesh == NULL)
            return 1;
        *pNumNodes = lMesh->getNN();
        *pNumSimplices = lMesh->getNT();
            
        // Get tensor mode
        MeshAndMetric::Metric::TensorMode lTensorMode;
        switch ( pTensorMode )
        {
            case 0: 
                lTensorMode = MeshAndMetric::Metric::TensorMode::Stationary;
                break;
            case 1: 
                lTensorMode = MeshAndMetric::Metric::TensorMode::PerSimplex;
                break;
            case 2: 
                lTensorMode = MeshAndMetric::Metric::TensorMode::PerNode;
                break;
        
            default:
                return 2;
        }
        
        const bool lIsScalar = (pNumElements == 1);
        
        // Create Metric
        MeshAndMetric::Metric lMetric( pMetricTensor, pNumElements, pNumTensors, lTensorMode, lIsScalar, pSectors, pNumSectorDimensions );
    
        // Try to store
        *pId = gMeshesAndMetrics.store( MeshAndMetric( *lMesh, lMetric ) );
    
        return 0;
    }
    
    // check whether meshAndMetric is already stored
    int meshAndMetric_check( const unsigned int pId )
    {
        const MeshAndMetric * const lPointer = gMeshesAndMetrics.load(pId);
        if (lPointer == NULL)
            // Return error since it was not found
            return 1;
        // Return okay since it was found
        return 0;
    }
    
    // Removes a MeshAndMetric if available
    int meshAndMetric_erase( const unsigned int pId )
    {
        // Load implicit mesh from storage
        return gMeshesAndMetrics.erase(pId);
    }
    



        
        
}   // End of extern "C"
