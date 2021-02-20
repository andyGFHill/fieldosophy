/* 
* C/C++ miscellaneous functions.
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


#include <math.h>
#include <omp.h>
#include "misc.hxx"




extern "C"
{



    inline void misc_computeCrossCorrTemplate( 
        const unsigned int pX1, const unsigned int pY1, 
        const unsigned int pX2, const unsigned int pY2, 
        const double pRadiusSq, const double pStartRadiusSq, const unsigned int pWidth, const unsigned int pHeight,
        const double * const pImage1, const double * const pImage2,
        double * const pCrossCorr, double * const pMeanImg1, double * const pMeanImg2, 
        double * const pVarImg1, double * const pVarImg2, unsigned int * const pNumIndices )
    {

        // Acquire current radius
        const double lTemplateOriginX = ( (double)pX1 - (double)pX2);
        const double lTemplateOriginY = ( (double)pY1 - (double)pY2 );
        const double lCurTemplateRadius = lTemplateOriginX * lTemplateOriginX + lTemplateOriginY * lTemplateOriginY;
        // If radius is outside allowed
        if ( (lCurTemplateRadius > pRadiusSq) || (lCurTemplateRadius < pStartRadiusSq) )
            // Escape
            return;
        
        // Acquire pixel values
        const double lCurValImg1 = pImage1[ pX1 + pY1 * pWidth ];
        const double lCurValImg2 = pImage2[ pX2 + pY2 * pWidth ];
        // If any value is nan
        if ( isnan(lCurValImg1) || isnan(lCurValImg2)  )
            // Escape
            return;
        
        // Sum up
        *pCrossCorr += lCurValImg1 * lCurValImg2;
        *pMeanImg1 += lCurValImg1;
        *pMeanImg2 += lCurValImg2;
        *pVarImg1 += lCurValImg1 * lCurValImg1;
        *pVarImg2 += lCurValImg2 * lCurValImg2;
        
        (*pNumIndices)++;
        
        return;
    }









    inline void misc_computeCrossCorrSearch( const unsigned int pX1, const unsigned int pY1, 
        const unsigned int pX2, const unsigned int pY2, const unsigned int pWidth, const unsigned int pHeight,
        const double pSearchRadiusSq, const double pSearchStartRadiusSq,
        const unsigned int pTemplateRadius, const unsigned int pTemplateSkip, const unsigned int pTemplateStart,
        const double * const pImage1, const double * const pImage2,
        double * const pMaximum, unsigned int * const pMaximumIndex )
    {
    
        // Acquire current search to current differences
        const double lSearchOriginX = ( (double)pX1 - (double)pX2 );
        const double lSearchOriginY = ( (double)pY1 - (double)pY2 );
        // Get current distance between search point and current point                    
        const double lCurSearchRadius = lSearchOriginX * lSearchOriginX + lSearchOriginY * lSearchOriginY;
        // If radius is outside allowed
        if ( ( lCurSearchRadius > pSearchRadiusSq ) || ( lCurSearchRadius < pSearchStartRadiusSq ) )
            // Escape
            return;
            
        // Initialize cross-corr
        double lCrossCorr = 0.0;
        double lMeanImg1 = 0.0;
        double lMeanImg2 = 0.0;
        double lVarImg1 = 0.0;
        double lVarImg2 = 0.0;
        unsigned int lNumIndices = 0;
        const double lTemplateRadiusSq = ((double)pTemplateRadius) * ((double)pTemplateRadius);
        const double lTemplateStartRadiusSq = ((double)pTemplateStart) * ((double)pTemplateStart);
        
        // Loop through each pixel in template
        for (unsigned int lIterTempX = 0 ; lIterTempX <= pTemplateRadius ; lIterTempX += pTemplateSkip + 1 )
            for (unsigned int lIterTempY = 0 ; lIterTempY <= pTemplateRadius ; lIterTempY += pTemplateSkip + 1 )
            {
                // Handle 0 case
                if ( (lIterTempX == 0) && (lIterTempY == 0) )
                    // Compute cross correlation for no shift
                    misc_computeCrossCorrTemplate( 
                        pX1, pY1, pX2, pY2, 
                        lTemplateRadiusSq, lTemplateStartRadiusSq, pWidth, pHeight,
                        pImage1, pImage2,
                        &lCrossCorr, &lMeanImg1, &lMeanImg2, &lVarImg1, &lVarImg2, &lNumIndices );
                else
                {
                    // If current x is not too big
                    if ( (pX1 + lIterTempX < pWidth) && (pX2 + lIterTempX < pWidth) )
                    {
                        // If current y is not too big
                        if ( (pY1 + lIterTempY < pHeight) && (pY2 + lIterTempY < pHeight) )
                            // Compute cross correlation for positive shifts
                            misc_computeCrossCorrTemplate( pX1 + lIterTempX, pY1 + lIterTempY, 
                                pX2 + lIterTempX, pY2 + lIterTempY,
                                lTemplateRadiusSq, lTemplateStartRadiusSq, pWidth, pHeight,
                                pImage1, pImage2,
                                &lCrossCorr, &lMeanImg1, &lMeanImg2, &lVarImg1, &lVarImg2, &lNumIndices );
                            
                        // If current y is not too small
                        if ( (pY1 >= lIterTempY) && (pY2 >= lIterTempY) )
                            // Compute cross correlation for pos.-neg. shifts
                            misc_computeCrossCorrTemplate( pX1 + lIterTempX, pY1 - lIterTempY, 
                                pX2 + lIterTempX, pY2 - lIterTempY,
                                lTemplateRadiusSq, lTemplateStartRadiusSq, pWidth, pHeight,
                                pImage1, pImage2,
                                &lCrossCorr, &lMeanImg1, &lMeanImg2, &lVarImg1, &lVarImg2, &lNumIndices );
                    }
                    // If current x is not too small
                    if ( (pX1 >= lIterTempX) && (pX2 >= lIterTempX) )
                    {
                        // Compute cross correlation for neg.-pos. shifts
                        if ( (pY1 + lIterTempY < pHeight) && (pY2 + lIterTempY < pHeight) )
                            misc_computeCrossCorrTemplate( pX1 - lIterTempX, pY1 + lIterTempY, 
                                pX2 - lIterTempX, pY2 + lIterTempY,
                                lTemplateRadiusSq, lTemplateStartRadiusSq, pWidth, pHeight,
                                pImage1, pImage2,
                                &lCrossCorr, &lMeanImg1, &lMeanImg2, &lVarImg1, &lVarImg2, &lNumIndices );
                            
                        // Compute cross correlation for neg.-neg. shifts
                        if ( (pY1 >= lIterTempY) && (pY2 >= lIterTempY) )
                            misc_computeCrossCorrTemplate( pX1 - lIterTempX, pY1 - lIterTempY, 
                                pX2 - lIterTempX, pY2 - lIterTempY,
                                lTemplateRadiusSq, lTemplateStartRadiusSq, pWidth, pHeight,
                                pImage1, pImage2,
                                &lCrossCorr, &lMeanImg1, &lMeanImg2, &lVarImg1, &lVarImg2, &lNumIndices );
                    }
                }
            } // Stop looping over template
            
        // Acquire means
        lMeanImg1 = lMeanImg1 / ( (double)lNumIndices);
        lMeanImg2 = lMeanImg2 / ( (double)lNumIndices);
        // Acquire variances
        lVarImg1 = ( lVarImg1 - ( (double)lNumIndices ) * lMeanImg1 * lMeanImg1 ) / ( (double)lNumIndices - 1.0 );
        lVarImg2 = ( lVarImg2 - ( (double)lNumIndices ) * lMeanImg2 * lMeanImg2 ) / ( (double)lNumIndices - 1.0 );
        // Acquire cross-correlation
        lCrossCorr = lCrossCorr / ((double)lNumIndices) - lMeanImg1 * lMeanImg2;
        lCrossCorr /= sqrt( lVarImg1 * lVarImg2 );
        
        // If current cross correlation is the maximum
        if ( lCrossCorr > *pMaximum )
        {
            // Update
            *pMaximum = lCrossCorr;
            *pMaximumIndex = pX2 + pY2 * pWidth;
        }
        
        return;
    }



    
    // local cross-correlation between two images
    int misc_localMaxCrossCorr2D(
        const double * const pImage1, const double * const pImage2, 
        const unsigned int pWidth, const unsigned int pHeight,
        const unsigned int pTemplateRadius, const unsigned int pSearchRadius,
        const unsigned int pTemplateSkip, const unsigned int pSearchSkip,
        const unsigned int pTemplateStart, const unsigned int pSearchStart,
        unsigned int * const pOutput, const bool * const pEstimInd, double * const pCrossCorr
        )
    {
        // Get number of pixels
        const unsigned int lN = pWidth * pHeight;
    
        // Loop through each pixel in image 1
        #pragma omp parallel for
        for (unsigned int lIterPixel = 0; lIterPixel < lN; lIterPixel++)
        {
            // If defined which pixels to estimate
            if (pEstimInd != NULL)
                // If current pixel should not be estimated
                if ( !pEstimInd[lIterPixel] )
                    // Skip ahead
                    continue;
        
            // Get X and Y coordinates of current pixel
            const unsigned int lCurX = lIterPixel % pWidth;
            const unsigned int lCurY = lIterPixel / pWidth;
            
            // Initialize index and maximum cross-correlation
            double lMaximum = - 1.0/0.0;
            unsigned int lMaximumIndex = lN;
            const double lSearchRadiusSq = ((double)pSearchRadius) * ((double)pSearchRadius);
            const double lSearchStartRadiusSq = ((double)pSearchStart) * ((double)pSearchStart);
    
            // Loop through search region
            for (unsigned int lIterSearchX = 0 ; lIterSearchX <= pSearchRadius ; lIterSearchX += pSearchSkip + 1 )
                for (unsigned int lIterSearchY = 0 ; lIterSearchY <= pSearchRadius ; lIterSearchY += pSearchSkip + 1 )
                {
                    // Handle 0 case
                    if ( (lIterSearchX == 0) && (lIterSearchY == 0) )
                        // Compute cross correlation for no shift
                        misc_computeCrossCorrSearch( lCurX, lCurY, lCurX, lCurY,
                            pWidth, pHeight,
                            lSearchRadiusSq, lSearchStartRadiusSq,
                            pTemplateRadius, pTemplateSkip, pTemplateStart,
                            pImage1, pImage2,
                            &lMaximum, &lMaximumIndex );
                    else
                    {
                        // If current x is not too big
                        if ( lCurX + lIterSearchX < pWidth )
                        {
                            // If current y is not too big
                            if ( lCurY + lIterSearchY < pHeight )
                                // Compute cross correlation for positive shifts
                                misc_computeCrossCorrSearch( lCurX, lCurY, 
                                    lCurX + lIterSearchX, lCurY + lIterSearchY, 
                                    pWidth, pHeight,
                                    lSearchRadiusSq, lSearchStartRadiusSq,
                                    pTemplateRadius, pTemplateSkip, pTemplateStart,
                                    pImage1, pImage2,
                                    &lMaximum, &lMaximumIndex );
                                
                            // If current y is not too small
                            if ( lCurY >= lIterSearchY )
                                // Compute cross correlation for pos.-neg. shifts
                                misc_computeCrossCorrSearch( lCurX, lCurY, 
                                    lCurX + lIterSearchX, lCurY - lIterSearchY,
                                    pWidth, pHeight,
                                    lSearchRadiusSq, lSearchStartRadiusSq,
                                    pTemplateRadius, pTemplateSkip, pTemplateStart,
                                    pImage1, pImage2,
                                    &lMaximum, &lMaximumIndex );
                        }
                        // If current x is not too small
                        if ( lCurX >= lIterSearchX )
                        {
                            // Compute cross correlation for neg.-pos. shifts
                            if ( lCurY + lIterSearchY < pHeight )
                                misc_computeCrossCorrSearch( lCurX, lCurY, 
                                    lCurX - lIterSearchX, lCurY + lIterSearchY,
                                    pWidth, pHeight,
                                    lSearchRadiusSq, lSearchStartRadiusSq,
                                    pTemplateRadius, pTemplateSkip, pTemplateStart,
                                    pImage1, pImage2,
                                    &lMaximum, &lMaximumIndex );
                                
                            // Compute cross correlation for neg.-neg. shifts
                            if ( lCurY >= lIterSearchY )
                                misc_computeCrossCorrSearch( lCurX, lCurY, 
                                    lCurX - lIterSearchX, lCurY - lIterSearchY,
                                    pWidth, pHeight,
                                    lSearchRadiusSq, lSearchStartRadiusSq,
                                    pTemplateRadius, pTemplateSkip, pTemplateStart,
                                    pImage1, pImage2,
                                    &lMaximum, &lMaximumIndex );
                        }
                    }
                }   // Stop looping over search region
            
            // Insert maximum cross-correlation index for current point
            pOutput[ lCurX + lCurY * pWidth ] = lMaximumIndex;
            if ( (pCrossCorr != NULL) && (lMaximumIndex != lN) )
                pCrossCorr[ lCurX + lCurY * pWidth ] = lMaximum;
        }
        
        // Return success
        return 0;    
    }
    
        
}





