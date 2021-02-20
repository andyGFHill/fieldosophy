/* 
* C functions for the NIG marginal distribution.
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


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>



extern "C"
{
    
    // Make sure that alpha tilde is not too large
    void NIG_filterParams( double *pAlphaTilde, double *pBetaTilde)
    {        
        if (*pAlphaTilde > 100.0d)
        {
            *pBetaTilde = *pBetaTilde * ( 100.0d / (*pAlphaTilde) );
            *pAlphaTilde = 100.0d;            
        }
    
        return;    
    }
      
    


    // Modified Bessel function of the second kind
    double Kv(const int v, const double x)
    {
        double lOut = 0.0d;
        int status;
        
        gsl_sf_result result;
    
        switch(v)
        {
        case 0:
            status = gsl_sf_bessel_K0_e(x, &result);
            break;
        case 1:
            status = gsl_sf_bessel_K1_e(x, &result);
            break;
        default:
            status = gsl_sf_bessel_Kn_e(v, x, &result);
            break;                
        }
    
        // If some kind of error
        if (status != 0)
        {
            //printf("Error code: %d in Kv", status);
            lOut = GSL_NAN;
        }
        else
        {
            lOut = result.val;
        }
            
    
        return lOut;
    }
    
    // Logarithm of modified Bessel function of second kind
    double lKv(const int v, const double x)
    {
        double lOut = Kv( v, x );      
        // Logarithmize
        lOut = log( lOut );
    
        return lOut;
    }    



    // logarithm of PDF for NIG distribution
    unsigned int NIG_lpdf( double * const pX, const unsigned int pN, const double alphaTilde, const double betaTilde, const double mu, const double delta ) 
    {
        // turn off gsl error handler
        gsl_set_error_handler_off();
        
        // Loop through each element in vector   
        #pragma omp parallel for
        for ( unsigned int iter = 0; iter < pN; iter++)
        {
            // Get normalized data value
            const double lCur = (pX[iter] - mu) / delta;
            // Compute q function
            const double lQ = sqrt( (1.0d + lCur * lCur) );
            // Compute bessel function
            const double lLogK = lKv(1, alphaTilde * lQ);
              
            // compute log pdf  
            double lCurOut = lLogK;
            lCurOut -= log(lQ);
            lCurOut += betaTilde * lCur;
            // Add constants
            lCurOut += log( alphaTilde / M_PI ) + sqrt( alphaTilde * alphaTilde - betaTilde * betaTilde );
            lCurOut -= log(delta);
            
            // Store full log pdf
            pX[iter] = lCurOut;
        }
        
        return pN;
    }
    
    
    // Compute log-likelihood
    double NIG_logLikelihood( double * lOut, const double * const pX, const unsigned int pN, const double alphaTilde, const double betaTilde, const double mu, const double delta )
    {        
        // Copy content 
        memcpy( lOut, pX, sizeof(double) * pN );
        
        // Compute log-pdf
        NIG_lpdf( lOut, pN, alphaTilde, betaTilde, mu, delta );
        // Sum 
        double lSum = 0.0d;
        for (unsigned int iter = 0; iter < pN; iter++)
        {
            lSum += lOut[iter];
        }        
        
        return lSum;
    }
    
    
    
    // EM-algorithm for finding MLE of NIG
    unsigned int NIG_MLE( double * const pLogLik, double * const pX, const unsigned int pN, 
        double * const pAlphaTilde, double * const pBetaTilde, double * const pMu, double * const pDelta,
        const double pXBar, const unsigned int pM, const double pTol ) 
    {
    
        // turn off gsl error handler
        gsl_set_error_handler_off();
    
        // Init
        double lAlphaTilde = *pAlphaTilde;
        double lBetaTilde = *pBetaTilde;
        double lMu = *pMu;
        double lDelta = *pDelta;
        // Filter tilde parameters
        NIG_filterParams( &lAlphaTilde, &lBetaTilde);
                
        // Preallocate arrays
        double * lTempArray1 = (double*) malloc(sizeof(double) * pN);
        double * lTempArray2 = (double*) malloc(sizeof(double) * pN);
        if (lTempArray1 == NULL || lTempArray2 == NULL)
            return 1;
        
        // Compute log-likelihood        
        *pLogLik = NIG_logLikelihood( lTempArray1, pX, pN, lAlphaTilde, lBetaTilde, lMu, lDelta );
        
        // If likelihood is bad
        if (isnan(*pLogLik))
            return 1;
                
    
        // Iterate EM-algorithm
        for (unsigned int iter = 0; iter < pM; iter++)
        {
            // --- E-step ---
            
            double lSBar = 0.0d;
            double lWBar = 0.0d;    
            #pragma omp parallel for reduction(+:lSBar) reduction(+:lWBar)
            for (unsigned int iter2 = 0; iter2 < pN; iter2++)
            {    
                const double lPhiSqrt = sqrt( 1.0d + ( (pX[iter2]-lMu)/lDelta ) * ( (pX[iter2]-lMu)/lDelta )  );    
                // compute besselk function values
                const double lK0 = Kv( 0, lAlphaTilde * lPhiSqrt );
                const double lK1 = Kv( 1, lAlphaTilde * lPhiSqrt );
                const double lK2 = Kv( 2, lAlphaTilde * lPhiSqrt );
                // Compute pseudo values
                lTempArray1[iter2] = ( lDelta * lPhiSqrt * lK0 ) / ( (lAlphaTilde * lK1) / lDelta ); // S
                lTempArray2[iter2] = ( (lAlphaTilde * lK2) / lDelta ) / ( lDelta * lPhiSqrt * lK1 ); // W
                lSBar += lTempArray1[iter2];
                lWBar += lTempArray2[iter2];
            }
            lSBar /= ((double)pN);
            lWBar /= ((double)pN);
            
                
            // --- M-step ---     
                   
                                        
            // Compute delta            
            lDelta = sqrt( 1.0d / ( lWBar - (1.0d / lSBar) ) );
            double lGamma = lDelta / lSBar;
            // Compute beta
            double lBeta = 0.0d;
            for (unsigned int iter2 = 0; iter2 < pN; iter2++)
            {
                lBeta += pX[iter2] * lTempArray2[iter2];
            }
            lBeta = lBeta - pXBar * lWBar * ((double)pN);
            lBeta /= ( (double)pN - lSBar * lWBar * ((double)pN) );
            lBetaTilde = lBeta * lDelta;
            // Compute alpha            
            double lAlpha = sqrt( lGamma * lGamma + lBeta * lBeta );            
            lAlphaTilde = lAlpha * lDelta;
            // Filter tilde values
            NIG_filterParams( &lAlphaTilde, &lBetaTilde);
            lGamma = sqrt( lAlphaTilde * lAlphaTilde - lBetaTilde * lBetaTilde ) / lDelta;
            // Compute mu
            lMu = pXBar - lBetaTilde / lGamma;
                        
            // Compute loglik
            const double llLikNew = NIG_logLikelihood( lTempArray1, pX, pN, lAlphaTilde, lBetaTilde, lMu, lDelta );            
                
            // If likelihood is bad
            if (isnan(llLikNew))
                return(iter+2);    // Mark that likelihood went bad during iterations
                
            // Compute likelihood difference
            const double lLLikDiff = llLikNew - *pLogLik;
                        
            // If new likelihood is higher
            if (lLLikDiff > 0) 
            {
                // Set new log-likelihood as old one
                *pLogLik = llLikNew;
                // Update output parameters
                *pAlphaTilde = lAlphaTilde;
                *pBetaTilde = lBetaTilde;
                *pMu = lMu;
                *pDelta = lDelta;
                
                // If should break due to tolerance
                if ( lLLikDiff < pTol )
                    break;                         
            }
            
        }   // end of EM-iterations
                    
        // Free allocated arrays
        free(lTempArray1);
        free(lTempArray2);
            
         // Return no problem   
         return( 0 );       
    }    
    
    
    

}


