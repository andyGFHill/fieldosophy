/* 
* This file is part of Fieldosophy, a toolkit for random fields.
*
* Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
*
* This Source Code is subject to the terms of the BSD 3-Clause License.
* If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
*
*/

extern "C"
{

    // Make sure that alpha tilde is not too large
    void NIG_filterParams( double *pAlphaTilde, double *pBetaTilde);

    // Modified Bessel function of the second kind
    double Kv(const int v, const double x);
    
    // Logarithm of modified Bessel function of second kind
    double lKv(const int v, const double x);

    // logarithm of PDF for NIG distribution
    unsigned int NIG_lpdf( double * const pX, const unsigned int pN, const double alphaTilde, const double betaTilde, const double mu, const double delta );

    // Compute log-likelihood
    double NIG_logLikelihood( double * const pOut, const double * const pX, const unsigned int pN, 
        const double alphaTilde, const double betaTilde, const double mu, const double delta );

    // EM-algorithm for finding MLE of NIG
    unsigned int NIG_MLE( double * const pLogLik, double * const pX, const unsigned int pN, 
        double * const pAlphaTilde, double * const pBetaTilde, double * const pMu, double * const pDelta,
        const double pXBar, const unsigned int pM, const double pTol ); 
    
}


