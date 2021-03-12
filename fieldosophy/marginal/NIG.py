#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for normal-inverse Gaussian distributions.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import ctypes 
import numpy as np
import os
from scipy import stats as stats
from scipy import special as special
from scipy import interpolate
from scipy import optimize





c_double_p = ctypes.POINTER(ctypes.c_double)   



class NIGDistribution:
    """
    Class representing a NIG distribution.
    """

    c_double_p = ctypes.POINTER(ctypes.c_double) 
    
    _alpha = None
    _beta = None
    _mu = None
    _delta = None
    
    _log = None
    
    _libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" )
    _lib = None
        
    _stats = None
    _probs = None
    
    def __init__(self, params, libPath = None, log = False):
        
        if libPath is not None:
            self._libPath = libPath
        
        def flatter(x):
            if not (np.isscalar(x)):
                x = x.flatten()[0]
            return(x)
        
        self._alpha = np.float64(flatter(params["alpha"]))
        self._beta = np.float64(flatter(params["beta"]))
        self._mu = np.float64(flatter(params["mu"]))
        self._delta = np.float64(flatter(params["delta"]))
        
        self._log = log
        
        self._lib = ctypes.CDLL(self._libPath) 
        self._lib.NIG_lpdf.restype = ctypes.c_uint
        self._lib.NIG_lpdf.argtypes = [ self.c_double_p, ctypes.c_uint, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ]    
                
        
        
        
    
    def getDict(self):
        """
        Function for getting dictionary representation of distribution
        """
        
        return ( {"alpha":self._alpha, "beta":self._beta, "mu":self._mu, "delta":self._delta} )
        
        
    # function for copying object    
    def copy(self):
        """
        :return Deep copy os self.
        """
        
        out =  NIGDistribution( self.getDict(), libPath = self._libPath, log = self._log )
        
        out._stats = self._stats
        out._probs = self._probs
        
        return(out)
    
    
    def linearTrans( self, pMu, pDelta ):
        """
        Linear transformation
        """
        
        alphaTilde = self._alpha * self._delta
        betaTilde = self._beta * self._delta
        mu = self._mu + pMu
        delta = self._delta * pDelta
        alpha = alphaTilde / delta
        beta = betaTilde / delta
        
        out = NIGDistribution( {"alpha": alpha, "beta": beta, "delta": delta, "mu":mu}, libPath = self._libPath, log = self._log )
        return( out )
            
        
    
    def initProbs(self):
        """
        Compute CDF, PDF and Q approximations using spline interpolation
        """
        
        # Get points covering feasible region
        def findBorders(distr, sens = 1e-4, pointDens = 50.0/10.0):
            
            # Place points 
            x = distr.getStats()["mean"] + distr.getStats()["std"] * np.linspace(-10,10, num = int(np.round(pointDens * 10)) )
            y = distr.CDF( x, log = False )
        
            # Find points too far below maximum
            banPoints = (y < sens) | (y > 1-sens) | np.isnan(y)
            
            # If no banPoints at left
            while not banPoints[0]:
                # Extend to left
                xLeft = x[0] + distr.getStats()["std"] * np.linspace(-10,x[0]-x[1], num = int(np.round(pointDens * 5)) )
                yLeft = distr.CDF( xLeft, log = False )
                banPoints = np.append( (yLeft < sens) | np.isnan(yLeft), banPoints )
                x = np.append( xLeft, x )
                y = np.append( yLeft, y )
                
            # If no banPoints at right
            while not banPoints[-1]:
                # Extend to left
                xRight = x[-1] + distr.getStats()["std"] * np.linspace(x[-1]-x[-2], 10, num = int(np.round(pointDens * 5)) )
                yRight = distr.CDF( xRight, log = False )
                banPoints = np.append( banPoints, (yRight > 1-sens) | np.isnan(yRight) )
                x = np.append( x, xRight )
                y = np.append( y, yRight )
                    
            # If any points are not monotonically increasing
            banPoints[1:] = banPoints[1:] | (np.diff(y) <= 0)
                
            # Remove bad points    
            x = x[~banPoints]
            y = y[~banPoints]    
            
            return(x, y)
        
        
            
        # Adaptively add more points
        def adaptivelyRefineProb( x, y, distr, finalN = int(2e3), tol = 1e-8):
            badPoints = np.array( np.zeros(x.shape), dtype="bool" )
            badPoints[np.arange(1, len(badPoints)-1, step = 2)] = True
            
            # Interpolate current points
            tck = interpolate.splrep( x, y,  k = 3)
            tckQ = interpolate.splrep( y, x,  k = 3)
            
            TCKOld = [tck]
            TCKQOld = [tckQ]
            # Loop until no more bad points or maximum nodes have been reached
            while (len(x) < finalN) and (np.sum(badPoints) > 0):
                
                # Insert points in between bad ones
                xBadIndex = np.where(badPoints)[0]
                xNew = np.NaN * np.ones((2*np.sum(badPoints)))
                xNew[np.arange(0, len(xNew)-1, step = 2)] = ( x[xBadIndex-1] + x[xBadIndex] ) / 2.0
                xNew[np.arange(1, len(xNew), step = 2)] = ( x[xBadIndex] + x[xBadIndex+1] ) / 2.0
                yNew = distr.CDF( xNew, log = False )
                
                # Unmark if duplicate y
                duplicateIndex = np.isin( yNew, np.unique(np.concatenate( (y[xBadIndex], y[xBadIndex-1], y[xBadIndex+1])  )) )
                xNew = xNew[~duplicateIndex]
                yNew = yNew[~duplicateIndex]
                # If no new points left
                if (xNew.size == 0):
                    badPoints = np.array( np.zeros(x.shape), dtype="bool" )
                    continue
                    
                # SPLINE-interpolation
                yintrp = interpolate.splev( xNew, tck, der=0 )    
                xintrp = interpolate.splev( yNew, tckQ, der=0 )
                # Compute errors
                errors = np.abs(yNew - yintrp)
                errorsQ = np.abs(xNew - xintrp) / np.ptp(x)
                # Choose the q points with highest error        
                badPoints = (errors >= tol) | (errorsQ >= tol)
                    
                # Include the new points
                badPoints = np.append( np.array( np.zeros(x.shape), dtype="bool" ), badPoints )
                y = np.append( y, yNew )
                x = np.append(x, xNew)
                
                # Sort in increasing x
                sortInd = np.argsort( x )        
                x = x[sortInd]
                y = y[sortInd]        
                badPoints = badPoints[sortInd]   
                
                # Make sure that y is a structly monotone function
                while np.any((np.diff(y) <= 0)):                    
                    # Find too high bandit
                    bandit = np.concatenate( ( np.array([False]), (np.diff(y) > 0) ) ) & np.concatenate( ( (np.diff(y) <= 0), np.array([False]) ) )
                    # remove bandits                    
                    x = x[~bandit]
                    y = y[~bandit]        
                    badPoints = badPoints[~bandit]
                    
                    if np.any((np.diff(y) <= 0)):                    
                        # Find too low bandit
                        bandit = np.concatenate( ( np.array([False]), (np.diff(y) <= 0) ) ) & np.concatenate( ( (np.diff(y) > 0) , np.array([False]) ) )
                        # remove bandits                    
                        x = x[~bandit]
                        y = y[~bandit]        
                        badPoints = badPoints[~bandit]

                    
                    
                # Interpolate current points
                tck = interpolate.splrep( x, y,  k = 3)        
                TCKOld.append(tck)      
                tckQ = interpolate.splrep( y, x,  k = 3)        
                TCKQOld.append(tckQ)
        
            return( x,y, TCKOld, TCKQOld )
        
        def findTails( distr, pXL, pXR ):
        
            # Get range to approximate left tails on
            xL = pXL - np.linspace( 0, 1 * distr.getStats()["std"], num=200 )
            yL = distr.CDF( xL, log = False )
            if np.any(np.isnan(yL)):
                xL = xL + ( xL[ ~np.isnan(yL) ][-1] - xL[-1])                
                yL = distr.CDF( xL, log = False )
            
            # Get range to approximate right tails on
            xR = pXR + np.linspace( 0, 1 * distr.getStats()["std"], num=200 )
            yR = distr.CDF( xR, log = False )
            if np.any(np.isnan(yR)):
                xR = xR - ( xR[0] - xR[ np.isnan(yR) ][0] )
                yR = distr.CDF( xR, log = False )
            
            # Acquire linear coefficient of left tail
            xL = np.abs(xL - xL[0])
            yL = np.log( yL )
            yL = yL - yL[0] 
            slopeL = np.linalg.lstsq( xL.reshape(-1,1) , yL, rcond=None )[0][0]
            
            # Acquire linear coefficient of right tail
            xR = np.abs(xR - xR[0])
            yR = np.log( 1 - yR )
            yR = yR - yR[0]             
            slopeR = np.linalg.lstsq( xR.reshape(-1,1) , yR, rcond=None )[0][0]
                    
            # # Plot                
            # plt.clf()
            # plt.subplot(1,2,1)
            # plt.plot(xL, yL)
            # plt.plot( xL, slopeL * xL )
            # plt.subplot(1,2,2)
            # plt.plot(xR, yR)
            # plt.plot( xR, slopeR * xR )
            
            return { "slopeL":slopeL, "slopeR":slopeR }        
        
        
        # Find range of points
        xPoints, yPoints = findBorders(self, pointDens = 50.0/10, sens=1e-4)
        # Get points and spline interpolations
        xPoints, yPoints, TCKList, TCKQList = adaptivelyRefineProb( xPoints, yPoints, self, finalN = int(2e3), tol = 1e-7 )            
        # Get polynomials from splines
        polysCDF = interpolate.PPoly.from_spline(TCKList[-1], extrapolate = False)                                
        # compute limiting slopes
        slopes = findTails( self, xPoints[0], xPoints[-1] )
        
        # Interpolate quantile function
        polysQ = interpolate.PPoly.from_spline(TCKQList[-1], extrapolate = False)
         
        # Save parameters
        self._probs = { "x": xPoints, "y": yPoints, "polysCDF": polysCDF, "polysQ": polysQ, \
           "slopes": slopes, "TCKList": TCKList, "TCKQList": TCKQList }
                
        return        
    
    
    def getStats(self):
        """
        Get statistics of probability distribution.
        """
        
        if self._stats is None:        
            self._stats = self.computeStatistics()               
        
        return( self._stats.copy() )
    
    def getProbs(self):
        return( self._probs.copy() )
    
    

    
    def sample( self, size, log = None ):
        """
        Sample from NIG distribution.
        
        :param size: The sample size.
        :param log: Flag seting if the sample should be logarithmized.
        
        :return A sample from the distribution.
        """
        
        if log is None:
            log = self._log
        
        alpha = self._alpha
        beta = self._beta
        mu = self._mu
        delta = self._delta
        
        # Check if normal or NIG
        if (alpha < np.inf):
            out = stats.norminvgauss.rvs( a = alpha * delta, b = beta * delta, loc = mu, scale = delta, size = size )
        else:
            out = stats.norm.rvs( size = size, loc = mu, scale = delta )
            
        # If data is NIG after logarithmization
        if (log):
            out = np.exp(out)
            
        return(out)
    
    
    
    def limPDF( self, data, log = None ):
        """
        Limiting PDF
        """
        
        if log is None:
            log = self._log
        
        # Standardize
        x = (data - self._mu) / self._delta
        # Compute normalizing constant
        const =  np.sqrt( self._alpha / (2*np.pi) ) * np.exp( np.sqrt(self._alpha ** 2 - self._beta ** 2) )
        # Compute standardized pdf
        y = np.exp( self._beta * x - self._alpha * np.abs(x) ) / (np.abs(x) ** (3/2))
        y = const * y
        # Compensate for standardization
        y = y / self._delta
        
        # If data is NIG after logarithmization
        if (log):
            y = y / data
                
        return(y)
                
    

    
    def PDF( self, data, log = None ):    
        """
        PDF function of NIG
        """
        
        if log is None:
            log = self._log
            
        return( np.exp(self.lPDF(data, log = log)) )    
    
    
    def lPDF( self, data, log = None ):
        """
        log-PDF function of NIG
        """
        
        if log is None:
            log = self._log

        # Get parameters
        alpha = self._alpha
        beta = self._beta
        mu = self._mu
        delta = self._delta
        
        # copy data to output
        out = np.array(data.copy(), dtype="float64")

        # If data is NIG after logarithmization
        if (log):
            out = np.log(out)
        
        # Check if normal or NIG
        if (alpha < np.inf):            
            
            # Call log pdf            
            out_p = out.ctypes.data_as(self.c_double_p)
            self._lib.NIG_lpdf( out_p, ctypes.c_uint(len(data)), \
                ctypes.c_double(alpha * delta), \
                ctypes.c_double(beta * delta), \
                ctypes.c_double(mu), \
                ctypes.c_double(delta), \
                    )    
                
            # out2 = stats.norminvgauss.logpdf( data, \
                    #       a = alpha * delta , \
                    #       b = beta * delta, \
                    #       loc = mu, \
                    #       scale = delta \
                    #       )                    
                              
        else:
            out = stats.norm.logpdf( out, loc = mu, scale = delta )
            
            
        # If data is NIG after logarithmization
        if (log):
            out = out - np.array(np.log(data), dtype="float64")
            
#        assert(np.all(~np.isnan(out)))                        
        return(out)
        
        


    
    def CDF( self, data, log = None ):
        """
        CDF function of NIG.
        """
        
        if log is None:
            log = self._log
        
        # copy data to output
        out = np.array(data.copy(), dtype="float64")
        
        # If data is NIG after logarithmization
        if (log):
            out = np.log(out)
        
        # Check if normal or NIG
        if (self._alpha < np.inf):
            out = stats.norminvgauss.cdf( out, \
                       a = self._alpha * self._delta, \
                       b = self._beta * self._delta, \
                       loc = self._mu, \
                       scale = self._delta \
                       )            
        
        else:
            out = stats.norm.cdf( out, loc = self._mu, scale = self._delta )
        
        out[out > 1.0] = 1.0
        out[out < 0.0] = 0.0
        
        return( out )
    
    
    
    def approximateCDF( self, x, log = None ):
        """
        Approximate CDF function
        """
        
        if log is None:
            log = self._log
        
        # If approximate probabilities has not been initialized
        if self._probs is None:
            self.initProbs()
            
        # If data is NIG after logarithmization
        if (log):
            x = np.log(x.copy())
            
        def extrapolateCDFLeft( probs, x ):
            x = (x - self._probs["x"][0]).copy()
            badInds = x > 0
            y = - x * self._probs["slopes"]["slopeL"]
            y[~badInds] = y[~badInds] + np.log(self._probs["y"][0])
            y[~badInds] = np.exp(y[~badInds])
            y[badInds] = np.NaN
            
            return( y )    
        
        def extrapolateCDFRight( probs, x ):
            x = (x - probs["x"][-1]).copy()
            badInds = x < 0
            y = x * probs["slopes"]["slopeR"]
            y[~badInds] = y[~badInds] + np.log(1-probs["y"][-1])
            y[~badInds] = np.exp(y[~badInds])
            y[badInds] = np.NaN
            y = (1 - y) 
            
            return( y )            
        
        # Interpolate in middle region
        y = self._probs["polysCDF"](x)
        # Extrapolate in left tail
        y[ x < self._probs["x"][0] ] = extrapolateCDFLeft( self._probs, x[ x < self._probs["x"][0] ] )
        # Extrapolate in right tail
        y[ x > self._probs["x"][-1] ] = extrapolateCDFRight( self._probs, x[ x > self._probs["x"][-1] ] )
    
        return( y )
    
    def approximateQ( self, y, log = None ):
        """
        Approximate quantile function
        """
        
        if log is None:
            log = self._log
            
        # If approximate probabilities has not been initialized
        if self._probs is None:
            self.initProbs()
            
        def extrapolateQLeft( probs, y ):        
            x = np.log( y ) - np.log( self._probs["y"][0] )
            badInds = x > 0
            x[~badInds] = - x[~badInds] / self._probs["slopes"]["slopeL"]
            x[~badInds] = x[~badInds] + self._probs["x"][0]
            x[badInds] = np.NaN        
            
            return( x )
        
        def extrapolateQRight( probs, y ):                                
            x = 1 - y
            x = np.log( x ) - np.log( 1 - probs["y"][-1] )        
            badInds = x > 0
            x[~badInds] = x[~badInds] / probs["slopes"]["slopeR"]
            x[~badInds] = x[~badInds] + probs["x"][-1]
            x[badInds] = np.NaN        
            
            return( x )              
        
        # Interpolate in middle region
        x = self._probs["polysQ"](y)
        # Extrapolate in left tail
        x[ y < self._probs["y"][0] ] = extrapolateQLeft( self._probs, y[ y < self._probs["y"][0] ] )
        # Extrapolate in right tail
        x[ y > self._probs["y"][-1] ] = extrapolateQRight( self._probs, y[ y > self._probs["y"][-1] ] )
        
        # If data is NIG after logarithmization
        if (log):
            x = np.exp(x)
    
        return( x )    
    
        

    
    def Q( self, data, log = None ):
        """
        Quantile function of NIG
        """
        
        if log is None:
            log = self._log
        
        # copy data to output
        out = np.array(data.copy(), dtype="float64")        
        
        # Check if normal or NIG
        if (self._alpha < np.inf):
            out = stats.norminvgauss.ppf( out, \
                   a = self._alpha * self._delta , \
                   b = self._beta * self._delta, \
                   loc = self._mu, \
                   scale = self._delta \
                   )                        
        
        else:
            out = stats.norm.ppf( out, loc = self._mu, scale = self._delta )            
        
        # If data is NIG after logarithmization
        if (log):
            out = np.exp(out)
        
        return( out )



    
    def NIG2Gauss( self, data, log = None ):
        """
        Transform data from NIG distribution to Gaussian
        """
        
        if log is None:
            log = self._log
        
        # copy data to output
        out = np.array(data.copy(), dtype="float64")        
        
        # Check if normal or NIG
        if (self._alpha < np.inf):        
            # Transform to uniform
            out = self.approximateCDF( out, log = log )
            # Transform uniform to norm
            out = stats.norm.ppf( out )        
        
        else:
            # If data is NIG after logarithmization
            if (log):
                out = np.log(out)
            out = (out - self._mu) / self._delta
        
        return(out)

    
    
    
    def Gauss2NIG( self, data, log = None ):
        """
        Transform data from Gaussian distribution to NIG
        """
        
        if log is None:
            log = self._log
        
        # copy data to output
        out = np.array(data.copy(), dtype="float64")                
        
        # Check if normal or NIG
        if (self._alpha < np.inf):
            # Transform to uniform
            out = stats.norm.cdf( out )
            # Transform uniform to NIG
            out = self.approximateQ( out, log = log )
        
        else:
            out = out * self._delta + self._mu
            # If data is NIG after logarithmization
            if (log):
                out = np.exp(out)
        
        
        return(out)
                
            
        
    

    # compute statistics
    def computeStatistics( self ):        
        
        # Check if normal or NIG
        if (self._alpha < np.inf):
            gamma = np.sqrt(self._alpha ** 2 - self._beta ** 2)
            
            mean = self._mu + self._delta * self._beta / gamma
            var = self._delta * (self._alpha ** 2) / (gamma ** 3)
            skew = 3 * self._beta / ( self._alpha * np.sqrt( self._delta * gamma ) )
            kurt = 3.0 * ( 1.0 + 4.0 * ( self._beta ** 2) / ( self._alpha ** 2) ) / ( self._delta * gamma )            
            
            return( {"mean" : mean, "variance" : var, "std":np.sqrt(var), "skewness" : skew, "kurtosis" : kurt} )
        
        else:            
            return( {"mean" : self._mu, "variance" : self._delta ** 2, "std":np.sqrt(self._delta), "skewness" : 0, "kurtosis" : 0} )
    
    

    
    
    
    
    
class NIGEstimation:
    """
    Class for estimating NIG distributions from data in different ways.
    """
    
    c_double_p = ctypes.POINTER(ctypes.c_double) 
    
        
        
    
    def EMMLE( x, init = None, maxIter = 100, tol = 0, libPath = os.path.join( os.path.dirname( __file__), "../libraries/libSPDEC.so" ) ):
        """
        Compute maximum likelihood estimates of the NIG parameters    
        
        Algorithm acquired from Karlis, D. (2002), "An EM type algorithm for maximum likelihood estimation of the normal-inverse Gaussian distribution"
        """
        
    
        # If init is not supplied    
        if init is None:
            init = NIGEstimation.MOM( x )
            init = [init, NIGEstimation.Gauss( x ) ]
        
        # Mkae sure that init is a list
        if not isinstance( init, list):
            init = [init]  
            
        # Make sure x has right dimensions
        assert(len(x.shape) == 2)
        d = x.shape[1]
        
        # Load C library
        lib = ctypes.CDLL(libPath) 
        c_double_p = ctypes.POINTER(ctypes.c_double) 
        lib.NIG_MLE.restype = ctypes.c_uint
        lib.NIG_MLE.argtypes = [ c_double_p, c_double_p, ctypes.c_uint, \
           c_double_p, c_double_p, c_double_p, c_double_p, \
           ctypes.c_double, ctypes.c_uint, ctypes.c_double]    
            
            
        # Preallocate log-lik values
        llVals = -np.Inf * np.ones( (d, len(init)+1) )
        # Preallocate parameteter values
        params = [None] * d
                    
        # Loop through each variable
        for iterVar in np.arange(0,d):
    
            # Preallocate paramter values        
            params[iterVar] = [None] * (len(init)+1)
        
            # Remove all nans for current variable
            data = np.asfortranarray( x[ ~np.any(np.isnan(x), axis = 1), iterVar ], dtype = "float64")
        
            # Init    
            xBar = np.mean(data) 
            s2 = np.var(data) 
            n = len(data)
            
            # Loop through each initiation
            for iterInit in np.arange(0, len(init)):
    
                # Init
                delta = ctypes.c_double(init[iterInit]["delta"].flatten()[iterVar])
                alphaTilde = ctypes.c_double(init[iterInit]["alpha"].flatten()[iterVar] * delta)
                betaTilde = ctypes.c_double(init[iterInit]["beta"].flatten()[iterVar] * delta)
                mu = ctypes.c_double(init[iterInit]["mu"].flatten()[iterVar])
                loglik = ctypes.c_double(-np.inf)
            
                # Call EM-algorithm             
                status = lib.NIG_MLE( ctypes.byref(loglik), \
                    data.ctypes.data_as(c_double_p), ctypes.c_uint(n), \
                    ctypes.byref(alphaTilde), ctypes.byref(betaTilde), \
                    ctypes.byref(mu), ctypes.byref(delta), \
                    ctypes.c_double(xBar), ctypes.c_uint(maxIter), ctypes.c_double(tol) )                             
    
                if (status == 0) or (status > 1):
                    # Store loglikelihood
                    llVals[iterVar, iterInit] = np.float64(loglik) / n
                    # Store parameters                
                    params[iterVar][iterInit] = { \
                     "delta" : np.float64(delta), "mu" : np.float64(mu), \
                     "alpha" : np.float64(alphaTilde) / np.float64(delta), "beta" : np.float64(betaTilde) / np.float64(delta) \
                     }
                   
                    
            # Check if normal distribution is better        
            # Store loglikelihood
            llVals[iterVar, len(init)] = np.sum( stats.norm.logpdf( data, loc = xBar, scale = np.sqrt(s2) ) ) / n
            # Store parameters                
            params[iterVar][len(init)] = { \
                 "delta" : np.float64(np.sqrt(s2)), "mu" : np.float64(xBar), \
                 "alpha" : np.float64(np.inf), "beta" : np.float64(0) \
                 }
                     
        # Get the best initial value for all dimensions
        bestInit = np.argmax( llVals, axis = 1 )
        
        # Merge the best ones
        out = [None] * d
        for iterVar in np.arange(0,d):
            out[iterVar] = { \
                "alpha" : params[iterVar][bestInit[iterVar]]["alpha"], \
                "beta" : params[iterVar][bestInit[iterVar]]["beta"], \
                "mu" : params[iterVar][bestInit[iterVar]]["mu"], \
                "delta" : params[iterVar][bestInit[iterVar]]["delta"] \
                }    
             
        # Return best estimates
        return out
                
                
                
                
                


    def gradientMLE( x, init = None, maxIter = 100, tol = 0 ):
        """
        Estimate a NIG distribution from data using gradient based optimization on maximum likelihood function.
        """
        
        # If init is not supplied    
        if init is None:
            init = NIGEstimation.MOM( x )
            init = [init, NIGEstimation.Gauss( x ) ]
        
        # Mkae sure that init is a list
        if not isinstance( init, list):
            init = [init]  
            
        # Make sure x has right dimensions
        assert(len(x.shape) == 2)
        d = x.shape[1]            
            
        # Preallocate log-lik values
        llVals = -np.Inf * np.ones( (d, len(init)+1) )
        # Preallocate parameteter values
        params = [None] * d
                
        
        def optimTrans( x, scaler = 1 ):
            # Function transforming from unconstained to real value            
            
            x = x.copy()
            
            # Delta
            delta = np.exp( x[3] )
            # Alpha
            alpha = np.exp( x[0] ) / delta
            # Beta
            beta = ( stats.logistic.cdf(x[1]) * 2 - 1 ) * alpha
            # mu
            mu = x[2]
            
            x[0] = alpha / scaler
            x[1] = beta / scaler
            x[2] = mu * scaler
            x[3] = delta * scaler
            
            return x
        
        def optimTransInv( x ):
            # Function transforming from real values to uncosntrained
            
            x = x.copy()
            
            alpha = x[0]
            beta = x[1]
            mu = x[2]
            delta = x[3]
            
            # Alpha to log alphatilde
            x[0] = np.log( alpha * delta )
            # Beta to logistic beta/alpha
            x[1] = stats.logistic.ppf( (beta / alpha + 1)/2 )
            # mu to mu
            x[2] = mu
            # Delta
            x[3] = np.log(delta)
            
            return x
                        
                    
        
        # Loop through each variable
        for iterVar in np.arange(0,d):
    
            # Preallocate paramter values        
            params[iterVar] = [None] * (len(init)+1)
        
            # Remove all nans for current variable
            data = np.asfortranarray( x[ ~np.any(np.isnan(x), axis = 1), iterVar ], dtype = "float64")
        
            # Loop through each initiation
            for iterInit in np.arange(0, len(init)):
    
                # Init
                delta = init[iterInit]["delta"].flatten()[iterVar]
                alpha = init[iterInit]["alpha"].flatten()[iterVar]
                beta = init[iterInit]["beta"].flatten()[iterVar]
                mu = init[iterInit]["mu"].flatten()[iterVar]
#                filtered = NIGEstimation.filterParams( np.array([alpha]), np.array([beta]), np.array([delta]) )
#                alpha = filtered["alpha"][0]
#                beta = filtered["beta"][0]
#                delta = filtered["delta"][0]
                
                
                def optimFunc( x ):
                    # Cost function for optimization
                    
                    # Transform from unconstrained to constrained value
                    x = optimTrans( x )            
                    
                    logLik = -np.inf
                    
                    if (x[0]*x[3] < 2e2):          
                        # Compute log-lik
                        distr = NIGDistribution( {"alpha":x[0], "beta":x[1], "mu":x[2], "delta":x[3]} )
                        logLik = np.sum( distr.lPDF( data ) / data.size )
                    # Return minus log-likelihood
                    return - logLik
                
                
                # Set initial value 
                x0 = [ optimTransInv( np.array([alpha, beta, mu, delta]) ) ]
                # Optimize ("BFGS")
                resultOptim = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': False, "maxiter":maxIter, "gtol": tol} )
#                resultOptim = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': False, "maxiter":maxIter} )
                # Get result
                xEst = optimTrans( resultOptim.x )
                loglik = resultOptim.fun
                if np.isnan(loglik) or np.isinf(loglik):
                    xEst = optimTrans(x0[0])
                    loglik = optimFunc(x0[0])
                
                
                # Store loglikelihood
                llVals[iterVar, iterInit] = np.float64(loglik)
                # Store parameters                
                params[iterVar][iterInit] = { \
                 "delta" : np.float64(xEst[3]), "mu" : np.float64(xEst[2]), \
                 "alpha" : np.float64(xEst[0]) , "beta" : np.float64(xEst[1])  \
                 }
                   
                     
        # Get the best initial value for all dimensions
        bestInit = np.argmax( llVals, axis = 1 )
        
        # Merge the best ones
        out = [None] * d
        for iterVar in np.arange(0,d):
            out[iterVar] = { \
                "alpha" : params[iterVar][bestInit[iterVar]]["alpha"], \
                "beta" : params[iterVar][bestInit[iterVar]]["beta"], \
                "mu" : params[iterVar][bestInit[iterVar]]["mu"], \
                "delta" : params[iterVar][bestInit[iterVar]]["delta"] \
                }    
             
        # Return best estimates
        return out
    
    
    
    
    def EMMLEOld( data, alpha, beta, mu, delta, maxIter, tol, libPath ):
    
        xBar = np.mean(data)
        n = len(data)
                        
        # compute initial log-lik
        distrOld = NIGDistribution({"delta":delta,"mu":mu,"alpha":alpha, "beta":beta}, libPath = libPath)
        distrNew = distrOld.copy()
        loglikOld = np.sum( distrOld.lPDF(data) ) / n
    
        # Iterate in EM-algorithm
        for iterEM in np.arange(0, maxIter):
        
            # --- E-step ---
            
            phiSqrt = np.sqrt( 1 + ( (data-distrNew.mu)/distrNew.delta ) ** 2 )    
            # compute besselk function values
            k0 = special.kn( 0, distrNew.delta * distrNew.alpha * phiSqrt )
            k1 = special.kn( 1, distrNew.delta * distrNew.alpha * phiSqrt )
            k2 = special.kn( 2, distrNew.delta * distrNew.alpha * phiSqrt )
            # Compute pseudo values
            s = ( distrNew.delta * phiSqrt * k0 ) / ( distrNew.alpha * k1 )
            w = ( distrNew.alpha * k2 ) / ( distrNew.delta * phiSqrt * k1 )    
        
            
            # --- M-step ---
        
            # Compute pseudo values
            sBar = np.mean(s, axis=0)
            wBar = np.mean(w, axis=0)
            Lambda = 1.0 / np.mean( w - 1.0 / sBar, axis=0 )
                    
            # Update parameters
            distrNew.delta = np.sqrt(Lambda)
            gamma = distrNew.delta / sBar
            distrNew.beta = ( np.sum( data * w, axis=0 ) - xBar * wBar * n ) / ( n - sBar * wBar * n )            
            distrNew.alpha = np.sqrt( (gamma ** 2) + (distrNew.beta ** 2) )
            
            # Filter too large alphas
            filtered = NIGEstimation.filterParams(np.array([distrNew.alpha]), np.array([distrNew.beta]), np.array([distrNew.delta]))
            distrNew.alpha = filtered["alpha"][0]
            distrNew.beta = filtered["beta"][0]
            distrNew.delta = filtered["delta"][0]
            gamma = filtered["gamma"][0] 
            distrNew.mu = xBar - distrNew.beta * distrNew.delta / gamma
                                        
            # --- Compute log-lik ---                            
            loglikNew = np.sum( distrNew.lPDF(data) ) / n       
            loglikdiff = loglikNew - loglikOld
            
            # Set new log-likelihood as old one
            if (loglikdiff > 0):
                loglikOld = loglikNew
                distrOld = distrNew.copy()
                
                # If should break due to tolerance
                if ( loglikdiff < tol):
                    break   
            
        return ( distrNew, loglikOld)
    

    
    
    
    # Filter params
    def filterParams(a, b, d):    
        
        # Handle too large alphas
        badInds = (a * d > 1e2)     
        
        multiplier = (1e2) / (a[badInds] * d[badInds])
        
        d[badInds] = d[badInds] * multiplier
        b[badInds] = b[badInds] * multiplier
        a[badInds] = a[badInds] * multiplier
        
        gamma = np.sqrt(a**2 - b**2)            
        
        return( {"alpha" : a, "beta" : b, "delta" : d, "gamma" : gamma} )  
    
    
    
    # Find the best Gaussian fit     
    def Gauss( x ):
        
        mu = np.nanmean( x, axis = 0 ).reshape(1,-1)
        s2 = np.nanvar( x, axis = 0 ).reshape(1,-1)
        
        alphaTilde = np.float64(100)
        delta = np.sqrt(s2 * alphaTilde)
        alpha = alphaTilde / delta
        
        return( {"alpha" : alpha * np.ones( mu.shape, dtype="float64" ), "beta" : np.zeros( mu.shape ), "delta" : delta, "mu" : mu} )
    
    
    
    
    '''    
    ' Compute method of moments estimates of the NIG parameters    
    '
    ' Algorithm acquired from Karlis, D. (2002), "An EM type algorithm for maximum likelihood estimation of the normal-inverse Gaussian distribution"
    ''' 
    def MOM( x ):
        """
        Compute method of moments estimates of the NIG parameters    

        Algorithm acquired from Karlis, D. (2002), "An EM type algorithm for maximum likelihood estimation of the normal-inverse Gaussian distribution"
        """
        
        # Preallocate
        mu = np.zeros(x.shape[1], dtype = "float64")
        gamma = np.zeros(x.shape[1], dtype = "float64")
        delta = np.ones(x.shape[1], dtype = "float64")
        beta = np.zeros(x.shape[1], dtype = "float64")
        
        # Estimate moments
        xBar = np.nanmean( x, axis=0 )
        s2 = np.nanvar( x, axis=0 )
        s = np.sqrt(s2)
        mu3 = np.nanmean( (x-xBar)**3, axis=0 )
        mu4 = np.nanmean( (x-xBar)**4, axis=0 )
        # Compute intermediary    
        gamma1 = mu3 / (s ** 3)
        gamma2 = mu4 / (s2 ** 2) - 3
        gamma2[ gamma2 <= 0 ] = 0.1
        # Find bad estimates
        badInds = 3*gamma2 < 5*(gamma1 ** 2)        
        gamma2[badInds] = 6.0/3.0 * (gamma1[badInds]**2)
        
        # Estimate parameters                
        gamma = 3 / ( s * np.sqrt( 3 * gamma2 - 5 * (gamma1 ** 2) ) )
        beta = ( gamma1 * s * (gamma ** 2) ) / 3
        beta[badInds] = 0.0
        alpha = np.sqrt( gamma ** 2 + beta ** 2 )
        delta = ( (s ** 2) * (gamma ** 3) ) / (alpha**2)
        
        # Filter too large alphas
        filtered = NIGEstimation.filterParams(alpha, beta, delta)
        alpha = filtered["alpha"]
        beta = filtered["beta"]
        delta = filtered["delta"]
        gamma = filtered["gamma"]            
        mu = xBar - beta * delta / gamma
            
        
        return( {"alpha" : alpha.reshape(1, -1), "beta" : beta.reshape(1, -1), "delta" : delta.reshape(1, -1), "mu" : mu.reshape(1, -1)} )
    
    