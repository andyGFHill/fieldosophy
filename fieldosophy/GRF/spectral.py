#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for spectral approximations of Gaussian random fields.

This file is part of Fieldosophy, a toolkit for random fields.
Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>
This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
from numpy import fft

    


class Fourier:
    """
        Class for spectral representation of Gaussian random field on R^d
    """
    
    
    
    shape = None
    region = None
    
    mu = None
    gamma = None
    
    def __init__(self, shape, region, mu = None):
        
        self.shape = shape
        self.region = region
        self.mu = 0
       

    def getOmega(self):
        
        omega = np.zeros( (np.prod(self.shape), len(self.shape)) )
        for iter in range(self.shape.size):
            omega[:, iter] = np.repeat( \
                  np.tile( \
                    np.concatenate( (np.arange(np.floor(self.shape[iter]/2+1)), -np.arange(np.ceil(self.shape[iter]/2))[-1:0:-1]) ), \
                    reps= np.prod(self.shape[:iter]) ), \
                  repeats = np.prod(self.shape[(iter+1):]) ) * 2 * np.pi
                
        return omega


    def generate( self, size, indices = None ):
        
        realization = np.random.normal( size = np.prod(self.shape)*size ) + np.random.normal( size = np.prod(self.shape)*size ) * 1j
        realization = realization.reshape( np.append(self.shape, size) ) * np.sqrt(self.gamma).reshape( np.append( self.shape, 1 ) ) * np.sqrt(np.prod(self.shape))
        
        realization = np.real(fft.ifft2( realization, axes = range(self.shape.size) ))
        realization = realization + self.mu
        
        if indices is not None:
        
            realization = realization.reshape( np.append( np.prod(self.shape), -1 ) )
            realization = realization[indices, :]
        
        return realization
    
    
    def multiplyCov( self, data, power = 1, input_indices = None, output_indices = None ):
        
        d = len(self.shape)
        
        output = None
        
        # If input indices are used
        if input_indices is not None:
            
            if len(data.shape) == 1:
                output = np.zeros( (np.prod(self.shape), 1) )
            else:
                output = np.zeros( (np.prod(self.shape), data.shape[1]) )
            output[ input_indices, : ] = data.reshape((data.shape[0], -1))
            output = output.reshape( np.append( self.shape, -1) )
            
        else:
        
            if len(data.shape) < d:
                raise Exception("The shape of data does not match random field!")
            if ( np.all( data.shape[0:d] != self.shape ) ):
                raise Exception("The shape of data does not match random field!")
            output = data.reshape( np.append( self.shape, -1 ) ).copy()
                
        
        gammaShape = np.append( self.shape, -1)
        
        output = fft.fft2( output, axes = np.arange(self.shape.size) ) / np.sqrt(np.prod(self.shape))
        output = output * self.gamma.reshape(gammaShape) ** power
        output = np.real(fft.ifft2( output, axes = np.arange(self.shape.size) ) * np.sqrt(np.prod(self.shape)))
        
        # Handle if output should be on a subset of all pixels
        if output_indices is not None:
            
            if input_indices is None:
                output = output.reshape( np.concatenate( (np.prod(self.shape), data.shape[len(self.shape):] ) ) )
            else:
                output = output.reshape( np.append( np.prod(self.shape), -1 ) )
                
            output = output[output_indices, :]
            
        
        return output
    
    
    def logDetCov( self, power = 1.0 ):
        
        output = power * np.sum(np.log(self.gamma))
        
        return output
    
    
    
    def logLik(self, data ):
        
        d = len(self.shape)
            
        n = np.prod(data.shape[d:])
        
        centeredData = data - self.mu
        
        output = - 0.5*self.logDetCov() - d*0.5*np.log( 2 * np.pi )
        output = output - 0.5 * np.sum( centeredData * self.multiplyCov( centeredData, power = - 1.0) ) / n
        
        return output
    
    
    
    def anisotropicMatern( self, nu, G ):
        
        if G is None:
            self.G = np.eye(self.shape.size)
        else:
            self.G = G
                
        alpha = nu + self.shape.size / 2
        omega = self.getOmega()
        
        gamma = (1 + np.sum( omega * np.matmul(omega, G) / np.abs(np.diff(self.region, axis = 1)).reshape((-1,self.shape.size))**2  , axis=1) )**(-alpha)
        gamma = gamma / np.mean(gamma) 
        
        return gamma
    
    
    def setSpectralDensity( self, gamma, sigma=1.0 ):
        
        self.gamma = sigma**2 * gamma
        
        return
        




    