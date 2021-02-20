#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates template matching using cross correlation.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
from skimage import feature as skim
from scipy import stats

from matplotlib import pyplot as plt
from scipy import ndimage
from fieldosophy.misc.misc_templateMatching import TemplateMatching as templ


# %% Create


# Create image
w = int(100)
h = int(100)
y = np.linspace( 0, 1, num = h )
x = np.linspace( 0, 1, num = w )
X,Y = np.meshgrid( x,y )

fig = plt.figure(1)
plt.clf()

# Generate Gaussian white noise
img = stats.norm.rvs(size = w*h).reshape((h,w))

ax = plt.subplot(221)
ax.cla()
plt.imshow( img )

# Smooth
img = ndimage.gaussian_filter(img, sigma=(6, 6), order=0)

ax = plt.subplot(222)
ax.cla()
plt.imshow( img )

# Acquire template from image
temp = img[ 5:9, 7:11 ]
# Match template
tempMatch = skim.match_template( img, temp )

ax = plt.subplot(223)
ax.cla()
plt.imshow( (tempMatch)**20 )




# %% Deform

# Define deformation vector field
def detF(points): 
    out = np.ones(points.shape) * 3e-2
    out[:,1] = 0
    out[:,1] = out[:,1] + np.cos( points[:,0]*2*np.pi ) * 3e-2
    return out
vectorField = detF( np.concatenate( ( X.reshape((-1,1)), Y.reshape((-1,1)) ), axis=1 ) )

# Define points with data
points1 = np.stack((X.flatten(), Y.flatten()), axis=1)
points2 = points1 + vectorField

# Set bounding box
boundingBox = np.array( [ \
     [ np.min( (np.min(points1[:,0]), np.min(points2[:,0])) ), np.max( (np.max(points1[:,0]), np.max(points2[:,0])) ) ], \
     [ np.min( (np.min(points1[:,1]), np.min(points2[:,1])) ), np.max( (np.max(points1[:,1]), np.max(points2[:,1])) ) ], \
     ] )

# Acquire resampled data
img1 = templ.resampleOnGrid( points1, img.flatten(), resolution = (200,300), boundingBox = boundingBox )
img2 = templ.resampleOnGrid( points2, img.flatten(), resolution = (200,300), boundingBox = boundingBox )
# img1 = np.zeros( img.shape + np.array([5,5]) )
# img2 = np.zeros( img.shape + np.array([5,5]) )
# img1[:-5,:-5] = img
# img1 = {"image":img1, "x":np.arange(0,img2.shape[0]), "y":np.arange(0,img2.shape[1]), "boundingBox":np.array([[0,1],[0,1]])}
# img2[2:-3,3:-2] = img
# img2 = {"image":img2, "x":img1["x"], "y":img1["y"], "boundingBox":np.array([[0,1],[0,1]])}

# Get grid of resampled data
X2, Y2 = np.meshgrid( img1["x"], img1["y"] )



# Plot
fig = plt.figure(1)
plt.clf()

ax = plt.subplot(221)
ax.cla()
plt.imshow( img1["image"] )

ax = plt.subplot(222)
ax.cla()
plt.imshow( img2["image"] )


ax = plt.subplot(223)
ax.cla()
plt.quiver( X[::5,::5].flatten(), Y[::5,::5].flatten(), vectorField[::5,0], vectorField[::5,1] )



# %% Estimate

# Acquire local template matching between the two images
templateMatcher = templ()
estimateInds = np.zeros( X2.shape, dtype=np.bool, order='C' )
# estimateInds[::20,::20] = True
estimateInds = None
indices, maxCrossCorr = templateMatcher.griddedTemplateMatching( \
        img1["image"], img2["image"], templateRadius = 7, searchRadius = 25, \
        estimateInds = estimateInds, searchSkip = 0, searchStart = 0 )

# Acquire vector field representation of template matching
vectorFieldEstimX, vectorFieldEstimY = templ.acquireVectorField( \
      indices, maxCrossCorr, X2, Y2, crossCorrThresh = 0.9, medianSize = (4,4) )

# Plot
fig = plt.figure(1)

ax = plt.subplot(224)
ax.cla()
# plt.scatter( X2.flatten(), Y2.flatten() )
jumpInds = 20
plt.quiver( X2[::jumpInds,::jumpInds], Y2[::jumpInds,::jumpInds], \
       vectorFieldEstimX[::jumpInds,::jumpInds], vectorFieldEstimY[::jumpInds,::jumpInds] )
    
ax = plt.subplot(223)
ax.cla()
plt.imshow(maxCrossCorr)


plt.show()