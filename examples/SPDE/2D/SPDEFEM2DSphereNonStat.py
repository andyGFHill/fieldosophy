#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates:
    * Creating an spherical Matérn FEM approximation model in 2 dimensions.
    * Generate samples from this model.
    * Compute covariances.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""


# Import package
from fieldosophy.GRF import FEM
from fieldosophy.GRF import GRF
from fieldosophy import mesh as mesher

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize


        
print("Running two-dimensional FEM Manifold test case")


plt.figure(1)
plt.clf()
#plt.figure(2)
#plt.clf()


# %% Create 2D mesh

# define boundary [in degrees]
lon = [-180.0, 180.0]
lat = [-60.0, 60.0]

# Define smallest correlation range [in degrees]
corrMin = 50.0
# Define extension range [in degrees]
extension = corrMin

# Create data points
dataGrid = np.meshgrid( \
   np.linspace(lon[0], lon[1], num = int(np.ceil( np.diff(lon)[0]/extension*1.2 )) ), \
   np.linspace(lat[0], lat[1], num = int(np.ceil( np.diff(lat)[0]/extension*1.2 )) ) \
   )
dataPoints = np.hstack( (dataGrid[0].reshape(-1,1), dataGrid[1].reshape(-1,1)) )

# Mesh
print("Compute Mesh")
meshPlane = None

# Create spherical mesh
meshSphere = mesher.Mesh.meshOnSphere( maxDiam = 2 * np.sin( 2 * extension / 180.0 * np.pi / 2.0 ), maxNumNodes = int(1e4), radius = 1)


# Cut away unwanted regions
meshSphere = meshSphere.cutOutsideMeshOnSphere( \
   mesher.geometrical.lonlat2Sphere( dataPoints.transpose() ), \
   distance = 1.1 * extension / 180.0 * np.pi )


# Create refined sphere
meshSphere = meshSphere.refine( \
    maxDiam = 1 * np.sin( extension / 180.0 * np.pi / 2.0 ), \
    maxNumNodes = meshSphere.N + 10000, \
    transformation = mesher.geometrical.mapToHypersphere )
    
# Cut away unwanted regions
meshSphere = meshSphere.cutOutsideMeshOnSphere( \
    mesher.geometrical.lonlat2Sphere( dataPoints.transpose() ), \
    distance = 1.0 * extension / 180.0 * np.pi )    

# Create refined sphere
meshSphere = meshSphere.refine( \
    maxDiam = 2/5 * np.sin( corrMin / 180.0 * np.pi / 2.0 ), \
    maxNumNodes = meshSphere.N + 10000, \
    transformation = mesher.geometrical.mapToHypersphere )



print("Plot mesh")

fig = plt.figure(1)
ax = fig.add_subplot(221, projection='3d')
ax.cla()
ax.set_title( "Mesh" )    

# Plot mesh
meshPlotter = mesher.MeshPlotter(meshSphere)
edges = meshPlotter.getLines()
ax.plot(edges[0], edges[1], edges[2], color="blue")
edges = meshPlotter.getBoundaryLines()
ax.plot(edges[0], edges[1], edges[2], color="red")


temp = mesher.geometrical.lonlat2Sphere(dataPoints.transpose())
ax.scatter( temp[0,:], temp[1,:], temp[2,:], color="red" )    





# %% Create FEM system

print("Set up FEM system")

# Define the random field
nu = 2
sigma = 1
sigmaEps = 1e-3

# Get mid points of triangles
triPoints = np.mean( meshSphere.nodes[ meshSphere.triangles, : ], axis=1 )
# Set ranges in longitudal and latitudal directions
r = np.array([1*corrMin, 3*corrMin]) / 180.0 * np.pi
r = np.repeat( r.reshape((1,-1)), repeats = meshSphere.NT, axis=0)
# Compute local basis of tangent spaces
vectors = FEM.tangentVectorsOnSphere( triPoints, northPole = np.array([0.0,0.0,1.0]) )

def mapFEMParams( params ):
    # Function to map own parameters to FEM parameters
    
    # Compute kappa and H
    logGSqrt, GInv = FEM.orthVectorsToG( vectors, params["r"]/np.sqrt(8*nu) )

    return (logGSqrt, GInv)


BCDirichlet = np.NaN * np.ones((meshSphere.N))
BCDirichlet[meshSphere.getBoundary()["nodes"]] = 0
BCDirichlet = None
BCRobin = np.ones( (meshSphere.getBoundary()["edges"].shape[0], 2) )
BCRobin[:, 0] = 0  # Association with constant
BCRobin[:, 1] = - 1 # Association with solution
# BCRobin = None

# Create FEM object
fem = FEM.nonStatFEM( mesh = meshSphere, childParams = { "r":r, "f":mapFEMParams}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )


# temp = triPoints
# ax.scatter( temp[:, 0], temp[:,1], temp[:,2], color="green" )    





# %% Sample

# Acquire realizations
print("Generate realizations")

M = int(5e3)

Z = fem.generateRandom( M )


obsPoints = ( \
     np.linspace(0.99*lon[0]+0.01*lon[1],0.01*lon[0]+0.99*lon[1], num=80), \
     np.linspace(0.99*lat[0]+0.01*lat[1],0.01*lat[0]+0.99*lat[1], num=80) \
     )
obsPoints = np.meshgrid( obsPoints[0], obsPoints[1] )
obsPoints3D = mesher.geometrical.lonlat2Sphere( np.stack( (obsPoints[0].flatten(), obsPoints[1].flatten()), axis=0 ) ) * 0.99
obsPoints3D = np.ascontiguousarray( obsPoints3D.transpose() )

# temp = obsPoints3D
# ax.scatter( temp[:, 0], temp[:,1], temp[:,2], color="green" ) 

# Get observation matrix
print("Acquire observation matrix")
obsMat = fem.mesh.getObsMat( obsPoints3D, embTol = 0.05, centersOfCurvature = np.zeros( (1,3) ) ) 
# obsMat = fem.mesh.getObsMat( triPoints, embTol = 20 )
# obsMat = fem.mesh.getObsMat( meshSphere.nodes[meshSphere.triangles[:,0], :], embTol = 2/5 * np.sin( corrMin / 180.0 * np.pi / 2.0 ) / 10 )
ZObs = obsMat.tocsr() * Z + stats.norm.rvs( loc = 0, scale = sigmaEps, size = M*obsMat.shape[0] ).reshape((obsMat.shape[0], M))







# %% Plot 


fig = plt.figure(1)
# ax = fig.add_subplot(122, projection='3d')
ax = fig.add_subplot(222)
ax.cla()
ax.set_title( "A realization" )    

# m = Basemap(projection='ortho',lat_0=0,lon_0=0) 
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='aquamarine')
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmeridians(np.arange(0,360,30)) # grid every 30 deg
m.drawparallels(np.arange(-90,90,30))

x,y = m(obsPoints[0], obsPoints[1])

m.contourf( x, y, ZObs[:,0].reshape( obsPoints[0].shape ) )


# lon, lat = mesher.geometrical.sphere2Lonlat( triPoints )
# x,y = map( lon, lat )
# ax.tricontourf(lon,lat, ZObs[:,0])

# collec = ax.plot_trisurf( meshSphere.nodes[:,0], meshSphere.nodes[:,1], meshSphere.nodes[:,2], \
#     triangles = meshSphere.triangles, cmap = cm.jet, shade = False )
# collec.set_array( ZObs[:,0] )
# collec.autoscale()






print("Plot covariances")


fig = plt.figure(1)
ax = fig.add_subplot(223)
ax.cla()
ax.set_title( "Covariance" )


# Get point to compare covariance with
covPoint = np.array( [ [0,0] ] )
covPoint3D = mesher.geometrical.lonlat2Sphere( covPoint.transpose() ) * 0.95
covPoint3D = np.ascontiguousarray( covPoint3D.transpose() )
covObsMat = fem.mesh.getObsMat( covPoint3D, embTol = 0.05, centersOfCurvature = np.zeros( (1,3) ) )


# Compute SPDE covariance
runy = fem.multiplyWithCovariance(covObsMat.transpose())
runy = obsMat.tocsr() * runy

# m = Basemap(projection='ortho',lat_0=0,lon_0=0) 
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='aquamarine')
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmeridians(np.arange(0,360,30)) # grid every 30 deg
m.drawparallels(np.arange(-90,90,30))

m.contourf( x, y, runy.reshape( obsPoints[0].shape ) )









fig = plt.figure(1)
ax = fig.add_subplot(224)
ax.cla()
ax.set_title( "Conditional" )


# Get points to condition on
condPoints = np.array( [ [0,0], [12,57] ] )
condPoints3D = mesher.geometrical.lonlat2Sphere( condPoints.transpose() ) * 0.99
condPoints3D = np.ascontiguousarray( condPoints3D.transpose() )
condObsMat = fem.mesh.getObsMat( condPoints3D, embTol = 0.05, centersOfCurvature = np.zeros( (1,3) ) )
condVal = np.array( [1, -1] )

# Compute conditional distribution
condDistr = fem.cond(condVal, condObsMat, sigmaEps)
# Get conditional mean at observation points
condMean = obsMat.tocsr() * condDistr.mu

# m = Basemap(projection='ortho',lat_0=0,lon_0=0) 
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='aquamarine')
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmeridians(np.arange(0,360,30)) # grid every 30 deg
m.drawparallels(np.arange(-90,90,30))

m.contourf( x, y, condMean.reshape( obsPoints[0].shape ) )




# %% Plot marginal standard deviation


fig = plt.figure(1)
ax = fig.add_subplot(224)
ax.cla()
ax.set_title( "Marginal std" )

# m = Basemap(projection='ortho',lat_0=0,lon_0=0) 
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='aquamarine')
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmeridians(np.arange(0,360,30)) # grid every 30 deg
m.drawparallels(np.arange(-90,90,30))

temp = m.contourf( x, y, np.std(ZObs,axis=1).reshape(obsPoints[0].shape) )

fig.colorbar(temp, ax=ax, orientation='horizontal')





