#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script highlight the meshGraphing of a mesh. Showing which simplices that are associated with a given node in the mesh graph.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

# Import package
from fieldosophy import mesh as mesher

from matplotlib import pyplot as plt
import numpy as np



coordinateLims = np.array([ [0,1], [0,1] ])
mesh = mesher.regularMesh.meshInPlaneRegular( coordinateLims, 1e-1 * np.ones((2)) )

# Create mesh graph
graph = mesher.MeshGraph(mesh.triangles, mesh.nodes, minGraphDiam = 0.2, maxNumGraphNodes = 40, minNumTrianglesInGraph = 1)
    

# %% Plot 


plt.figure(1)
plt.clf()


meshPlotter = mesher.MeshPlotter( mesh )
# Plot interior edges
edges = meshPlotter.getLines()
plt.plot(edges[0], edges[1], color="blue")
# Plot boundary edges
edges = meshPlotter.getBoundaryLines()
plt.plot(edges[0], edges[1], color="red")


# Plot mesh graph
plotGraph = graph.prepareForPlotting()
plt.plot( plotGraph[0,:], plotGraph[1,:], color="black", linewidth=2 )

# Plot chosen node
choosenNode = 2
# Plot triangles belonging to chosen node
for iterTri in graph.triangleList[choosenNode]:
    nodeInds = mesh.triangles[iterTri, [0,1,2,0]]
    plt.plot( mesh.nodes[nodeInds, 0], mesh.nodes[nodeInds, 1], color="green", linewidth=5 )
plt.plot(plotGraph[0, choosenNode*(2**2+2) + np.arange(2**2+1) ], plotGraph[1, choosenNode*(2**2+2) + np.arange(2**2+1) ], color="red", linewidth=2)
