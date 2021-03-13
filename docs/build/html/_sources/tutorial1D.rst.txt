
Tutorial: 1D SPDE modeling using Fieldosophy
=============================================

This tutorial explains the basics of modeling a Gaussian random field in one dimension using the SPDE-approach. 
Let us consider modeling a continuously indexed Gaussian random field in one dimension on the interval [0,1] of the real line using the SPDE approach.
Since we are using a FEM approximation we first need to specify a simplicial mesh over the domain of interest.


Constructing the mesh
----------------------



A simplicial mesh is made up of nodal points and corresponding simplices. 
The nodal points, for a one-dimensional mesh, are defined as an array of size N x 1 where 'N' is the number of nodal points and the value at the i:th element of the array is the location of the i:th nodal point.
For this example, let us define 500 nodal points evenly distributed over the unit interval using the numpy 'linspace' command.

.. code-block:: python

    import numpy as np
    nodes = np.linspace( 0,1, 500 ).reshape((-1,1))

For one-dimension, a simplex is a line between two nodes. The simplices are therefore defined in an array of size NT x 2, where each row defines a simplex by giving the indices of its two nodal points. 
To make a sensible mesh in one-dimension we want the nodal points to be connected through a simplex with the former and next nodal point on the interval. 
Since our nodal points are in the order from lowest to highest, we can make such a collection of simplices by stacking two one-dimensional arrays, one with values from 0 to N-1 and one from 1 to N.
Since the available indices of the nodal points ranges from 0 to N-1, we can construct the simplices using the 'arange' command from numpy.

.. code-block:: python

    simplices = np.stack( (np.arange( 0, 499 ), np.arange(1,500)) )
    
Given the nodal points and simplices we create a mesh object using the 'Mesh' class from fieldosophy.mesh.

.. code-block:: python

    from fieldosophy import mesh as mesher
    mesh = mesher.Mesh( triangles = simplices, nodes = nodes )





Constructing a Matérn model
----------------------------

If we want to work with the Matérn model on the defined mesh we first want to choose the values of the parameters. 
The general Matérn model has three parameters, a range parameter (:math:`\kappa`), a smoothness parameter (:math:`\nu`), and a marginal standard deviation parameter (:math:`\sigma`).
In the *MaternFEM* class of Fieldosophy.FEM we actually parameterize the range by the, more intuitive, correlation range :math:`\left( r := \frac{\sqrt{8\nu}}{\kappa} \right)' instead of :math:`\kappa`.
Let us set the correlation range to 0.4, the smoothness to 1.5, and the marginal standard deviation to 2. We can then create the Matérn FEM object.

.. code-block:: python

    # Define the Matérn random field
    r = 0.4 # Set correlation range (range for which two points have approximately 0.13 correlation)
    nu = 1.5   # Set smoothness (basically the Hölder constant of realizations)
    sigma = 2   # Set standard deviation
    # Create MaternFEM object 
    from fieldosophy.GRF import FEM
    fem = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma )
    
Notice how *sigma* and *nu* are direct parameters to FEM.MaternFEM while *r* is a member of the dictionary *childParams*, which itself is a parameter of FEM.MaternFEM. The reason for this is that :math:`\sigma` and :math:`\nu` are parameters of the general differential operator of :eq:`hildemanSPDE`, while :math:`r` is a parameter of the differential operator of :eq:`lindgrenSPDE`.
Since :eq:`lindgrenSPDE` is just a special case of :eq:`hildemanSPDE`, the *r* parameter is added to the FEM.MaternFEM class while the *nu* and *sigma* parameters are actually added to a class higher up in the inheritance hierarchy.

We can now generate realizations of our Matérn Gaussian random field on [0,1]. Let us generate two realizations at the nodal points.
    
.. code-block:: python

    # Acquire realizations on nodes
    Z = fem.generateRandom( 2 )
    
And plot these using matplotlib.

.. code-block:: python

    # Plot realizations
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z[:,0] )
    plt.title( "Realization 1" )
    plt.figure(2)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z[:,1] )
    plt.title( "Realization 2" )
    
.. image:: https://drive.google.com/uc?export=view&id=1K1zWpkfYvykeWN5rEpx40JYpydHPWEmI
    :width: 49%


.. image:: https://drive.google.com/uc?export=view&id=1I8Vse1okxEqAP2DMVNWY9lAz0fnAvSRU
    :width: 49%
    
As can be seen, we have two different functions on the unit interval. 
Both of them are exhibiting a correlation range of 0.4; which is indicated by all trends having a length of no more than approximately 0.4.


Quality of FEM-approximation and numerical stability
-----------------------------------------------------

It should be noted that the smoothness parameter (:math:`\nu`), the range (:math:`r`), and the number of nodes (N) per distance all together strongly affects how well the FEM solution approximates the theoretical random field. 
As a rule of thumb, there should be at least 5 nodes per correlation range. 
That means that, anywhere on the mesh, one has to pass atleast 5 nodes when moving a distance corresponding to :math:`r`.
The reason for this is that the finite element approximation of the true solution otherwise becomes bad.

How many nodes per correlation range one should have to get a good approximation depends on several factors: 

#. Piecewise linear basis functions
    The FEM-approximation assumes that the realizations are piecewise linear (linear within a simplex), considering several points within the same simplex, for the same realization, can only give you a linear relationship between them. 
    Hence, it is often necessary to have a closer space between nodes in regions where there are a lot of observation points.
    This such that there should be no more than one observed point in the same simplex (in the same realization). 
#. Demand on approximative power
    How similar the correlation function should be to the theoretical one (Matérn in this case) depend on the application. 
    Therefore, the number of nodes per correlation range should be adjusted for your needs.
#. Computational cost
    More nodes means a higher computational cost. In one dimension the computational cost (of the FEM approximation) is linear in the number of nodes.
    In higher dimensions it is larger ( :math:`\mathcal{O}^{\frac{2}{3}}` for two dimensions and more than quadratic for three-dimensions).
    Therefore, one does never want to have more nodes in the mesh than what is needed.
    
Our personal experience is that 7-12 nodes per correlation range is usually sufficient for the applications we have encountered.

Another issue is numerical stability. This can be a problem if 'N' is very large. 
More than so though, numerical stability is strongly affected by the smoothness parameter (:math:`\nu`).
The reason is twofold:

#. Number of FEM-approximations
    A FEM approximation is actually only made on the stochastic partial differential equation of :eq:`generalSPDE` with :math:`\beta = 1`.
    For every other value of :math:`\beta`, the approximation is made by doing FEM approximations on the FEM approximation, i.e., *iterated FEM*. 
    For instance, :math:`\beta = 3` means that two levels of FEM approximations are made while :math:`\beta=5` means that three levels of FEM approximations are made.
    For every level of FEM approximation, the resulting precision matrix becomes less stable, i.e., instabilities propagate and amplifies for each iteration.
    In Fieldosophy, the maximum number of iterations allowed are 2, i.e., no more than two levels of FEM approximations can be performed, regardless of :math:`\beta`.
    The limit of 2 is to reduce this issue of propagating instabilities.
#. Rational approximation
    Remember that :math:`\beta = \frac{\nu}{2} + \frac{d}{4}` and :math:`\nu` can be any positive value.
    For the cases when :math:`\beta` is not an integer (or due to the limit of 2, also when :math:`\beta` > 3), the approximation to the solution is a linear combination of several (iterated-) FEM approximations, see :cite:`lit:bolin2`.
    Although no single iterated FEM approximation in this case has a higher level than 2, they are summed up with varying values of coefficients.
    For some combinations of these coefficients (each combination corresponding to a single value of :math:`\beta`), the precision matrix can become unstable.
    This is typically the case for small values of :math:`\beta`, i.e., small values of :math:`\nu`. 
    
Hence, one should be aware that small values of :math:`\nu` can give a worse approximation of the theoretical covariance function. 
In worst case, the precision matrix might actually become so unstable that the Choelsky factorization fails, throwing the error "PositiveDefiniteException: Matrix is not positive definite".
If you are experiencing instabilities, one remedy can be to reduce the number of nodes in the mesh, if the application allows. A more obvious solution, if the application allow, is to increase :math:`\nu`.

To complicate it further, the stability of the precision matrix can also be strongly affected by the boundary conditions.
Especially Dirichlet boundary conditions that are improbable with the given random field model can cause strong instabilities and cause the system to throw the error "PositiveDefiniteException: Matrix is not positive definite".



Boundary conditions
----------------------



So far we have one problem though. In both realizations we see that time series near the ends of the unit interval is going flat. 
This is due to the homogeneous Neumann boundary condition that we implicitly enforced on the random field when we generated the FEM.MaternFEM object. 
Remember (see :ref:`Introduction`) that the solution to a differential equation is not well-defined without boundary/initial conditions. 
In other words, one has to give some kind of condition to even consider a solution to a stochastic partial differential equation.
In the FEM implementation in Fieldosophy this causes any SPDE-model without an explicit boundary condition to get a homogeneous Neuman condition on the whole boundary domain. 
In this one-dimensional case, the boundary domain is the end points of the unit interval and the Neumann boundary condition states that the derivative at the boundary should be zero.
This is also something we can observe in the figures above.

It is possible to change the boundary conditions. The supported types of boundary conditions are *Dirichlet*, *periodic*, and *Robin*, where *Neumann* is a special case of a Robin boundary condition. 
Different boundary conditions can also be applied to different parts of the boundary.
In fact, the Dirichlet boundary condition does not even need to be set on the boundary. It is possible to set Dirichlet conditions on any nodes in the mesh, i.e., fixing their values.
However, the Robin and periodic conditions can only be set at the boundaries.

Just to show how this is done, let us apply a Dirichlet boundary condition of 3 to the left boundary and enforce that the 300:th node in the mesh is -0.5, while keeping the homogeneous Neumann boundary condition on the right boundary.
This is done by defining the *BCDirichlet* parameter when creating the fem object. The BCDirichlet parameter should be an array the size of the number of nodal points in the mesh. 
Each element in the array correspond to the Dirichlet condition on the associated nodal point. 
A value of 'np.nan' signals no Dirichlet condition on that specific node. 
Hence, we can apply our new boundary conditions as,

.. code-block:: python

    # Create Dirichlet boundary condition
    BCDirichlet = np.NaN * np.ones((mesh.N))
    BCDirichlet[[0, 300]] = np.array([3, -0.5])
    # Create new fem model
    fem2 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )

We can now generate two new realizations and plot these.
    
.. code-block:: python

    # Acquire realizations on nodes
    Z2 = fem2.generateRandom( 2 )    

    plt.figure(1)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z2[:,0] )
    plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
    plt.title( "Realization 1 with Dirichlet conditions" )
    plt.figure(2)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z2[:,1] )
    plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
    plt.title( "Realization 2 with Dirichlet conditions" )

.. image:: https://drive.google.com/uc?export=view&id=1V-nEqLrActie2YibSniR21NMj-WehqMH
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=10dElv8qiJr9xLWxZ6pOeFGgGtNkNk18m
    :width: 49%

As can be seen, the Dirichlet boundary conditions are indeed enforced in the realizations while the homogeneous Neumann boundary condition still applies to the right boundary.


We know that we will only have a true Matérn covariance function on the unit interval if we extend the modeled domain to the whole real line. 
Of course, practically we do only need to extend it a bit outside of the unit interval to get a good approximation of a Matérn covariance function on the whole unit interval.
The amount of extension needed depends on the smoothness and range parameters but also on the boundary conditions.
A rule of thumb is that the extension outside of the spatial domain of interest should be about 1.5 to 2 times the correlation range when using homoegeneous Dirichlet conditions and a centered Gaussian random field (a mean function that is zero).
A similar extension range also apply to Neumann conditions. 

It is quite intuitive to think that if we enforce the values at the boundary (Dirichlet condition) the variations around the boundary will be small. 
Hence, close to the boundary we will get artifacts (as compared to a Matérn field).
In the same way, enforcing the derivative to be fixed close to the boundary also causes restrictions on the variability.
This since points close to the boundary all tend to have an almost deterministic relationship among themselves (since they should approximately follow the line of the derivative).
Due to this, our experience have shown that one can acquire a shorter range from the boundary artifacts if one instead uses a more clever boundary condition.
If we do not want to put a cap on the derivative (as in a Neumann condition) and we do not want to put a cap on the actual value (as in the Dirichlet condition), we can instead enforce a certain Robin condition.

A Robin condition on a piece of a boundary, :math:`\Gamma`, is,

.. math::
    \boldsymbol{n} \cdot \nabla X  = a X + b, \boldsymbol{s} \in \Gamma.
    :label: RobinBC
    
By setting :math:`b=0` and :math:`a = -1`, we are saying that the derivative can vary, and the function value can vary to. 
We do however force them to vary together such that the derivative is equal to the negative of the actual value.
This means that there will still be boundary artifacts but their effective range is diminished.  
In our experience it is sufficient to extend by only 1 times the correlation range when using such a boundary condition.
One might also want to adjust the magnitude of 'a' depending on the smoothness and the marginal variance at the boundary. 
The reason being that one might otherwise cap the values of either its derivative or the random field itself.

An arbitrary Robin boundary condition can be defined by the BCRobin parameter. The number of rows of the array should correspond to the number of simplices at the edges. 
Each row is associated with a specific boundary simplex, which one can be seen by the command 'mesh.getBoundary()["edges"]'.
The first column in 'BCRobin' correspond to the :math:`b`-value in :eq:`RobinBC`. To set the boundary condition on the right boundary as suggested above, we do:

.. code-block:: python

    # Create Robin boundary condition
    BCRobin = np.ones( (2, 2) )
    BCRobin[1, 0] = 0 # Association with constant
    BCRobin[1, 1] = -1 # Association with function
    # Update new fem model
    fem3 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )

And once again, generate two realizations and plot these.

.. code-block:: python

    # Acquire realizations on nodes
    Z3 = fem3.generateRandom( 2 )
    
    plt.figure(1)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z3[:,0] )
    plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
    plt.title( "Realization 1 with Dirichlet and Robin conditions" )
    plt.figure(2)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), Z3[:,1] )
    plt.scatter( mesh.nodes[[0, 300]], np.array([3,-0.5]), color="red" )
    plt.title( "Realization 2 with Dirichlet and Robin conditions" )


.. image:: https://drive.google.com/uc?export=view&id=1bGdeZos4EXO9GQjVWKt7e-7X9iRzLXKQ
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1iexPdE0mAqBbrKSs-ZlnA-K4EBKGt0fp
    :width: 49%


As can be seen, this Robin condition forces the realizations to have a derivative on the right boundary that is aimed towards zero. That is, a realization that is above zero at 1 will have a tendency to move towards zero and a realization that is below zero at 1 will have a tendency to move towards zero as well.
Note that, if we would place a Robin condition and a Dirichlet condition affecting the same part of the boundary, the Dirichlet condition has presedence. 

It should be noted that in this example we used :math:`\nu = 1.5`. This was not a coincidence since that corresponds to :math:`\beta = 1` in :eq:`generalSPDE`. In fact, the boundary conditions given are only exact as long as :math:`\beta = 1`. 
This is because we are using an iterative finite element solution for other values of :math:`\beta`. 
This means that we will generate the system matrix of the finite element several times. Each time we will enforce the boundary conditions. 
Hence, this will generally correspond to a different boundary condition on the final random field then what was given to the "single" FEM solution.
However, using a Robin condition to reduce the extension of the boundary artifacts, as suggested above, still apply and is strongly recomended. 




Extending the mesh to remove boundary artifacts
------------------------------------------------

So given the above we will have to redo our mesh in order to incorporate this mesh extension. 
When we extend we might still want to keep the nodal density. Hence we have to increase N accordingly.

.. code-block:: python
    
    # Create new mesh extended by r in each direction
    N = int( 500 * ( (2*r+1)/1 ) )
    nodes = np.linspace( 0-r,1+r, N ).reshape((-1,1))
    simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
    extendedMesh = mesher.Mesh( triangles = simplices, nodes = nodes )

Then we create a new FEM object, but this time with the new mesh and with the suggested Robin boundary conditions on both boundary points.  
    
.. code-block:: python
    
    # Create new mesh extended by r in each direction
    N = int( 500 * ( (2*r+1)/1 ) )
    nodes = np.linspace( 0-r,1+r, N ).reshape((-1,1))
    simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
    extendedMesh = mesher.Mesh( triangles = simplices, nodes = nodes )
    
    # Create Robin boundary condition
    BCRobin = np.ones( (2, 2) )
    BCRobin[:, 0] = 0 # Association with constant
    BCRobin[:, 1] = -1 # Association with function
    
    # Create fem model
    fem4 = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCRobin = BCRobin )
    
Let us generate two realizations and plot these to compare.

.. code-block:: python
    
    # Acquire realizations on nodes
    Z4 = fem4.generateRandom( 2 )
    
    plt.figure(1)
    plt.clf()
    plt.plot( extendedMesh.nodes.flatten(), Z4[:,0] )
    plt.title( "Realization 1 with extended mesh" )
    plt.figure(2)
    plt.clf()
    plt.plot( extendedMesh.nodes.flatten(), Z4[:,1] )
    plt.title( "Realization 2 with extended mesh" )



.. image:: https://drive.google.com/uc?export=view&id=1F2uZadDFnEadSd-qGLPXS_DHovqOrUqB
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1XLT_7Spygb2rGiR6P6tdfDMce032ACIe
    :width: 49%

Not much new except that the x-axis not show values in the range of [-0.4 to 1.4].




Computing and comparing covariances
------------------------------------


It would be interesting to compute the actual covariance function and see if it does indeed correspond to a Matérn covariance. 
This is possible but remember, a covariance function takes two arguments. 
Since the Matérn covariance is stationary it does not matter which two points we choose; the covariance should only depend on the distance between the two points.
Therefore, let us choose one point and compute the covariance between that point and all other points in the mesh.

We choose the nodal point with index 500, that happens to be placed at 0.601. 
To acquire the covariance between all nodal points and the point at 0.601 we can use the function 'multiplyWithCovariance' in the FEM class. 

.. code-block:: python
    
    referenceNode = np.zeros((extendedMesh.N,1))
    referenceNode[500] = 1
    covSPDE = fem4.multiplyWithCovariance( referenceNode )
    
We can compare this to the true Matern covariance by using the 'Fieldosophy.GRF.GRF.MaternCorr' function. Remember that the distance should be computed from the nodal point at index 500.    
    
.. code-block:: python

    # Compare with actual matern covariance
    from fieldosophy.GRF import GRF
    covMatern = sigma**2 * GRF.MaternCorr( np.abs(extendedMesh.nodes[500,0] - extendedMesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )
    
    # Plot covariances
    plt.figure(1)
    plt.clf()
    plt.plot( extendedMesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
    plt.vlines( extendedMesh.nodes[500,0], 0, 4, linestyle='--' )
    plt.title( "Covariance between point 0.601 and all other" )
    plt.plot( extendedMesh.nodes.flatten(), covMatern, color="red", label="Matern" )
    plt.legend()
    
  
.. image:: https://drive.google.com/uc?export=view&id=114kj9MGS0GTCNC2kYjWp4GT1jZ53FsmA    
    :width: 80%
    :align: center    
    
As can be seen, the two plots fit perfectly.

We can compare this with the covariances if we would not have extended the mesh and left the original homogeneous Neumann boundary conditions. 
Here using the two different nodal points to see the effect.
    
.. code-block:: python
    
    # Compute covariance
    referenceNode = np.zeros((mesh.N,1))
    referenceNode[300] = 1
    covSPDE = fem.multiplyWithCovariance( referenceNode )
    
    # Compare with actual matern covariance
    from fieldosophy.GRF import GRF
    covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[300,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )
    
    # Plot covariances
    plt.figure(1)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
    plt.vlines( mesh.nodes[300,0], 0, 4, linestyle='--' )
    plt.title( "Covariance between point 0.601 and all other" )
    plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
    plt.legend()
    
    # Compute covariance
    referenceNode = np.zeros((mesh.N,1))
    referenceNode[0] = 1
    covSPDE = fem.multiplyWithCovariance( referenceNode )
    
    # Compare with actual matern covariance
    from fieldosophy.GRF import GRF
    covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[0,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )
    
    # Plot covariances
    plt.figure(2)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
    plt.vlines( mesh.nodes[0,0], 0, 4, linestyle='--' )
    plt.title( "Covariance between point 0 and all other" )
    plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
    plt.legend()
    


.. image:: https://drive.google.com/uc?export=view&id=1s3bIO_zjtRxuNvCWlWHUkjEQCV5YBBi6
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1nw1mZNxJ2GFTz9iTuEmYE5IVPZXDgeqN
    :width: 49%    

It is clear from this picture that the boundary artifacts have an impact on the corresponding Gaussian random field. 
In the left figure we see the covariances between the point at the left boundary of the unit interval and all other nodal points.
Here, the variance is greatly overestimated by the SPDE approximation. In the right figure we see the covariance between the point at 0.601 again. 
Without the mesh extension we see the boundary artifacts in the increased covariance when approaching the boundaries.

As can be seen, a Neumann boundary condition will overshoot the true covariance; making correlation ranges longer than intended. 
On the other hand, a Dirichlet condition will undershoot it; making the correlation smaller than intended. 
This gives a rationale for why a Robin condition can mitigate the boundary artifacts since it is a mixture between a Dirichlet condition (when both 'a' and 'b' are both large) and a Neumann condition (when 'a' is zero).

Before leaving the boundary conditions, let us just demonstrate the difference between the homogeneous Dirichlet and the Robin condition, when 'a' is  tuned for this particular example (setting :math:`a = -0.26`).

.. code-block:: python
    
    # Create Dirichlet boundary condition
    BCDirichlet = np.NaN * np.ones((mesh.N))
    BCDirichlet[0] = 0
    # Create Robin boundary condition
    BCRobin = np.ones( (2, 2) )
    BCRobin[1, 0] = 0 # Association with constant
    BCRobin[1, 1] = -0.26 # Association with function
    # Update new fem model
    fem5 = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCDirichlet = BCDirichlet, BCRobin = BCRobin )
    
    temp = 300
    
    # Compute covariance
    referenceNode = np.zeros((mesh.N,1))
    referenceNode[temp] = 1
    covSPDE = fem5.multiplyWithCovariance( referenceNode )
    
    # Compare with actual matern covariance
    from fieldosophy.GRF import GRF
    covMatern = sigma**2 * GRF.MaternCorr( np.abs(mesh.nodes[temp,0] - mesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )
    
    # Plot covariances
    plt.figure(2)
    plt.clf()
    plt.plot( mesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
    plt.vlines( mesh.nodes[temp,0], 0, 4, linestyle='--' )
    plt.title( "Covariance between point 0.601 and all other" )
    plt.plot( mesh.nodes.flatten(), covMatern, color="red", label="Matern" )
    plt.legend()
    


.. image:: https://drive.google.com/uc?export=view&id=19Zj-wqFqE_VI0s2gLascacIi5pnz4AQJ
    :width: 80%  
    :align: center

Here, the left boundary has a homogeneous Dirichlet condition while the right boundary has the suggested Robin condition (:math:`a=-0.26, b=0`).
When comparing the covariance between all points and the point at 0.601, no boundary artifact is visible at the right side while the left side undershoots. 
This is important since the point 0.601 is about 1 x :math:`r` away from the right boundary but 1.5 x :math:`r` from the left boundary.
So remember, always use the Robin condition in order reduce the number of nodes needed in the mesh extension; it becomes even more important when working in higher dimensions.




Conditional distributions
---------------------------

One of the most powerful properties of probabilistic modeling is the ability to acquire conditional distributions. 
That is, acquiring the random field given that data has been observed at some locations.

We saw earlier how the Dirichlet boundary condition could be enforced also in the interior of the spatial domain. 
This is one way of conditioning the random field, i.e., when knowing the value at some nodal points we can use those values as Dirichlet conditions to acquire a random field that includes the extra information we got from our data.
However, this only work properly when :math:`\beta=1` and the observed values are at nodal points. 
Although nodal points can be inserted in the exact locations of observations, this become more complicated if observation points differ between different replicates.

Often there are some additive noise to an observations. One speaks about the random field being a *latent model* that is not observed directly.
Instead, the observations are the latent model with some "noise" added. This noise is typically modeled as independent between observations.
Fieldosophy allows for conditioning given such an additive noise model,

.. math::
    Y_i = X(\boldsymbol{s}_i) + \epsilon_i, \boldsymbol{s}_i \in \mathcal{D}.

Here, :math:`\{Y_i\}_i` are the "measurement" random variables at locations :math:`\boldsymbol{s}_i`. 
These random variables are dependent through the random field, :math:`X`, but also has their independent noise, :math:`\epsilon_i`.

In a typical application we are not interested in the observational noise, :math:`\epsilon_i`, which typically depend on imperfections in the measurement equipment. 
Instead, we are interested in modeling the actual process which the measurements try to observe at some points in space.
That is, we are interested in the random field or in a realization of the random field. 
This scenario can be considered as a Bayesian analysis. We have our prior (the random field model), our likelihood (the distribution of :math:`Y` given :math:`X`), and we are interested in our posterior (:math:`X` given Y).
Fieldosophy allow for computing these posterior random fields using the function 'cond' in the FEM class (given that the noise distribution is Gaussian).

Assume that we have the FEM model with the extended mesh and we have measurements at points 0.2 and 0.9. 
The measured values were 2 and -1. We also know that the first measurement was made with a noise variance of 0.3 while the second measurement was made with a noise variance of 0.05.
We can then acquire the posterior random field.

.. code-block:: python

    # Set points of measurement
    condPoints = np.array( [0.2, 0.9 ] ).reshape((-1,1))
    # Get observation matrix for these points
    condObsMat = fem4.mesh.getObsMat( condPoints ).tocsc()
    # Set conditional values
    condVal = np.array( [2.0, -1.0] )
    # Set measurement noise
    sigmaEps = np.sqrt(np.array([0.3, 0.05]))
    # Compute conditional distribution
    condDistr = fem4.cond(condVal, condObsMat, sigmaEps = sigmaEps)



Before continuing, let us look at the first two rows above. What is hapening there is that we define two points to "look at". 
We then generate an "observation matrix" for those points using the command 'getObsMat' from the 'fieldosophy.mesh.Mesh' class.
An observation matrix maps nodal point values to the value at the given points. 
This is possible since our FEM representation assumes picewise linear basis functions. 
It is therefore possible to map values at the nodal points to corresponding values at any points in the interior of the mesh using a linear transformation. 

The next lines defines the values at the measurement points as well as the standard deviation of their noise processes.
The final line of code then generates 'condDistr', which is a FEM object just as the original random field, 'fem4'. 
That means that we will be able to perform the same operations on it as on 'fem4'.


Before plotting the posterior random field, remember that we are only interested in the unit interval. 
Our mesh is extended to [-0.4,1.4] due to boundary effects but we only care about what is hapening on [0,1].
Just as with the measurement points, we can define which locations we want to "look" at.
Let us for now consider 100 points evenly distributed on the unit interval. 
We can define an "observation matrix" for these as well.
Let us do that and then generate five realizations of the posterior random field as well as ploting the posterior mean and posterior marginal 90% prediction interval.

.. code-block:: python

    # Define analysis points on the unit interval
    anPoints = np.linspace(0,1,100).reshape((-1,1))
    # Get observation matrix for analysis points
    anObsMat = fem4.mesh.getObsMat( anPoints ).tocsc()
    # Get posterior mean mean at analysis points
    condMean = anObsMat.tocsr() * condDistr.mu
    # Generate from the posterior random field and get values at analysis points
    condZ = anObsMat.tocsr() * condDistr.generateRandom( 5 )
    # Get covariance matrix for posterior Gaussian random field at analysis points
    condVar = anObsMat.tocsr() * condDistr.multiplyWithCovariance( anObsMat.transpose().tocsc() ) 
    # Use only the variances, i.e., the diagonal of the covariance matrix
    condVar = np.diag(condVar)
    
    # Plot conditional distribution
    plt.figure(1)
    plt.clf()
    plt.plot( anPoints.flatten(), condZ[:,0], color="gray", label="Realization 1", linestyle="--", linewidth=1)
    plt.plot( anPoints.flatten(), condZ[:,1], color="gray", label="Realization 2", linestyle="--", linewidth=1)
    plt.plot( anPoints.flatten(), condZ[:,2], color="gray", label="Realization 3", linestyle="--", linewidth=1)
    plt.plot( anPoints.flatten(), condZ[:,3], color="gray", label="Realization 4", linestyle="--", linewidth=1)
    plt.plot( anPoints.flatten(), condZ[:,4], color="gray", label="Realization 5", linestyle="--", linewidth=1)
    plt.plot( anPoints.flatten(), condMean, color="red", label="Mean", linewidth=2)
    plt.scatter( condPoints.flatten(), condVal, label="Measurements" )
    
    # Plot conditional marginal 90% prediction interval
    from scipy import stats
    plt.fill_between( anPoints.flatten(), \
         stats.norm.ppf( 0.05, loc = condMean, scale = np.sqrt(condVar) ), \
         stats.norm.ppf( 0.95, loc = condMean, scale = np.sqrt(condVar) ), color = [0,1,0], label="Uncertainty" )
    plt.legend( loc='upper right' )


.. image:: https://drive.google.com/uc?export=view&id=1bO3AwCmdPVI48H7xq7oEkaIp6L0teOa-
    :width: 80%  
    :align: center

The green region is the marginal 90% prediction interval, i.e., looking at only one location, there is a 90% probability that the value at this point will be inside the green inteval.
Remember that the green region should be interpreted marginally, i.e., for each point by itself.
If one realization happens to have a point outside of the green region, the probability that a point close to it will be outside of the green region is much higher than 10%.

As can be seen from the green region, close to the measured locations the variability is lower, i.e., the prediction interval is more narrow.
The variability then increases as points gets further away from the measured points. 
We can see the same with the mean function that is very close to the measured values but then moves towards 0, since the original random field was centered.
Also the five realizations from the posterior random field show how the variability increases further away from the measurement locations.

It should be added that one can condition on more than one replicate by letting 'condVal' be a matrix where the columns represent different replicates while the rows still represents different locations. 




Likelihood based inference
---------------------------


The likelihood function is important in statistics. 
It can be used for several things, e.g., to estimate parameters through the maximum likelihood method, model validation through likelihood-ratio tests, or in Bayesian statistics, for instance to calculate acceptance ratios in Markov chain Monte-Carlo algorithms.

Fieldosophy allow computing the logarithm of the likelihood of a FEM object using the 'loglik' function. 
Let us use the measurements from the former section and compute the log-likelihood function.


.. code-block:: python

    logLik = fem4.loglik( condVal, condObsMat.tocsc(), sigmaEps=sigmaEps)
    
Giving a value of :math:`-3.86`.

A high value of the likelihood (and therefore also of the log-likelihood) indicates that the model explain the observed data well. 
However, one can only talk about high and low values in relative terms since the value of the likelihood depend strongly on the observed values.
Therefore, as an example, let us create a model which is identical to 'fem4' in every way except that it has a mean function that is equal to the posterior mean.
Hence, this model should have a higher likelihood than 'fem4' since it is more probable that it would generate the "measured" data.

.. code-block:: python

    # Create Robin boundary condition
    BCRobin = np.ones( (2, 2) )
    BCRobin[:, 0] = 0 # Association with constant
    BCRobin[:, 1] = -0.26 # Association with function
    # Create FEM object
    femLoglik = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':r}, nu = nu, sigma = sigma, BCRobin = BCRobin, mu = condDistr.mu )
    
    # compute
    logLik2 = femLoglik.loglik( condVal, condObsMat.tocsc(), sigmaEps=sigmaEps)

The value of 'loglik2' is :math:`-3.27` and hence higher than 'loglik', as expected. The above code also show how we can define a non-centered GRF model by the 'mu' parameter. One simply supply an array with the mean value for each node.




Estimate parameters
--------------------


We can also use the log-likelihood function to estimate the parameters of our random field model using the maximum likelihood method. 
Let us first create a new random field model with a different range and smoothness, and with some measurement noise. 


.. code-block:: python

    # Create Robin boundary condition
    BCRobin = np.ones( (2, 2) )
    BCRobin[:, 0] = 0 # Association with constant
    BCRobin[:, 1] = -0.26 # Association with function

    # Define new Matérn random field
    rTrue = 0.2 # Set correlation range (range for which two points have approximately 0.13 correlation)
    nuTrue = 2.5   # Set smoothness (basically the Hölder constant of realizations)
    # Create MaternFEM object 
    femTrue = FEM.MaternFEM( mesh = extendedMesh, childParams = {'r':rTrue}, nu = nuTrue, sigma = sigma )
    
    # Generate 100 realizations
    ZTrue = femTrue.generateRandom( 100 )
    
    # Define 20 "measurement points" randomly over the unit interval
    measPoints = np.random.rand( (20) ).reshape((-1,1))
    # Get observation matrix 
    measObsMat = femTrue.mesh.getObsMat( measPoints ).tocsc()
    # Get observations to measurement points
    ZMeas = measObsMat.tocsr() * ZTrue
    # Define measurement standard deviation (the same for all points)
    sigmaEpsTrue = 0.2
    # Add measurement noise
    ZMeas = ZMeas + stats.norm.rvs( loc = 0, scale = sigmaEpsTrue, size = 20*100 ).reshape((20, 100))
    
To get a feeling for the random field, lets observe the values of 'Z' and 'ZMeas' for two different realizations.

.. code-block:: python

    plt.figure(1)
    plt.clf()
    # Plot realizations
    plt.plot( extendedMesh.nodes.flatten(), ZTrue[:,0], label="Z" )
    plt.scatter( measPoints.flatten(), ZMeas[:,0], label = "ZMeas")
    plt.legend( loc='upper right' )
    plt.title("Realization 1")
    plt.figure(2)
    plt.clf()
    # Plot realizations
    plt.plot( extendedMesh.nodes.flatten(), ZTrue[:,1], label="Z" )
    plt.scatter( measPoints.flatten(), ZMeas[:,1], label = "ZMeas")
    plt.legend( loc='upper right' )
    plt.title("Realization 2")



.. image:: https://drive.google.com/uc?export=view&id=1pEbexBXK9OY6xo8WNfXGWcpDbaB7nP3M
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1GDGYmU_ZryOscGNliJHl9t2woC5-y92i
    :width: 49%


    
    

Next, we do only want to search for the parameters in a reasonable range. Let us only look for a smoothness in between [1, 4], a correlation range in between [0.1, 1], and a noise standard deviation in between [0.01, 1].
Remember that we want the correlation range to be no smaller than 5 times the minimum nodal distance and no longer than the mesh extension. 
At the same time we want to use as few nodes as possible. 
One of the benefits with the observation matrices is that we remove the direct connection between points we are intersted in and nodal points.
This can enhance performance since the mesh can often be coarser due to this, compared to if we could only model values at the nodal points.
Before we had 500 points on the unit interval.
We do not need that many nodes and when optimizing we want speed so let us make a new mesh. 
Say that we do not want the nodal distance to be longer than a seventh of the correlation range in order to keep a good approximation.
We know that the smallest range we will allow is 0.1 and therefore we should have :math:`7\cdot 10 = 70` nodes on the unit interval. 
At the same time we want to search for the correlation range in an interval with a maximum of 1. Hence, we want to extend by 1 on each side.


.. code-block:: python

    # Create new mesh
    N = 70 * 3
    nodes = np.linspace( -1,2, N ).reshape((-1,1))
    simplices = np.stack( (np.arange( 0, N-1 ), np.arange(1,N)), axis=1 )
    mesh = mesher.Mesh( triangles = simplices, nodes = nodes )
    # Create observation matrix for points
    obsMat = mesh.getObsMat( measPoints ).tocsc()
 
We can now define our initial guess of the true random field on this new mesh. Let us assumed the same parameters as before as our initial guess.


.. code-block:: python

    # Define the Matérn random field
    r = 0.4 # Set correlation range (range for which two points have approximately 0.13 correlation)
    nu = 1.5   # Set smoothness (basically the Hölder constant of realizations)
    sigma = 2   # Set standard deviation
    sigmaEps = 0.1 # Set the measurement noise standard deviation
    femNew = FEM.MaternFEM( mesh = mesh, childParams = {'r':r}, nu = nu, sigma = sigma )
    
Compare the log-likelihood function of the two models.

.. code-block:: python

    # loglik
    loglik = femNew.loglik( ZMeas, obsMat.tocsc(), sigmaEps=sigmaEps )
    loglik2 = femTrue.loglik( ZMeas, measObsMat.tocsc(), sigmaEps=sigmaEpsTrue )

The values were, -32.59 for the initial guess and -23.20 for the true model. Obviously the true model has a higher log-likelihood.

We want to make use of a numerical optimizer to maximize the log-likelihood. It is generally easier to optimize over an unconstrained parameter space. 
Therefore, we define a transformation of the parameters such that the search range becomes the real line. 
Let us define a sigmoid function and its inverse through the CDF and quantile function of the logistic probability distribution. 
Hence, we can map [1,4], [0.1,1], and [0.01, 1] to the real line with a well-behaving function.


.. code-block:: python

    def optimTrans( x ):
        # Function for mapping nu and r back to original range
        
        y = x.copy()
        
        y[0] = stats.logistic.cdf(y[0]) * 3.0 + 1.0
        y[1] = stats.logistic.cdf(y[1]) * 0.9 + 0.1
        y[2] = stats.logistic.cdf(y[2]) * 0.99 + 0.01
        
        return y
        
    def optimTransInv( x ):
        # Function for mapping nu and r to an unbounded space
        
        y = x.copy()
        
        y[0] = stats.logistic.ppf( (y[0] - 1.0 ) / 3.0 )
        y[1] = stats.logistic.ppf( (y[1] - 0.1 ) / 0.9 )
        y[2] = stats.logistic.ppf( (y[2] - 0.01 ) / 0.99 )
        
        return y

Now we can perform the numerical maximization of the log-likelihood function using 'scipy.optimize.minimize' and defining the cost function as the negative of the log-likelihood.


.. code-block:: python

    def optimFunc( x ):
        # function to optmize, in this case the log-likelihood after transformation
        
        # Transform from unconstrained to constrained value
        y = optimTrans( x )
        nuCur = y[0]
        rCur = y[1]
        sigmaEpsCur = y[2]
        
        # Update current system
        femNew.updateSystem( childParams = {'r':rCur}, nu=nuCur, sigma=sigma, BCRobin = BCRobin )
        # Compute log-lik
        logLik = femNew.loglik( ZMeas, obsMat.tocsc(), sigmaEps=sigmaEpsCur)
        # Return minus log-likelihood
        return - logLik    
    
    
    from scipy import optimize
    
    # Set initial value
    x0 = optimTransInv( [ nu, r, sigmaEps ] )
    # Optimize ("BFGS")
    # resultOptim = optimize.minimize( optimFunc, x0, method='BFGS', options={'disp': True, "maxiter":20, "gtol": 1e-1} )
    resultOptim = optimize.minimize( optimFunc, x0, method='Nelder-Mead', options={'disp': True, "maxiter":200} )
    # Get result
    nuEst, rEst, sigmaEpsEst = optimTrans( resultOptim.x )    

The parameters were estimated to [nu = 3.6, r = 0.19, and sigmaEps = 0.20 ] as compared to [nu = 2.5, r = 0.2, and sigmaEps = 0.2 ].
The correlation range and noise standard deviation was almost exact while the smoothness parameter was a little bit off. 
It should be noted that for large values of :math:`\nu`, a small change in the smoothness does not change the covariance function significantly.
This is intuitive if you consider that :math:`\nu` correspond to the Hölder constant of the realizations. 
It is hard to see if a function is differentiable up to the fourth order or just the third order, while a smoothness of 1 is significantly different than 0.5.
Let us therefore compare the two Matérn correlations, the true and the estimated, to see how much difference it really makes to have :math:`\nu=3.6` instead of :math:`\nu = 2.5`.

.. code-block:: python
    
    plt.figure(1)
    plt.clf()
    # Plot realizations
    plt.plot( np.linspace(0.01, 0.4, 500), (GRF.MaternCorr( np.linspace(0.01, 0.4, 500), nu = nuEst, kappa = np.sqrt(8*nuEst)/rEst ) ), label="Estimated", color = "black", linewidth = 2 )
    plt.plot( np.linspace(0.01, 0.4, 500), (GRF.MaternCorr( np.linspace(0.01, 0.4, 500), nu = nuTrue, kappa = np.sqrt(8*nuTrue)/rTrue ) ), label="True", color="red" )
    plt.legend()



.. image:: https://drive.google.com/uc?export=view&id=1CaZi6iLEHjyisll3bc8e2nwfWH8U27E3
    :width: 80%
    :align: center
    
As can be seen, for smoothness values as large as 2.5 it is no visible difference to a value of 3.6. The log-likelihood of the estimated model is :math:`-23.01`, which is actually a little bit higher than the true model. 
    
    
    
    
    
    
    
Constructing the non-stationary model
--------------------------------------

Fieldosophy was created mainly to focus on modeling non-stationary random fields using the SPDE-approach.
So far in this tutorial we have only been looking at the regular Matérn model. 
However, the non-stationary model is an extension of the Matérn model derived from extending the stochastic partial differential equation of the Matérn covariance function.
Hence, since we now know the basics of modeling the Matérn model in one dimension using Fieldosophy, going non-stationary is a small step.

The non-stationary models are defined using the SPDE of :eq:`hildemanSPDE`.
As such, they are defined by the metric tensor (the :math:`G` matrix).
One intuitive way of looking at the non-stationary models is to remember that :math:`G` is a matrix which eigenvalues explains how much space should be compressed or expanded in order to get a Matérn covariance with :math:`\kappa = 1` (in the direction of the corresponding eigenvector).
Hence, if :math:`G` was a constant matrix with an eigenvalue of :math:`\frac{\sqrt{8\nu}}{r}`, this would mean that the correlation range in the direction of the corresponding eigenvector would be :math:`r`.
This is easy to see if you remember that the correlation range is :math:`\sqrt{8\nu}` when :math:`\kappa = 1`, and that :math:`\kappa` is inversely proportional to the correlation range.
Hence, an alternative way of creating the Matérn field of 'fem4' is by using the 'FEM.nonStatFEM' function of 'fieldosophy.GRF'.

.. code-block:: python

    def mapFEMParamsToG( params ):
        
        GInv = [ (params["r"] / np.sqrt(8*nu) )**2 ]
        logGSqrt = - 0.5 * np.log( GInv[0] )
        
        return (logGSqrt, GInv)
    
    # Create FEM object
    nonstatfem = FEM.nonStatFEM( mesh = extendedMesh, childParams = {"r":r, "f":mapFEMParamsToG}, nu = nu, sigma = sigma, BCRobin = BCRobin )

When defining the random field of class 'FEM.nonStatFEM', the 'childParams' dictionary should hold all parameters needed to construct G.
One can define a function that assembles these paramers and returns the inverse of G and the logarithm of the square root of the determinant of G.
Here, 'GInv' should be a list, each list element corresponding to an element in the G-matrix (flattened to one dimension).
The elements of 'Ginv', as well as 'logGSqrt', operate on an array the size of the number of simplices in the mesh. 
Hence, they either have to be scalars or arrays with size the number of simplices in the mesh.
In the stationary case they can be scalars since we want to use the same value for all simplices. 
However, when we actually want to define a non-stationary model we want the values to vary with the simplices.


Let us continue with our extended mesh from before, 'extendedMesh'.
This mesh was created as to allow for a correlation range of 0.4. 
When we speak about a non-stationary model the concept of correlation range become somewhat confusing. 
We earlier defined correlation range as the distance at which two points have a correlation of 0.1 (or in our parameterization 0.13).
For a non-stationary model such a definition does not make sense.
Instead, we will now define a non-stationary model based on what we can call *local correlation ranges*.
That is, we manipulate the :math:`G`-matrix for each simplex using :math:`r`, just as above. However, :math:`r` will vary between different simplices.

Let us in this example assume that we want short local correlation ranges close to 0 and long correlation ranges close to 1.

.. code-block:: python

    # Compute the middle point in each simplex 
    simplexMeanPoints = np.mean( extendedMesh.nodes[ extendedMesh.triangles , 0], axis=1 )
    # Use simplixes middle points to set the local correlation range
    rLocal = simplexMeanPoints - 0.1
    # Set the extensions to the original value of 0.4
    rLocal[simplexMeanPoints < 0 ] = 0.4
    rLocal[simplexMeanPoints > 1 ] = 0.4
    

The last two rows just make sure that the extension region still have the old value of 0.4. 
This to make sure that we do not get boundary artifacts close to the right extension region (which would otherwise have had longer local correlation ranges), and that we do not get infeasible or too small correlation ranges for our mesh on the left extension region.

Let us now define the non-stationary model using these local correlation ranges instead.


.. code-block:: python

    # Create FEM object
    nonstatfem = FEM.nonStatFEM( mesh = extendedMesh, childParams = {"r":rLocal, "f":mapFEMParamsToG}, nu = nu, sigma = sigma, BCRobin = BCRobin )

As can be seen, we could reuse our 'mapFEMParamsToG' function since it also works with array input.

To understand what this model is doing let us plot the local correlation ranges agains the middle value of each simplex and compare this with the covariance between the point at 0.5 and all other points.


.. code-block:: python

    # Get covariance between the point in the middle of the interval and all other points
    referenceNode = np.zeros((extendedMesh.N,1))
    referenceNode[450] = 1
    covSPDE = nonstatfem.multiplyWithCovariance( referenceNode )
    covSPDE = nonstatfem.multiplyWithCovariance( referenceNode )
    # Compare with actual matern covariance
    covMatern = sigma**2 * GRF.MaternCorr( np.abs(extendedMesh.nodes[450,0] - extendedMesh.nodes.flatten()), nu = nu, kappa = np.sqrt(8*nu)/r )
    
    
    # Plot covariances
    plt.figure(1)
    plt.clf()
    plt.plot( extendedMesh.nodes.flatten(), covSPDE, color="black", label="SPDE", linewidth=3)
    plt.vlines( extendedMesh.nodes[450,0], 0, 4, linestyle='--' )
    plt.title( "Covariance between point 0.5 and all other" )
    plt.plot( extendedMesh.nodes.flatten(), covMatern, color="red", label="Matern" )
    plt.legend()
    
    # Plot non-stationarity
    plt.figure(2)
    plt.clf()
    plt.plot( simplexMeanPoints, rLocal, color="black")
    plt.title( "Local correlation ranges" )


.. image:: https://drive.google.com/uc?export=view&id=1lot0WEDwcI5nHml7kvYUpSzOWq_v3jZA
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1RLqfOr9B6wiO0pYWYzJ4pTHezs0kCwp6
    :width: 49%

As seen, the covariance on the left side of 0.5 drops off much faster than the original Matérn covariance.
On the right side we see the opposite.

Let us also plot two realization to see what type of behavior this actually correspond to among realizations from this non-stationary random field.


.. image:: https://drive.google.com/uc?export=view&id=1JZyUpvzZWcQn9iNnZw3t-X-tz1aDEQhG
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1RHUE1DBvULml0vuMF87VKPILIqDDQ8CE
    :width: 49%

On the left side we see more dramatic behavior since the original scales are compressed to a shorter region (0.1 instead of 0.4).
On the right side we see slowly varying ups and downs, since here, the original scales are instead elongated.

