.. _Introduction:

Introduction
============

The Fieldosophy package provides tools for working with random fields. 
Most of its functionality is based around describing a certain class of Gaussian random fields (GRF) as a solution to a specific stochastic partial differential equation (SPDE). 
The main advantage of such models is that they can be defined on complicated domains and manifolds, more complex than just :math:`\mathbb{R}^d`, while at the same time allow for easy and intuitive parameterization.
Most compuations regarding these GRFs can also be computed efficiently using a finite element method (FEM) approximation of the SPDE.



Gaussian random fields
-----------------------
A Gaussian random field can be seen as a collection of Gaussian random variables, each random variable exclusively associated with a point in "space". 
In this context "space" being a topological space, often a subset of a manifold embedded into :math:`\mathbb{R}^d`.
For example, let us denote a GRF as :math:`X \equiv \{X(\boldsymbol{s})\}_{\boldsymbol{s} \in \mathcal{D}}`. Here, :math:`X` is the random field, :math:`\mathcal{D}` is the topological space and :math:`\boldsymbol{s}` is a point in :math:`\mathcal{D}`.

Since a GRF is a collection of Gaussian random variables and a multivariate Gaussian random variable is defined solely by its first and second moments, the GRF can be completely characterized by its first-order mean function, :math:`\mu(\boldsymbol{s})`, and its second-order covariance function, :math:`\mathcal{C}(\boldsymbol{s}_1, \boldsymbol{s}_2)`. 
That is, given any finite set of points :math:`\mathcal{I} \subset \mathcal{D}`, the probability distribution of the associated random variables is a multivariate Gaussian distribution characterized by a mean vector, :math:`\boldsymbol{\mu} := \{\mu(\boldsymbol{s}_i)\}_{i \in I}`, and a covariance matrix, :math:`\Sigma := \{\mathcal{C}(\boldsymbol{s}_i, \boldsymbol{s}_j)\}_{i,j \in I}`.
It should be noted that :math:`\mathcal{D}` does not have to be a finite set itself. In fact, in Fieldosophy we consider continuously indexed Gaussian random fields, i.e., :math:`\mathcal{D}` is a continuous space with an uncountable number of points. 
As such, we can view the GRF as a random function in :math:`\mathcal{D}`, i.e., each realization of :math:`X` is a function in :math:`\mathcal{D}`.



.. image:: https://drive.google.com/uc?export=view&id=1r5mqn0BoO7co6WG2puE3FEEzRQr9LMbV
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1x5E_y9xJYu51FGOAN3mm0RbPQgzqthuD
    :width: 49%

As an example, above are two different realizations from the same GRF. 
This particular GRF being defined on the interval :math:`[0,1]` of the real line.

.. image:: https://drive.google.com/uc?export=view&id=12Y-KKa27H5_EBxFveiKwHSwipNxL5peg
    :width: 49%

.. image:: https://drive.google.com/uc?export=view&id=1_TUd2U409F8SHWaQBrs1YSmLqaZzSE3q
    :width: 49%

Just as a one-dimensional random field on the unit interval will have realizations being real-valued functions on the unit interval. A two-dimensional random field on the unit rectangle will have realizations being two-dimensional functions on the unit rectangle, see figures above.

So as can be seen in both the one-dimensional and two-dimensional examples above, different realizations do not look exactly the same but tend to have similar "qualities". 
These qualities depend on the random field model and the difference among the realizations depend on the probabilistic nature of the random fields. 


Purpose
--------

So what is the point of Gaussian random fields? Gaussian random fields is a class of models for the probability distribution of spatial data (remember that longitudal data such as time-series is a special case of spatial data). 
In applications, where they have spatial data, GRFs are typically used for one or several of the following tasks:

* Analyze the probability of joint events occuring.
    Often, an important event is characterized by the values at several points in space being in certain ranges at the same time, ie., joint events. 
    In these cases, it is not enough to analyze the probability distribution on each site alone since there is dependency among them. 
    The GRF is a joint probability distribution of all points in :math:`\mathcal{D}`, hence, it can be used to assess the probability of such joint events.
* Predict the value at points in space given knowledge of the value at other points in space.
    This task goes under many names depending on the application, e.g., forecasting, Kriging, interpolation, estimation, conditional prediction. 
    Using GRFs for this purpose is a key element in disciplines such as, meterology, finance, geoscience.
    
    .. image:: https://drive.google.com/uc?export=view&id=1ystcfaoF-k8KsYZTTKS1mjveZW9s9KWZ    
        :width: 49%

    .. image:: https://drive.google.com/uc?export=view&id=1umED1GBUhGAX8PvDCjUXo2NF6wRwsCwc
        :width: 49%
        
    The above figures show the same example of two realizations from a Gaussian random field on the unit interval as before. 
    However, now the true value of the time series has been observed at three distinct points (red dots). 
    By knowing the random field model and the value at these three points it is possible to acquire the conditional probability distribution for the value at all other points in the unit interval.
    The blue curve corresponds to the conditional mean (being a possible choice of a point prediction given the observed data) and the green regions being the conditional, pointwise, 90% prediction interval (it is a 90% probability that the true value at a point is inside the green region). 
    As can be seen, close to the observations there is almost no uncertainty. Further away the uncertainty increases.
    
* Simulate from the GRF.
    A very strong tool when the interest lies not in assessing the value at the points of the random field directly, but rather that these values are input to another process. 
    By performing such simulations (Monte-Carlo) it is possible to "push" the probability distribution of the underlying random field to generate a probabilistic analysis of the end process, which might have a complicated and highly non-linear dependency on :math:`X`.
* Estimate the values of parameters of the GRF model from data.
    The value of the parameters can be interesting in their own since they might explain important behaviors in the studied process. However, what is even more common is that the parameters need to be estimated as a first step before performing tasks such as conditional prediction or simulation.
* Compare several GRF models to data to analyze which model that does the best job of explaining the process studied.
    Sometimes, the interest lies in comparing several models. Either to see how much complexity that needs to be added to explain a phenomena, or to choose which theory that seems to comply best with observations. Comparisons can be performed in many ways and Fieldosophy allow for evaluation of the likelihood function as well as conditional simulations in order to asses model fit.

For the classes of GRFs that Fieldosophy can handle, it is possible to perform all the above tasks using a computationally efficient methodology.


Why Gaussian random fields?
---------------------------

So far we know what a Gaussian random field is and some examples of what it can be used for. 
One strong assumption that is made when using GRFs is the assumption of Gaussianity. 
If we remove this assumption we have a general random field, i.e., a collection of random variables indexed by their associated points in space.
The joint distribution of this collection of random varibles is not neccessarily Gaussian for a general random field.
There are several reason why we focus solely on Gaussian random fields in this package, and not other probability distributions.

#. Less complex
    Multivariate Gaussian distributions are solely defined by their first and second moments. 
    That means that we only need to care about the mean function and covariance function. 
    Although that can be complicated enough, it is a dramatic restriction compared to allowing flexibility in higher order moments.
    Another important, and often forgotten, reason is due to data scarcity. 
    More complex models require more observed data in order to be estimated properly. 
    This relationship can often scale super-linearly since the number of required parameters for adding further moments often depend on the dimensionality of :math:`\mathcal{D}`.
    In many applications data scarcity is a strong restriction. Data is expensive to generate and since there is dependencies in space and/or time, a lot of data points might be needed to acquire an adequate effective sample size.
    Furthermore, the computational complexity of working with more complex random field models are often increased considerably. 
#. Central limit theorem
    Due to the central limit theorem, the probability distribution of a sum of independent and identically distributed random variables converges towards a Gaussian distribution. This is also true for random fields, i.e. a sum of random fields are converging towards a Gaussian random field if they are independent and identically distributed. 
#. Flexible
    Even when the studied process is not a sum of many independent and identically distributed random fields, it can often be approximated as a GRF. 
    The GRF can model the mean and correlation between the values at the different points. 
    This is often enough for conditional predictions and probability assessment in the "bulk" of the distribution, i.e., when very rare events are not of concern.
    However, one should remember that there are certainly cases when a GRF is not an appropriate approximation. Typical such cases are multimodal distributions, highly skewed distributions, and distributions with compact support on intervals such that the Gaussian approximation would yield rather high probabilities outside of their support.
#. Marginal transformation
    In the case when a GRF is not a good approximation of the actual random field, it is often possible to marginally transform it into Gaussianity.
    The idea is simple, consider the marginal distribution for the points in space separately. 
    Look to find a good mapping such that the marginal distributions become Gaussian. 
    Map all data to Gaussianity and model them with a GRF.     
    If separate mappings should be considered for each point in space, or if all points in space can use the same mapping, depend on the data and its scarcity. 
    One of the most important special cases of marginal transformations is the log-transformation. This transformation is important since it is used vastly in, for instance, finance. It is particularly useful since multiplications become additions after log-transformation. In other words, multiplicative noise becomes additive noise and additive noise converges to Gaussianity. Hence, it can be used to model processes with multiplicative noise using GRFs.
    


    




The stochastic partial differential equation
--------------------------------------------

The class of continuously-indexed Gaussian random fields that are of concern to Fieldosophy can be described by the stochastic partial differential equation,
    
.. math::
    \mathcal{L}^{\beta} \left(\tau X\right) = \mathcal{W}.
    :label: generalSPDE
    
Here, :math:`\mathcal{L}` is a differential operator on :math:`\mathcal{D}`, :math:`\beta > \frac{d}{4}` is a real valued constant denoting the order of the (fractional) derivative of the differential operator, :math:`\tau` is a positive scalar-valued function controlling the marginal variance of :math:`X`, and :math:`\mathcal{W}` is the (generalized) field of Gaussian white noise on :math:`\mathcal{D}`.
Since :math:`\mathcal{W}` is a GRF, the solution to the SPDE, i.e. :math:`X`, will be a GRF. In this sense, we can consider :math:`X` as being a mapping of :math:`\mathcal{W}`, i.e., 

.. math::
    X = \frac{1}{\tau} \mathcal{L}^{-\beta} \mathcal{W}.

Note that the solution is not well-defined until appropriate boundary and/or initial conditions are given, as well as in what sense we have defined a solution.
These are technicalities and for an in-depth understanding of the details please read :cite:`lit:lindgren, lit:hildeman1, lit:bolin2` and the references there within.


Matérn covariance
-----------------

The original differential operator of :cite:`lit:lindgren` is

.. math::
    \mathcal{L} := \left( \kappa^2 - \Delta \right),
    :label: lindgrenSPDE
    
where :math:`\kappa` is a real-valued scalar and :math:`\Delta` is the Laplacian operator. When this differential operator is used in equation :eq:`generalSPDE` and :math:`\mathcal{D} = \mathbb{R}^d` with the Euclidean metric, :math:`X` will be a GRF with zero mean function and a Matérn covariance function. 
The Matérn covariance function has the form, 

.. math::
    \mathcal{C}(\boldsymbol{s}_1, \boldsymbol{s}_2) = \sigma^2 \mathcal{C}_{\nu}\left( \kappa \|\boldsymbol{s}_1 - \boldsymbol{s}_2\| \right) \\
    
where, :math:`\sigma^2` is the marginal variance, and

.. math::
    \mathcal{C}_{\nu}(h) = \frac{h^{\nu}K_{\nu}(h)}{2^{2\nu -1} \Gamma(\nu)}.
    
Here, :math:`K_{\nu}` denotes the modified Bessel function of the second kind and :math:`\beta = \frac{\nu}{2} + \frac{d}{4}`.
For this equation, the relationship between the marginal variance and the :math:`\tau`-parameter is,

.. math::
    \sigma = \sqrt{\frac{\Gamma(\nu)}{\Gamma\left( \nu + d/2 \right) (4\pi)^{d/2} }} \frac{1}{\kappa^{\nu}\tau},  
    
and :math:`d` is the dimensionality of :math:`\mathcal{D}`.   

    
The Matérn covariance is very popular in spatial analysis. 
Mostly since it is quite flexible while yielding a positive definite covariance matrix when applied to any arbitrary number of distinct points in :math:`\mathbb{R}^d`; a necessary condition for any covariance function on :math:`\mathbb{R}^2`.
It also adheres to Tobler's first law of philosophy, viz., "Everything is related to everything else, but near things are more related than distant things".
All this while having only three easily interpreted scalar-valued parameters.


* The :math:`\kappa`-parameter controls the correlation range. 
    The correlation range is the distance between two points at which their correlation becomes lower than 10%. 
    However, :math:`\kappa` is not equal to the correlation range, but is has a one-to-one relationship with it. 
    A good approximation is that the correlation at distance :math:`\frac{\sqrt{8\nu}}{\kappa}` is 13% :cite:`lit:lindgren`.
* The :math:`\nu`-parameter controls the smoothness of realizations from the random field. 
    In fact, :math:`\nu` is equivalent to the Hölder constant, almost everywhere, of realizations for the GRF.
    This is an important parameter since a higher smoothness means that, for a fixed correlations range, the correlation becomes higher for short distances but drops off faster. 
    In that sense it can be seen as a shape parameter of the covariance function.
    :ref:`fig-maternExample` highlight how the shape is changing when changing the :math:`\nu`-parameter.
* The :math:`\sigma` parameter controls the marginal standard deviation.
    The standard deviation for any fixed point in :math:`\mathcal{D}`. 

.. _fig-maternExample:

.. figure:: https://drive.google.com/uc?export=view&id=1AEyEXBQe95d3AwMfhtJqp8Bg0o1DaJ1b

    Figure 1 

It might be hard to get an intuition of the covariance function by the equations above. 
In :ref:`fig-maternExample` two different Matérn functions are shown. 
These two differ by the value of :math:`\nu`, while having unit correlation range and marginal variance (:math:`r=1`). 
Changing the :math:`\kappa`-parameter would scale the x-axis, while changing the :math:`\sigma`-parameter would scale the y-axis.
The black dashed line just show the correlation range. 
As can be seen, the blue curve, with a smaller :math:`\nu`-parameter than the red curve, has comparably smaller correlation for distances shorter than the correlation range but higher correlation for longer distances.

It is important to realize that the Matérn covariance function is diminishing with distance between the two points. 
This implies that points close to each other will be more similar than points far away. 
A property that often holds in real life phenomena.  

Two important special cases of the Matérn covariance is the Gaussian covariance function and the exponential covariance function. 
It should be noted that the Matérn covariance function has some further theoretically attractive properties, see :cite:`lit:stein`. 




Extending the Matérn covariance
--------------------------------

The Matérn covariance function is restricted to a limited set of use cases. 

#. It is stationary
    The covariance only depend on the relative position between the two points.
#. It is isotropic
    The covariance does not depend on the angle between the two points.
#. It is only defined on :math:`\mathcal{D} \subseteq \mathbb{R}^d`. 
    Many real-world processes are observed on manifolds such as the surface of the earth or on a curve in space.

However, the SPDE of :eq:`generalSPDE` using the operator :eq:`lindgrenSPDE` can still be defined on general Riemannian manifolds. 
Although its solution will no longer have a Matérn covariance function, the resulting covariance function will keep many of the attractive properties of the Matérn covariance (control over smoothness, correlation range, and marginal variance as well as enforcing that points nearby should be more correlated than points far away).
Through this SPDE-trick we therefore extend the Matérn covariance function to a richer class of covariance functions.

In fact, it can be hard to explicitly define a covariance function at all on arbitrary Riemannian manifolds. 
Using stochastic partial differential equations give us a general approach to implicitly construct attractive covariance functions on complicated Riemannian manifolds.



Moreover, considering that a Riemannian manifold is made up of a diffeomorphism together with a metric, one can also change metric while staying on a subset of :math:`\mathbb{R}^d`.
By considering different metrics, it is possible to acquire non-stationary and anisotropic covariance functions by studying how the differential operator of :eq:`lindgrenSPDE` changes under a change of metric. 
This was done in :cite:`lit:hildeman1`, the result being the differential operator,

.. math::
    \mathcal{L} := |G|^{\frac{1}{4\beta}} \left( I - \frac{1}{\sqrt{|G|}} \nabla \cdot \left( \sqrt{|G|}G^{-1} \right) \nabla \right).
    :label: hildemanSPDE

Here, :math:`I` is the identity operator, :math:`\nabla \cdot` the divergence operator, and :math:`\nabla` the gradient operator.
The matrix-valued function, :math:`G`, defines a metric in the following way:
A metric is defined by an inner product in each point of :math:`\mathcal{D}`, i.e., :math:`g[\boldsymbol{v}_1, \boldsymbol{v}_2](\boldsymbol{s})` where :math:`\boldsymbol{v}_1, \boldsymbol{v}_2` are tangent vectors of the manifold in point :math:`\boldsymbol{s}`. 
Considering manifolds embedded in :math:`\mathbb{R}^D`, for some :math:`D`, such an inner product can be represented by a positive definite matrix-valued function, :math:`G`, operating on the natural basis vectors, i.e., :math:`g[\boldsymbol{v}_1, \boldsymbol{v}_2](\boldsymbol{s}) := \sqrt{\boldsymbol{v}_1^T G(\boldsymbol{s}) \boldsymbol{v}_2}`.
In other words, :math:`G` is a matrix-valued function describing the deviation between the metric defined by :math:`g` and the "natural" metric induced on :math:`\mathcal{D}` by considering a euclidean metric on :math:`\mathbb{R}^D`, in which :math:`\mathcal{D}` is embedded. 

It should be mentioned that the equation corresponding to eq. :eq:`hildemanSPDE` in :cite:`lit:hildeman1` looks slightly different. 
This is because in :cite:`lit:hildeman1` the equation was defined with respect to the Jacobian matrix of the mapping from using the "natural" metric of the manifold together with the differential operator of :eq:`lindgrenSPDE` with :math:`\kappa = 1`.
In short, this means that using the differential operator of eq. :eq:`hildemanSPDE` in :eq:`generalSPDE` is equivalent to considering the covariance function induced by the differential operator :eq:`lindgrenSPDE` but changing the metric from the "natural" metric of :math:`\mathcal{D}` to :math:`g`.

In the way the Riemannian manifold and its corresponding differential operator, :eq:`hildemanSPDE`, is defined, the parameter :math:`\nu = \frac{4\beta - d}{2}` is still equivalent to the Hölder constant of realizations from the GRF.
Also, the parameter :math:`\tau` alone controls the marginal variance by the relationship,

.. math::
    \sigma = \sqrt{\frac{\Gamma(\nu)}{\Gamma\left( \nu + d/2 \right) (4\pi)^{d/2} }} \frac{1}{\tau}. 

Comparing the original differential operator of the Matérn covariance, :eq:`lindgrenSPDE`, with the more versatile :eq:`hildemanSPDE`, the :math:`\kappa` parameter of :eq:`lindgrenSPDE` has been replaced by a matrix valued :math:`G`-function.
This function now allow for anisotropy (when eigenvalues are not all equal) and non-stationarity (when :math:`G` is not constant in space). 

As an example, in the special case when :math:`\mathcal{D} = \mathbb{R}^d` and :math:`G = c I` (that is, a constant scaling of the identity matrix), this model induces a Matérn covariance with parameters:

.. math::
    \sigma &= \sqrt{\frac{\Gamma(\nu)}{\Gamma\left( \nu + d/2 \right) (4\pi)^{d/2} }} \frac{1}{\tau}, \\
    \nu &= \frac{4\beta - d}{2}, \\
    \kappa &= \sqrt{c}.
    

 

Finite element approximations
------------------------------

We now know that we can use the stochastic partial differential equation of :eq:`generalSPDE` together with the differential operator of :eq:`hildemanSPDE` to model a wide range of spatial correlation structures on arbitrary Riemannian manifolds.
The last piece of the puzzle is approximating the solution of these SPDEs using the finite element method (FEM). 

The benefit of the finite element method is twofold:

#. For arbitrary regions of general Riemannian manifolds, the solution is not known explicitly. Instead, we need to approximate the solution numerically. 
    The finite element method is one such approximation. Moreover, with FEM it is possible to have control over the approximation error and acquire a solution that is continous in space.
#. With FEM, using basis functions that are mostly orthogonal to each other yields a precision matrix (inverse of the covariance matrix) that is sparse. 
    The sparsity of the precision matrix is key to reducing the computational complexity of most operations that might lie in our interest, see :cite:`lit:lindgren, lit:rue`.

The first step of utilizing the finite element method is to rewrite the differential equation into weak form, i.e., the SPDE of :eq:`generalSPDE` (when :math:`\beta = 1`) becomes,

.. math::
    \left< \mathcal{L}\left( \tau X \right), \phi \right> = \left< \mathcal{W}, \phi \right>, \forall \phi \in \mathcal{V}.

The weak form solution considers the inner product between the left hand side of :eq:`generalSPDE` and an arbitrary member, :math:`\phi`, of function space :math:`\mathcal{V}`, to be equal to the inner product between the right hand side of :eq:`generalSPDE` and the very same function, :math:`\phi`.
The weak form is a actually the "correct way" of interpretating :eq:`generalSPDE` since the Wiener noise, :math:`\mathcal{W}`, does not have pointwise meaning and is only defined in weak form.
For the differential equations of :eq:`lindgrenSPDE` and :eq:`hildemanSPDE` with :math:`\beta=1`, :math:`\mathcal{V}` is usually considered to be the Sobolev space :math:`W^{1,2}`. 
Here, :math:`W^{1,2}` denoting the space of functions for which both the function and its derivative (in any direction) is bounded in :math:`L^2`-sense.
When this function space is used, also the solution, :math:`X`, is considered to be part of :math:`\mathcal{V}`.

The finite element method relaxes the requirements on the solution by just saying that :math:`X, \phi \in \hat{\mathcal{V}}`, where :math:`\hat{\mathcal{V}} \subset \mathcal{V}`. 
In fact, since the numerical approximation of the solution has to be possible to compute with a computer in finite time, :math:`\hat{\mathcal{V}}` is chosen as a finite dimensional function space that is practically manageable.
Although many choices exists for :math:`\hat{\mathcal{V}}`, in Fieldosophy only piecewise linear functions on simplices are considered.
This is the same as saying that we are looking for the solution which approximates the true solution best while being a piecewise linear function. 

Part of FEM is to choose a mesh over :math:`\mathcal{D}`. That is, divide the spatial domain into many simplices such that each simplex is small enough such that the linear approximation in the given simplex is a good approximation. At the same time, the finer the simplicial mesh, the higher the computational burden. 
Therefore, the trick is to make the simplices small enough, but not smaller than so.

Given such a simplicial mesh with :math:`N` nodes (points connecting lines in the simplicial mesh), the FEM function space, :math:`\hat{\mathcal{V}}`, is :math:`N`-dimensional.
This function space is :math:`\hat{\mathcal{V}} = \{ \phi_i\}_{i=1}^N`, where each :math:`\phi_i` is linear in all simplices for which the :math:`i`:th node is a member of, and zero in all other simplices.
The relaxed weak solution can then be described as a system of linear equations, ie.,

.. math::
    \sum_{i = 1}^N x_i \left< \mathcal{L}\left( \tau \phi_i \right), \phi \right> = \left< \mathcal{W}, \phi \right>, \forall \phi \in \{\phi_j\}_{j=1}^N \Leftrightarrow K \boldsymbol{x} = W.

Here, :math:`\hat{X}(\boldsymbol{s}) := \sum_{i=1}^N x_i \phi_i(\boldsymbol{s})`, is the FEM approximation of the real solution :math:`X`.
Since the inner product between two basis functions, :math:`\phi_i, \phi_j`, are only nonzero when the nodes are part of the same simplex, :math:`K` is a sparse matrix.
This can be leveraged to accomplish a Cholesky decomposition of the precision matrix with a reduced computational complexity :cite:`lit:lindgren`. 
The computational complexity is :math:`\mathcal{O}(N)` in one dimension, :math:`\mathcal{O}(N^{3/2})` in two dimensions, and :math:`\mathcal{O}(N^{5/2})` in three dimensions.
This should be compared to the general computational complexity of :math:`\mathcal{O}(N^3)` for a Cholesky decomposition.

.. image:: https://drive.google.com/uc?export=view&id=1Scvs8DUKzeUJEENd7YA1w_fbrZXgg__g
    :align: center

The mesh above was used in the two-dimensional examples on the unit rectangle shown earlier in this chapter.
In two dimensions, a simplex is a triangle and the mesh is a collection of triangles that completely covers the spatial domain. 

Remember that the solution to :eq:`generalSPDE` is not fully defined without boundary conditions.
That means that different boundary conditions will correspond to different covariance functions for the random field model, even though the differential operator is the same.
The original theory of :eq:`lindgrenSPDE` as being the differential operator corresponding to a Matérn covariance function only holds when :math:`\mathcal{D} = \mathbb{R}^d`.
However, meshing over all of :math:`\mathbb{R}^d` is infeasible and in real-life applications we are only interested in a compact domain thereof. 
Hence, the mesh is often *extended* such that points inside our :math:`\mathcal{D}` have a neglible correlation with points at the boundary of the mesh. 
This effectively removes the ambivalence due to boundary effects since they do not affect the region of interest anyway.

Such a *mesh extension* can be seen in the above figure, where the mesh extends out 0.6 in all directions from the unit rectangle. 
This is a *mesh extension* that was added such that the FEM approximation of :eq:`generalSPDE` with differential operator :eq:`lindgrenSPDE` will behave as a Matérn covariance function inside domain :math:`\mathcal{D}`.


It should be noted that there are ocassions when we do not want to extend the mesh. 
As been noted earlier, one of the advantages with expressing the covariance function implicitly through a differential equation is that it can be modified to more complex models/spaces while still maintaining many of the good properties of the Matérn covariance.
One such extension is to define appropriate boundary conditions for the specific problem. Sometimes one knows that the value at the boundary should be zero, then a homogeneous Dirichlet boundary condition is the correct choice, while mesh extension would lead to an unwanted solution.
Similarly, when modeling a random field on a periodic manifold, such as a sphere, a periodic boundary condition should be used. 
Fieldosophy support Dirichlet, Neumann, Robin, and periodic boundary conditions. The different boundary conditions can be mixed as well, i.e., having one boundary condition on one part of the boundary domain and another boundary condition on another part.





Rational finite element approximations
----------------------------------------


The section above explained how to approximate the solution to :eq:`generalSPDE` when :math:`\beta = 1`. What about the cases when :math:`\beta \neq 1`?

Above we referred to :math:`\beta` as the fractional derivative of the differential operator. 
This is most easily understood as the power of the eigenvalues for the eigendecomposition of :math:`\mathcal{L}`. 
Given appropriate boundary/initial-conditions, a differential operator can be characterized by a sum of eigenvalue/eigenfunction pairs. That is, if

.. math::
    \left( \mathcal{L} X \right)(\boldsymbol{s}) = \sum_{i=1}^{\infty} \lambda_i \langle X, \psi_i \rangle \psi_i(\boldsymbol{s}),

then the fractional derivative, :math:`\beta`, is defined as

.. math::
    \left( \mathcal{L}^{\beta} X \right)(\boldsymbol{s}) := \sum_{i=1}^{\infty} \lambda_i^{\beta} \langle X, \psi_i \rangle \psi_i(\boldsymbol{s}).

However, the eigendecomposition of the differential operators are generally unknown on arbitrary Riemannian manifolds, and even when known they might not give the computational benefits of the finite element method. 
In the case of integer valued :math:`\beta` it is possible to interpret it as an iterative differentiation. In this way one can construct iterative finite element solutions such as in :cite:`lit:lindgren`.
This idea was further extended to fractional derivatives in :cite:`lit:bolin2` by using a rational approximation.
The idea here is to find two polynomials, :math:`P` and :math:`Q`, such that :math:`x^{\beta} \approx P(x)/Q(x)`. This is important since then,

.. math::
    \left( \mathcal{Q}^{-1} \mathcal{P} X \right)(\boldsymbol{s}) = \sum_{i=1}^{\infty} \frac{P(\lambda_i)}{Q(\lambda_i)} \langle X, \psi_i \rangle \psi_i(\boldsymbol{s}) \approx \sum_{i=1}^{\infty} \lambda_i^{\beta} \langle X, \psi_i \rangle \psi_i(\boldsymbol{s}).
    
Here, :math:`\mathcal{P} := \sum_{j=1}^{k} a_j \mathcal{L}^{j}` and :math:`\mathcal{Q} := \sum_{j=1}^{l} b_j \mathcal{L}^{j}`, where :math:`\{a_j\}_j^k` and :math:`\{b_j\}_j^l` are the coefficients for the two polynomials :math:`P` and Q respectively.

What is actually happening is that the non-integer values of :math:`\beta` are handled by using a summation of integer valued powers; which can be handled using the iterative finite element solution as in :cite:`lit:lindgren`.


When using higher order iterative finite element solutions, the computational complexity increases while the numerical stability decreases. 
Therefore, Fieldosophy uses polynomials with, at most, 2 degrees for one polynomial and 1 degree for the other. That is, either 

.. math::
    P(x) &= a_0 + a_1 x + a_2 x^2 \\
    Q(x) &= b_0 + b_1 x,

or

.. math::
    P(x) &= a_0 + a_1 x \\
    Q(x) &= b_0 + b_1 x + b_2 x^2.

In Fieldosophy you only have to specify the :math:`\nu`-parameter (:math:`\nu := 2\beta - \frac{d}{2}`) and the rational approximation is computed automatically for optimal performance.





