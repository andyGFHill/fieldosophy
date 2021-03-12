Structure of Fieldosophy
=========================


The Fieldosophy package provides tools for Gaussian random fields.
It is composed of four different sub-packages. 

The GRF sub-package
------------------------

The GRF sub-package is concerned with operations with or on random fields. 
It is itself made up of four different modules:

#. The Cholesky module
    This module is used for factoring a sparse (and assumably large) matrix by the Cholesky decomposition. It involves functions for storing, decomposing, permuting, multiplying and solving the matrix. 
    It includes both an inhouse low-level implementation using the Eigen-library and a wrapper for the **scikit-sparse** Cholesky decomposition.
    The inhouse implementation is deprecated since it is outperformed by the scikit-sparse counterpart, and since the Eigen-library does not allow using its Cholesky factorization under the MPL2 license.
    
#. The FEM module
    This module holds the functionality related to the finite element approximation of a Gaussian random field defined by the stochastic partial differential equation of :eq:`generalSPDE`.
    This means functionality for, defining various versions of the SPDE model, conditioning the model on observed data (possibly under noisy measurements), compute the likelihood function, compute correlations, generate data from the model. 
    
#. The GRF module
    This module includes more general functionality regarding Gaussian random fields. Here resides covariance functions such as the Matérn and anisotropic Matérn.




The marginal sub-package
------------------------------

The marginal sub-package is concerned with pointwise modeling of probability distributions. This sub-package is mainly included in Fieldosophy in order to transform non-Gaussian data into Gaussianity such that it can be modeled with GRFs.
It is made up of two modules:

#. The boxcox module
    Includes functionality for transforming data to (and from) Gaussianity using the Box-Cox transform. 
    
#. The NIG module
    This module includes functionality for working with the normal-inverse Gaussian distribution. 
    This includes functionality for computing the pdf, CDF, and quantile-function, as well as functionality for efficiently approximating the CDF and quantile-functions. 
    The latter being important since those functions can be used to transform data into (and from) Gaussianity.
    This module is included in Fieldosophy since the NIG distribution is more flexible than the Gaussian one and can therefore be used for transforming data into Gaussianity such that it can be modeled with GRFs.
    The module also includes functionality for fitting the NIG-distribution to data (parameter estimation). Also this important for marginal transformation to Gaussianity.


The mesh sub-package
---------------------------

The mesh sub-package is dedicated to functionality for generating and working on meshes (mainly simplicial meshes). 
The mesh is an integral part of the finite element method. However, meshing a spatial domain can be useful for many other purposes as well. 

    



The misc sub-package
---------------------------

This sub-package holds functionality that is necessary, or convenient, and does not really fit into the categories of the other sub-packages.
So far it includes functionality for visualizing random fields in three dimensions, template matching in order to find velocity fields, rational approximations of arbitrary functions, and cosine basis for function-valued parameters.



