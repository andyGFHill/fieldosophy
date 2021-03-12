.. SPDEC documentation master file, created by
   sphinx-quickstart on Wed Sep 23 08:59:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fieldosophy: a toolkit for random fields!
====================================================

.. figure:: https://drive.google.com/uc?export=view&id=17fSqlCPBd06zf0jM2ghjKJdPrpoQUyqN

    
If you are accessing this documentation from a local copy, the most up-to-date version of the Fieldosophy package is available online at https://github.com/andyGFHill/fieldosophy and the complementary documentation can be accessed by https://andygfhill.github.io/fieldosophy
    
What is **Fieldosophy**?
-------------------------

The Fieldosophy package is a toolkit for random fields. 
As such the package provides functionality for prediction, parameter estimation, uncertainty quantification, and data generation in the context of spatially dependent data. 
Spatial here referring to data that is observed at different points in some, typically continuous, space.

Random fields can be thought of as random functions. 
A random field model is a model describing the probability distribution of such a random function.
These models are useful since they can be used to analyze data and explain observed processes. 
Specifically, they are important since many processes observed in real-life are not deterministic. 
The origin of this randomness is most often a lack of information, either information that was too costly to acquire or information for which its relevance to the observed process was unknown.
In either case, the unexplained variability can be accounted for by a probabilistic model. 

A random field model can, for instance, be used to compute probabilities for certain events associated to values at several points in a "space" at the same time. 
Typical applications are, but not limited to, finance, meterology, and biology where the "spaces" are either an interval of time, regions of the surface of the earth, or a region of the three-dimensional atmosphere. 
However, most of the functionality provided by **Fieldosophy** can also be applied on arbitrary Riemannian manifolds.

Computing probabilities and other important quantities from a random field model is generally quite complicated and demand a high computational cost. 
The **Fieldosophy** package provides important tools for making these tasks easier and quicker.



Contact
--------

If you have questions, suggestions, want to make a contribution, or want to get into contact with the Fieldosophy project of any other reason, do not hesitate to contact: fieldosophyspdec@gmail.com.



Table of content
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   introduction
   structure
   tutorials
   api_reference
   bibliography



.. #automodule:: spdec.GRF.FEM

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
