#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.
"""

# from distutils.core import setup, Extension
from setuptools import setup

NAME = "Fieldosophy"
VERSION = "0.1"




# Setup package
setup( name = NAME,
      version = VERSION,
      description = "Python package for working with Gaussian random fields.",
      author = "Anders Gunnar Felix Hildeman",
      author_email = "fieldosophySPDEC@gmail.com",
      packages = ['fieldosophy', 'fieldosophy.GRF', 'fieldosophy.mesh', 'fieldosophy.marginal', 'fieldosophy.misc'],
      # data_files = [ ('' , ['libraries/libSPDEC.so']) ],
      package_data = {"": ["libraries/*.so"]},
      install_requires=[ 'numpy', 'scipy', 'matplotlib', 'scikit-sparse', 'meshio' ],
      include_package_data=True )
