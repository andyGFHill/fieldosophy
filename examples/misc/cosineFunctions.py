#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the cosine basis.


This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import  numpy as np
from matplotlib import pyplot as plt
from fieldosophy.misc import misc_functions as misc



boundingBox = np.array([[0,1], [0,1]])



# %% Plot sinus basis



x = np.linspace(boundingBox[0,0], boundingBox[0,1],100)
y = np.linspace(boundingBox[1,0], boundingBox[1,1],100)
x,y = np.meshgrid(x,y)
z = misc.cosinusBasis( np.concatenate(( x.reshape((-1,1)), y.reshape((-1,1)) ), axis=1), \
   np.array([ [0,0,0,1]]).transpose(), boundingBox )
z = z.reshape(x.shape)

fig = plt.figure(1)
plt.cla()
plt.title( "Sinusoid" )  

plt.plot(x[1,:], z[1,:])



