# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 4

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Test the FlatSurf class for a flat surface by propagating a beam.

Notes
-----
Testing passed.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# initialise a beam to be imaged
beam = ort.Beam([-50, 0, 0], [1, 0, 2], 4, 10, profile='hline')

# initialise a screen to be used for imaging
screen = ort.Screen([0, 0, 200], [0, 0, -1])

# check initialisation of the FlatSurf class
surf = ort.FlatSurf([0, 0, 100], [2, 0, -3], 1, 2, 10, color='green')

# check the representation method
surf

# check string method
print(surf)

# check the propagate() method
surf.propagate(beam)
screen.intercept(beam)


# check the show() method
beam.show()
surf.show()
screen.show()
plt.show()

# check what happens if part of beam misses the flat surface
beam = ort.Beam([-10, 0, 0], [0, 0, 1], 4, 10, profile='hline')
surf.propagate(beam)
screen.intercept(beam)
beam.show()
surf.show()
screen.show()
plt.show()

#%% check exception for initialisation with wrong shape of p

surf = ort.FlatSurf([0, 0, 100, 0], [0, 0, -1], 1, 2, 10)

#%% check exception for initialisation with wrong shape of n

surf = ort.FlatSurf([0, 0, 100], [0, 0, -1, 0], 1, 2, 10)

#%% check exception for initialisation with normal vector n with zero magnitude

surf = ort.FlatSurf([0, 0, 100], [0, 0, 0], 1, 2, 10)

#%% check exception for when rays propagate away from the surface

beam = ort.Beam([0, 0, 0], [0, 0, -1], 4, 10, profile='hline')
surf.propagate(beam)









