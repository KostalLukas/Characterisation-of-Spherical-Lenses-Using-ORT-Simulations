# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 5

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Test the SpherSurf class for a spherical surface by propagating a beam.

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
beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 10, profile='hline')

# initialise a screen to be used for imaging
screen = ort.Screen([0, 0, 200], [0, 0, -1])

# check initialisation of the SpherSurf class
surf = ort.SpherSurf([0, 0, 100], 0.03, 1, 2, 10, color='green')

# check the representation method
surf

# check string method
print(surf)

# check the propagate() method
surf.propagate(beam)
screen.intercept(beam)

# check the get_fparax() method
f_parax = surf.get_fparax(show=True)
print(f_parax)

# check the show() method
beam.show()
surf.show()
screen.show()
plt.show()

# check initialising and propagating when curvature is negative
surf = ort.SpherSurf([0, 0, 100], -0.03, 1, 2, 10)
beam.reset()
surf.propagate(beam)
screen.intercept(beam)
beam.show()
surf.show()
screen.show()
plt.show()

# check what happens if part of beam misses the surface
beam = ort.Beam([-10, 0, 0], [0, 0, 1], 4, 10, profile='hline')
surf.propagate(beam)
screen.intercept(beam)
beam.show()
surf.show()
screen.show()
plt.show()

#%% check exception for initialisation with wrong shape of p

surf = ort.SpherSurf([0, 0, 100, 0], 0.03, 1, 2, 10)

#%% check exception for initialisation with a zero curvature

surf = ort.SpherSurf([0, 0, 100, 0], 0, 1, 2, 10)

#%% check exception for when rays propagate away from the surface

beam = ort.Beam([0, 0, 0], [0, 0, -1], 4, 10, profile='hline')
surf.propagate(beam)
