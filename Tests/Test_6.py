# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 6

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Test the SpherSurf class for a spherical lens by propagating a beam.

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
beam = ort.Beam([0, 0, 0], [0, 0, 1], 5, 10, profile='hline')

# initialise a screen to be used for imaging
screen = ort.Screen([0, 0, 200], [0, 0, -1])

# check initialisation of the SpherLens class first for a biconvex lens
lens = ort.SpherLens([0, 0, 100], 10, 0.01, 0.01, 1, 2, 10, color='green')

# check the representation method
lens

# check string method
print(lens)

# check the propagate() method
lens.propagate(beam)
screen.intercept(beam)

# check the get_fparax() method
f_parax = lens.get_fparax(show=True)
print(f_parax)

# check the show() method
beam.show()
lens.show()
screen.show()
plt.show()

# check initialising and propagating through a lens with zero curvature
lens = ort.SpherLens([0, 0, 100], 10, 0.03, 0, 1, 2, 10,)
beam.reset()
lens.propagate(beam)
screen.intercept(beam)
beam.show()
lens.show()
screen.show()
plt.show()

# check initialising and propagating through a lens with negative curvature
lens = ort.SpherLens([0, 0, 100], 10, -0.02, 0, 1, 2, 10,)
beam.reset()
lens.propagate(beam)
screen.intercept(beam)
beam.show()
lens.show()
screen.show()
plt.show()

# check what happens if part of beam misses the lens
beam = ort.Beam([-10, 0, 0], [0, 0, 1], 4, 10, profile='hline')
lens.propagate(beam)
screen.intercept(beam)
beam.show()
lens.show()
screen.show()
plt.show()

# check get_shape() method for calcualting the shape factor of the lens
lens = ort.SpherLens([0, 0, 100], 20, 0.01, 0, 1, 2, 10)
q = lens.get_shape()
print(q)

lens = ort.SpherLens([0, 0, 100], 20, 0, 0.01, 1, 2, 10)
q = lens.get_shape()
print(q)

lens = ort.SpherLens([0, 0, 100], 20, 0.01, 0.01, 1, 2, 10)
q = lens.get_shape()
print(q)

#%% check exception for initialisation with wrong shape of p

lens = ort.SpherLens([0, 0, 100, 0], 20, 0.1, 0.1, 1, 2, 10)

#%% check exception for aperture greater than the smaller radius of curvature

lens = ort.SpherLens([0, 0, 100], 20, 0.1, 0.1, 1, 2, 30)

#%% check exception for two surfaces of lens intersecting

lens = ort.SpherLens([0, 0, 100], 10, 0.1, 0.1, 1, 2, 10)


#%% check exception for when rays propagate away from the surface
beam = ort.Beam([0, 0, 0], [0, 0, -1], 4, 10, profile='hline')
lens.propagate(beam)

