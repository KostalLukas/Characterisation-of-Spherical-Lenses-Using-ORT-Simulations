# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Example 3

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Model a compound optical system made of multiple spherical lenses and propagate
a beam of light through it.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# initialise a beam with with a linear profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 20, 15, profile='hline')

# initialise multiple spherical lenses of different parameters
lens_1 = ort.SpherLens([0, 0, 60], 5, 0.01, 0.01, 1.0, 1.5, 20, color='purple') 
lens_2 = ort.SpherLens([0, 0, 100], 5, -0.03, -0.03, 1.0, 1.5, 10, color='blue')
lens_3 = ort.SpherLens([0, 0, 140], 5, 0.02, 0.02, 1.0, 2, 10, color='green')

# initialise a screen
screen = ort.Screen([0, 0, 200], [0, 0, -1])

# propagate the rays through the lenses
lens_1.propagate(beam)
lens_2.propagate(beam)
lens_3.propagate(beam)

# intercept the rays which pass throguh the prism with the screen
screen.intercept(beam)

# plot a ray diagram while specifying the axis to have equal scales
beam.show()
lens_1.show()
lens_2.show()
lens_3.show()
screen.show()

plt.show()