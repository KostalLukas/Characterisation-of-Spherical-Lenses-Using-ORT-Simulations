# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Example 1

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Define two spherical surfaces of different curvature and propagate a beam of
light through them. Show the initial beam profile as well as the final image on
the same plot to show the pincushion distortion effects of field curvature 
aberration.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# initialise a beam with with a square profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 11, profile='square')

# initialise a spherical surface
surf_1 = ort.SpherSurf([0, 0, 60], 0.02, 1.0, 1.5, 20)
# initialise a spherical surface
surf_2 = ort.SpherSurf([0, 0, 140], -0.05, 1.0, 2, 20)

# initialise a screen
screen = ort.Screen([0, 0, 200], [0, 0, -1])

# propagate the rays through the two spherical surfaces
surf_1.propagate(beam)
surf_2.propagate(beam)

# intercept the rays which pass throguh the prism with the screen
screen.intercept(beam)

# plot a ray diagram while specifying the axis to have equal scales
beam.show(equal=True)
surf_1.show()
surf_2.show()
screen.show()
plt.show()

# plot the initial profile of the beam as well as the produced image to show
# field curvature aberration
beam.profile()
screen.image(beam)
plt.show()