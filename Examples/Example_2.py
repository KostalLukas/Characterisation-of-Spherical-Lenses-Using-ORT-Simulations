# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Example 2

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Model an isosceles triangle prism with a vertex angle of 90° with a refractive
index of 1.2 and propagate a beam of light through it.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# initialise a beam with with a linear profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 10, profile='hline')

# initialise the two surfaces of the prism specifying their orientatio with
# the vector normal to their surface
surf_1 = ort.FlatSurf([0, 0, 85.858], [1, 0, -1], 1, 1.2, 20)
surf_2 = ort.FlatSurf([0, 0, 114.142], [-1, 0, -1], 1.2, 1, 20)

# initialise a screen
screen = ort.Screen([0, 0, 150], [-1, 0, -2])

# propagate the rays through the two surfaces of the prism
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