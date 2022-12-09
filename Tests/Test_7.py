# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 7

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Test methods of the Beam class which utilise RMS spread.

Notes
-----
Testing passed.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# initialise a beam with a spherical profile, a lens and a screen
beam = ort.Beam([0, 0, 0], [0, 0, 1], 5, 10)
lens = ort.SpherLens([0, 0, 100], 10, 0.01, 0.01, 1, 2, 20)
screen = ort.Screen([0, 0, 200], [0, 0, 1])

# propagate the beam through the lens
lens.propagate(beam)

# calculate the paraxial focal length of the lens for comparison
f_parax = lens.get_fparax(show=True)
print(f_parax)

# check the get_fnum() method by finding the numerical estiamte for the
# position of the focus and show it on the ray diagram
r_focus, RMS_focus = beam.get_fnum(0.01, show=True, color='orange')
print(r_focus)
print(RMS_focus)

# intercept the beam with the screen
screen.intercept(beam)

# check the get_RMS() method by finding the RMS sepration at the screen
RMS = beam.get_RMS()
print(RMS)

# plot a ray diagram

beam.show()
lens.show()
screen.show()
plt.show()

#%% check exception for when increment dz is too large
beam.reset()
lens.propagate(beam)
r_focus, RMS_focus = beam.get_fnum(200)

#%% check exception for when rays diverge so need a diverging lens
beam.reset()
lens = ort.SpherLens([0, 0, 100], 10, -0.01, 0, 1, 2, 20)
lens.propagate(beam)
r_focus, RMS_focus = beam.get_fnum(0.01)
