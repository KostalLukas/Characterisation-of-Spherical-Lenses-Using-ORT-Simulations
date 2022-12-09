# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 2

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Test all of the methods of the Beam class which can be tested without defining
another class.

Notes
-----
Testing passed.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# check the initialisation of the Beam class
beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 3, color='green')

# check the representation method
beam

# check the string method
print(beam)

# check the profile() method
beam.profile()
plt.show()

# check initialising a beam with a horizontal line profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 10, profile='hline')
beam.profile()
plt.show()

# check initialising a beam with a square profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 5, profile='square')
beam.profile()
plt.show()

# check initialising a beam with a circular profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 3, profile='circle')
beam.profile()
plt.show()

# check the show() method
beam.show()
plt.show()

# cehck the get_RMS() method
print(beam.get_RMS())

#%% check exception for initialisation with wrong shape of p

beam = ort.Beam([0, 0, 0, 0], [0, 0, 1], 4, 3)

#%% check exception for initialisation with wrong shape of p

beam = ort.Beam([0, 0, 0], [0, 0, 1, 0], 4, 3)

#%% check exception for initialisation with k with zero magnitude

beam = ort.Beam([0, 0, 0], [0, 0, 0], 4, 3)

#%% check exception for initialisation with negative radius or length r

beam = ort.Beam([0, 0, 0], [0, 0, 1], -4, 3)

#%% check exception for initialisation with non-integer number of rays n

beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 3.1)

#%% check exception for initialisation with undefined beam profile

beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 3, profile='bruh')
