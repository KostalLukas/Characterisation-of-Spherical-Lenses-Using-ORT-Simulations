# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 3

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Test the Screen class by imaging rays and beams so also further test the Ray
and Beam classes. If everything works for a beam then it must work for a ray
since the Beam class is a composition of the Ray class.

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
beam = ort.Beam([0, 0, 0], [1, 0, 2], 4, 10, profile='hline')


# check initialisation of the Screen class
screen = ort.Screen([0, 0, 100], [-1, 0, -4], color='green')

# check representation method
screen

# check string method
print(screen)

# check intercept() method
screen.intercept(beam)
beam.show()
plt.show()

# check image() method
screen.image(beam)
plt.show()

# check show() method
beam.show()
screen.show()
plt.show()


#%% check exception for initialisation with wrong shape of p

screen = ort.Screen([0, 0, 100, 0], [0, 0, -1])

#%% check exception for initialisation with wrong shape of n

screen = ort.Screen([0, 0, 100], [0, 0, -1, 0])

#%% check exception for initialisation with normal vector n with zero magnitude

screen = ort.Screen([0, 0, 100], [0, 0, 0])

#%% check exception for when rays form a virtual image

beam = ort.Beam([0, 0, 0], [0, 0, 1], 4, 10, profile='hline')
screen = ort.Screen([0, 0, -100], [0, 0, -1], color='green')
screen.intercept(beam)

#%% check that virtual images can still be formed if virt=True

screen = ort.Screen([0, 0, -100], [0, 0, -1], color='green', virt=True)
screen.intercept(beam)
beam.show()
screen.show()
plt.show()





