# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Test 1

Lukas Kostal, 3.12.2022, ICL

Objective
---------
Test all of the methods of the Ray class.

Notes
-----
Testing passed.
"""


import numpy as np
import matplotlib.pyplot as plt
import importlib
import ORT as ort

importlib.reload(ort)


# check initialisation of the Ray calss
ray = ort.Ray([0, 0, 0], [0, 0, 1], color='green')

# check the representation method
ray

#check the string method
print(ray)

# check the append method
ray.append([1, 1, 1], [1, 1, 1])
ray.append([0, 0, 2], [2, 2, 2])
print(ray)

# check the get_p() method
print(ray.get_p())

# check the get_k() method
print(ray.get_k())

# check the get_vert() method
print(ray.get_vert())

# check the show() method
ray.show()
plt.show()

# check the reset() method
ray.reset()
print(ray)

#%% check exception for initialisation with wrong shape of p

ray = ort.Ray([0, 0, 0, 0], [0, 0, 1])

#%% check exception for initialisation with wrong shape of k

ray = ort.Ray([0, 0, 0], [0, 0, 1, 0])

#%% check exception for initialisation with k with a zero magnitude

ray = ort.Ray([0, 0, 0], [0, 0, 0])

#%% check exception for appending with wrong shape of p

ray.append([0, 0, 0, 0], [0, 0, 1])

#%% check exception for appending with wrong shape of k

ray.append([0, 0, 0], [0, 0, 1, 0])

#%% check exception for appending with k with a zero magnitude

ray.append([0, 0, 0], [0, 0, 0])