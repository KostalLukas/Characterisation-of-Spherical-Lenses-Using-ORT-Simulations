# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Tasks for the Spherical Lens

Lukas Kostal, 3.12.2022, ICL

Tasks: 5
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(ort)


# define number of samples
n = 50

# array of beam radii to be tested
r_arr = np.linspace(0.1, 10, n)

#%% Task 15 plano-convex lens with -1 shape factor

# initialise a narrow beam at z = 0mm with a circular profile
beam = ort.Beam([0, 0, 0,], [0, 0, 1], 0.1, 7, profile='circle')

# initialise a plano-convex lens with -1 shape factor at z = 100mm
lens = ort.SpherLens([0, 0, 100], 5, 0, 0.02, 1.0, 1.5168, 20)

# propagate the narrow beam through the lens
lens.propagate(beam)

# determine a numerical estimate for the new position of the focal point
p_f, RMS_f = beam.get_fnum(0.01, show=False)

# define a screen at the found focal point
screen = ort.Screen(p_f, [0, 0, 1])

# calculate the RMS spot radius at the screen for beams of increasing radii
RMS_neg = np.empty(n)
for i in range(0, n):
    beam = ort.Beam([0, 0, 0], [0, 0, 1], r_arr[i], 10)
    lens.propagate(beam)
    screen.intercept(beam)
    
    RMS_neg[i] = beam.get_RMS()

# plot a ray diagram for the final beam radius
beam.show()
lens.show()
screen.show()

# set parameters for plotting
plt.title("Plano-Convex Lens q = -1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T15_RD_neg.png', dpi=300)
plt.show()

#%% Task 15 plano-convex lens with +1 shape factor

# initialise a narrow beam at z = 0mm with a circular profile
beam = ort.Beam([0, 0, 0,], [0, 0, 1], 0.1, 7, profile='circle')

# initialise a plano-convex lens with a +1 shape factor at z = 100mm
lens = ort.SpherLens([0, 0, 100], 5, 0.02, 0, 1.0, 1.5168, 20)

# propagate the narrow beam through the lens
lens.propagate(beam)

# determine a numerical estimate for the position of the focal point
p_f, RMS_f = beam.get_fnum(0.01, show=False)

# define a screen at the found focal point
screen = ort.Screen(p_f, [0, 0, 1])

# calculate the RMS spot radius at the screen for beams of increasing radii
RMS_pos = np.empty(n)
for i in range(0, n):
    beam = ort.Beam([0, 0, 0], [0, 0, 1], r_arr[i], 10)
    lens.propagate(beam)
    screen.intercept(beam)
    
    RMS_pos[i] = beam.get_RMS()

# plot a ray diagram for the final beam radius
beam.show()
lens.show()
screen.show()

# set parameters for plotting
plt.title("Plano-Convex Lens q = +1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T15_RD_pos.png', dpi=300)
plt.show()

#%% Task 15 compare the variation of RMS spot radius for the two lenses

# plot the RMS spot radius against the beam diameter
plt.plot(r_arr, RMS_pos, color='red', label='plano-convex q = +1')
plt.plot(r_arr, RMS_neg, color='blue', label='plano-convex q = -1')

# set parameters for plotting
plt.title("Variation of RMS Spot Radius with Beam Radius")
plt.xlabel("beam radius $r$ (mm)")
plt.ylabel("RMS spot radius $\sigma$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/T15_RMS.png', dpi=300)
plt.show()