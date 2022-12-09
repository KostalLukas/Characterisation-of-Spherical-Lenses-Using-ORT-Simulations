# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Investigation 4

Lukas Kostal, 5.12.2022, ICL

Objective
---------
Investigate the relationship between the RMS spot radius at the screen and the 
curvatures of the two sides of the lens centered around the optimum curvature.
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

importlib.reload(ort)


# define number of samples
n = 40

# curvature of optimised lens in mm^-1
curv_opt = [0.016681863019222653, 0.00250925785101088]

#%% investigating the dependance

# curvatures to be tested for each side of the lens
curv_1 = curv_opt[0] + 1e-4 * np.linspace(-1, 1, n)
curv_2 = curv_opt[1] + 1e-4 * np.linspace(-1, 1, n)

# initialise a wide beam at z = 0mm
beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 10)

# initialise a screen at z = 200mm
screen = ort.Screen([0, 0, 200], [0, 0, 1])

# test all possible combinations of the curvatures and calcualte the RMS spot
# radius for each combination
RMS_grid = np.empty((n, n))
for i in range(0, n):
    for j in range(0, n):
        lens = ort.SpherLens([0, 0, 100], 5, curv_1[i], curv_2[j], \
                             1.0, 1.5168, 10)
        beam.reset()
        lens.propagate(beam)
        screen.intercept(beam)
        
        RMS_grid[i, j] = beam.get_RMS()

#%% plotting the dependance

# arrays of curvature for 3D plot
curv1_grid, curv2_grid = np.meshgrid(curv_1, curv_2)

# set axis for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(30, 130)
ax.dist = 13
plt.tight_layout()

# plot the RMS spot radius surface in 3D
ax.plot_surface(curv1_grid, curv2_grid, RMS_grid, cmap='plasma', edgecolor='none')

# set plotting parameters
plt.title("RMS Spot Radius against Lens Curvatures")
ax.set_xlabel("curvature $C_1$ ($mm^{-1}$)", labelpad=12)
ax.set_ylabel("curvature $C_2$ ($mm^{-1}$)", labelpad=16)
ax.set_zlabel("RMS spot radius $\sigma$ (mm)", labelpad=8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I4.png', dpi=300)
plt.show()