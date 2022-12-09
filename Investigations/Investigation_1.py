# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Investigation 1

Lukas Kostal, 5.12.2022, ICL

Objective
---------
Investigate spherical abberation and its dependance on the radius of the beam
for lenses of common shape factors.
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(ort)


# define function which finds the focal point of a lens and propagates beams
# of different radii to find the RMS spot radius at each beam radius
def OpSys(lens, r_arr):
    n = len(r_arr)
    
    beam = ort.Beam([0, 0, 0,], [0, 0, 1], 0.1, 7, profile='circle')
    lens.propagate(beam)
    p_f, RMS_f = beam.get_fnum(0.01, show=False)
    screen = ort.Screen(p_f, [0, 0, 1])
    
    RMS_arr = np.empty(n)
    for i in range(0, n):
        beam = ort.Beam([0, 0, 0], [0, 0, 1], r_arr[i], 10)
        lens.propagate(beam)
        screen.intercept(beam)
        
        RMS_arr[i] = beam.get_RMS()
    
    beam.show()
    lens.show()
    screen.show()
    
    return RMS_arr


# define number of samples
n = 50

# array of beam radii to be tested
r_arr = np.linspace(0.1, 20, n)

# curvature of optimised lens in mm^-1
curv_opt = [0.016681863019222653, 0.00250925785101088]

#%% shape factor q = -1 plano-convex lens so same as task 15

# initialise the plano-convex lens
lens = ort.SpherLens([0, 0, 100], 5, 0, 0.02, 1.0, 1.5168, 20)

RMS_neg = OpSys(lens, r_arr)

# set parameters for plotting
plt.title("Plano-Convex Lens q = -1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I1_neg.png', dpi=300)
plt.show()

#%% shape factor q = +1 plano-convex lens so same as task 15

# initialise the plano-convex lens in opposite direction
lens = ort.SpherLens([0, 0, 100], 5, 0.02, 0, 1.0, 1.5168, 20)

RMS_pos = OpSys(lens, r_arr)

# set parameters for plotting
plt.title("Plano-Convex Lens q = +1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I1_pos.png', dpi=300)
plt.show()

#%% shape factor q = 0 bi-convex lens

# initialise the bi-convex lens
lens = ort.SpherLens([0, 0, 100], 5, 0.01, 0.01, 1.0, 1.5168, 20)

RMS_zero = OpSys(lens, r_arr)

# set parameters for plotting
plt.title("Bi-Convex Lens q = 0")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I1_zero.png', dpi=300)
plt.show()


#%% shape factor q = 0.738 optimised lens 

# initialise the optimised lens
lens = ort.SpherLens([0, 0, 100], 5, curv_opt[0], curv_opt[1], 1.0, 1.5168, 22)

RMS_opt = OpSys(lens, r_arr)

# calculate the shape factor of the optimised lens
q = lens.get_shape()

# set parameters for plotting
plt.title("Optimised Lens q = %.3g" % q)
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I1_opt.png', dpi=300)
plt.show()

#%% plotting the dependance

# plot the calculated RMS spot radius for each lens
plt.plot(r_arr, RMS_neg, color='blue', label='plano-convex q = -1')
plt.plot(r_arr, RMS_pos, color='red', label='plano-convex q = +1')
plt.plot(r_arr, RMS_zero, color='orange', label='bi-convex       q = 0')
plt.plot(r_arr, RMS_opt, color='green', label='optimised       q = %.3g' % q)

# set parameters for plotting
plt.title("Spherical Abberation against Beam Radius")
plt.xlabel("beam radius $r$ (mm)")
plt.ylabel("RMS spot radius $\sigma$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/I1.png', dpi=300)
plt.show()
