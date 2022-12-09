# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Investigation 2

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Investigate coma aberration and its dependance on the angle between the beam
and the principal plane of the lens lenses of common shape factors.
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(ort)


# define function similar to the one in Investigation 1 but this time vary the
# angle of incidence between the beam and the lens
def OpSys(lens, x_arr):
    n = len(x_arr)
    
    beam = ort.Beam([0, 0, 0], [0, 0, 1], 0.1, 10)
    
    lens.propagate(beam)
    p_f, RMS_f = beam.get_fnum(0.1)
    screen = ort.Screen(p_f, [0, 0, 1])
    
    RMS_arr = np.empty(n)
    for i in range(0, n):
        beam = ort.Beam([x_arr[i], 0, 0], [-x_arr[i]/100, 0, 1], 10, 10)
        lens.propagate(beam)
        screen.intercept(beam)
        
        RMS_arr[i] = beam.get_RMS()
    
    beam.show()
    lens.show()
    screen.show()
    
    return beam, screen, RMS_arr


# define number of samples
n = 50

# array of displacements along x axis to be tested
x_arr = np.linspace(0, 20, n)

# curvature of optimised lens in mm^-1
curv_opt = [0.016681863019222653, 0.00250925785101088]

#%% shape factor q = -1 plano-convex

# initialise the plano-convex lens
lens = ort.SpherLens([0, 0, 100], 5, 0, 0.02, 1.0, 1.5168, 20)

beam, screen, RMS_neg = OpSys(lens, x_arr)

# set parameters for plotting
plt.title("Plano-Convex Lens q = -1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)

# save the plot
plt.savefig('Plots/I2_neg.png', dpi=300)
plt.show()

# plot the image produced at the screen
screen.image(beam)

# set parameters for plotting
plt.title("Coma Aberration for a Plano-Convex Lens q = -1") 
plt.xlabel("y (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I2_neg_image.png', dpi=300)
plt.show()

#%% shape factor q = +1 plano-convex

# initialise the plano-convex lens in opposite direction
lens = ort.SpherLens([0, 0, 100], 5, 0.02, 0, 1.0, 1.5168, 20)

beam, screen, RMS_pos = OpSys(lens, x_arr)

# set parameters for plotting
plt.title("Plano-Convex Lens q = +1")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I2_pos.png', dpi=300)
plt.show()

#%% shape factor q = 0 bi-convex lens

# initialise the bi-convex lens
lens = ort.SpherLens([0, 0, 100], 5, 0.01, 0.01, 1.0, 1.5168, 20)

beam, screen, RMS_zero = OpSys(lens, x_arr)

# set parameters for plotting
plt.title("Bi-Convex Lens q = 0")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I2_zero.png', dpi=300)
plt.show()

#%% shape factor q = 0.738 optimised lens 

# initialise the optimised lens
lens = ort.SpherLens([0, 0, 100], 5, curv_opt[0], curv_opt[1], 1.0, 1.5168, 20)

beam, screen, RMS_opt = OpSys(lens, x_arr)

# calculate the shape factor of the optimised lens
q = lens.get_shape()

# set parameters for plotting
plt.title("Optimised Lens q = %.3g" % q)
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/I2_opt.png', dpi=300)
plt.show()

#%% plotting the dependance

# calcualte the angle of incidence of the beam
tht_arr = np.arctan(x_arr / 100)

# remove the offset caused by spherical aberration at tht = 0
RMS_neg -= np.amin(RMS_neg)
RMS_pos -= np.amin(RMS_pos)
RMS_zero -= np.amin(RMS_zero)
RMS_opt -= np.amin(RMS_opt)

# plot the calculated RMS spot radius for each lens
plt.plot(tht_arr, RMS_neg, color='blue', label='plano-convex q = -1')
plt.plot(tht_arr, RMS_pos, color='red', label='plano-convex q = +1')
plt.plot(tht_arr, RMS_zero, color='orange', label='bi-convex       q = 0')
plt.plot(tht_arr, RMS_opt, color='green', label='optimised       q = %.3g' % q)

# set parameters for plotting
plt.title("Coma Abberation against Angle of Incidence")
plt.xlabel("angle of incidence $\\theta_i$ (rad)")
plt.ylabel("RMS spot radius $\sigma$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/I2.png', dpi=300)
plt.show()