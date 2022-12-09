# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Investigation 5

Lukas Kostal, 5.12.2022, ICL

Objective
---------
Investigate the effects of the separation of lens surfaces on the RMS spot 
radius at the focal point of the lens as well as the focal lenght of the lens
for lenses of various common shape factors.
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

importlib.reload(ort)


def OpSys(curv, z_arr):
    n = len(z_arr)
    
    beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 10)
    
    fparax_arr = np.empty(n)
    fnum_arr = np.empty(n)
    RMS_arr = np.empty(n)
    for i in range(0, n):
        lens = ort.SpherLens([0, 0, 100], z_arr[i], curv[0], curv[1], 1.0, \
                             1.5168, 12)
        
        beam.reset()
        lens.propagate(beam)
        p_f, RMS_f = beam.get_fnum(0.1)
        
        fparax_arr[i] = lens.get_fparax()
        fnum_arr[i] = p_f[2] - 100
        RMS_arr[i] = RMS_f
    
    return fparax_arr, fnum_arr, RMS_arr


# number of samples
n = 50

# separation of surfaces to test
z_arr = np.linspace(4, 40, n)

# curvature of optimised lens in mm^-1
curv_opt = [0.016681863019222653, 0.00250925785101088]

#%% testing common lens shapes

# test lens with shape factor q = -1 so plano-convex
fparax_neg, fnum_neg, RMS_neg = OpSys([0, 0.02], z_arr)

# test lens with shape factor q = +1 so plano-convex
fparax_pos, fnum_pos, RMS_pos = OpSys([0.02, 0], z_arr)

# test lens with shape factor q = 0 so bi-convex lens
fparax_zero, fnum_zero, RMS_zero = OpSys([0.01, 0.01], z_arr)

# test lens with shape factor q = 0.738 optimised lens 
fparax_opt, fnum_opt, RMS_opt = OpSys(curv_opt, z_arr)

#%% plotting the dependance

# calculate the shape factor of the optimised lens
lens = ort.SpherLens([0, 0, 100], 5, curv_opt[0], curv_opt[1], 1.0, 1.5168, 10)
q = lens.get_shape()

# plot the calculated RMS spot radius against lens thickness for each lens
plt.plot(z_arr, RMS_neg, color='blue', label='plano-convex q = -1')
plt.plot(z_arr, RMS_pos, color='red', label='plano-convex q = +1')
plt.plot(z_arr, RMS_zero, color='orange', label='bi-convex       q = 0')
plt.plot(z_arr, RMS_opt, color='green', label='optimised       q = %.3g' % q)

# set parameters for plotting
plt.title("RMS Spot Radius against Surface Separation")
plt.xlabel("surface separation $z$ (mm)")
plt.ylabel("RMS Spot Radius $\sigma$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/I5_RMS.png', dpi=300)
plt.show()

# plot the focal length against lens thickness for each lens
plt.plot(z_arr, fnum_neg, color='blue', label='plano-convex q = -1')
plt.plot(z_arr, fparax_neg, linestyle='--', color='blue', linewidth=2)
plt.plot(z_arr, fnum_pos, color='red', label='plano-convex q = +1')
plt.plot(z_arr, fparax_pos, linestyle='--', color='red')
plt.plot(z_arr, fnum_zero, color='orange', label='bi-convex       q = 0')
plt.plot(z_arr, fparax_zero, linestyle='--', color='orange')
plt.plot(z_arr, fnum_opt, color='green', label='optimised       q = %.3g' % q)
plt.plot(z_arr, fparax_opt, linestyle='--', color='green')

# empty plots to add text the legend
plt.plot([], [], linestyle='-', color='black', \
         label='represents numerical $f$')
plt.plot([], [], linestyle='--', color='black', \
         label='represents paraxial $f$')

# set parameters for plotting
plt.title("Focal Length against Surface Separation")
plt.xlabel("surface separation $z$ (mm)")
plt.ylabel("focal length $f$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/I5_f.png', dpi=300)
plt.show()