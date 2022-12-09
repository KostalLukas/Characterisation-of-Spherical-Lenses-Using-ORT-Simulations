# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Investigation 3

Lukas Kostal, 4.12.2022, ICL

Objective
---------
Investigate the relationship between spherical aberration and lens shape factor
as well as the relatioship between coma aberration and lens shape factor.
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

importlib.reload(ort)


# define a function to initialise a lens of a given curvature and propagate a
# beam through it to find the RMS spot radius
def OpSys(beam, z, curv_1, curv_2, n1, n2):
    n = len(curv_1)
    
    RMS_arr = np.empty(n)
    for i in range(0, n):
        lens = ort.SpherLens([0, 0, 100], z, curv_1[i], curv_2[i], n1, n2, 12)    
        beam.reset()
        lens.propagate(beam)
        
        p_f, RMS_f = beam.get_fnum(0.1)
        
        RMS_arr[i] = RMS_f

    return RMS_arr
    

# paraxial focal length of lens in mm
f = 100

# refractive indices of surrounding medium and lens respectively
n1 = 1.0
n2 = 1.5168

# separation of surfaces at at optical axis in mm
z = 5

# define number of samples
n = 50

# array of lens shape factors to be tested
q_arr = np.linspace(-3, 3, n)

# convert shape factor to curvature of lens
curv_1 = 0.5 * (1 + q_arr) / (n2 - n1) / f
curv_2 = 0.5 * (1 - q_arr) / (n2 - n1) / f

#%% testing for spherical and coma aberration

screen = ort.Screen([0, 0, 200], [0, 0, 1])

# initialise a wide beam perpendicular to the principial plane of the lens
beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 10)
#test for spherical aberration
RMS_spher = OpSys(beam, z, curv_1, curv_2, n1, n2)

# initialise a wide beam with a small angle of incidence to the lens
beam = ort.Beam([20, 0, 0], [-20/100, 0, 1], 10, 10)
# test for coma aberration at small angle
RMS_coma1 = OpSys(beam, z, curv_1, curv_2, n1, n2)

# initialise a wide beam with a larger angle of incidence to the lens
beam = ort.Beam([40, 0, 0], [-40/100, 0, 1], 10, 10)
# test for coma aberration at a larger angle
RMS_coma2 = OpSys(beam, z, curv_1, curv_2, n1, n2)

#%% plotting the dependance

# find the shape factor at which each curve has a minimum
q_spher = q_arr[np.argmin(RMS_spher)]
q_coma1 = q_arr[np.argmin(RMS_coma1)]
q_coma2 = q_arr[np.argmin(RMS_coma2)]

print("q_spher = %.4g" % q_spher)
print("q_coma1 = %.4g" % q_coma1)
print("q_coma2 = %.4g" % q_coma2)

# calcualte the angles of incidence for the coma aberration
tht_1 = np.arctan(20 / 100)
tht_2 = np.arctan(40 / 100)

# plot the RMS spot radius against the lens shape factor
plt.plot(q_arr, RMS_spher, color='blue', \
         label='spherical aberration    $\\theta_i = 0$ rad')
plt.plot(q_arr, RMS_coma1, color='green', \
         label='coma aberration         $\\theta_i = %.3g$ rad' % tht_1)
plt.plot(q_arr, RMS_coma2, color='red', \
         label='coma aberration         $\\theta_i = %.3g$ rad' % tht_2)

# plot minima of each curve
plt.plot(q_spher, np.amin(RMS_spher), 'x', color='blue')
plt.plot(q_coma1, np.amin(RMS_coma1), 'x', color='green')
plt.plot(q_coma2, np.amin(RMS_coma2), 'x', color='red')
plt.axvline(x = q_spher, ls=':', color='blue')
plt.axvline(x = q_coma1, ls=':', color='green')
plt.axvline(x = q_coma2, ls=':', color='red')

# set parameters for plotting
plt.title("Aberration against Lens Shape Factor")
plt.xlabel("lens shape factor $q$ (unitless)")
plt.ylabel("RMS spot radius $\sigma$ (mm)")
plt.legend()
plt.tight_layout()

# save the plot
plt.savefig('Plots/I3.png', dpi=300)
plt.show()
