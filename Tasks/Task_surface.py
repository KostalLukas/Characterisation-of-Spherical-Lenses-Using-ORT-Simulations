# -*- coding: utf-8 -*-
"""
Optical Ray Tracer Tasks for the Spherical Refracting Surface 

Lukas Kostal, 3.12.2022, ICL

Tasks: 9, 10, 12, 13, 14
"""


import importlib
import ORT as ort
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(ort)


#%% Task 9

# initialise a spherical refracting surface at z = 100mm
surf = ort.SpherSurf([0, 0, 100], 0.03, 1.0, 1.5, 25)

# initialise a screen at z = 250mm
screen = ort.Screen([0, 0, 250], [0, 0, 1])

# list of colors to plot randomised rays with
colors = ['magenta', 'red', 'orange', 'limegreen', 'teal', ]

# loop to create and propagate 5 randomised rays
for i in range(0, 5):
    
    # position with z = 0mm and x, y randomised on the interval [-25, 25] mm
    p = 50 * np.random.rand(3) - 25
    p[2] = 0
    
    # direction of propagation vector with z = 4 and x, y randomised on the
    # interval [-0.5, 0.5]
    k = np.random.rand(3) - 1/2
    k[2] = 6

    # create a ray with the randomised position and direction of propagation
    ray = (ort.Ray(p, k, color=colors[i]))
    
    # propagate the ray through the surface and intercept with the screen
    surf.propagate(ray)
    screen.intercept(ray)
    
    # plot the ray on a ray diagram
    ray.show()
    
# plot the surface and screen on the ray diagram
surf.show()
screen.show()

# set parameters for plotting
plt.title("Ray Diagram for Randomised Rays")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T9.png', dpi=300)
plt.show()

#%% Task 10

# initialise a narrow beam at z = 0mm with a linear profile
beam = ort.Beam([0, 0, 0,], [0, 0, 1], 0.1, 11, profile='hline')

# propagate the beam through the spherical surface
surf.propagate(beam)

# determine a numerical estimate for the position of the focal point
p_f, RMS_f = beam.get_fnum(0.01, show=False)

# intercept the beam with the screen
screen.intercept(beam)

# plot the ray and optical elements on a ray diagram
beam.show()
surf.show()
screen.show()

# determine the paraxial focal length and plot it using a dotted line
f = surf.get_fparax(show=True)

# print the estiamted and calculated paraxial focal length
print("Comparison of estimated and calculated focal length:")
print("f_estimated  = %.4g mm (4sf)" % (p_f[2] - 100))
print("f_paraxial   = %.4g mm (4sf)" % f)

# set parameters for plotting
plt.title("Narrow Beam Ray Diagram")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.ylim(-0.3, 0.3)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T10.png', dpi=300)
plt.show()
#%% Task 12

# initialise a beam at z = 0mm with a circular profile
beam = ort.Beam([0, 0, 0,], [0, 0, 1], 10, 7, profile='circle')

# initialise a screen at the previously found paraxial focal point
screen = ort.Screen(p_f, [0, 0, 1])

# propagate the beam through the surface and intercept with the screen
surf.propagate(beam)
screen.intercept(beam)

# determine the RMS spot radius at the screen
RMS = beam.get_RMS()

# plot the ray and optical elements on a ray diagram
beam.show()
surf.show()
screen.show()

# set plot title and axis labels and make a grid
plt.title("Wide Beam Ray Diagram")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T12.png', dpi=300)
plt.show()

#%% Task 13

# plot the image formed at the screen
screen.image(beam)

# set plot title and axis labels and make a grid
plt.title("Wide Beam Spot Diagram")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.grid(ls='--', alpha=0.8)
plt.tight_layout()

# save the plot
plt.savefig('Plots/T13.png', dpi=300)
plt.show()

#%% Task 14

# calcualte the radius of the Airy disc pattern for the longest and shortest
# wavelengths of visible light assuming the aperture radius is the radius of
# the beam which is 5mm
R_red = 1.22 * (700e-9) * 100 / (2 * 5e-3)
R_blue = 1.22 * (400e-9) * 100 / (2 * 5e-3)

print("RMS spot radius                  = %.4g mm (4sf)" % RMS)
print("Radius of Airy disc at 700nm     = %.4g mm (4sf)" % (R_red * 1e3))
print("Radius of Airy disc at 400nm     = %.4g mm (4sf)" % (R_blue * 1e3))

# so the system would be diffraction limited for a 5mm aperture radius