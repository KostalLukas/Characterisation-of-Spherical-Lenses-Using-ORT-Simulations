"""
Optical Ray Tracer Lens Optimisation Task

Lukas Kostal, 4.12.2022, ICL

Tasks: Lens Optimisation
"""


import importlib
import ORT as ort
import matplotlib.pyplot as plt
import scipy.optimize as spo

importlib.reload(ort)


# function which takes curvatures and returns RMS spot radius to be minimised
def OpSys(curv, beam, screen):
    # initialise a lens at z = 100mm with the specified curvatures
    lens = ort.SpherLens([0, 0, 100], 5, curv[0], curv[1], 1, 1.5168, 12)

    beam.reset()
    lens.propagate(beam)
    screen.intercept(beam)
    
    return beam.get_RMS()

# initialise beam at z = 0mm with a circular profile
beam = ort.Beam([0, 0, 0], [0, 0, 1], 10, 10)

# initialise a screen at z = 200mm
screen = ort.Screen([0, 0, 200], [0, 0, 1])

# optimise the curvatures using downhill simplex algorithm
curv_opt, curv_arr = spo.fmin(OpSys, x0=[0.01 ,0.01], args=(beam, screen), \
                              xtol=1e-8, retall=True)

# initialise a lens using the optimised curvatures
lens = ort.SpherLens([0, 0, 100], 5, curv_opt[0], curv_opt[1], 1, 1.5168, 20)

# reset the beam to propagate it through the system again
beam.reset()

# propagate the beam through the lens and intercept with the screen
lens.propagate(beam)
screen.intercept(beam)

# determine a numerical estimate for the position of the focal point
RMS_opt = beam.get_RMS()
f = lens.get_fparax()

# determine the Coddington shape factor of the optimised lens
q = lens.get_shape()

# plot the numerical results
print()
print("Parameters of optimised lens:")
print("Curvature_1      = %.4g mm^-1 (4sf)" % curv_opt[0])
print("Curvature_2      = %.4g mm^-1 (4sf)" % curv_opt[1])
print("Shape factor     = %.4g (4sf)" % q)
print("RMS radius       = %.4g mm (4sf)" % RMS_opt)
print("f_parax          = %.4g mm (4sf)" % f)
print()

# plot a ray diagram for the optimised lens parameters
beam.show()
lens.show()
screen.show()

# set parameters for plotting
plt.title("Optimised Spherical Lens")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.grid(ls='--', alpha=0.8)

# save the plot
plt.savefig('Plots/Topt.png', dpi=300)
plt.show()
plt.tight_layout()

print(curv_opt[0])
print(curv_opt[1])