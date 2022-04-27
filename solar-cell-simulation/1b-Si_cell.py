# In this first example, we will look at a very simple planar Si solar cell.

# First, lets import some very commonly-used Python packages:

import numpy as np
import matplotlib.pyplot as plt

from solcore import material, si
from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.interpolate import interp1d
from solcore.absorption_calculator import calculate_rat, OptiStack
import seaborn as sns

# To define our solar cell, we first want to define some materials. Then we want to organise those materials into Layers,
# organise those layers into a Junction, and then finally define a SolarCell with that Junction.
# 
# First, let's define an Si material. Silicon, along with many other semiconductors, dielectrics, and metals common in
# solar cells, is included in Solcore's database.

Si = material("Si")
GaAs = material("GaAs")
SiN = material("Si3N4")()
Ag = material("Ag")()

# This creates an instance of the Si material. However, to use this in a solar cell we need to do specify some more
# information, such as the doping level.

Si_n = Si(Nd=si("1e21cm-3"), hole_diffusion_length=si("10um"), relative_permittivity=11.7)
Si_p = Si(Na=si("1e16cm-3"), electron_diffusion_length=si("400um"), relative_permittivity=11.7)

Si_thicknesses = np.linspace(np.log(0.4e-6), np.log(300e-6), 8)
Si_thicknesses = np.exp(Si_thicknesses)

wavelengths = si(np.linspace(300, 1200, 400), "nm")

options = {
    "recalculate_absorption": True,
    "optics_method": "TMM",
    "wavelength": wavelengths
           }


colors = sns.color_palette('rocket', n_colors=len(Si_thicknesses))
colors.reverse()

ARC_layer = Layer(width=si('75nm'), material=SiN)

plt.figure()

for i1, Si_t in enumerate(Si_thicknesses):

    base_layer = Layer(width=Si_t, material=Si_p)
    solar_cell = OptiStack([ARC_layer, base_layer])
    RAT_c = calculate_rat(solar_cell, wavelengths*1e9, no_back_reflection=False)
    RAT_i = calculate_rat(solar_cell, wavelengths*1e9, no_back_reflection=False,
                          coherent=False, coherency_list=['c', 'i'])
    plt.plot(wavelengths*1e9, RAT_c["A"], color=colors[i1],
             label=str(round(Si_t*1e6, 1)), alpha=0.7)
    plt.plot(wavelengths*1e9, RAT_i["A"], '--', color=colors[i1])

plt.legend(title=r"Thickness ($\mu$m)")
plt.xlim(300, 1300)
plt.ylim(0, 1.02)
plt.ylab("Absorption")
plt.title("Absorption in Si with varying thickness")
plt.show()


plt.figure()

for i1, Si_t in enumerate(Si_thicknesses):

    base_layer = Layer(width=Si_t, material=Si_p)
    solar_cell = OptiStack([ARC_layer, base_layer], substrate=Ag)
    RAT_c = calculate_rat(solar_cell, wavelengths*1e9, no_back_reflection=False)
    RAT_i = calculate_rat(solar_cell, wavelengths*1e9, no_back_reflection=False,
                          coherent=False, coherency_list=['c', 'i'])
    plt.plot(wavelengths*1e9, RAT_c["A"], color=colors[i1],
             label=str(round(Si_t*1e6, 1)), alpha=0.7)
    plt.plot(wavelengths*1e9, RAT_i["A"], '--', color=colors[i1])

plt.legend(title=r"Thickness ($\mu$m)")
plt.xlim(300, 1300)
plt.ylim(0, 1.02)
plt.ylab("Absorption")
plt.title("Absorption in Si with varying thickness (Ag substrate)")
plt.show()


angles = [0, 30, 60, 70, 80, 89]

colors = sns.cubehelix_palette(n_colors=len(angles))

plt.figure()

for i1, theta in enumerate(angles):

    ARC_layer = Layer(width=si('75nm'), material=SiN)
    base_layer = Layer(width=si("100um"), material=Si_p)
    solar_cell = OptiStack([ARC_layer, base_layer])
    RAT_s = calculate_rat(solar_cell, wavelengths*1e9, angle=theta,
                          pol='s',
                          no_back_reflection=False,
                          coherent=False, coherency_list=['c', 'i'])
    RAT_p = calculate_rat(solar_cell, wavelengths*1e9, angle=theta,
                          pol='p',
                          no_back_reflection=False,
                          coherent=False, coherency_list=['c', 'i'])
    plt.plot(wavelengths*1e9, RAT_s["A"], color=colors[i1], label=str(round(theta)))
    plt.plot(wavelengths*1e9, RAT_p["A"], '--', color=colors[i1])

plt.legend(title=r"$\theta (^\circ)$")
plt.xlim(300, 1300)
plt.ylim(0, 1.02)
plt.ylab("Absorption")
plt.title("Absorption in Si with varying thickness")
plt.show()
