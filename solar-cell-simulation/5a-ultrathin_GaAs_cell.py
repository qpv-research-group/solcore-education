# In previous examples, we have considered a few different methods used to improve absorption in solar cells:
# anti-reflection coatings, to decrease front-surface reflection, metallic rear mirrors to reduce transmission and increase
# the path length of light in the cell, and textured surfaces (with pyramids) which are used on Si cells to reduce
# reflection and increase the path length of light in the cell. Another method which can be used for light-trapping is the
# inclusion of periodic structures such as diffraction gratings or photonic crystals; here, we will consider an ultra-thin
# (80 nm) GaAs cell with a diffraction grating.

# This example is based on the simulations done relating to this work: https://doi.org/10.1002/pip.3463
# This example requires that you have a working S4 installation: https://rayflare.readthedocs.io/en/latest/Installation/installation.html

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from solcore import si, material
from solcore.structure import Layer
from solcore.light_source import LightSource
from solcore.solar_cell import SolarCell
from solcore.constants import q
from solcore.absorption_calculator import search_db

from rayflare.rigorous_coupled_wave_analysis.rcwa import rcwa_structure
from rayflare.transfer_matrix_method.tmm import tmm_structure
from rayflare.options import default_options

# Defining all the materials. We are just going to do an optical simulation, so don't have to worry about doping levels
# and other parameters which would affect the electrical performance of the cell.

InAlP = material('AlInP')(Al=0.5)
GaAs = material('GaAs')()
InGaP = material('GaInP')(In=0.5)
SiN = material('Si3N4')()
Al2O3 = material('Al2O3P')()

Air = material('Air')()

# The optical constants used for the silver are very important for the grating performance (see https://doi.org/10.1016/j.solmat.2018.11.008);
# search for a known reliable dataset.

Ag_pageid = search_db("Ag/Jiang")[0][0]
Ag = material(str(Ag_pageid), nk_db=True)()

wavelengths = np.linspace(303, 1000, 200) * 1e-9

# AM0 spectrum (photon flux) for calculating currents. For space applications (i.e. above the atmosphere) we are often
# interested in AM0.
AM0_ls = LightSource(source_type='standard', version='AM0', x=wavelengths, output_units="photon_flux_per_m")

AM0 = AM0_ls.spectrum(x=wavelengths)[1] # Photon flux; used to calculate photogenerated current later on

# Setting options. We choose 's' polarization because, for normal incidence, there will not be a difference in the results
# for s and p polarization (and thus for unpolarized light, 'u', which would be calculated as the average of the results for
# s and p polarization. We could set the polarization to 'u' for equivalent results, but this would take longer because then
# RayFlare has to run two calculations (for 's' and 'p' polarization) instead of one.
# The other key option is the number of Fourier order: rigorous coupled-wave analysis (RCWA) is a Fourier-space method, and
# we have to specify how many Fourier orders should be retained in the calculation. As we increase the number of orders, the
# calculation should converge, but the computation time increases (it scales with the cube of the number of orders).

options = default_options()
options.pol = 's'
options.wavelengths = wavelengths
options.orders = 200 # Reduce the number of orders to speed up the calculation.

# ================================================= #
print("Calculating on-substrate device...")
# on-substrate device (planar)

struct = SolarCell([Layer(si('20nm'), InAlP), Layer(si('85nm'), GaAs),
                   Layer(si('20nm'), InGaP)])


# make TMM structure for planar device
TMM_setup = tmm_structure(struct, incidence=Air, transmission=GaAs)

# calculate
RAT_TMM_onsubs = TMM_setup.calculate(options)

Abs_onsubs = RAT_TMM_onsubs['A_per_layer'][:,1]  # absorption in GaAs
# indexing of A_per_layer is [wavelengths, layers]

R_onsubs = RAT_TMM_onsubs['R']
T_onsubs = RAT_TMM_onsubs['T']

# ================================================= #
# Device with planar silver
print("Calculating planar Ag mirror device...")

solar_cell_TMM = SolarCell([Layer(material=InGaP, width=si('20nm')),
                        Layer(material=GaAs, width=si('85nm')),
                        Layer(material=InAlP, width=si('20nm'))],
                           substrate=Ag)

TMM_setup = tmm_structure(solar_cell_TMM, incidence=Air, transmission=Ag)

RAT_TMM = TMM_setup.calculate(options)

Abs_TMM = RAT_TMM['A_per_layer'][:, 1]
Abs_TMM_InAlPonly = RAT_TMM['A_per_layer'][:, 2]
Abs_TMM_InGaPonly = RAT_TMM['A_per_layer'][:, 0]
R_TMM = RAT_TMM['R']
T_TMM = RAT_TMM['T']

# ================================================= #
# Setting things up for RCWA calculation - DTL device
print("Calculating nanophotonic grating device...")

x = 600

# lattice vectors for the grating. Units are in nm!
size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

# RCWA options for S4 (https://web.stanford.edu/group/fan/S4/python_api.html)

ropt = dict(LatticeTruncation='Circular',
            DiscretizedEpsilon=False,
            DiscretizationResolution=8,
            PolarizationDecomposition=True,
            PolarizationBasis='Default',
            LanczosSmoothing=dict(Power=2, Width=1),
            #LanczosSmoothing=False,
            SubpixelSmoothing=False,
            ConserveMemory=False,
            WeismannFormulation=False,
            Verbosity=0)

options.S4_options = ropt

# grating layers
grating = [Layer(width=si(100, 'nm'), material=SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                 'radius': x/3, 'angle': 0}])] # actual grating part of grating


# DTL device without anti-reflection coating
solar_cell = SolarCell([Layer(material=InGaP, width=si('20nm')),
                        Layer(material=GaAs, width=si('85nm')),
                        Layer(material=InAlP, width=si('20nm'))] + grating,
                       substrate=Ag)

# make RCWA structure
S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

# calculate

RAT = S4_setup.calculate(options)

Abs_DTL = RAT['A_per_layer'][:,1] # absorption in GaAs

R_DTL = RAT['R']
T_DTL = RAT['T']

# ================================================= #
# Setting things up for RCWA calculation - DTL device with ARC
print("Calculating nanophotonic grating device with ARC...")

# DTL device with anti-reflection coating
solar_cell = SolarCell([Layer(material=Al2O3, width=si('70nm')),
                        Layer(material=InGaP, width=si('20nm')),
                        Layer(material=GaAs, width=si('85nm')),
                        Layer(material=InAlP, width=si('20nm'))] + grating,
                       substrate=Ag)

# make RCWA structure
S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

# calculate
RAT_ARC = S4_setup.calculate(options)

Abs_DTL_ARC = RAT_ARC['A_per_layer'][:,2]     # absorption in GaAs + InGaP

R_DTL_ARC = RAT_ARC['R']
T_DTL_ARC = RAT_ARC['T']

# ================================================= #
# plotting

pal = sns.color_palette("husl", 4)

# assume 10% shading loss for the optical simulations when comparing with EQE
fig = plt.figure(figsize=(6.4, 4.8))

plt.plot(wavelengths*1e9, 100*Abs_onsubs, color=pal[0], label="On substrate")
plt.plot(wavelengths*1e9, 100*Abs_TMM, color=pal[1], label="Planar mirror")
plt.plot(wavelengths*1e9, 100*Abs_DTL, color=pal[2], label="Nanophotonic grating (no ARC)")
plt.plot(wavelengths*1e9, 100*Abs_DTL_ARC, color=pal[3], label="Nanophotonic grating (with ARC)")

plt.xlim(300, 950)
plt.ylim(0, 100)
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE (%)')
plt.legend(loc='upper left')
plt.show()

fig = plt.figure(figsize=(6.4, 4.8))
plt.stackplot(wavelengths*1e9,
              [100*Abs_TMM, 100*Abs_TMM_InGaPonly, 100*Abs_TMM_InAlPonly],
              colors=pal,
              labels=['Absorbed in GaAs', 'Absorbed in InGaP', 'Absorbed in InAlP'])
plt.xlim(300, 950)
plt.ylim(0, 90)
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE (%)')
plt.legend(loc='upper right')
plt.show()

# Calculate photogenerated currents:

onsubs = 0.1 * q * np.trapz(Abs_onsubs*AM0, wavelengths)
Ag = 0.1 * q * np.trapz(Abs_TMM*AM0, wavelengths)
DTL = 0.1 * q * np.trapz(Abs_DTL*AM0, wavelengths)
DTL_ARC = 0.1 * q * np.trapz(Abs_DTL_ARC*AM0, wavelengths)


print('On substrate device current: %.1f mA/cm2 ' % onsubs)
print('Planar Ag mirror device current: %.1f mA/cm2 ' % Ag)
print('Nanophotonic grating (no ARC) device current: %.1f mA/cm2 ' % DTL)
print('Nanophotonic grating (with ARC) device current: %.1f mA/cm2 ' % DTL_ARC)
