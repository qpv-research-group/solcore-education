import numpy as np
import os

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

file_path = os.path.dirname(os.path.abspath(__file__))

# If True, run calculation; otherwise load from file.
calc = True

# defining all the materials. We are just going to do an optical simulation, so don't have to worry about doping levels
# and other parameters which would affect the electrical performance of the cell.

InAlP = material('AlInP')(Al=0.5)
GaAs = material('GaAs')()
InGaP = material('GaInP')(In=0.5)
SiN = material('Si3N4')()
Al2O3 = material('Al2O3P')()

Air = material('Air')()

Ag_pageid = search_db("Ag/Jiang")[0][0]
Ag = material(str(Ag_pageid), nk_db=True)()

wavelengths = np.linspace(303, 1000, 200) * 1e-9

# AM0 spectrum (photon flux) for calculating currents
AM0_ls = LightSource(source_type='standard', version='AM0', x=wavelengths,
                     output_units="photon_flux_per_m")

AM0 = AM0_ls.spectrum(x=wavelengths)[1]

options = default_options()
options.pol = 's'
options.wavelengths = wavelengths
options.orders = 100 # number of Fourier orders (this is the total number of orders, so not the values of m along one side)


# ================================================= #
print("Calculating on-substrate device...")
# on-substrate device (planar)

struct = SolarCell([Layer(si('17nm'), InAlP), Layer(si('87nm'), GaAs),
                   Layer(si('19nm'), InGaP), Layer(si('25nm'), GaAs),
                   Layer(si('149nm'), InAlP)])


# make TMM structure for planar device
TMM_setup = tmm_structure(struct, incidence=Air, transmission=GaAs)

# calculate
RAT_TMM_onsubs = TMM_setup.calculate(options)

Abs_onsubs = RAT_TMM_onsubs['A_per_layer'][:,1] + RAT_TMM_onsubs['A_per_layer'][:,2]     # absorption in GaAs + InGaP
# indexing of A_per_layer is [wavelengths, layers]

R_onsubs = RAT_TMM_onsubs['R']

# ================================================= #
# Device with planar silver
print("Calculating planar Ag mirror device...")

solar_cell_TMM = SolarCell([Layer(material=InGaP, width=si('19nm')),
                        Layer(material=GaAs, width=si('87nm')),
                        Layer(material=InAlP, width=si('17nm'))],
                           substrate=Ag)

TMM_setup = tmm_structure(solar_cell_TMM, incidence=Air, transmission=Ag)

RAT_TMM = TMM_setup.calculate(options)

Abs_TMM = RAT_TMM['A_per_layer'][:, 0] + RAT_TMM['A_per_layer'][:, 1]
Abs_TMM_GaAsonly = RAT_TMM['A_per_layer'][:, 1]
Abs_TMM_InAlPonly = RAT_TMM['A_per_layer'][:, 2]
Abs_TMM_InGaPonly = RAT_TMM['A_per_layer'][:, 0]
R_TMM = RAT_TMM['R']

# ================================================= #
# Setting things up for RCWA calculation - DTL device
print("Calculating nanophotonic grating device...")

x = 500

# lattice vectors for the grating. Units are in nm!
size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

# options for RCWA. angles are in degrees.
# A_per_order = True means the absorption per order per layer is calculated
options.A_per_order = True

# RCWA options for S4
ropt = dict(LatticeTruncation='Circular',
            DiscretizedEpsilon=False,
            DiscretizationResolution=8,
            PolarizationDecomposition=True,
            PolarizationBasis='Default',
            LanczosSmoothing=dict(Power=2, Width=1),
            SubpixelSmoothing=False,
            ConserveMemory=False,
            WeismannFormulation=False,
            Verbosity=0)

options.S4_options = ropt

# grating layers
grating = [Layer(width=si(100, 'nm'), material=SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                 'radius': 115, 'angle': 0}])] # actual grating part of grating


# DTL device without anti-reflection coating
solar_cell = SolarCell([Layer(material=InGaP, width=si('19nm')),
                        Layer(material=GaAs, width=si('87nm')),
                        Layer(material=InAlP, width=si('17nm'))] + grating,
                       substrate=Ag)

# make RCWA structure
S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

# calculate
if calc:
    RAT = S4_setup.calculate(options)
    np.save(file_path+'/results/DTL_noARC_results_' + str(options.orders) + 'orders.npy', RAT)

else:
    RAT = np.load(file_path+'/results/DTL_noARC_results_' + str(options.orders) + 'orders.npy', allow_pickle=True).item()

Abs_DTL = RAT['A_per_layer'][:,1] + RAT['A_per_layer'][:,0]     # absorption in GaAs + InGaP

Abs_DTL_GaAs = RAT['A_per_layer'][:, 1]

R_DTL = RAT['R']

# ================================================= #
# Setting things up for RCWA calculation - DTL device with ARC
print("Calculating nanophotonic grating device with ARC...")

# DTL device without anti-reflection coating
solar_cell = SolarCell([Layer(material=Al2O3, width=si('70nm')),
                        Layer(material=InGaP, width=si('19nm')),
                        Layer(material=GaAs, width=si('87nm')),
                        Layer(material=InAlP, width=si('17nm'))] + grating,
                       substrate=Ag)

# make RCWA structure
S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

# calculate
if calc:
    RAT_ARC = S4_setup.calculate(options)
    np.save(file_path + '/results/DTL_ARC_results_' + str(options.orders) + 'orders.npy', RAT_ARC)

else:
    RAT_ARC = np.load(file_path+'/results/DTL_ARC_results_' + str(options.orders) + 'orders.npy', allow_pickle=True).item()


Abs_DTL_ARC = RAT_ARC['A_per_layer'][:,2] + RAT_ARC['A_per_layer'][:,1]    # absorption in GaAs + InGaP

R_DTL_ARC = RAT_ARC['R']

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
plt.legend(loc='upper right')
plt.show()

fig = plt.figure(figsize=(6.4, 4.8))
plt.stackplot(wavelengths*1e9,
              [100*Abs_TMM_GaAsonly, 100*Abs_TMM_InGaPonly, 100*Abs_TMM_InAlPonly],
              colors=pal,
              labels=['Absorbed in GaAs', 'Absorbed in InGaP', 'Absorbed in InAlP'])
plt.xlim(300, 950)
plt.ylim(0, 90)
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE (%)')
plt.legend(loc='upper right')
plt.show()

# Calculate photogenerated currents:
Ag = 0.1 * q * np.trapz(Abs_TMM*AM0, wavelengths)
DTL = 0.1 * q * np.trapz(Abs_DTL*AM0, wavelengths)
onsubs = 0.1 * q * np.trapz(Abs_onsubs*AM0, wavelengths)

print('On substrate device current: %.1f mA/cm2 ' % onsubs)
print('Planar Ag mirror device current: %.1f mA/cm2 ' % Ag)
print('Nanophotonic grating device current: %.1f mA/cm2 ' % DTL)