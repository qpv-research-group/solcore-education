# In example 4a, we looked at a 'solar cell' made of a single layer of Si with pyramidal texturing. In reality, a
# solar cell will have a more complicated structure with thin layers deposited on the front side to act as e.g. selective
# transport layers for carriers. This adds a layer of complication to the ray-tracing process, because we can no longer
# rely on the Fresnel equations to calculate the angle and wavelength-dependent reflection and transmission probabilities;
# we might get absorption in the surface layers, and we need to take into account interference in the surface layers.
# To do this, we can combine ray-tracing and the transfer-matrix method; we can calculate the reflection, absorption and
# transmission probabilities using TMM, and use thosee probabilities in our ray-tracing calculations. In RayFlare, this
# functionality is implemented as part of the angular redistribution matrix functionality.

from solcore import material, si
from solcore.light_source import LightSource
from solcore.constants import q
from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import default_options as defaults_solcore, solar_cell_solver

from rayflare.textures import regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options
from rayflare.angles import make_angle_vector
from rayflare.utilities import make_absorption_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

from cycler import cycler


import os

# new materials from data - uncomment to add to database

from solcore.material_system import create_new_material
cur_path = os.path.dirname(os.path.abspath(__file__))
create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
create_new_material('ITO_measured', os.path.join(cur_path, 'data/front_ITO_n.txt'), os.path.join(cur_path, 'data/front_ITO_k.txt'))

# matrix multiplication
wavelengths = np.linspace(300, 1200, 80)*1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'HIT_example'
options.n_rays = 10000 # Reduce this (or the number of wavelengths) to speed up the example!
options.n_theta_bins = 20
options.nx = 5
options.ny = 5
options.I_thresh = 0.005
options.bulk_profile = True
_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])
options.bulk_profile = True

Si = material('Si')()
Air = material('Air')()
ITO = material('ITO_measured')()

Ag = material('Ag')()
aSi = material('aSi_i')()

# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [Layer(80e-9, ITO), Layer(13e-9, aSi)]
back_materials = [Layer(13e-9, aSi), Layer(240e-9, ITO)]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model

surf = regular_pyramids(elevation_angle=55, upright=True)
surf_back = regular_pyramids(elevation_angle=55, upright=False)

front_surf = Interface('RT_TMM', texture=surf, layers=front_materials, name='HIT_front', coherent=True)
back_surf = Interface('RT_TMM', texture=surf_back, layers=back_materials, name='HIT_back', coherent=True)


bulk_Si = BulkLayer(170e-6, Si, name='Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options, save_location="current")

results = calculate_RAT(SC, options, save_location="current")

RAT = results[0]
results_per_pass = results[1]


R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)

results_per_layer_back = np.sum(results_per_pass['a'][1], 0)


allres = np.hstack((RAT['T'].T, results_per_layer_back,
                    RAT['A_bulk'].T, results_per_layer_front)).T

# calculated photogenerated current (Jsc with 100% EQE)

spectr_flux = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                           output_units='photon_flux_per_m', concentration=1).spectrum(wavelengths)[1]

Jph_Si = q * np.trapz(RAT['A_bulk'][0] * spectr_flux, wavelengths)/10 # mA/cm2

print("Photogenerated current in Si = %.1f mA/cm2" % Jph_Si)

pal = sns.cubehelix_palette(allres.shape[0] + 1, start=.5, rot=-.9)
pal.reverse()
cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)



# plot total R, A, T
fig = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
ax.plot(options['wavelengths']*1e9, R_escape + R_0, '--k', label=r'$R_{total}$')
ax.plot(options['wavelengths']*1e9, R_0, '-.k', label=r'$R_0$')
ax.stackplot(options['wavelengths']*1e9, allres,
             labels=['Ag (transmitted)', 'Back ITO', 'a-Si (back)', 'Bulk Si',
                     'a-Si (front)', 'Front ITO'
                     ])
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel('Absorption/Emissivity')
ax.set_xlim(min(options['wavelengths']*1e9), max(options['wavelengths']*1e9))
ax.set_ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


ysmoothed = gaussian_filter1d(np.vstack((allres, RAT["R"])), sigma=2, axis=1)

# plot total R, A, T - smoothed
fig = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
ax.stackplot(options['wavelengths']*1e9, ysmoothed,
             labels=['Ag (transmitted)', 'Back ITO', 'a-Si (back)', 'Bulk Si',
                     'a-Si (front)', 'Front ITO', 'R'
                     ])
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel('Absorption/Emissivity')
ax.set_xlim(min(options['wavelengths']*1e9), max(options['wavelengths']*1e9))
ax.set_ylim(0, 1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

profile_Si = results[3][0]
external_R = RAT['R'][0, :]

positions, absorb_fn = make_absorption_function([None, profile_Si, None], SC, options, True)

Si_SC = material("Si")
GaAs_SC = material("GaAs")
T = 300

p_material_Si = Si_SC(T=T, Na=si(1e21, "cm-3"), electron_diffusion_length=si("10um"),
                hole_mobility=50e-4, relative_permittivity=11.68)
n_material_Si = Si_SC(T=T, Nd=si(1e16, "cm-3"), hole_diffusion_length=si("290um"),
                electron_mobility=400e-4, relative_permittivity=11.68)


options_sc = defaults_solcore
options_sc.optics_method = "external"
options_sc.position = positions
options_sc.light_iv = True
options_sc.wavelength = wavelengths
options_sc.mpp = True
options_sc.theta = options.theta_in*180/np.pi
V = np.linspace(0, 2.5, 250)
options_sc.voltages = V

solar_cell = SolarCell([Layer(80e-9, ITO),
                   Layer(13e-9, aSi),
                   Junction([Layer(500e-9, p_material_Si, role="emitter"),
                             Layer(bulk_Si.width-500e-9, n_material_Si, role="base")], kind="DA"),
                   Layer(13e-9, aSi),
                   Layer(240e-9, ITO)],
                  external_reflected = external_R,
                                       external_absorbed = absorb_fn)



solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(options['wavelengths']*1e9, RAT["A_bulk"][0], 'r-')
plt.plot(wavelengths*1e9, solar_cell.absorbed, 'k--', label='Absorbed (integrated)')
plt.plot(wavelengths*1e9, solar_cell[2].eqe(wavelengths), 'b-', label='Si EQE')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()


plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], '-k')
plt.ylim(-20, 370)
plt.xlim(0, 0.85)
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)')
plt.show()