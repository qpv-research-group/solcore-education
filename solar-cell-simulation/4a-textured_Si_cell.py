# In this example, we will introduce RayFlare, which is a package which is closely interlinked with Solcore and extends
# its optical capabilities. One of the features it has is a ray-tracer, which is useful when modelling e.g. Si solar
# cells with textured surfaces. We will compare the result with PVLighthouse's wafer ray tracer.

import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from rayflare.utilities import make_absorption_function

from solcore import material, si
from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver

file_path = os.path.dirname(os.path.abspath(__file__))

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)

# setting up Solcore materials
Air = material('Air')()
Si_base = material("Si")
Si_RT = Si_base()

# number of x and y points to scan across. Decrease this to speed up the calculation (but increase noise in results).
nxy = 25

# If True, run the calculation; if False, load the result of the previous calculation. Will need to run at least once to
# generate the results!
calc = True
# We use this 'switch' to avoid re-running the whole ray-tracing calculation (which can be time-consuming) each time we
# want to look at the results.

# setting options
wl = np.linspace(300, 1201, 50) * 1e-9
options = default_options()
options.wavelengths = wl

# Number of point to scan across in the x & y directions in the unit cell:
options.nx = nxy
options.ny = nxy

# Number of rays to be traced at each wavelength:
options.n_rays = 2 * nxy ** 2
options.depth_spacing = si('50nm')
options.parallel = True  # this is the default - if you do not want the code to run in parallel, change to False

PVlighthouse = np.loadtxt(file_path + '/data/RAT_data_300um_2um_55.csv', delimiter=',', skiprows=1)

flat_surf = planar_surface(size=2) # pyramid size in microns
triangle_surf = regular_pyramids(55, upright=False, size=2)

# set up ray-tracing options
rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                    materials = [Si_RT],
                    widths=[si('300um')], incidence=Air, transmission=Air)

if calc:

    result = rtstr.calculate_profile(options)

    result_RAT = np.vstack((options['wavelengths']*1e9,
                        result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T
    profile_rt = result['profile']

    np.savetxt(file_path + '/results/rayflare_fullrt_300um_2umpyramids_300_1200nm.txt', result_RAT)
    np.savetxt(file_path + '/results/rayflare_fullrt_300um_2umpyramids_300_1200nm_profile.txt', result['profile'])

else:

    result_RAT = np.loadtxt(file_path + '/results/rayflare_fullrt_300um_2umpyramids_300_1200nm.txt')
    profile_rt = np.loadtxt(file_path + '/results/rayflare_fullrt_300um_2umpyramids_300_1200nm_profile.txt')


plt.figure()
plt.plot(result_RAT[:,0], result_RAT[:,1], '-o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(result_RAT[:,0], result_RAT[:,2], '-o', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(result_RAT[:,0], result_RAT[:,3], '-o', color=pal[2], label=r'T', fillstyle='none')
plt.plot(result_RAT[:,0], result_RAT[:,4], '-o', color=pal[3], label=r'A', fillstyle='none')
plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 2], '--', color=pal[0])
plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 9], '--', color=pal[2])
plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 3], '--', color=pal[1])
plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 5], '--', color=pal[3])
plt.plot(-1, -1, '-ok', label='RayFlare')
plt.plot(-1, -1, '--k', label='PVLighthouse')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)
plt.legend()
plt.show()

# Now, we have just done an optical calculation; however, if we want to use this information to do a profile calculation,
# we can
_, diff_absorb_fn = make_absorption_function(profile_rt, rtstr, options, matrix_method=False)


# Now feed profile into Solcore

n_material_Si_width = si("500nm")
p_material_Si_width = rtstr.widths[0] - n_material_Si_width

n_material_Si = Si_base(Nd=si(1e21, "cm-3"), hole_diffusion_length=si("10um"),
                electron_mobility=50e-4, relative_permittivity=11.68)
p_material_Si = Si_base(Na=si(1e16, "cm-3"), electron_diffusion_length=si("290um"),
                hole_mobility=400e-4, relative_permittivity=11.68)

from solcore.solar_cell_solver import default_options as defaults_solcore

options_sc = defaults_solcore
options_sc.optics_method = "external"
options_sc.position = np.arange(0, rtstr.width, options.depth_spacing)
options_sc.light_iv = True
options_sc.wavelength = wl
options_sc.theta = options.theta_in*180/np.pi
V = np.linspace(0, 1, 200)
options_sc.voltages = V

_, diff_absorb_fn = make_absorption_function(profile_rt, rtstr, options, matrix_method=False)

solar_cell = SolarCell(
    [
        Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                  Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                 sn=1, sp=1, kind='DA')
    ],
    external_reflected=result_RAT[:,1],
    external_absorbed=diff_absorb_fn)

solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(wl*1e9, solar_cell.absorbed, 'k-', label='Absorbed (integrated)')
plt.plot(wl*1e9, solar_cell[0].eqe(wl), 'r-', label='EQE')
plt.plot(wl*1e9, result_RAT[:,4], 'r--', label='Absorbed - RT')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()

plt.figure()
plt.plot(V, -solar_cell[0].iv(V), 'r', label='Si')
plt.ylim(-20, 400)
plt.xlim(0, 0.8)
plt.legend()
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)') #The expected values of Isc and Voc are 372 A/m^2 and 0.63 V respectively
plt.show()