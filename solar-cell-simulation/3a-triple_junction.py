import numpy as np
import matplotlib.pyplot as plt

from solcore import siUnits, material, si
from solcore.solar_cell import SolarCell
from solcore.structure import Junction, Layer
from solcore.solar_cell_solver import solar_cell_solver
from solcore.light_source import LightSource
from solcore.absorption_calculator import search_db

all_materials = []

wl = np.linspace(300, 1850, 700) * 1e-9

light_source = LightSource(source_type='standard', x=wl, version='AM1.5g')
# We need to build the solar cell layer by layer.
# We start from the AR coating. In this case, we load it from an an external file

# Note: you need to have downloaded the refractiveindex.info database for these to work. See []

MgF2_pageid = search_db("MgF2/Rodriguez-de Marcos")[0][0]
ZnS_pageid = search_db("ZnS/Querry")[0][0]
MgF2 = material(str(MgF2_pageid), nk_db=True)()
ZnS = material(str(ZnS_pageid), nk_db=True)()

ARC = [Layer(si("100nm"), MgF2), Layer(si("15nm"), ZnS), Layer(si("15nm"), MgF2), Layer(si("50nm"), ZnS)]

# TOP CELL - GaInP
# Now we build the top cell, which requires the n and p sides of GaInP and a window layer.
# We also load the absorption coefficient from an external file. We also add some extra parameters needed for the
# calculation such as the minority carriers diffusion lengths
AlInP = material("AlInP")
InGaP = material("GaInP")
window_material = AlInP(Al=0.52)
top_cell_n_material = InGaP(In=0.49, Nd=siUnits(2e18, "cm-3"), hole_diffusion_length=si("200nm"))
top_cell_p_material = InGaP(In=0.49, Na=siUnits(1e17, "cm-3"), electron_diffusion_length=si("1um"))

all_materials.append(window_material)
all_materials.append(top_cell_n_material)
all_materials.append(top_cell_p_material)

# MID CELL  - InGaAs
# We add manually the absorption coefficient of InGaAs since the one contained in the database doesn't cover
# enough range, keeping in mind that the data has to be provided as a function that takes wavelengths (m) as input and
# returns absorption (1/m)
GaAs = material("GaAs")

mid_cell_n_material = GaAs(Nd=siUnits(3e18, "cm-3"), hole_diffusion_length=si("500nm"))
mid_cell_p_material = GaAs(Na=siUnits(1e17, "cm-3"), electron_diffusion_length=si("5um"))

all_materials.append(mid_cell_n_material)
all_materials.append(mid_cell_p_material)

# BOTTOM CELL - Ge
# We add manually the absorption coefficient of Ge since the one contained in the database doesn't cover
# enough range.
Ge = material("Ge")

bot_cell_n_material = Ge(Nd=siUnits(2e18, "cm-3"), hole_diffusion_length=si("800nm"), hole_mobility=0.01)
bot_cell_p_material = Ge(Na=siUnits(1e17, "cm-3"), electron_diffusion_length=si("50um"), electron_mobility=0.1)

all_materials.append(bot_cell_n_material)
all_materials.append(bot_cell_p_material)

# And, finally, we put everything together, adding also the surface recombination velocities. We also add some shading
# due to the metallisation of the cell = 8%, and indicate it has an area of 0.7x0.7 mm2 (converted to m2)
solar_cell = SolarCell(
    ARC +
    [
        Junction([Layer(si("20nm"), material=window_material, role='window'),
                  Layer(si("100nm"), material=top_cell_n_material, role='emitter'),
                  Layer(si("560nm"), material=top_cell_p_material, role='base'),
                  ], sn=1, sp=1, kind='DA'),
        Junction([Layer(si("200nm"), material=mid_cell_n_material, role='emitter'),
                  Layer(si("3000nm"), material=mid_cell_p_material, role='base'),
                  ], sn=1, sp=1, kind='DA'),
        Junction([Layer(si("400nm"), material=bot_cell_n_material, role='emitter'),
                  Layer(si("100um"), material=bot_cell_p_material, role='base'),
                  ], sn=1, sp=1, kind='DA'),
    ], shading=0.05, R_series=2e-6)

position = len(solar_cell) * [0.1 * 1e-9]
position[-1] = 10e-9

plt.figure()

solar_cell_solver(solar_cell, 'qe', user_options={'wavelength': wl, 'optics_method': "TMM",
                                                  'position': position, 'recalculate_absorption': True})

plt.plot(wl * 1e9, solar_cell[4].eqe(wl) * 100, 'b', label='GaInP (TMM)')
plt.plot(wl * 1e9, solar_cell[5].eqe(wl) * 100, 'g', label='InGaAs (TMM)')
plt.plot(wl * 1e9, solar_cell[6].eqe(wl) * 100, 'r', label='Ge (TMM)')
plt.plot(wl * 1e9, 100 * (1 - solar_cell.reflected), 'k--', label='1-R (TMM)')
plt.plot(wl * 1e9, solar_cell[4].layer_absorption * 100, 'b--')
plt.plot(wl * 1e9, solar_cell[5].layer_absorption * 100, 'g--')
plt.plot(wl * 1e9, solar_cell[6].layer_absorption * 100, 'r--')

solar_cell_solver(solar_cell, 'qe', user_options={'wavelength': wl, 'optics_method': "BL",
                                                  'position': position, 'recalculate_absorption': True})

plt.plot(wl * 1e9, solar_cell[4].eqe(wl) * 100, 'b-', alpha=0.5, label='GaInP (BL)')
plt.plot(wl * 1e9, solar_cell[5].eqe(wl) * 100, 'g-', alpha=0.5, label='InGaAs (BL)')
plt.plot(wl * 1e9, solar_cell[6].eqe(wl) * 100, 'r-', alpha=0.5, label='Ge (BL)')
plt.legend()
plt.ylim(0, 100)
plt.ylabel('EQE (%)')
plt.xlabel('Wavelength (nm)')
plt.tight_layout()
plt.show()

V = np.linspace(0, 3, 300)
solar_cell_solver(solar_cell, 'iv', user_options={'voltages': V, 'light_iv': True,
                                                  'wavelength': wl, 'mpp': True,
                                                  'light_source': light_source})

plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], 'k', linewidth=3, label='Total')
plt.plot(V, -solar_cell[4].iv(V), 'b', label='GaInP')
plt.plot(V, -solar_cell[5].iv(V), 'g', label='InGaAs')
plt.plot(V, -solar_cell[6].iv(V), 'r', label='Ge')
plt.text(1.4, 200, 'Efficieny (%): ' + str(np.round(solar_cell.iv['Eta'] * 100, 1)))
plt.text(1.4, 180, 'FF (%): ' + str(np.round(solar_cell.iv['FF'] * 100, 1)))
plt.text(1.4, 160, r'V$_{oc}$ (V): ' + str(np.round(solar_cell.iv["Voc"], 2)))
plt.text(1.4, 140, r'I$_{sc}$ (A/m$^2$): ' + str(np.round(solar_cell.iv["Isc"], 2)))

plt.legend()
plt.ylim(0, 250)
plt.xlim(0, 3)
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)')

plt.show()

# Effects of concentration:

concentration = np.linspace(np.log(1), np.log(3000), 20)
concentration = np.exp(concentration)

Effs = np.zeros_like(concentration)
Vocs = np.zeros_like(concentration)
Iscs = np.zeros_like(concentration)

V = np.linspace(0, 3.5, 300)

for i1, conc in enumerate(concentration):
    light_conc = LightSource(source_type='standard', x=wl, version='AM1.5d', concentration=conc)
    solar_cell_solver(solar_cell, 'iv', user_options={'voltages': V, 'light_iv': True,
                                                      'wavelength': wl, 'mpp': True,
                                                      'light_source': light_conc})

    Effs[i1] = solar_cell.iv["Eta"] * 100
    Vocs[i1] = solar_cell.iv["Voc"]
    Iscs[i1] = solar_cell.iv["Isc"]

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.semilogx(concentration, Effs, '-o')
plt.ylabel('Efficiency (%)')
plt.xlabel('Concentration')

plt.subplot(132)
plt.semilogx(concentration, Vocs, '-o')
plt.ylabel(r'V$_{OC}$ (V)')
plt.xlabel('Concentration')

plt.subplot(133)
plt.plot(concentration, Iscs / 10000, '-o')
plt.ylabel(r'J$_{SC}$ (A/cm$^2$)')
plt.xlabel('Concentration')
plt.tight_layout()
plt.show()
