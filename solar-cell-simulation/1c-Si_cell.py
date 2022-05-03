# In this first set of examples, we will look at a very simple planar Si solar cell.

# In the first two scripts, we mostly focused on different optical models and how they can be applied to an Si cell.
# Here we will look at different electrical models, roughly in increasing order of how realistic they are expected to be:
# 1. Detailed balance (DB)
# 2. Two-diode model (2D)
# 3. Depletion approximation (DA)
# 4. Poisson drift-diffusion solver (PDD)


import numpy as np
import matplotlib.pyplot as plt

from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.absorption_calculator import OptiStack, calculate_rat

from solcore import material, si

from solcore.interpolate import interp1d

Si = material("Si")()

OS = OptiStack([Layer(si("300um"), Si)])

wavelengths = si(np.linspace(300, 1200, 200), "nm")

RAT = calculate_rat(OS, wavelength=wavelengths*1e9, coherent=False, coherency_list=["i"])

eqe_func = interp1d(wavelengths, RAT["A"])


db_junction_A1 = Junction(kind='DB', Eg=1.1, A=1, R_shunt=np.inf, n=3.5)
db_junction = Junction(kind='DB', Eg=1.1, A=0.8, R_shunt=np.inf, n=3.5)

twod_junction = Junction(kind='2D', n1=1, n2=2, j01=4.53e-11/0.6084, j02=3.02e-6/0.6084, Rseries=1.4e-2*0.6084,
                         Rshunt=103.3*0.6084, eqe=eqe_func)

V = np.linspace(0, 1, 200)

opts = {'db_mode': 'top_hat', 'voltages': V, 'light_iv': True, 'wavelength': wavelengths,
                                    'mpp': True}

solar_cell_db_A1 = SolarCell([db_junction_A1])
solar_cell_db = SolarCell([db_junction])
solar_cell_2d = SolarCell([twod_junction])

solar_cell_solver(solar_cell_db_A1, 'iv', user_options=opts)
solar_cell_solver(solar_cell_db, 'iv', user_options=opts)
solar_cell_solver(solar_cell_2d, 'iv', user_options=opts)

plt.figure()
plt.plot(*solar_cell_db_A1.iv["IV"], label='Detailed balance (Eg = 1.1 eV, A = 1)')
plt.plot(*solar_cell_db.iv["IV"], label='Detailed balance (Eg = 1.1 eV, A = 0.8)')
plt.plot(*solar_cell_2d.iv["IV"], '--', label='Two-diode')
plt.xlim(0, 1)
plt.ylim(0, 500)
plt.legend()
plt.show()


Si_emitter = material("Si")(Na=si(1e17, "cm-3"))
Si_base = material("Si")(Nd=si(1e15, "cm-3"))

SC_layers = [Layer(si("200nm"), Si_emitter, role="emitter"),
             Layer(si("5um"), Si_base, role="base")]

opts["optics_method"] = "TMM"

solar_cell_da = SolarCell([Junction(SC_layers, kind="DA")])

solar_cell_pdd = SolarCell([Junction(SC_layers, kind="PDD")])

# solar_cell_solver(solar_cell_da, 'iv', user_options=opts)

# solar_cell_solver(solar_cell_pdd, 'iv', user_options=opts)
# solar_cell_solver(solar_cell_da, 'qe', user_options=opts)

solar_cell_solver(solar_cell_pdd, 'qe', user_options=opts)

# plt.figure()
# plt.plot(*solar_cell_da.iv["IV"])
# plt.plot(*solar_cell_pdd.iv["IV"], '--')
# plt.xlim(0, 1)
# plt.ylim(0, 500)
# plt.show()


plt.figure()
# plt.plot(wavelengths*1e9, 100*solar_cell_da[0].eqe(wavelengths), 'k-', label="Depletion approximation")
plt.plot(wavelengths*1e9, 100*solar_cell_pdd[0].eqe(wavelengths), 'k--', label ="PDD")
# plt.plot(wavelengths*1e9, 100*solar_cell_da[0].layer_absorption, 'r-')
plt.plot(wavelengths*1e9, 100*solar_cell_pdd[0].layer_absorption, 'b--')
plt.legend()
plt.show()