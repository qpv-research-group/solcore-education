import numpy as np
import matplotlib.pyplot as plt

from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.light_source import LightSource
from solcore.absorption_calculator import OptiStack, calculate_rat

from solcore import material, si

from solcore.interpolate import interp1d

Si = material("Si")()

OS = OptiStack([Layer(si("300um"), Si)])

wavelengths = si(np.linspace(300, 1200, 200), "nm")

RAT = calculate_rat(OS, wavelength=wavelengths*1e9, coherent=False, coherency_list=["i"])

eqe_func = interp1d(wavelengths, RAT["A"])


db_junction = Junction(kind='DB', Eg=1.1, A=1, R_shunt=np.inf, n=3.5)

twod_junction = Junction(kind='2D', n1=1, n2=2, j01=1e-8, j02=1e-6, Rseries=1e-5, Rshunt=1e15, eqe=eqe_func)

V = np.linspace(0, 1, 200)


opts = {'db_mode': 'top_hat', 'voltages': V, 'light_iv': True, 'wavelength': wavelengths,
                                    'mpp': True}

solar_cell_db = SolarCell([db_junction])
solar_cell_2d = SolarCell([twod_junction])

solar_cell_solver(solar_cell_db, 'iv', user_options=opts)
solar_cell_solver(solar_cell_2d, 'iv', user_options=opts)

plt.figure()
plt.plot(*solar_cell_db.iv["IV"])
plt.plot(*solar_cell_2d.iv["IV"], '--')
plt.xlim(0, 1)
plt.ylim(0, 500)
plt.show()


Si_emitter = material("Si")(Na=si(1e21, "cm-3"), electron_diffusion_length=si("500nm"),
                hole_mobility=50e-4)
Si_base = material("Si")(Nd=si(1e16, "cm-3"), hole_diffusion_length=si("100um"),
                electron_mobility=50e-4)
SC_layers = [Layer(si("500nm"), Si_emitter, role="emitter"),
             Layer(si("100um"), Si_base, role="base")]

solar_cell_da = SolarCell([Junction(SC_layers, kind="DA")])

solar_cell_pdd = SolarCell([Junction(SC_layers, kind="PDD")])

solar_cell_solver(solar_cell_da, 'iv', user_options=opts)

solar_cell_solver(solar_cell_pdd, 'iv', user_options=opts)

plt.figure()
plt.plot(*solar_cell_da.iv["IV"])
plt.plot(*solar_cell_pdd.iv["IV"], '--')
plt.xlim(0, 1)
plt.ylim(0, 500)
plt.show()