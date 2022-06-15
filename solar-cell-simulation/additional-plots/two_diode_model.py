import numpy as np
import matplotlib.pyplot as plt

from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.absorption_calculator import OptiStack, calculate_rat

from solcore import material, si

from solcore.interpolate import interp1d

import seaborn as sns

GaAs = material("GaAs")()
Al2O3 = material("Al2O3")()
Ag = material("Ag")()

wavelengths = si(np.linspace(300, 950, 200), "nm")

OS = OptiStack([Layer(si("3um"), GaAs)], substrate=Ag)

RAT = calculate_rat(OS, wavelength=wavelengths*1e9, no_back_reflection=False)

eqe_func = interp1d(wavelengths, RAT["A"])

V = np.linspace(0, 1.4, 200)

N_R = 6
R_series = np.insert(10 ** np.linspace(-6, -2, N_R - 1), 0, 0)
R_shunt = 10 ** np.linspace(-4, 1, N_R)

cols = sns.cubehelix_palette(N_R)

for light_iv, title in zip([False, True], ['Dark IV', 'Light IV']):

    opts = {'voltages': V, 'wavelength': wavelengths, 'light_iv': light_iv}

    plt.figure(figsize=(10,4))
    plt.subplot(121)

    for i1, Rs in enumerate(R_series):
        twod_junction = Junction(kind='2D', n1=1, n2=2, j01=1e-8, j02=1e-6,
                                 R_series=Rs, R_shunt=R_shunt[-1], eqe=eqe_func)
        solar_cell_2d = SolarCell([twod_junction])
        solar_cell_solver(solar_cell_2d, 'iv', user_options=opts)
        plt.semilogy(*np.abs(solar_cell_2d.iv["IV"]), color=cols[i1], label='%.2E' % Rs)

    plt.xlim(0, 1.5)
    plt.xlabel("V (V)")
    plt.ylabel("|J| (A/m$^2$)")
    plt.legend(title='R$_{series}$')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.title(title, loc='left')

    plt.subplot(122)

    for i1, Rsh in enumerate(R_shunt):
        twod_junction = Junction(kind='2D', n1=1, n2=2, j01=1e-8, j02=1e-6,
                                 R_series=R_series[1], R_shunt=Rsh, eqe=eqe_func)
        solar_cell_2d = SolarCell([twod_junction])
        solar_cell_solver(solar_cell_2d, 'iv', user_options=opts)
        plt.semilogy(*np.abs(solar_cell_2d.iv["IV"]), color=cols[i1], label='%.2E' % Rsh)

    plt.xlim(0, 1.5)
    plt.xlabel("V (V)")
    plt.ylabel("|J| (A/m$^2$)")
    plt.legend(title='R$_{shunt}$')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.show()

