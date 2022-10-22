import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import lambertw

from solcore.light_source import LightSource
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver
from solcore.structure import Junction

# Define fundamental physical constants
q=1.60217662E-19  # electronic charge [C]
k=1.38064852E-23/q   # Boltzmann constant [eV/K]
h=6.62607004E-34/q  # Planck constant expressed in [eV.s]
c=299792458  # Speed of light [m.s^-1]

# Load the AM1.5G solar spectrum
wl = np.linspace(300, 4000, 4000) * 1e-9    #wl contains the x-ordinate in wavelength
am15g = LightSource(source_type='standard', x=wl, version='AM1.5g')

# EG is the semiconductor bandgap.
eg=1.3   #  GaAs bandgap is 1.42eV
V = np.linspace(0, 1.3, 500)  # Define some voltages for IV calculation

# Detailed Balance calculation. We only need Eg and temperature T.
db_junction = Junction(kind='DB', T=298, Eg=eg, A=1, R_shunt=np.inf, n=1)
# A is cell area, R_shunt here is infinity, n is the refractive index

my_solar_cell = SolarCell([db_junction], T=298, R_series=0)
# In detail balance model, we only supply a value for Eg.
# The smallest possible J01 is calculated internally. J02 = 0 because the solar is ideal (perfect, no unecessary losses)
# The jsc value is also calculated internally.

# Performs the calculation (solves the solar cell), the results are left in my_solar_cell.
solar_cell_solver(my_solar_cell, 'iv',
                      user_options={'T_ambient': 298, 'db_mode': 'top_hat', 'voltages': V, 'light_iv': True,
                                    'internal_voltages': np.linspace(0, 1.3, 400), 'wavelength': wl,
                                    'mpp': True, 'light_source': am15g})


# Plotting the IV curve
plt.figure(1)   # Define a figure.

# Plot a graph of (V,I), so here the I (current) are stored in my_solar_cell.iv.IV[1]
plt.plot(V, my_solar_cell.iv.IV[1], 'k')

# Define x and y limits for graph
plt.ylim(0, 380)
plt.xlim(0, 1.2)

# Add text to the graph for Isc, Voc and Pmax
plt.text(0.1,300,f'Jsc {my_solar_cell.iv.Isc:.2f}')
plt.text(0.1,280,f'Voc {my_solar_cell.iv.Voc:.2f}')
plt.text(0.1,260,f'Pmax {my_solar_cell.iv.Pmpp:.2f}')

# Add labels to the axes,
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')

# Finally display the graph! 
plt.show()