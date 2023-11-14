import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver
from solcore.structure import Junction

# Load the AM1.5G solar spectrum
wl = np.linspace(300, 4000, 4000) * 1e-9    # wl contains the x-coordinate in wavelength
am15g = LightSource(source_type='standard', x=wl, version='AM1.5g')

V = np.linspace(0, 3, 500)
db_gainp = Junction(kind='DB', T=300, Eg=1.95, A=1, R_shunt=np.inf, n=1) 

db_gaas = Junction(kind='DB', T=300, Eg=1.42, A=1, R_shunt=np.inf, n=1) 

my_solar_cell = SolarCell([db_gainp, db_gaas], T=300, R_series=0)

solar_cell_solver(my_solar_cell, 'iv',
                      user_options={'T_ambient': 300, 'db_mode': 'top_hat', 'voltages': V, 'light_iv': True,
                                    'wavelength': wl,
                                    'mpp': True, 'light_source': am15g})

plt.figure()
plt.title('Limiting Efficiency IV curve for GaInP/GaAs tandem')
plt.plot(V, my_solar_cell.iv.IV[1], 'k', label="Total")

# plot I-V of individual junctions:
plt.plot(V, -my_solar_cell[0].iv(V), 'r', label="GaInP")
plt.plot(V, -my_solar_cell[1].iv(V), 'b', label="GaAs")

plt.ylim(0, 200)
plt.xlim(0, 3)
plt.text(0.1,150,f'Jsc: {my_solar_cell.iv.Isc:.2f}')
plt.text(0.1,130,f'Voc: {my_solar_cell.iv.Voc:.2f}')
plt.text(0.1,110,f'Pmax: {my_solar_cell.iv.Pmpp:.2f}')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.legend()
plt.show()

# saving data in a .txt file:
IV_data = np.transpose(np.array(my_solar_cell.iv.IV))

np.savetxt("2J_DB_IV.txt", IV_data)


# test that the data saved correctly:

loaded_data = np.loadtxt("2J_DB_IV.txt")

plt.figure()
plt.title('Check saved/loaded data')
plt.plot(loaded_data[:,0], loaded_data[:,1], 'k', label="Total")

plt.ylim(0, 200)
plt.xlim(0, 3)
plt.text(0.1,150,f'Jsc: {my_solar_cell.iv.Isc:.2f}')
plt.text(0.1,130,f'Voc: {my_solar_cell.iv.Voc:.2f}')
plt.text(0.1,110,f'Pmax: {my_solar_cell.iv.Pmpp:.2f}')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.legend()
plt.show()