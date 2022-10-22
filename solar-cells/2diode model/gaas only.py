import numpy as np

from solcore.solar_cell import SolarCell
from solcore.light_source import LightSource
from solcore.solar_cell_solver import solar_cell_solver
from solcore.structure import Junction
from solcore.spice.pv_module_solver import spice_junction
from solcore.spice.pv_module_solver import solve_pv_module

gaas_junction = Junction(kind='2D', T=300,  A=1,n1=1,n2=2,R_series=0.000,R_shunt=1e14, j01=6E-18, j02=1E-14,jsc=300) 


my_solar_cell = SolarCell([gaas_junction],T=300, R_series=0.0, area=1)

wl = np.linspace(350, 2000, 301) * 1e-9
light_source = LightSource(source_type='standard', version='AM1.5g', x=wl, output_units='photon_flux_per_m')

# options = {'light_iv': True, 'wavelength': wl, 'light_source': light_source}
V = np.linspace(0, 5, 500)
solar_cell_solver(my_solar_cell, 'iv',
                  user_options={'T_ambient': 300, 'db_mode': 'top_hat', 'voltages': V, 'light_iv': True,
                                        'internal_voltages': np.linspace(-6, 5, 1100), 'wavelength': wl,
                                        'mpp': True, 'light_source': light_source})

print(my_solar_cell.iv["Eta"])
print(my_solar_cell.iv["Pmpp"])
print(my_solar_cell.iv["FF"])
print(my_solar_cell.iv["Voc"])
print(my_solar_cell.iv["Isc"])
	
	