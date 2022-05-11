# In this first set of examples, we will look at a very simple planar Si solar cell.

# In the first two scripts, we mostly focused on different optical models and how they can be applied to an Si cell.
# Here we will look at different electrical models, roughly in increasing order of how realistic they are expected to be:
# 1. Two-diode model (2D)
# 2. Detailed balance (DB)
# 3. Depletion approximation (DA)
# 4. Poisson drift-diffusion solver (PDD)

import numpy as np
import matplotlib.pyplot as plt

from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.absorption_calculator import OptiStack, calculate_rat

from solcore import material, si

from solcore.interpolate import interp1d

# define some materials:

GaAs = material("GaAs")()
Al2O3 = material("Al2O3")()
Ag = material("Ag")()

wavelengths = si(np.linspace(300, 950, 200), "nm")

# we are going to do an optical calculation first to get absorption for a GaAs layer; we will use this as an
# estimate for the EQE as input for the two-diode model

OS = OptiStack([Layer(si("3um"), GaAs)], substrate=Ag)

# calculate reflection/absorption/transmission (note that we have to give the wavelength to this function in
# nm rather than m!)
RAT = calculate_rat(OS, wavelength=wavelengths*1e9)

# create a function which interpolates the absorption - note that we pass a function which returns the absorption
# when given a wavelength to the Junction, rather than a table of values!
eqe_func = interp1d(wavelengths, RAT["A"])

# Define the 2D junction with reasonable parameters for GaAs. The units of j01 and j01 are A/m^2. The units
# for the resistances are (Ohm m)^2. We use the standard ideality factors (1 and 2 respectively) for the two
# diodes
twod_junction = Junction(kind='2D', n1=1, n2=2, j01=3e-17, j02=1e-7, Rseries=4e-3,
                         Rshunt=1e4, eqe=eqe_func)

# Define two instances of a detailed-balance type junction. In both cases, there will be a sharp absorption onset
# at the bandgap (1.42 eV for GaAs). By specifying A, we set the fraction of light above the bandgap that is absorbed
# (A = 1 means 100% absorption above the gap).
db_junction_A1 = Junction(kind='DB', Eg=1.42, A=1, R_shunt=1e4, n=3.5)
db_junction = Junction(kind='DB', Eg=1.42, A=0.8, R_shunt=1e4, n=3.5)

V = np.linspace(0, 1.5, 200)
Vin = np.linspace(-2, 2.61, 201)

# set some options and define solar cells based on these junctions:

opts = {'voltages': V, 'light_iv': True, 'wavelength': wavelengths,
                                    'mpp': True, "internal_voltages": Vin}

solar_cell_db_A1 = SolarCell([db_junction_A1])
solar_cell_db = SolarCell([db_junction])
solar_cell_2d = SolarCell([twod_junction])

# calculate and plot the IV curves:

solar_cell_solver(solar_cell_db_A1, 'iv', user_options=opts)
solar_cell_solver(solar_cell_db, 'iv', user_options=opts)
solar_cell_solver(solar_cell_2d, 'iv', user_options=opts)

plt.figure()
plt.plot(*solar_cell_db_A1.iv["IV"], label='Detailed balance (Eg = 1.44 eV, A = 1)')
plt.plot(*solar_cell_db.iv["IV"], label='Detailed balance (Eg = 1.44 eV, A = 0.8)')
plt.plot(*solar_cell_2d.iv["IV"], '--', label='Two-diode')
plt.xlim(0, 1.5)
plt.ylim(0, 500)
plt.legend()
plt.show()

# As we expect, the two DB solar cells have a very similar shape, but the A = 1 case has a higher Jsc.


# Now let's consider the two slightly more complex models, which will actually take into account the absorption profile
# of light in the cell:

T = 293 # ambient temperature

window = material('AlGaAs')(T=T, Na=si("5e18cm-3"), Al=0.8)
p_GaAs = material('GaAs')(T=T, Na=si("1e18cm-3"), electron_diffusion_length=si("400nm"))
n_GaAs = material('GaAs')(T=T, Nd=si("8e16cm-3"), hole_diffusion_length=si("8um"))
bsf = material('GaAs')(T=T, Nd=si("2e18cm-3"))

SC_layers = [
                   Layer(width=si('150nm'), material=p_GaAs, role="Emitter"),
                   Layer(width=si('2850nm'), material=n_GaAs, role="Base"),
                   Layer(width=si('200nm'), material=bsf, role="BSF")]

# sn and sp are the surface recombination velocities (in m/sec). sn is the SRV for the n-doped junction, sp for the
# p-doped junction.

solar_cell_da = SolarCell(
    [Layer(width=si("90nm"), material=Al2O3), Layer(width=si('20nm'), material=window, role="Window"),
     Junction(SC_layers, sn=5e4, sp=5e4, kind='DA')],
    R_series=0, substrate=Ag
)

solar_cell_pdd = SolarCell(
    [Layer(width=si("90nm"), material=Al2O3), Layer(width=si('20nm'), material=window, role="Window"),
     Junction(SC_layers, sn=5e4, sp=5e4, kind='PDD')],
    R_series=0, substrate=Ag
)

opts["optics_method"] = "TMM"
opts["position"] = 1e-10
opts["no_back_reflection"] = False

solar_cell_solver(solar_cell_da, "iv", user_options=opts)
solar_cell_solver(solar_cell_da, "qe", user_options=opts)

solar_cell_solver(solar_cell_pdd, "iv", user_options=opts)
solar_cell_solver(solar_cell_pdd, "qe", user_options=opts)

plt.figure()
plt.plot(*solar_cell_da.iv["IV"], label="Depletion approximation")
plt.plot(*solar_cell_pdd.iv["IV"], '--', label="Poisson Drift Diffusion")
plt.xlim(0, 1.2)
plt.ylim(0, 330)
plt.legend()
plt.show()


plt.figure()
plt.plot(wavelengths*1e9, 100*solar_cell_da[2].eqe(wavelengths), 'k-', label="EQE (DA)")
plt.plot(wavelengths*1e9, 100*solar_cell_pdd[2].eqe(wavelengths), 'k--', label="EQE (PDD)")
plt.plot(wavelengths*1e9, 100*solar_cell_da[2].layer_absorption, 'r-', label="A (DA)")
plt.plot(wavelengths*1e9, 100*solar_cell_pdd[2].layer_absorption, 'b--', label="A (PDD)")
plt.legend()
plt.show()