# The Shockley Queisser IV curve implemented in SolCore
# Comparing the internal IV model in SolCore with the analytical expressions from [Pusch et al., JPV (2019)](doi.org/10.1109/JPHOTOV.2019.2903180)

# NED Unresolved issue - J01 calculated by Solcore in DB model is different to J01 calculated using the abrupt junction and Boltzmann approximation.  Otherwise the analytical model gives excellent agreement with SolCore's internal, numerical IV solution.


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

#############################################
# Step 1 : SolCore IV curve implementation
# Take a GaAs solar cell with a 1.42eV band-gap and plot the IV curve

eg=1.42
V = np.linspace(0, 1.3, 500)
db_junction = Junction(kind='DB', T=298, Eg=eg, A=1, R_shunt=np.inf, n=1)
my_solar_cell = SolarCell([db_junction], T=298, R_series=0)

solar_cell_solver(my_solar_cell, 'iv',
                      user_options={'T_ambient': 298, 'db_mode': 'top_hat', 'voltages': V, 'light_iv': True,
                                    'internal_voltages': np.linspace(0, 1.3, 400), 'wavelength': wl,
                                    'mpp': True, 'light_source': am15g})

plt.figure(1)
plt.plot(V, my_solar_cell.iv.IV[1], 'k')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc {my_solar_cell.iv.Isc:.2f}')
plt.text(0.1,280,f'Voc {my_solar_cell.iv.Voc:.2f}')
plt.text(0.1,260,f'Pmax {my_solar_cell.iv.Pmpp:.2f}')
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.show()

# Unclear to NED how to access the internal J01 value!  Will work this out from Isc Voc
j01solcore=my_solar_cell.iv.Isc/math.exp(my_solar_cell.iv.Voc/(k*298))
print(f'SolCore J01= {j01solcore:.3e}')

#############################################
# Step 2 : Perform the same calculation using the analytical expressions of Pusch et al.

# Need to calculate the limit to Jsc
# Transform the AM1.5G spectrum to photon flux to enable quick limiting Jsc calculation
solarflux = LightSource(source_type='standard', version='AM1.5g', x=wl, output_units='photon_flux_per_nm')
# Establish an interpolation function to allow integration over arbitrary limits
solarfluxInterpolate = InterpolatedUnivariateSpline(solarflux.spectrum()[0], solarflux.spectrum()[1], k=1)


# Analytical expressions to find IMax and VMax using LambertW function
# Find Jsc [limits are expressed in eV]
def getJsc(lowlim,upplim) :
    return q*solarfluxInterpolate.integral(1240/upplim, 1240/lowlim)
# Find J01 assuming abrupt junction & Boltzmann approximation
def getJ01(eg,t) :
   return ((2*math.pi* q )/(h**3 * c**2))* k*t * (eg**2 + 2*eg*(k*t) + 2*(k*t)**2)*math.exp(-(eg)/(k*t))
# Find Vmax
def getVmax(eg,emax,t) :
    return (k*t*(lambertw(math.exp(1)*(getJsc(eg,emax)/getJ01(eg,t)))-1)).real
# Find Imax
def getImax(eg,emax,t) :
    return getJsc(eg,emax) - getJ01(eg,t)*math.exp((getVmax(eg,emax,t))/(k*t))


# Calculate PV parameters
jsc=getJsc(eg,10)
j01=getJ01(eg,298)
voc=k*298*math.log(jsc/j01)
vmax=getVmax(eg,10,298)
imax=getImax(eg,10,298)
print(f'Analytical J01= {j01:.3e}')
# Calculate IV curve
I=np.array([jsc-j01*math.exp(vi/(k*298)) for vi in V])

plt.figure(2)
plt.plot(V, I, 'k')
plt.plot([0,voc,vmax],[jsc,0,imax],'o',color='red')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc {jsc:.2f}')
plt.text(0.1,280,f'Voc {voc:.2f}')
plt.text(0.1,260,f'Pmax {vmax*imax:.2f}')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.show()

#############################################
# Step 3 : Comparison between internal SolCore and analytical models
# The same band-gap (1.42eV) is used for each model with unity refactive index.  This should result in identical J01 values but does not.
# In addition, SolCore yields a higher Pmax yet uses a _higher_ value of J01.
# Warrants further investigation.  NED has

plt.figure(3)
plt.plot(V, my_solar_cell.iv.IV[1], color='blue',label='SolCore')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc SC={my_solar_cell.iv.Isc:.2f}, AN={jsc:.2f}')
plt.text(0.1,280,f'Voc SC={my_solar_cell.iv.Voc:.2f}, AN={voc:.2f}')
plt.text(0.1,260,f'Pmax {my_solar_cell.iv.Pmpp:.2f}, AN={vmax*imax:.2f}')
plt.plot(V, I,color='green')
plt.plot([0,voc,vmax],[jsc,0,imax],'o',color='red',label='Analytical')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.legend()
plt.show()

#############################################
# Step 4 : Recalculating the analytical model using SolCore J01 value

jsc=getJsc(eg,10)
voc=k*298*math.log(jsc/j01solcore)
vmax=(k*298*(lambertw(math.exp(1)*(getJsc(eg,10)/j01solcore))-1)).real
imax=jsc - j01solcore*math.exp(vmax/(k*298))

# Calculate IV curve
I=np.array([jsc-j01solcore*math.exp(vi/(k*298)) for vi in V])

#Replot comparison chart:
plt.figure(4)
plt.plot(V, my_solar_cell.iv.IV[1], color='blue',label='SolCore')
plt.ylim(0, 350)
plt.xlim(0, 1.2)
plt.text(0.1,300,f'Jsc SC={my_solar_cell.iv.Isc:.2f}, AN={jsc:.2f}')
plt.text(0.1,280,f'Voc SC={my_solar_cell.iv.Voc:.2f}, AN={voc:.2f}')
plt.text(0.1,260,f'Pmax {my_solar_cell.iv.Pmpp:.2f}, AN={vmax*imax:.2f}')
plt.plot(V, I,color='green',label='Analytical')
plt.plot([0,voc,vmax],[jsc,0,imax],'o',color='red')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A/m$^2$)')
plt.legend()
plt.show()