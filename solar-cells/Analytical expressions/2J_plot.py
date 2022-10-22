# The Multi-junction Efficiency Limit
# Uses the analytical expressions from [Pusch et al., JPV (2019)](doi.org/10.1109/JPHOTOV.2019.2903180)

import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import lambertw
from solcore.light_source import LightSource

# Define fundamental physical constants
q=1.60217662E-19  # electronic charge [C]
k=1.38064852E-23/q   # Boltzmann constant [eV/K]
h=6.62607004E-34/q  # Planck constant expressed in [eV.s]
c=299792458  # Speed of light [m.s^-1]

# Load the AM1.5G solar spectrum
wl = np.linspace(300, 4000, 4000) * 1e-9    #wl contains the x-ordinate in wavelength
am15g = LightSource(source_type='standard', x=wl, version='AM1.5g')

# Convert the AM1.5G spectrum to photon flux
solarflux = LightSource(source_type='standard', version='AM1.5g', x=wl, output_units='photon_flux_per_nm')
# Establish an interpolation function to allow integration over arbitary limits
solarfluxInterpolate = InterpolatedUnivariateSpline(solarflux.spectrum()[0], solarflux.spectrum()[1], k=1)

# Analytical expressions to find IMax using LambertW function
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

# Function to return pmax for a series connected solar cell with n junctions.
# Argument is a list of bandgaps in descending order.
def getPmax(egs) :
    # Define arrays with the same dimensions as egs
    jscs=egs.copy()
    Vmaxs=egs.copy()
    Imaxs=egs.copy()
    j01s=egs.copy()
    upperlimit=10

    #Iterate through band-gaps to find the lowest Imax
    for i in range(0,len(egs),1) :
        eg=egs[i]
        j01s[i]=getJ01(eg,298)
        jscs[i]=getJsc(eg,upperlimit)
        Vmaxs[i]=getVmax(eg,upperlimit,298)
        Imaxs[i]=getImax(eg,upperlimit,298)
        upperlimit=egs[i]
#    Find the minimum Imaxs
        minImax=np.amin(Imaxs)

#   Find tandem sub-cell voltage at the minImax current:
    vTandem=0
    for i in range(0,len(egs),1) :
        vsubcell=k*298*math.log((jscs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

    return vTandem*minImax  # Return the tandem power


# Plot a 2 junction tandem map
# Reproduces Fig 5a in Pusch et al.,

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getPmax([yy,xx])/1000

x=np.linspace(0.7, 1.28, 30)
y=np.linspace(1.3, 1.9, 30)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

plt.figure(4)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45])
plt.clabel(contours,inline=True,fontsize=8)
plt.colorbar()
plt.ylabel('Eg2')
plt.xlabel('Eg1')
plt.show()