# The Multi-junction Efficiency Limit with radiative coupling
# Uses the analytical expressions from [Pusch et al., JPV (2019)](doi.org/10.1109/JPHOTOV.2019.2903180)

## Component functions
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
# Establish an interpolation function to allow integration over arbitrary limits
solarfluxInterpolate = InterpolatedUnivariateSpline(solarflux.spectrum()[0], solarflux.spectrum()[1], k=1)

# Routine to find the illuminated light current density JL [limits are expressed in eV]
def getJL(lowlim,upplim) :
    return q*solarfluxInterpolate.integral(1240/upplim, 1240/lowlim)
# Routine to find J01
def getJ01(eg,t) :
   return ((2*math.pi* q )/(h**3 * c**2))* k*t * (eg**2 + 2*eg*(k*t) + 2*(k*t)**2)*math.exp(-(eg)/(k*t))
# Routine to find Vmax
def getVmax(IL,j01,t) :
    return (k*t*(lambertw(math.exp(1)*(IL/j01))-1)).real
# Routine to find Imax
def getImax(IL,j01,vmax,t) :
    return IL - j01*math.exp((vmax)/(k*t))

def getJLbotTotal(etaext,n,JLbot,JLtop) : # Eqn 15 in Pusch et al
    return JLbot*(1/etaext + n**2)/(1/etaext+2*n**2)+JLtop*(n**2/(1/etaext+2*n**2))

def getJ0botTotal(etaext,n,eg,t) : # Eqn 16 in Pusch et al, expanded to include j0bot
    return (1/etaext+n**2)/(1/etaext+2*n**2)*getJ01(eg,t)

def getJ0topTotal(etaext,n,eg,t) : # Second term in Eqn 18 in Pusch et al
    return (1/etaext+n**2)*getJ01(eg,t)

#######################################
# Taking the specific case of a 2J, radiatively coupled solar cell
# Repeating the calculation shown in Figure 5b, Pusch et al. where n=3.5 and etaext=1.

# In the specific instance of a 2J solar cell
def getPmax2J(eg1,eg2,etaext,n,t) :
    # Evaluate the top junction first:
    JLtop=getJL(eg1,10)  # Top cell light current
    j01toptotal=getJ0topTotal(etaext,n,eg1,t)
    topVmax=getVmax(JLtop,j01toptotal,t)
    topImax=getImax(JLtop,j01toptotal,topVmax,t)

    # Evaluate the bottom junction next:
    JLbot=getJLbotTotal(etaext,n,getJL(eg2,eg1),JLtop)  # Total bottom cell light current
    j01bottotal=getJ0botTotal(etaext,n,eg2,t)
    botVmax=getVmax(JLbot,j01bottotal,t)
    botImax=getImax(JLbot,j01bottotal,botVmax,t)

#   Find the minimum Imaxs
    minImax=np.amin([topImax,botImax])

#   Find tandem voltages
    vtop=k*298*math.log((JLtop-minImax)/j01toptotal)
    vbot=k*298*math.log((JLbot-minImax)/j01bottotal)
    vTandem = vtop+vbot
    return vTandem*minImax

@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getPmax2J(yy,xx,1,3.5,298)/1000

x=np.linspace(0.7, 1.28, 30)
y=np.linspace(1.3, 1.9, 30)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

plt.figure(1)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45])
plt.clabel(contours,inline=True,fontsize=8)
plt.colorbar()
plt.ylabel('Eg2')
plt.xlabel('Eg1')
plt.show()

# Figure 7 in Pusch et al. is not precisely reproduced since that calculation includes reflectivity losses.
@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getPmax2J(xx,1.2,10**yy,3.5,298)/1000

x=np.linspace(1.4, 2, 30)
y=np.linspace(0, -3, 30)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

plt.figure(4)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.3,0.33,0.36,0.38,0.39,0.40,0.41,0.42,0.43])
plt.clabel(contours,inline=True,fontsize=8)
plt.colorbar()
plt.ylabel('Log10(Radiative Efficiency)')
plt.xlabel('Eg1')
plt.show()


#################################################
## Generalising to allow for any number of radiatively coupled junctions
# Code reproduces the 2J implementation above but in a general form:

#Function to return pmax with radiative coupling for a n J cell
#Argument is a list of band-gaps, external raditive efficiencies and refractive indecies in descending order.
def getPmax(egs,etaexts,ns) :
    t=298
    # Need to store all the effective JL, J01 and Imax values
    j01s=egs.copy()  # Quick way of defining jscs with same dimensions as egs
    JLs=egs.copy()
    Imaxs=egs.copy()

    # Initialise by calculating the top junction
    JLs[0]=getJL(egs[0],10)  # Top cell light current
    j01s[0]=getJ0topTotal(etaexts[0],ns[0],egs[0],t)
    Imaxs[0]=getImax(JLs[0],j01s[0],getVmax(JLs[0],j01s[0],t),t)

    for i in range(1,len(egs),1) :  # loop through remaining junctions.
        # Evaluate the bottom junction next:
        JLs[i]=getJLbotTotal(etaexts[i],ns[i],getJL(egs[i],egs[i-1]),JLs[i-1])  # Total bottom cell light current
        j01s[i]=getJ0botTotal(etaexts[i],ns[i],egs[i],t)
        Imaxs[i]=getImax(JLs[i],j01s[i],getVmax(JLs[i],j01s[i],t),t)

#    Find the minimum Imaxs
    minImax=np.amin(Imaxs)

#   Find tandem voltage
    vTandem=0
    for i in range(0,len(egs),1) :  # loop through all junctions, including the top
        vsubcell=k*298*math.log((JLs[i]-minImax)/j01s[i])
        vTandem=vTandem+vsubcell

    return vTandem*minImax


@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *decending* list of band-gap energies
    return getPmax([yy,xx],[1,1],[3.5,3.5])/1000

x=np.linspace(0.7, 1.28, 30)
y=np.linspace(1.3, 1.9, 30)
X,Y=np.meshgrid(x,y)
Z=getEff(X,Y)

plt.figure(3)
contours=plt.contour(X,Y,Z,colors='black',levels=[0.35,0.375,0.4,0.42,0.44,0.45])
plt.clabel(contours,inline=True,fontsize=8)
plt.colorbar()
plt.ylabel('Eg2')
plt.xlabel('Eg1')
plt.show()

#Try a 3J InGaP/GaAs/x/Ge with radiative coupling from InGaP to GaAs and GaAs to x
@np.vectorize
def getEff(xx,yy) :
    # Make sure getPmax receives *descending* list of band-gap energies
    return getPmax([1.9,1.42,xx],[yy,yy,0.0001],[3.5,3.5,3.5])/1000

x=np.linspace(0.8, 1.2, 30)
radlimit=np.array([getEff(xi,1) for xi in x])  # Radiative limit
conv=np.array([getEff(xi,0.03) for xi in x])    # Conventional 3J rad.eff of 3%

plt.figure(2)
plt.plot(x,radlimit,label='rad limit')
plt.plot(x,conv,label='conv')
plt.legend()
plt.ylabel('Efficiency')
plt.xlabel('Eg1')
plt.show()

#NED Not sure this is working correctly.  Needs more investigation.  Maybe compare with SolCore's internal radiative coupling models...

