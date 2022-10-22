# Single diode IV parameterisation with series and shunt resistances

# Using LambertW formalism to calculate the IV curve in the presence of series & shunt resistance.
# [Tripathy (2017)](http://dx.doi.org/10.1016/j.solener.2017.10.007)
# PMax is calculated numerically
# NED has tried to find analytical methods to find Pmax but seems to be not possible.

from scipy.special import lambertw
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# Define fundamental physical constants
q=1.60217662E-19  # electronic charge [C]
k=1.38064852E-23/q   # Boltzmann constant [eV/K]
h=6.62607004E-34/q  # Planck constant expressed in [eV.s]
c=299792458  # Speed of light [m.s^-1]

# Calculate current using LambertW formulation of Tripathy
def getI(v,jsc,rs,rsh,j01,t) :
    return ((rsh*jsc+j01)-v)/(rs+rsh)-(k*t/rs)*lambertw((rs*rsh*j01)/(k*t*(rs+rsh))*np.exp(rsh*(rs*jsc+rs*j01+v)/(k*t*(rs+rsh)))).real

# Plot a graph, using a silicon PV cell as an example

x=np.linspace(0,0.73,50)  # voltage range

plt.figure(1)
area=272.13*1e-4
#Plot calculation result
plt.plot(x,getI(x,401*area,1e-3,1e4,1.7e-11,298))
plt.ylim([0,12])
plt.xlim([0,0.75])
plt.xlabel("Bias / V")
plt.ylabel("Current /A")
plt.show()


from scipy import optimize

# Routine to find power
def getP(v,iph,rs,rsh,j0,t) :
    return (-v*getI(v,iph,rs,rsh,j0,t))

# Routine to find Pmax parameters
def getPmax(iph,rs,rsh,j0,t) :
    # The bounds for the PV cell have to be between zero and the zero resistance Voc
    vocbound=k*t* np.log((j0 + iph)/j0)
    # Find Vmax by looking for the minimum of -Power.
    result=optimize.minimize_scalar(lambda x: getP(x,iph,rs,rsh,j0,t),bounds=(0,vocbound))
    vmax=result.x
    imax=getI(vmax,iph,rs,rsh,j0,t)
    pmax=-getP(vmax,iph,rs,rsh,j0,t)
    return([vmax,imax,pmax])

def getVoc(iph,rs,rsh,j0,t) :
    # The bounds for the PV cell have to be between zero and the zero resistance Voc
    vocbound=k*t* np.log((j0 + iph)/j0)
    # Find Voc by looking for the minimum of -Power.
    result=optimize.minimize_scalar(lambda x: getI(x,iph,rs,rsh,j0,t)**2,bounds=(0,vocbound))
    voc=result.x
    return(voc)

area=272.13/1e4
x=np.linspace(0,0.7,30)
plt.figure(2)
plt.plot(x,-getP(x,401*area,1e-3,1e4,1.7e-11,298))
plt.xlabel("Bias / V")
plt.ylabel("Power / W")
plt.show()

print(getPmax(401*area,1e-3,1e4,1.7e-11,298))
print('Pmax=',getPmax(401*area,1e-3,1e4,1.7e-11,298)[2])
print('Eff=',getPmax(401*area,1e-3,1e4,1.7e-11,298)[2]/(1000*area))
print('Voc=',getVoc(401*area,1e-3,1e4,1.7e-11,298))
print('IIsc=',getI(0,401*area,1e-3,1e4,1.7e-11,298))
print('FF=',(getPmax(401*area,1e-3,1e4,1.7e-11,298)[1]*getPmax(401*area,1e-3,1e4,1.7e-11,298)[0])/(getVoc(401*area,1e-3,1e4,1.7e-11,298)*getI(0,401*area,1e-3,1e4,1.7e-11,298)))