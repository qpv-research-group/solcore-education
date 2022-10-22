
#Import some useful libraries
from solcore.light_source import LightSource  # Functions that load solar spectra
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# Define the  wavelength range to load the solar spectrum, 300nm to 4000nm with 4000 steps.
wl = np.linspace(300, 4000, 4000) * 1e-9

#Load the AM1.5G global terrestial spectrum
am15g = LightSource(source_type='standard', x=wl, version='AM1.5g')
#Load the AM1.5d direct terrestrial spectrum
am15d = LightSource(source_type='standard', x=wl, version='AM1.5d')
#Load the AM0 spectrum
am0 = LightSource(source_type='standard', x=wl, version='AM0')

plt.figure(1)  # Setup a figure for plotting
plt.plot(*am0.spectrum(), 'k',label='AM0')      #Plot AM0 spectrum
plt.plot(*am15g.spectrum(), 'b',label='AM1.5G') #Plot AM1.5G spectrum
plt.plot(*am15d.spectrum(), 'g',label='AM1.5D') #Plot AM1.5D spectrum

#Question:   .spectrum() returns a tuple, what does the * do?
plt.xlim(300, 3000)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power density (Wm$^{-2}$nm$^{-1}$)')
plt.legend()
plt.show()


#Integrate the solar spectra to provide the overall solar irradiance
# We will use the numpy trapz implementation of the trapezium rule
# The SolCore .spectrum function returns a tuple (x,y) but np.trapz requires data in format (y,x) these are extracted into separate 1D np arrays that are selected with [0] and [1].
yval=am0.spectrum()[1]
xval=am0.spectrum()[0]
am0_irradiance=np.trapz(yval,xval)  # Perform integration using trapezium rule

#Print the irradiance value for AM0
print('AM0 Irradiance = ',am0_irradiance,'W.m^2')

#The output above contains too many decimal places, can format the output using the .format command
am0_formatted = "{:.0f}".format(am0_irradiance)
print('AM0 Irradiance = ',am0_formatted,'W.m^2')

#We can write the above more concisely for the AM1.5G and D spectra

print('AM1.5G Irradiance = ',"{:.0f}".format(np.trapz(am15g.spectrum()[1],am15g.spectrum()[0])),'W.m^2')

print('AM1.5D Irradiance = ',"{:.0f}".format(np.trapz(am15d.spectrum()[1],am15d.spectrum()[0])),'W.m^2')