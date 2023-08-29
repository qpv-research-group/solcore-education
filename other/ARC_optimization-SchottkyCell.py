# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:33:59 2022

@author: z5228379
"""

""" Optimizing a double-layer MgF2/Ta2O5 anti-reflection coating for "infinitely-thick"
GaAs. Minimize reflection * AM0 spectrum (weighted reflectance).

To use yabox for the DE, we need to define a class which sets up the problem and has an
'evaluate' function within it, which will actually calculate the value we are trying to
minimize for each set of parameters.

The "if __name__ == "__main__" construction is used to avoid issues with parallel processing on Windows.
The issues arises because the multiprocessing module uses a different process on Windows than on UNIX
systems which will throw errors if this construction is not used.
"""
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from solcore import material
from solcore.optics.tmm import OptiStack, calculate_rat
from solcore.light_source import LightSource


from solcore.optimization import DE
from solcore.interpolate import interp1d


class CalcRDiff:
    def __init__(self):
        """ Make the wavelength and the materials n and k data object attributes.

        The n and k data are extracted from the Solcore materials rather than using
        the material directly because there is currently an issue with using the
        Solcore material class in parallel computations.
        """
        self.wl = np.linspace(300, 2000, 200)
        

        
        wl_n, n, wl_k, k = np.loadtxt("C:/Users/z5228379/Downloads/ZnO.csv", 
                                    delimiter=",", unpack=True,encoding='utf-8-sig')
        
        wl_n = wl_n[~np.isnan(wl_n)]
        n = n[~np.isnan(n)]

        n_wl = interp1d(wl_n, n)
        k_wl = interp1d(wl_k, k)
        
        
        self.ZnO= [
            self.wl,
            n_wl(self.wl),
            k_wl(self.wl)
        ]
        
    

        
        wl_n, n, wl_k, k = np.loadtxt("C:/Users/z5228379/Downloads/ALn&k.csv", 
                                    delimiter=",", unpack=True,encoding='utf-8-sig')

        
        n_wl = interp1d(wl_n, n)
        k_wl = interp1d(wl_k, k)
        
        
        self.Al= [
            self.wl,
            n_wl(self.wl),
            k_wl(self.wl)
        ]
        
        self.Si = [
            self.wl,
            material("Si")().n(self.wl * 1e-9),
            material("Si")().k(self.wl * 1e-9),
        ]

   
        spectr = LightSource(
            source_type="standard",
            version="AM1.5g",
            x=self.wl,
            output_units="photon_flux_per_m",
            concentration=1,
        ).spectrum(self.wl * 1e-9)[1]

 
        self.spectrum = spectr / max(spectr)
        

    def reflectance(self, x: Sequence[float]) -> float:
        """ Create a list with the format [thickness, wavelengths, n_data, k_data] for
        each layer.

        This is one of the acceptable formats in which OptiStack can take information
        (look at the Solcore documentation or at the OptiStack code for more info)
        We set no_back_reflection to True because we DO  NOT want to include reflection
        at the back surface (assume GaAs is infinitely thick)

        :param x: List with the thicknesses of the two layers in the ARC.
        :return: Array with the reflection at each wavelength
        """
        
        Si = material("Si")()
        
        arc = [[x[0]] + self.ZnO, [x[1]] + self.Al]
        full_stack = OptiStack(arc, no_back_reflection=False, substrate=Si)
        return calculate_rat(full_stack, self.wl, no_back_reflection=False)["R"]
    
    def absorption(self, x: Sequence[float]) -> float:
     
        
        Si = material("Si")()
        
        arc = [[x[0]] + self.ZnO, [x[1]] + self.Al]
        full_stack = OptiStack(arc, no_back_reflection=False, substrate=Si)
        return calculate_rat(full_stack, self.wl, no_back_reflection=False)["A"]
    def transmission(self, x: Sequence[float]) -> float:

        
        Si = material("Si")()
        
        arc = [[x[0]] + self.ZnO, [x[1]] + self.Al]

       
        full_stack = OptiStack(arc, no_back_reflection=False, substrate=Si)
        return calculate_rat(full_stack, self.wl, no_back_reflection=False)["T"]
    def evaluate(self, x: Sequence[float]) -> float:
        """ Returns the number the DA algorithm has to minimise.

        In this case, this is the weighted reflectance

        :param x: List with the thicknesses of the two layers in the ARC.
        :return: weighted reflectance
        """
        return np.mean(self.reflectance(x) * self.spectrum)

    def plot(self, x: Sequence[float]) -> None:
        """ Plots the reflectance

        :param x: List with the thicknesses of the two layers in the ARC.
        :return: None
        """
        plt.figure()
        plt.plot(self.wl, self.reflectance(x), label="Reflectance")
        plt.plot(self.wl, self.absorption(x), label="Absorption")
        plt.plot(self.wl, self.transmission(x), label="Transmission")
        
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("R/A/T")
        plt.legend()
        plt.show()

    def plot_weighted(self, x: Sequence[float]) -> None:
        """ Plots the weighted reflectance.

        :param x: List with the thicknesses of the two layers in the ARC.
        :return: None
        """
        plt.figure()
        plt.plot(self.wl, self.reflectance(x) * self.spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("R weighted by AM0")
        plt.show()


def main():


    # number of iterations for Differential Evolution
    maxiters = 70

    # class the DE algorithm is going to use, as defined above
    PDE_class = CalcRDiff()

    # Pass the function which will be minimized to the PDE (parallel differential evolution)
    # solver. PDE calculates the results for each population in parallel to speed up the
    # overall process

# =============================================================================
#     PDE_obj = DE(PDE_class.evaluate, bounds=[[0, 250], [0, 250], [0, 250], [0, 250]], maxiters=maxiters)
# =============================================================================
    PDE_obj = DE(PDE_class.evaluate, bounds=[[0, 100], [10,15]], maxiters=maxiters)
    # PDE_obj = DE(PDE_class.evaluate, bounds=[[0, 500]], maxiters=maxiters)
    # solve, i.e. minimize the problem
    res = PDE_obj.solve()

    """
    PDE_obj.solve() returns 5 things:
    - res[0] is a list of the parameters which gave the minimized value
    - res[1] is that minimized value
    - res[2] is the evolution of the best population (the best population from each 
        iteration
    - res[3] is the evolution of the minimized value, i.e. the fitness over each iteration
    - res[4] is the evolution of the mean fitness over the iterations
    """
    best_pop = res[0]
    print("Parameters for best result:", best_pop, res[1])

    PDE_class.plot(best_pop)
    PDE_class.plot_weighted(best_pop)

if __name__ == '__main__':
    main()

