__author__ = 'hartelt'
'''
This file provides data evaluation scripts for photoemission spectroscopy (PES) data. This includes anything that is
not experiment-specific, but can be applied for all photoemission spectra.
For furter info about data structures, see:
data.imports.tiff.py
data.datasets.py
'''

import snomtools.calcs.units as u
import numpy as np
import snomtools.data.datasets
from scipy.optimize import curve_fit
import scipy.special
import snomtools.calcs.constants as const

Kb = const.k_B # The Boltzmann constant
Temp = 300*u.to_ureg("Kelvin") # The Temperature, for now hardcoded as room temperature.

def fermi_edge(E,E_f,b,c,d):
	"""
	The typical shape of a fermi edge for constant DOS. Suitable as a fit function

	:param E: The x-Axis of the data consists of energies in eV

	:param E_f: The Fermi energy in eV.

	:param b: The width of the Fermi edge on top of the thermal broadening, which is introduced by all experimental
	errors, in eV.

	:param c: The height of the Fermi edge, in whichever units the data is given, e.g. "counts".

	:param d: Offset of the lower level of the fermi edge, e.g. "dark counts".

	:return: The value of the Fermi distribution at the energy E.
	"""
	return 0.5*(1-scipy.special.erf((E_f-E)/(np.sqrt(((1.7*Kb*Temp)**2)+b**2)*np.sqrt(2))))*c+d

def fermi_fit(data, energy_axis=None):
	pass