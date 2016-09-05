__author__ = 'hartelt'
'''
This file provides data evaluation scripts for PEEM data.
For furter info about data structures, see:
data.imports.tiff.py
data.datasets.py
'''

import snomtools.calcs.units as u
import snomtools.data.datasets
import snomtools.data.imports.tiff
import os.path
import numpy

def fit_powerlaw(powers,intensities):
	"""
	This function shall fit a powerlaw to data.
	:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.
	:param intensities: Quantity or array: The corresponding intensity values to powers.
	:return: TBD
	"""
	pass