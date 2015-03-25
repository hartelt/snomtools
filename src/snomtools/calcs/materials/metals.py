__author__ = 'hartelt'
'''
This script provides a class for calculating the properties of metals, as well as instances for metals we know.
Functions take and give SI values! Use conversion tools in snomtools/calcs/conversions for different formats.
'''

import numpy

class Metal:
	def __init__(self, name):
		self.name = name

	def __str__(self):
		return self.name