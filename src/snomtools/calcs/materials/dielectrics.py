__author__ = 'hartelt'
'''
This script provides a class for the properties of dielectric materials, as well as instances for materials we know.
Functions take and give SI values! Use conversion tools in snomtools/calcs/conversions for different formats.
Functions should be coded compatible to the ones in metals.py, and should work with numpy arrays.
'''

import numpy

#TODO: Compatibility with pint.
#TODO: We should implement tabulated literature data. Then we can always take the nearest value or so.
class Dielectric:
	"""
	The base class for dielectric materials.
	"""
	def __init__(self,name,epsilon=1.,n=1.):
		"""
		The constructor.
		Can be initialized with an epsilon or an n value. If you give neither, you'll end up with Vacuum.
		All values can be complex.
		:param name: The name of the material. E.g. ITO
		:param epsilon: The relative dielectric constant of the medium.
		:param n: The refraction index,
		:return: nothing
		"""
		self.name = name
		if epsilon!=1.:
			self.dielconstant = epsilon
			self.refracindex = numpy.sqrt(epsilon)
		elif n != 1.:
			self.dielconstant = n**2
			self.refracindex = n
		else:
			self.dielconstant=1.0
			self.refracindex=1.0

	def epsilon(self,omega=0):
		"""
		The dielectric constant of the medium.
		:param omega: just for compatibility with the metal function, and for determining the output type.
		:return: the dielectric function. will be the same type as omega
		"""
		eps = omega * 0 + self.dielconstant #to conserve format
		return eps

	def n(self,omega=0.):
		"""
		The complex refraction index of the medium.
		:param omega: just for compatibility with the metal function, and for determining the output type.
		:return: the refraction indes. will be the same type as omega
		"""
		myn = omega * 0 + self.refracindex #to conserve format
		return myn

Vacuum = Dielectric("Vacuum")

#ITO @ 800nm http://refractiveindex.info/?shelf=other&book=In2O3-SnO2&page=Konig
ITO = Dielectric("ITO",n=1.5999+0.0056700j)

#for testing:
if __name__=="__main__":
	import snomtools.calcs.prefixes as pref
	test = 100.
	hz = test * pref.tera
	moep = Vacuum
	print(moep.n(hz))