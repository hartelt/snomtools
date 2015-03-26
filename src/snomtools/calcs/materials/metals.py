__author__ = 'hartelt'
'''
This script provides a class for calculating the properties of metals, as well as instances for metals we know.
Functions take and give SI values! Use conversion tools in snomtools/calcs/conversions for different formats.
'''

import numpy

class InterbandTransition:
	"""
	This is the base class for an interband transition.
	It consists of the parameters for the transition and a method for calculating the effect on the dielectric function.
	"""
	def __init__(self, plasma_freq, damping_freq, central_freq):
		"""
		The constructor. All frequencies should be given as angular frequencies (rad/s)
		:param plasma_freq: the (quasi-)plasma frequency of the bound electrons
		:param damping_freq: the damping frequency of the bound electrons
		:param central_freq: the central frequency of the interband transition
		"""
		self.omega_p = plasma_freq
		self.gamma = damping_freq
		self.omega_0 = central_freq

	def epsilon(self,omega):
		"""
		This method calculates the contribution of the interband transition to the complex dielectric function.
		Caution: The offset 1 from the vacuum is NOT included, to make it possible to just add all contributions to get
		the final dielectric function.
		The formula is:
		omega_p^2 / (omega_0^2 - omega^2 - i omega gamma)
		:param omega: the frequency at which the dielectric function shall be calculated (float or numpy array) in rad/s
		:return: the contribution to the dielectric function (dimensionless)
		"""
		omegasquare = omega**2
		denominator = (self.omega_0*self.omega_0) - omegasquare - ((0+1j)*self.gamma*omega)
		eps = self.omega_p*self.omega_p / denominator
		return eps

#TODO: We should implement tabulated literature data. Then we can always take the nearest value or so.
class Metal:
	"""
	This is the base class for a metal.
	A metal is characterized for now, as something with a plasma frequency and one or more interband transitions.
	This might be extended later on.
	"""
	def __init__(self, name, plasma_freq, damping_freq, interband_transitions=None):
		"""
		The constructor.
		:param name: A string for the name of the material. e.g. "Au"
		:param plasma_freq: the plasma frequency of the free electrons in (rad/s)
		:param damping_freq: the damping frequency of the free electrons in (rad/s)
		:param interband_transitions: a list of interband transitions, given either as instance of the
		InterbandTransition class or as list or tuple of 3 elements containing the frequencies to initialize one.
		:return: nothing
		"""
		self.name = name
		self.omega_p = plasma_freq
		self.gamma = damping_freq
		self.interbands = []
		if interband_transitions:
			for trans in interband_transitions:
				if isinstance(trans,InterbandTransition):
					self.interbands.append(trans)
				elif isinstance(trans,list) or isinstance(trans,tuple) and len(trans) == 3:
					self.interbands.append(InterbandTransition(trans[0],trans[1],trans[2]))
				else:
					print("ERROR: Metal: Interband transition could not be cast. Ignoring the transition.")

	def epsilon_plasma(self,omega):
		"""
		This method calculates the contribution of the free electrons to the complex dielectric function.
		Caution: The offset 1 from the vacuum is NOT included, to make it possible to just add all contributions to get
		the final dielectric function.
		The formula is:
		- omega_p^2 / (omega^2 - i omega gamma)
		:param omega: the frequency at which the dielectric function shall be calculated (float or numpy array) in rad/s
		:return:the contribution to the dielectric function (dimensionless)
		"""
		eps = - self.omega_p**2 / (omega**2 - (0+1j)*omega*self.gamma)
		return eps

	def epsilon(self,omega):
		"""
		This method calculates the complex dielectric function of the metal.
		It therefore adds all contributions of the free electrons and the interband transitions.
		:param omega: the frequency at which the dielectric function shall be calculated (float or numpy array) in rad/s
		:return: the complex dielectric function (dimensionless)
		"""
		eps = 1 + self.epsilon_plasma(omega)
		for trans in self.interbands:
			eps = eps + trans.epsilon(omega)
		return eps

	def __str__(self):
		return self.name

	def n(self,omega):
		"""
		Returns the complex refraction index of the metal. Uses the dielectric function to calculate.
		:param omega: the frequency at which the refraction index shall be calculated (float or numpy array) in rad/s
		:return: the complex refraction index (dimensionless)
		"""
		return numpy.sqrt(self.epsilon(omega))

#Gold with the parameters determined in Christian Schneider's dissertation:
Au_Schneider = Metal("Au",13.202e15,102.033e12,[InterbandTransition(4.5e15,896e12,4.184e15)])

#for testing:
if __name__=="__main__":
	import snomtools.calcs.prefixes as pref
	test = numpy.linspace(2000,4000,20)
	hz = test * pref.tera
	print(Au_Schneider.epsilon(hz))
	print(Au_Schneider.epsilon_plasma(hz))