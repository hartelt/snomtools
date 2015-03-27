__author__ = 'hartelt'
'''
This file is meant as a collection of physical constants, that can be used in the calculations.
All constants must be given in SI! or prepare your self for a set of nonsense results.
'''
import numpy

#pi: the mathematical constant pi. from numpy. only for convenience
pi = numpy.pi

#c: the speed of light in vacuum
#Unit: m/s
c = 299792458.0

#mu_0: the magentic constant / permeability of vacuum
# Unit Vs/Am
mu_0 = 4e-7*numpy.pi

#epsilon_0: the dielectric constant / permittivity of vacuum
# Unit:  As/Vm
epsilon_0 = 1.0/(mu_0*c*c)

#e: the elementary charge
# Unit: Coulomb
e = 1.60217733e-19

#m_e: the electron mass
# Unit: kg
m_e = 9.1093897e-31

#h: the planck constant in Js
h = 6.6260693e-34

#hbar: the planck constant divided by 2 Pi in Js (/rad)
hbar = 1.05457168e-34

# Just for testing purposes:
if __name__ == "__main__":
	print pi
	print c
	print(mu_0)
	print(epsilon_0)
	print(e)
	print(m_e)