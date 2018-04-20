"""
This file is meant as a collection of physical constants, that can be used in the calculations.
All constants will be given in SI units as pint quantities.
Numerial float value can be cast straight forward or can be found in value_float.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from . import ureg

__author__ = 'hartelt'

# pi: the mathematical constant pi. from numpy. only for convenience
pi_float = numpy.pi
pi = pi_float * ureg('rad')

# c: the speed of light in vacuum
# Unit: m/s
c_float = 299792458.0
c = c_float * ureg('meters / second')

# mu_0: the magentic constant / permeability of vacuum
# Unit Vs/Am
mu_0_float = 4e-7 * numpy.pi
mu_0 = mu_0_float * ureg('V s / (A m)')

# epsilon_0: the dielectric constant / permittivity of vacuum
# Unit:  As/Vm
epsilon_0_float = 1.0 / (mu_0 * c * c)
epsilon_0 = 1.0 / (mu_0 * c * c)

# e: the elementary charge
# Unit: Coulomb
e_float = 1.60217733e-19
e = e_float * ureg('coulomb')

# m_e: the electron mass
# Unit: kg
m_e_float = 9.1093897e-31
m_e = m_e_float * ureg('kg')

# h: the planck constant in Js
h_float = 6.6260693e-34
h = h_float * ureg('J s')

# hbar: the planck constant divided by 2 Pi in Js (/rad)
hbar_float = 1.05457168e-34
hbar = hbar_float * ureg('J s / rad')

# k_B: the Boltzmann constant in J / K
k_B_float = 1.38064852e-23
k_B = k_B_float * ureg('J / K')


# Just for testing purposes:
def test():
	print(pi)
	print(c)
	print(mu_0)
	print(epsilon_0)
	print(e)
	print(m_e)
	print(h)
	print(hbar)
	print(k_B)
