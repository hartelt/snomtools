__author__ = 'hartelt'
'''
This script provides tools for conversion between types of representations.
As long as it's not specified otherwise, it should be kept in SI!
All functions must be programed to work with float variables, as well as numpy arrays.
'''

import numpy
import constants

def deg2rad(angle):
	'''
	Converts the given angle in degrees to radians
	:param angle: the angle in degrees
	:return: the angle in rad
	'''
	rad = angle / 180.0 * constants.pi
	return rad

def rad2deg(rad):
	'''
	Converts the given angle from radians to degrees
	:param rad: the angle in rad
	:return: the angle in degees
	'''
	angle = rad / constants.pi * 180.0
	return angle

def lambda2omega(lambda_):
	'''
	Converts a light wavelength to the corresponding angular frequency.
	:param lambda_: the wavelength value in m
	:return: the angular frequency in Hz (rad/s)
	'''
	omega=2.0*constants.c*constants.pi/lambda_ # c*2*pi/lambda
	return omega

def omega2lambda(omega):
	'''
	Converts the given angular frequency to the corresponding wavelength.
	:param omega: the angular frequency in Hz (rad/s)
	:return: the wavelength lambda in m
	'''
	lambda_ = 2.0 * constants.pi * constants.c / omega
	return lambda_

def k2lambda(k):
	'''
	Converts the given angular wavenumber, k, to the corresponding wavelength.
	:param k: the angular wavenumber in 1/m (rad/m)
	:return: the wavelength in m
	'''
	lambda_=2.0*constants.pi/k # lambda = 2pi/k
	return lambda_

def lambda2k(lambda_):
	'''
	Converts the given wavelength to the corresponding angular wavenumber, k.
	:param lambda_: the wavelength in meters
	:return: the angular wavenumber in 1/m (rad/m)
	'''
	k = 2.0*constants.pi/lambda_
	return k

def omega2energy(omega):
	'''
	Converts an angular frequency to the corresponding energy.
	E = hbar * omega
	:param omega: the angular frequency in rad/s
	:return:the energy in J
	'''
	E = constants.hbar * omega
	return E

def joule2ev(energy):
	'''
	Converts an energy value from J to eV.
	E/eV = E/J / e
	for e is the elementary charge.
	:param energy: the energy in J
	:return:the energy in eV
	'''
	eV = energy / constants.e
	return eV

def ev2joule(energy):
	'''
	Converts an energy value from eV to J.
	E/J = E/eV * e
	for e is the elementary charge.
	:param energy: the energy in eV
	:return: the energy in J
	'''
	J = energy * constants.e
	return J

def k_beat2k_spp(k_b,k_l,angle):
	'''
	Extracts the SPP wavenumber out of the PEEM beat pattern wavenumber
	for a certain angle of incidence.
	:param k_b: the beat pattern wavenumber in 1/m (rad/m)
	:param k_l: the laser wavenumber in 1/m (rad/m)
	:param angle: the incidence angle of light in degrees
	:return: the plasmon beat pattern wavenumber in 1/m (rad/m)
	'''
	k_spp=k_b+k_l*numpy.sin(angle*constants.pi/180.0)
	return k_spp

def n2epsilon(n):
	'''
	Converts a complex refraction index to the corresponding dielectric constant.
	(...which is simply the complex second power of it.)
	:param n: the complex refractive index
	:return: the complex dielectric constant
	'''
	eps = n**2
	return eps

def epsilon2n(epsilon):
	'''
	Converts a complex dielectric constant to the corresponding refraction index.
	(...which is simply the complex square root of it.)
	:param epsilon: the complex dielectric constant
	:return: the complex refractive index
	'''
	myn = numpy.sqrt(epsilon)
	return myn

# Just for testing purposes:
test = numpy.linspace(0,10,100)
if __name__ == "__main__":
	print (test + test*.5)