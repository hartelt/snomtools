__author__ = 'hartelt'
'''
This script provides tools for conversions of units.
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
	Converts the given wavelength to the corresponding angular frequency.
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

# Just for testing purposes:
test = numpy.linspace(0,10,100)
if __name__ == "__main__":
	print (test + test*.5)
