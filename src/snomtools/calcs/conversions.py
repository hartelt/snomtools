"""
This script provides tools for conversion between types of representations.
As long as it's not specified otherwise, it should be kept in SI!
All functions must be programed to work with float variables, as well as numpy arrays and pint quantities.

"""
__author__ = 'hartelt'

import numpy
import constants
import units as u


def deg2rad(angle, return_numeric=None):
	"""
	Converts the given angle in degrees to radians

	:param angle: the angle in degrees

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the angle in rad
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(angle)
	rad = u.to_ureg(angle, 'deg').to('rad')
	if return_numeric:
		return rad.magnitude
	else:
		return rad


def rad2deg(rad, return_numeric=None):
	"""
	Converts the given angle from radians to degrees

	:param rad: the angle in rad

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the angle in degees
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(rad)
	angle = u.to_ureg(rad, 'rad').to('deg')
	if return_numeric:
		return angle.magnitude
	else:
		return angle


def lambda2omega(lambda_, return_numeric=None):
	"""
	Converts a light wavelength to the corresponding angular frequency.

	:param lambda_: the wavelength value in m

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the angular frequency in Hz (rad/s)
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(lambda_)
	lambda_ = u.to_ureg(lambda_, 'm')
	omega = 2.0 * constants.c * constants.pi / lambda_  # c*2*pi/lambda
	if return_numeric:
		return omega.magnitude
	else:
		return omega


def omega2lambda(omega, return_numeric=None):
	"""
	Converts the given angular frequency to the corresponding wavelength.

	:param omega: the angular frequency in Hz (rad/s)

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the wavelength lambda in m
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(omega)
	omega = u.to_ureg(omega, 'Hz')
	lambda_ = 2.0 * constants.pi * constants.c / omega
	if return_numeric:
		return lambda_.magnitude
	else:
		return lambda_


def k2lambda(k, return_numeric=None):
	"""
	Converts the given angular wavenumber, k, to the corresponding wavelength.

	:param k: the angular wavenumber in rad/m

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the wavelength in m
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(k)
	k = u.to_ureg(k, 'rad/m')
	lambda_ = 2.0 * constants.pi / k  # lambda = 2pi/k
	if return_numeric:
		return lambda_.magnitude
	else:
		return lambda_


def lambda2k(lambda_, return_numeric=None):
	"""
	Converts the given wavelength to the corresponding angular wavenumber, k.

	:param lambda_: the wavelength in meters

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the angular wavenumber in 1/m (rad/m)
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(lambda_)
	lambda_ = u.to_ureg(lambda_, 'm')
	k = 2.0 * constants.pi / lambda_
	if return_numeric:
		return k.magnitude
	else:
		return k


def omega2energy(omega, return_numeric=None):
	"""
	Converts an angular frequency to the corresponding energy.
	E = hbar * omega

	:param omega: the angular frequency in rad/s

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return:the energy in J
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(omega)
	omega = u.to_ureg(omega, 'rad/s')
	E = constants.hbar * omega
	if return_numeric:
		return E.magnitude
	else:
		return E


def joule2ev(energy, return_numeric=None):
	"""
	Converts an energy value from J to eV.
	E/eV = E/J / e
	for e is the elementary charge.

	:param energy: the energy in J

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return:the energy in eV
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(energy)
	eV = u.to_ureg(energy, 'J').to('eV')
	if return_numeric:
		return eV.magnitude
	else:
		return eV


def ev2joule(energy, return_numeric=None):
	"""
	Converts an energy value from eV to J.
	E/J = E/eV * e
	for e is the elementary charge.

	:param energy: the energy in eV

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the energy in J
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(energy)
	J = u.to_ureg(energy, 'eV').to('J')
	if return_numeric:
		return J.magnitude
	else:
		return J


def k_beat2k_spp(k_b, k_l, angle, return_numeric=None):
	"""
	Extracts the SPP wavenumber out of the PEEM beat pattern wavenumber
	for a certain angle of incidence.

	:param k_b: the beat pattern wavenumber in 1/m (rad/m)

	:param k_l: the laser wavenumber in 1/m (rad/m)

	:param angle: the incidence angle of light in degrees

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the plasmon beat pattern wavenumber in 1/m (rad/m)
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(k_b)
	k_b = u.to_ureg(k_b, 'rad/m')
	k_l = u.to_ureg(k_l, 'rad/m')
	angle = u.to_ureg(angle, 'deg').to('rad')
	k_spp = k_b + k_l * numpy.sin(angle)
	if return_numeric:
		return k_spp.magnitude
	else:
		return k_spp


def n2epsilon(n, return_numeric=None):
	"""
	Converts a complex refraction index to the corresponding dielectric constant.
	(...which is simply the complex second power of it.)

	:param n: the complex refractive index

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the complex dielectric constant
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(n)
	n = u.to_ureg(n, 'dimensionless')
	eps = n ** 2
	if return_numeric:
		return eps.magnitude
	else:
		return eps


def epsilon2n(epsilon, return_numeric=None):
	"""
	Converts a complex dielectric constant to the corresponding refraction index.
	(...which is simply the complex square root of it.)

	:param epsilon: the complex dielectric constant

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: the complex refractive index
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(epsilon)
	epsilon = u.to_ureg(epsilon, 'dimensionless')
	myn = numpy.sqrt(epsilon)
	if return_numeric:
		return myn.magnitude
	else:
		return myn


def length2time(length, n=1., outunit="s", return_numeric=None):
	"""
	Converts a length to a time in the context of light travel time.

	:param length: A length, assumed in meters if given as numerical format.

	:param n: Float or dimensionless quantity: The refraction index of the medium.

	:param outunit: A valid time unit string.

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: The time, given in the unit specified in outunit.
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(length)
	length = u.to_ureg(length, 'meter', convert_quantities=False)
	n = u.to_ureg(n, 'dimensionless')
	time = n * length / constants.c
	time = time.to(outunit)
	if return_numeric:
		return time.magnitude
	else:
		return time


def time2length(time, n=1., outunit="m", return_numeric=None):
	"""
	Converts a time to a length in the context of light travel time.

	:param time: A time, assumed in seconds if given as numerical format.

	:param n: Float or dimensionless quantity: The refraction index of the medium.

	:param outunit: A valid length unit string.

	:param return_numeric: flag to return numeric formats instead of quantities. default is input format.

	:return: The length, given in the unit specified in outunit.
	"""
	if return_numeric is None:
		return_numeric = not u.is_quantity(time)
	time = u.to_ureg(time, 'second', convert_quantities=False)
	n = u.to_ureg(n, 'dimensionless')
	length = time * constants.c / n
	length = length.to(outunit)
	if return_numeric:
		return length.magnitude
	else:
		return length


# Just for testing purposes:
if __name__ == "__main__":
	test = numpy.linspace(0, 10, 100) * u.ureg('um')
	print (lambda2omega(test))
