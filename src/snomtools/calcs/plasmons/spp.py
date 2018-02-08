"""
This script provides a class for calculating the properties of Surface Plasmon Polaritons. All functions work on
Quantities see snomtools.calcs.units.
"""
__author__ = 'hartelt'

import snomtools.calcs.units as u
import snomtools.calcs.materials.metals
import snomtools.calcs.materials.dielectrics
import snomtools.calcs.constants as const
import numpy

default_metal = snomtools.calcs.materials.metals.Au_Schneider
default_diel = snomtools.calcs.materials.dielectrics.Vacuum


def Kspp(omega, metal=default_metal, diel=default_diel, unit='1/um'):
	omega = u.to_ureg(omega,'rad/s',convert_quantities=False)
	assert isinstance(metal,snomtools.calcs.materials.metals.Metal), "ERROR: No metal given to Kspp"
	assert isinstance(diel,snomtools.calcs.materials.dielectrics.Dielectric), "ERROR: No dielectric given to Kspp"

	wurzel = numpy.sqrt((metal.epsilon(omega) * diel.epsilon(omega)) / (metal.epsilon(omega) + diel.epsilon(omega)))
	k = omega / const.c * numpy.real(wurzel)
	k = u.to_ureg(k, unit)
	return k


def Kspp_wo_interband(omega, metal=default_metal, diel=default_diel, unit='1/um'):
	omega = u.to_ureg(omega,'rad/s',convert_quantities=False)
	assert isinstance(metal,snomtools.calcs.materials.metals.Metal), "ERROR: No metal given to Kspp"
	assert isinstance(diel,snomtools.calcs.materials.dielectrics.Dielectric), "ERROR: No dielectric given to Kspp"

	wurzel = numpy.sqrt(
		(metal.epsilon_wo_inter(omega) * diel.epsilon(omega)) / (metal.epsilon_wo_inter(omega) + diel.epsilon(omega)))
	k = omega / const.c * numpy.real(wurzel)
	k = u.to_ureg(k, unit)
	return k


def Kspp_real(omega, metal=default_metal, diel=default_diel, unit='1/um'):
	omega = u.to_ureg(omega,'rad/s',convert_quantities=False)
	assert isinstance(metal,snomtools.calcs.materials.metals.Metal), "ERROR: No metal given to Kspp"
	assert isinstance(diel,snomtools.calcs.materials.dielectrics.Dielectric), "ERROR: No dielectric given to Kspp"

	eps = numpy.real(metal.epsilon_wo_inter(omega))
	wurzel = numpy.sqrt((eps * diel.epsilon(omega)) / (eps + diel.epsilon(omega)))
	k = omega / const.c * numpy.real(wurzel)
	k = u.to_ureg(k, unit)
	return k


def Vspp(omega, metal=default_metal, diel=default_diel, unit='m/s'):
	omega = u.to_ureg(omega,'rad/s',convert_quantities=False)
	assert isinstance(metal,snomtools.calcs.materials.metals.Metal), "ERROR: No metal given to Vspp"
	assert isinstance(diel,snomtools.calcs.materials.dielectrics.Dielectric), "ERROR: No dielectric given to Vspp"

	wurzel = numpy.sqrt((metal.epsilon(omega) * diel.epsilon(omega)) / (metal.epsilon(omega) + diel.epsilon(omega)))
	return const.c / wurzel