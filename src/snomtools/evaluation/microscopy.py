__author__ = 'hartelt'
'''
This file provides data evaluation scripts for microscopy data measured e.g. with PEEM, SNOM or any microscope.
Explicitly, methods for image calibration are provided.
The methods work on DataSets obtained by importing images with the snomtools.data.imports package.
For furter info about data structures, see:
data.imports
data.datasets.py
'''

import snomtools.data.datasets
import snomtools.calcs.units as u


def fov_scale_absolute(pixel_axis, fov, unit='m'):
	"""
	Transforms an Axis from pixels to a real space length from a known field of view.
	:param pixel_axis: The Axis to transform.
	:param fov: Quantity: The absolute size of the FoV. If a number instead of a quantity is given, m is assumed as
	unit.
	:param unit: String: Specifies the output unit for the Axis, Must evaluate to a length unit.
	:return: The converted Axis.
	"""
	assert isinstance(pixel_axis, snomtools.data.datasets.Axis), "No Axis instance given to scale function."
	length_per_pixel = fov / len(pixel_axis)
	length_per_pixel = u.to_ureg(length_per_pixel, 'meters/pixel')
	pixels = pixel_axis.get_data()
	return pixel_axis.scale_linear(length_per_pixel,unit=unit)