#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
This file provides functions to evaluate k-space data measured e.g. with PEEM.
Explicitly functions to scale dldpixels into inverse Angstrom.
The methods work on a 4D DataSet imported with snomtools.
For furter info about data structures, see:
data.imports
data.datasets.py
"""

import snomtools.data.datasets as ds
import snomtools.calcs.units as u
import snomtools.data.fits
import snomtools.data.transformation.project
import snomtools.plots.datasets
from snomtools.calcs.constants import m_e, hbar
import matplotlib.pyplot as plt

__author__ = 'Lukas Hellbrück'


def load_dispersion_data(data, y_axisid='y', x_axisid='x', e_axisid='energy', d_axisid='delay',
						 x_center=None, x_offset=0, x_window=10,
						 delay_center=None, delay_offset=0, delay_window=10):
	"""
	Loads a n-D HDF5 file and projects it onto the energy- and pixel axis of the dispersion data (default 'energy, 'y')
	to create a dispersion plot.
	For better statistics, a number of slices along the *other* pixel axis (default 'x') and time delay axis
	(default 'delay') are summed up (10,10 by default).

	:param data: n-D-DataSet with y-pixel, x-pixel, energy and a k-space dimension.

	:param y_axisid: The name (label) of the y-axis of the data, used as dispersion k direction.

	:param x_axisid: The name (label) of the x-axis of the data, used to sum over. If set to `False` or `None`,
		no pixel summation is done and other `x_`...-Parameters are ignored.

	:param e_axisid: The name (label) of the energy-axis of the data, used for the dispersion.

	:param d_axisid: The name (label) of the delay-axis of the data, used to sum over. If set to `False` or `None`,
		no summation is done and other `delay_`...-Parameters are ignored.

	:param x_center: The center position index along the x Axis around which shall be summed.
		Default: The "middle" of the axis, defined as half its length.
	:type x_center: int

	:param x_offset: An offset in pixels (array indices) relative to `x_center`. For example using this with
		default `x_center` allows to provide a relative rather than absolute origin to sum over.
	:type x_offset: int

	:param x_window: A number of pixels around the center to sum over. Default: `10`
	:type x_window: int

	:param delay_center: The center position index along the delay Axis around which shall be summed.
		Default: The "middle" of the axis, defined as half its length.
	:type delay_center: int

	:param delay_offset: An offset in energy channels (array indices) relative to `delay_center`. For example using this
		with default `delay_center` allows to provide a relative rather than absolute origin to sum over.
	:type delay_offset: int

	:param delay_window: A number of energy channels around the center to sum over. Default: `10`
	:type delay_window: int

	:return: The projection of the n-D-Dataset on the pixel and energy axis,
		with a summation over slices around time zero and pixel mean point
	"""
	# Define RoI boundaries to sum over for better statistics:
	sum_boundaries_index = {}
	if d_axisid:
		if delay_center is None:
			delay_center = int(len(data.get_axis(d_axisid)) / 2)
		sum_boundaries_index[d_axisid] = [delay_center + delay_offset - int(delay_window / 2),
										  delay_center + delay_offset + int(delay_window / 2)]
	if x_axisid:
		if x_center is None:
			x_center = int(len(data.get_axis(x_axisid)) / 2)
		sum_boundaries_index[x_axisid] = [x_center + x_offset - int(x_window / 2),
										  x_center + x_offset + int(x_window / 2)]

	# Initialize RoI:
	sumroi = ds.ROI(data, sum_boundaries_index, by_index=True)

	# Project RoI to k_x-E-Plane and return data:
	return snomtools.data.transformation.project.project_2d(sumroi, e_axisid, y_axisid)


def show_kscale(dispersion_data, guess_zeropixel=None, guess_scalefactor=None, guess_energyoffset=None,
				k_axisid='y', e_axisid='energy'):
	"""
	Plots the 2d dispersion data along a free electron parable with given parameters. Useful to test k scale.

	:param dispersion_data: 2D-DataSet with an energy and a k-space dimension.

	:param guess_zeropixel: The origin pixel value of the parable, given in pixels or unscaled k-axis units.

	:param guess_scalefactor: The scalefactor translating unscaled k-axis units to k-space. Typically given in
	'angstrom**-1 per pixel'.

	:param guess_energyoffset: The origin of the parable on the energy axis. Typically, something like the drift
	voltage in PEEM.

	:param k_axisid: The name (label) of the k-axis of the data.

	:param e_axisid: The name (label) of the energy axis of the data.

	:return: A tuple of (scalefactor, zeropixel) that was used for the plot.
	"""
	# Define parabola and parameters for fit
	if guess_energyoffset is None:
		energy_offset = u.to_ureg(30, "eV")
	else:
		energy_offset = u.to_ureg(guess_energyoffset, "eV")
	dldpixels = dispersion_data.get_axis(k_axisid).data
	if guess_zeropixel is None:
		zeropoint = dldpixels.mean()
	else:
		zeropoint = u.to_ureg(guess_zeropixel, "pixel")
	if guess_scalefactor is None:
		scalefactor = u.to_ureg(1.5, "1/angstrom") / (dldpixels.max() - dldpixels.min())
	else:
		scalefactor = u.to_ureg(guess_scalefactor, "1/angstrom/pixel")

	# Calculate a free electron parabola with given parameters
	parab_data = freeElectronParabola(dldpixels, scalefactor, zeropoint, energy_offset)

	# Plot dispersion and ParabolaFit
	plt.figure()
	ax = plt.subplot(111)
	snomtools.plots.datasets.project_2d(dispersion_data, ax, e_axisid, k_axisid)
	ax.plot(dldpixels, parab_data, 'r-', label="Fitparabel")
	ax.invert_yaxis()
	plt.show()

	return (scalefactor, zeropoint)


def freeElectronParabola(x, kscale, zero, offset, energyunit='eV'):
	"""
	Calculates a standard free electron parabolawith nature constans and given scaling factor

	:param x: An array of x-pixels.

	:param kscale: The scalefactor translating unscaled k-axis units to k-space. Typically given in
	'angstrom**-1 per pixel'.

	:param zero: The origin pixel value of the parable, given in pixels or unscaled k-axis units.

	:param offset: The origin of the parable on the energy axis. Typically, something like the drift
	voltage in PEEM.

	:param energyunit: Desired unit, you want to use in your data. Typically electronVolts

	:return: Return the free electron parabola energy values for given x-pixel
	"""
	return (hbar ** 2 * (kscale * (x - zero)) ** 2 / (2 * m_e) + offset).to(energyunit)


def kscale_axes(data, scalefactor, yzero=None, xzero=None, yaxisid='y', xaxisid='x'):
	"""
	Scales the x- and y-axis of a given set of dldpixels from a 4D-Dataset to k-space, depending on a before
	determined scalefactor.

	:param data: 4D-DataSet with y-pixel, x-pixel, energy and a k-space dimension.

	:param scalefactor: The scalefactor translating unscaled k-axis units to k-space. Typically given in
	'angstrom**-1 per pixel'.

	:param yzero: The offset of the Gamma-point in k_y direction.

	:param xzero: The offset of the Gamma-point in k_x direction.

	:param yaxisid: The name (label) of the x-axis of the data.

	:param xaxisid: The name (label) of the y-axis of the data.

	:return: The k-scaled 4D-Dataset.
	"""
	if yzero is None:
		yzero = data.get_axis(yaxisid).mean()
	else:
		yzero = u.to_ureg(yzero, data.get_axis(yaxisid).units)
	data.get_axis(yaxisid).scale_linear(scalefactor, scalefactor * (-yzero), 'angstrom**-1',
										label='k_y',
										plotlabel="k_y / Angstroem^-1")
	if xzero is None:
		xzero = data.get_axis(xaxisid).mean()
	else:
		xzero = u.to_ureg(xzero, data.get_axis(xaxisid).units)
	data.get_axis(xaxisid).scale_linear(scalefactor, scalefactor * (-xzero), 'angstrom**-1',
										label='k_x',
										plotlabel="k_x / Angstroem^-1")


if __name__ == '__main__':
	file = "1. Durchlauf_binned.hdf5"

	full_data = ds.DataSet.from_h5(file, file.replace('.hdf5', '_kscaled.hdf5'))
	dispersion_data = load_dispersion_data(full_data, y_axisid='y binned x10', x_axisid='x binned x10')

	(scalefactor, zeropoint) = show_kscale(dispersion_data, k_axisid='y binned x10')
	save = True
	print((scalefactor, zeropoint))

	if save:
		kscale_axes(full_data, scalefactor, zeropoint, yaxisid='y binned x10', xaxisid='x binned x10')
		full_data.saveh5()
