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
import os

__author__ = 'Lukas Hellbrück'


def load_dispersion_data(data, y_axisid='y', x_axisid='x', e_axisid='energy', d_axisid='delay',
						 x_center=None, x_offset=0, x_window=10,
						 delay_center=None, delay_offset=0, delay_window=10):
	"""
	Loads a n-D HDF5 file and projects it onto the energy- and pixel axis of the dispersion data (default ``energy``,
	``y``) to create a dispersion plot.
	For better statistics, a number of slices along the *other* pixel axis (default ``x``) and time delay axis
	(default ``delay``) are summed up (``10``, ``10`` by default).

	:param data: n-D-DataSet with y-pixel, x-pixel, energy and a k-space dimension.

	:param y_axisid: The name (label) of the y-axis of the data, used as dispersion k direction. Default: ``y``
	:type y_axisid: str

	:param x_axisid: The name (label) of the x-axis of the data, used to sum over. If set to ``False`` or ``None``,
		no pixel summation is done and other ``x_``...-Parameters are ignored. Default: ``x``
	:type x_axisid: str

	:param e_axisid: The name (label) of the energy-axis of the data, used for the dispersion. Default: ``energy``
	:type e_axisid: str

	:param d_axisid: The name (label) of the delay-axis of the data, used to sum over. If set to ``False`` or ``None``,
		no summation is done and other ``delay_``...-Parameters are ignored. Default: ``delay``
	:type d_axisid: str

	:param x_center: The center position index along the x Axis around which shall be summed.
		Default: The "middle" of the axis, defined as half its length.
	:type x_center: int

	:param x_offset: An offset in pixels (array indices) relative to ``x_center``. For example using this with
		default ``x_center`` allows to provide a relative rather than absolute origin to sum over.
	:type x_offset: int

	:param x_window: A number of pixels around the center to sum over. Default: ``10``
	:type x_window: int

	:param delay_center: The center position index along the delay Axis around which shall be summed.
		Default: The "middle" of the axis, defined as half its length.
	:type delay_center: int

	:param delay_offset: An offset in energy channels (array indices) relative to ``delay_center``. For example using
		this with default ``delay_center`` allows to provide a relative rather than absolute origin to sum over.
	:type delay_offset: int

	:param delay_window: A number of energy channels around the center to sum over. Default: ``10``
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
				guess_kfov = None,
				k_axisid='y', e_axisid='energy', savefig=False, figname, **kwargs):
	"""
	Plots the 2d dispersion data along a free electron parable with given parameters. Useful to test k scale.

	:param dispersion_data: 2D-DataSet with an energy and a k-space dimension.

	:param guess_zeropixel: The origin pixel value of the parable, given in pixels or unscaled k-axis units.
	:type guess_zeropixel: float

	:param guess_scalefactor: The scalefactor translating unscaled k-axis units to k-space. Typically given in
		``angstrom**-1 per pixel``. Alternatively, ``guess_kfov`` can be used to give full kspace width instead,
		see below.
	:type guess_scalefactor: float

	:param guess_energyoffset: The origin of the parable on the energy axis. Typically, something like the drift
		voltage in PEEM.
	:type guess_energyoffset: float

	:param guess_kfov: Only used if ``guess_scalefactor`` is not given. Then, this can be given (in ``angstrom**-1``)
		to guess the kspace-Field-of-View (full kspace image width) instead of a factor per pixel.
		If neither ``guess_scalefactor`` or this parameter are given, a generic value for ``guess_kfov`` of
		``1.5 angstrom**-1`` is used.
	:type guess_kfov: float

	:param k_axisid: The name (label) of the k-axis of the data. Default: ``y``
	:type k_axisid: str

	:param e_axisid: The name (label) of the energy axis of the data. Default: ``energy``
	:type e_axisid: str

	:param savefig: Switch to determin if ploted figure should be saved or not.
	:type savefig: bool

	:param figname: Name of the file the plot should be saved into.
	:type figname: str

	:param kwargs: Keyword arguments for the plot() normalization of the plot object.

	:return: A tuple of (scalefactor, zeropixel) that was used for the plot. As this is just the replicated input
		parameters, it can be ignored or used for info/debugging.
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
	if guess_kfov is None:
		guess_kfov = u.to_ureg(1.5, "1/angstrom")
	else:
		guess_kfov = u.to_ureg(guess_kfov, "1/angstrom")
	if guess_scalefactor is None:
		scalefactor = guess_kfov / (dldpixels.max() - dldpixels.min())
	else:
		scalefactor = u.to_ureg(guess_scalefactor, "1/angstrom per pixel")

	# Calculate a free electron parabola with given parameters
	parab_data = freeElectronParabola(dldpixels, scalefactor, zeropoint, energy_offset)

	# Plot dispersion and ParabolaFit
	plt.figure()
	ax = plt.subplot(111)
	snomtools.plots.datasets.project_2d(dispersion_data, ax, e_axisid, k_axisid, **kwargs)
	ax.plot(dldpixels, parab_data, 'r-', label="Fitparabel") # Plot parabola as red line.
	ax.invert_yaxis() # project_2d flips the y axis as it assumes standard matrix orientation, so flip it back.)
	if savefig:
		plt.savefig(figname)
	plt.show()

	return (scalefactor, zeropoint)

def show_state_parabola(dispersion_data, guess_zeropixel=None, guess_mass=None, guess_energyoffset=None,
				k_axisid='y', e_axisid='energy', savefig=False, figname, **kwargs):
	"""
	Plots the 2d dispersion data along a parable for a intermediate state with given parameters. Useful
	for finding out the specific band mass.

	:param dispersion_data: 2D-DataSet with an energy and a k-space dimension.

	:param guess_zeropixel: The origin k value of the parable, given in ``1/angstrom``.
	:type guess_zeropixel: float

	:param guess_mass: The bandmass of the intermediate state you are interested. Typically given in
		units of m_e (electronmass).
	:type guess_mass: float

	:param guess_energyoffset: The origin of the parable on the energy axis. Typically, something like the drift
		voltage in PEEM.
	:type guess_energyoffset: float

	:param k_axisid: The name (label) of the k-axis of the data. Default: ``y``
	:type k_axisid: str

	:param e_axisid: The name (label) of the energy axis of the data. Default: ``energy``
	:type e_axisid: str

	:param savefig: Switch to determin if ploted figure should be saved or not.
	:type savefig: bool

	:param figname: Name of the file the plot should be saved into.
	:type figname: str

	:param kwargs: Keyword arguments for the plot() normalization of the plot object.

	:return: The value of the bandmass, that was used in the plot. Typically given in
		units of m_e (electronmass).
	"""
	# Define parabola and parameters for fit
	if guess_energyoffset is None:
		energy_offset = u.to_ureg(30, "eV")
	else:
		energy_offset = u.to_ureg(guess_energyoffset, "eV")
	k = dispersion_data.get_axis(k_axisid).data
	if guess_zeropixel is None:
		zeropoint = k.mean()
	else:
		zeropoint = u.to_ureg(guess_zeropixel, "1/angstrom")
	if guess_mass is None:
		bandmass = u.to_ureg(1, "m_e")
	else:
		bandmass = u.to_ureg(guess_mass, "m_e")

	# Calculate a parabola with set electronmass
	parab_data = bandDispersionRelation(k, bandmass, zeropoint, energy_offset)

	# Plot dispersion and ParabolaFit
	plt.figure()
	ax = plt.subplot(111)
	snomtools.plots.datasets.project_2d(dispersion_data, ax, e_axisid, k_axisid, **kwargs)
	ax.plot(k, parab_data, 'r-', label="Fitparabel") # Plot parabola as red line.
	ax.invert_yaxis() # project_2d flips the y axis as it assumes standard matrix orientation, so flip it back.
	if savefig:
		plt.savefig(figname)
	plt.show()

	return bandmass


def freeElectronParabola(x, kscale, zero, offset, energyunit='eV'):
	"""
	Calculates a standard free electron parabola with nature constants and given scaling factor.

	.. note:: This function can also be used to get a free electron parabola for already calibrated k-space DataSets:
		Just put in the k-space axis data as ``x``, set ``kscale`` to ``1`` and give ``offset`` corresponding to the
		scaled units on your k-space axis.

		Example::

			freeElectronParabola(mykspacedata.get_axis('k_y').data, 1, u.to_ureg(1.2, 'angstrom**-1'))

	:param x: An array of x-pixels.

	:param kscale: The scalefactor translating unscaled k-axis units to k-space. Typically given in
		``angstrom**-1 per pixel``.
	:type kscale: float

	:param zero: The origin pixel value of the parable, given in pixels or unscaled k-axis units.
	:type zero: float

	:param offset: The origin of the parable on the energy axis. Typically, something like the drift
		voltage in PEEM.
	:type offset: float

	:param energyunit: Desired unit, you want to use in your data. Default: ``eV``

	:return: Return the free electron parabola energy values for given x-pixel
	"""
	return (hbar ** 2 * (kscale * (x - zero)) ** 2 / (2 * m_e) + offset).to(energyunit)

def bandDispersionRelation(k, m, zero, offset, energyunit='eV'):
	"""
	Calculates a standard free electron parabola with nature constants and given scaling factor.

	:param k: An array of inverse Angstroem.

	:param m: The bandmass m, to fit a freeElectronParabola to a state in your dispersion plot.
	:type m: float

	:param zero: The origin pixel value of the parable, given in ``angstrom**-1``.
	:type zero: float

	:param offset: The origin of the parable on the energy axis. Depending on the state of interest.
	:type offset: float

	:param energyunit: Desired unit, you want to use in your data. Default: ``eV``

	:return: Return a parabola with a specific electron mass, to fit to your dispersion plot.
	"""
	return (hbar ** 2 * (k - zero) ** 2 / (2 * m) + offset).to(energyunit)


def kscale_axes(data, scalefactor, yzero=None, xzero=None, y_axisid='y', x_axisid='x'):
	"""
	Scales the x- and y-axis of a given set of dldpixels from a 4D-Dataset to k-space, depending on a before
	determined scalefactor.

	:param data: 4D-DataSet with y-pixel, x-pixel, energy and a k-space dimension.

	:param scalefactor: The scalefactor translating unscaled k-axis units to k-space. Typically given in
		``angstrom**-1 per pixel``.
	:type scalefactor: float

	:param yzero: The offset of the Gamma-point in k_y direction.
	:type yzero: float

	:param xzero: The offset of the Gamma-point in k_x direction.
	:type xzero: float

	:param yaxisid: The name (label) of the x-axis of the data. Default: ``y``
	:type y_axisid: str

	:param xaxisid: The name (label) of the y-axis of the data. Default: ``x``
	:type x_axisid: str

	:return: The k-scaled 4D-Dataset.
	"""
	if yzero is None:
		yzero = data.get_axis(y_axisid).mean()
	else:
		yzero = u.to_ureg(yzero, data.get_axis(y_axisid).units)
	data.get_axis(y_axisid).scale_linear(scalefactor, scalefactor * (-yzero), 'angstrom**-1',
										label='k_y',
										plotlabel="k_y / Angstroem^-1")
	if xzero is None:
		xzero = data.get_axis(x_axisid).mean()
	else:
		xzero = u.to_ureg(xzero, data.get_axis(x_axisid).units)
	data.get_axis(x_axisid).scale_linear(scalefactor, scalefactor * (-xzero), 'angstrom**-1',
										label='k_x',
										plotlabel="k_x / Angstroem^-1")


if __name__ == '__main__':
	# ___ Example for usage ___:
	# Load experimental data, copy to new target and project dispersion data:
	data_folder = os.path.abspath("E:\\Evaluation\\20200102_Au111")
	file = "1. Durchlauf_binned.hdf5"
	file_path = os.path.join(data_folder,file)
	full_data = ds.DataSet.from_h5(file_path, file_path.replace('.hdf5', '_kscaled.hdf5'))

	# Parameters for fitting the Parabola to your data
	scalefactor = None # example: u.to_ureg(0.02, 'angstrom**-1 per pixel')
	e_offset = None # example: u.to_ureg(30, 'eV')
	zero = None # example: u.to_ureg(650/2, 'pixel')
	kfov = None # example: u.to_ureg(1.5, '1/angstrom')

	dispersion_data = load_dispersion_data(full_data, y_axisid='y binned x10', x_axisid='x binned x10')

	# Show k-space scaling example by plotting parabola along data:
	(scalefactor, zeropoint) = show_kscale(dispersion_data, zero, scalefactor, e_offset, kfov, k_axisid='y binned x10')
	print((scalefactor, zeropoint))

	# Scale k-space axes according to some scaling factor and save the scaled DataSet:
	save = False
	if save:
		kscale_axes(full_data, scalefactor, zeropoint, yaxisid='y binned x10', xaxisid='x binned x10')
		full_data.saveh5()
