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
	return pixel_axis.scale_linear(length_per_pixel,unit=unit)


def fov_scale_relative(pixel_axis, length_per_pixel, unit='m'):
	"""
	Transforms an Axis from pixels to a real space length from a known length per pixel scaling.

	:param pixel_axis: The Axis to transform.

	:param length_per_pixel: Quantity: The relative scale in length per pixel. If a number instead of a quantity is
	given, m/pixel is assumed as unit.

	:param unit: String: Specifies the output unit for the Axis, Must evaluate to a length unit.

	:return: The converted Axis.
	"""
	assert isinstance(pixel_axis, snomtools.data.datasets.Axis), "No Axis instance given to scale function."
	length_per_pixel = u.to_ureg(length_per_pixel, 'meters/pixel',convert_quantities=False)
	return pixel_axis.scale_linear(length_per_pixel,unit=unit)


def normalize_by_flatfield_sum(data, flatfield_data, data_id=0, flat_id=0, newlabel='norm_int',
							   new_plotlabel="Normalized Intensity"):
	"""
	Normalizes a dataset by the data of another set, that was obtained on an unstructured surface and should
	therefore be "flat" (flatfield). The data is normalized by the sum over all other channels int the dataset,
	so only the spacial	image is normalized, while all other axes are kept at constant relative values.
	The normalized data is written into a new DataArray in the given DataSet.

	:param data: The DataSet instance of the data to normalize or a string with the filepath of the hdf5 file
	containing the data.

	:param flatfield_data: The DataSet instance of the flatfield correction to apply or a string with the filepath of
	the hdf5 file containing the data.

	:param data_id: A valid identifier of the DataArray in the DataSet instance to apply normalization to. Per
	default, the first DataArray is taken.

	:param flat_id: A valid identifier of the DataArray in the flatfield DataSet instance to take as reference. Per
	default, the first DataArray is taken.

	:param newlabel: The label to set for the created DataArray.

	:param new_plotlabel: The plotlabel to set for the created DataArray.

	:return: The modified dataset.
	"""
	if type(data) == str:
		filepath = os.path.abspath(data)
		filebase, ext = os.path.splitext(filepath)
		if ext == ".hdf5":
			data = snomtools.data.datasets.DataSet.from_h5file(filepath)
	if type(flatfield_data) == str:
		filepath = os.path.abspath(flatfield_data)
		filebase, ext = os.path.splitext(filepath)
		if ext == ".hdf5":
			flatfield_data = snomtools.data.datasets.DataSet.from_h5file(filepath)

	assert isinstance(data, snomtools.data.datasets.DataSet), "ERROR: No DataSet given or imported."
	assert isinstance(flatfield_data, snomtools.data.datasets.DataSet), "ERROR: No DataSet given or imported."

	# Assemble tuple of axis indices to project onto:
	axis1_id, axis2_id = "x", "y"
	ax1_index = flatfield_data.get_axis_index(axis1_id)
	ax2_index = flatfield_data.get_axis_index(axis2_id)
	sumlist = range(flatfield_data.dimensions)
	sumlist.remove(ax1_index)
	sumlist.remove(ax2_index)
	sumtup = tuple(sumlist)

	flatfield_sumimage = flatfield_data.get_datafield(flat_id).sum(sumtup)
	data_normalized = data.get_datafield(data_id) / flatfield_sumimage
	data_normalized[~ numpy.isfinite(data_normalized)] = 0  # set inf, and NaN to 0
	data.add_datafield(data_normalized, label=newlabel, plotlabel=new_plotlabel)
	return data