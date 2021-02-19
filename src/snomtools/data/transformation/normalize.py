"""
This script holds transformation functions for datasets, normalize data relative to reference data.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import snomtools.data.datasets as ds

__author__ = 'hartelt'

def normalize_by_reference(data, refdata, data_id=0, refdata_id=0, exclude_axes=None,
						   mode="division",
						   newlabel='normalizeddata',
						   new_plotlabel="Normalized Data"):
	"""
	Normalizes a dataset by the reference data of another set.
	The normalized data is written into a new DataArray in the given DataSet.
	Data and Reference dimensions must be compatible.

	:param data: The DataSet instance of the data to normalize.

	:param refdata: The DataSet instance of the reference data.

	:param data_id: A valid identifier of the DataArray in the DataSet instance to apply normalization to. Per
		default, the first DataArray is taken.

	:param flat_id: A valid identifier of the DataArray in the reference DataSet instance to take as reference. Per
		default, the first DataArray is taken.

	:param exclude_axes: A list of valid axes identifiers of the reference data to exclude during normalization. The
		reference data is then projected onto the axes which are not included in this list, so the included axes
		axes are kept at constant relative values.

	:param mode: The mode how the calculation between the data and reference should be done, Valid options:
		"division", "divide", "div": Divide every pixel of the data by the corresponding pixel of the reference.
		"subtraction", "subtract", "sub": Subtract every pixel of the data by the corresponding pixel of the reference.

	:param newlabel: The label to set for the created DataArray.

	:param new_plotlabel: The plotlabel to set for the created DataArray.

	:return: The modified dataset.
	"""
	assert isinstance(data, ds.DataSet), "ERROR: No DataSet given or imported."
	assert isinstance(refdata, (ds.DataSet, ds.ROI)), "ERROR: No DataSet or ROI given or imported."

	# Assemble tuple of axis indices to project onto:
	if exclude_axes:
		sumlist = []
		for exclude_axis in exclude_axes:
			sumlist.append(refdata.get_axis_index(exclude_axis))
		sumtup = tuple(sumlist)
		refquantity = refdata.get_datafield(refdata_id).sum(sumtup)
	else:
		refquantity = refdata.get_datafield(refdata_id).get_data()

	del refdata

	if mode in ["division", "divide", "div"]:
		data_normalized = data.get_datafield(data_id).get_data() / refquantity
	elif mode in ["subtraction", "subtract", "sub"]:
		if refquantity.dtype == np.dtype('uint'):
			refquantity = refquantity.astype('int')  # To avoid unsigned integer overflow.
		data_normalized = data.get_datafield(data_id).get_data() - refquantity
	else:
		raise ValueError("Unrecognized mode for normalize_by_reference.")

	del refquantity

	data_normalized[~ np.isfinite(data_normalized)] = 0  # set inf, and NaN to 0
	data.add_datafield(data_normalized, label=newlabel+'_reference_'+mode, plotlabel=new_plotlabel+'_reference_'+mode)
	return data


def normalize_along_axis(data, axes, data_id=0,
							   mode="div", ref='max',
							   newlabel='normalizeddata',
							   new_plotlabel="Normalized Data"):
	"""
	Normalizes a dataset along axes.
	The normalized data is written into a new DataArray in the given DataSet.

	:param data: The DataSet instance of the data to normalize.

	:param axes: Nomalization of the data along these axes.

	:param data_id: A valid identifier of the DataArray in the DataSet instance to apply normalization to. Per
		default, the first DataArray is taken.

	:param mode: The mode how the calculation between the data and reference should be done, Valid options:
		"division", "divide", "div": Divide every pixel of the data by the corresponding pixel of the reference.
		"subtraction", "subtract", "sub": Subtract every pixel of the data by the corresponding pixel of the reference.

	:param ref: The calculation of the reference data. Valid options:
		"max": Maximum as reference to normaize.
		"mean": Mean as reference to normaize.
		"absmax": Absolute maximum as reference.
		"absmin": Absolute minimum as reference.
		"sum": Sum as reference.

	:param newlabel: The label to set for the created DataArray.

	:param new_plotlabel: The plotlabel to set for the created DataArray.

	:return: The modified dataset.
	"""

	assert isinstance(data, ds.DataSet), "ERROR: No DataSet given or imported."

	axindexlist = []
	for axis in axes:
		axindexlist.append(data.get_axis_index(axis))
	axindexes = tuple(axindexlist)

	normalized_data = data.get_datafield(data_id).get_data()

	if ref == 'max':
		refquantity = normalized_data.max(axis=axindexes, keepdims=True)
	elif ref == 'mean':
		refquantity = normalized_data.mean(axis=axindexes, keepdims=True)
	elif ref == 'sum':
		refquantity =  normalized_data.sum(axis=axindexes, keepdims=True)
	elif ref == 'absmax':
		refquantity = normalized_data.absmax(axis=axindexes, keepdims=True)
	elif ref == 'absmin':
		refquantity = normalized_data.absmin(axis=axindexes, keepdims=True)
	else:
		raise ValueError("Unrecognized axis for normalize_along_axis.")

	if mode in ["division", "divide", "div"]:
		normalized_data = normalized_data / refquantity
	elif mode in ["subtraction", "subtract", "sub"]:
		if refquantity.dtype == np.dtype('uint'):
			refquantity = refquantity.astype('int')  # To avoid unsigned integer overflow.
		normalized_data = normalized_data - refquantity
	else:
		raise ValueError("Unrecognized mode for normalize_by_reference.")

	normalized_data[~ np.isfinite(normalized_data)] = 0  # set inf, and NaN to 0
	data.add_datafield(normalized_data, label=newlabel+'_axes_'+str(axes)[1:-1]+'_'+mode, plotlabel=new_plotlabel+'_axes_'+str(axes)[1:-1]+'_'+mode)
	return data


def normalize_sensitivity(data, sensitivity, data_id=0, sensitivity_id=0, include_axes=None,
						   mode="division",
						   newlabel='normalizeddata',
						   new_plotlabel="Normalized Data"):
	"""
	Normalizes a dataset by the sensitivity which is given by the data of another set.
	The normalized data is written into a new DataArray in the given DataSet.
	Data and sensitivity dimensions must be compatible.

	:param data: The DataSet instance of the data to normalize.

	:param sensitivity: The DataSet instance of the sensitivity (reference data).

	:param data_id: A valid identifier of the DataArray in the DataSet instance to apply normalization to. Per
		default, the first DataArray is taken.

	:param sensitivity_id: A valid identifier of the DataArray in the reference DataSet instance to take as reference. Per
		default, the first DataArray is taken.

	:param include_axes: A list of valid axes identifiers of the reference data to include during normalization.

	:param mode: The mode how the calculation between the data and reference should be done, Valid options:
		"division", "divide", "div": Divide every pixel of the data by the corresponding pixel of the reference.
		"subtraction", "subtract", "sub": Subtract every pixel of the data by the corresponding pixel of the reference.

	:param newlabel: The label to set for the created DataArray.

	:param new_plotlabel: The plotlabel to set for the created DataArray.

	:return: The modified dataset.
	"""
	assert isinstance(data, ds.DataSet), "ERROR: No DataSet given or imported."
	assert isinstance(sensitivity, ds.DataSet), "ERROR: No DataSet given or imported."

	# Assemble tuple of axis indices to project onto:
	if include_axes:
		#get the exclude axes
		exclude_axes = sensitivity.axlabels
		for include_axis in include_axes:
			exclude_axes.remove(include_axis)
		sumlist = []
		for exclude_axis in exclude_axes:
			sumlist.remove(sensitivity.get_axis_index(exclude_axis))
		sumtup = tuple(sumlist)
		refquantity = sensitivity.get_datafield(sensitivity_id).sum(sumtup)
	else:
		refquantity = sensitivity.get_datafield(sensitivity_id).get_data()

	del sensitivity

	if mode in ["division", "divide", "div"]:
		data_normalized = data.get_datafield(data_id).get_data() / refquantity
	elif mode in ["subtraction", "subtract", "sub"]:
		if refquantity.dtype == np.dtype('uint'):
			refquantity = refquantity.astype('int')  # To avoid unsigned integer overflow.
		data_normalized = data.get_datafield(data_id).get_data() - refquantity
	else:
		raise ValueError("Unrecognized mode for normalize_sensitivity.")

	del refquantity

	data_normalized[~ np.isfinite(data_normalized)] = 0  # set inf, and NaN to 0
	data.add_datafield(data_normalized, label=newlabel+'_reference_'+mode, plotlabel=new_plotlabel+'_reference_'+mode)
	return data