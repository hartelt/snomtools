__author__ = 'hartelt'
"""
This script holds transformation functions for datasets, normalize data relative to reference data.
"""

import numpy
import snomtools.data.datasets


def normalize_by_reference(data, refdata, data_id=0, refdata_id=0, exclude_axes=None,
						   mode = "division",
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
	assert isinstance(data, snomtools.data.datasets.DataSet), "ERROR: No DataSet given or imported."
	assert isinstance(refdata, snomtools.data.datasets.DataSet), "ERROR: No DataSet given or imported."

	# Assemble tuple of axis indices to project onto:
	if exclude_axes:
		sumlist = []
		for exclude_axis in exclude_axes:
			sumlist.append(refdata.get_axis(exclude_axis))
		sumtup = tuple(sumlist)
		refquantity = refdata.get_datafield(refdata_id).sum(sumtup)
	else:
		refquantity = refdata.get_datafield(refdata_id).get_data()

	if mode in ["division", "divide", "div"]:
		data_normalized = data.get_datafield(data_id) / refquantity
	elif mode in ["subtraction", "subtract", "sub"]:
		data_normalized = data.get_datafield(data_id) - refquantity
	else:
		raise ValueError("Unrecognized mode for normalize_by_reference.")

	data_normalized[~ numpy.isfinite(data_normalized)] = 0  # set inf, and NaN to 0
	data.add_datafield(data_normalized, label=newlabel, plotlabel=new_plotlabel)
	return data