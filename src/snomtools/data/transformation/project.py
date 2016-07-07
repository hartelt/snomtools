__author__ = 'hartelt'
"""
This script holds transformation functions for datasets, that project data onto given axes.
"""

import snomtools.data.datasets as datasets


def project_1d(data, axis_id=0, data_id=None, outlabel=None):
	"""
	Plots a projection of the data onto one axis. Therefore, it sums the values over all the other axes.

	:param data: The DataSet or ROI to plot.

	:param axis_id: An identifier of the axis to project onto.

	:param data_id: Optional: An identifier of the dataarray to take data from. If not given, all DataArrays of the
	Set are projected.

	:param outlabel: String, optional: A label to assign to the projected DataSet. Default: Label of the original
	DataSet.

	:return: A dataset instance with the projected data.
	"""
	assert isinstance(data, datasets.DataSet) or isinstance(data, datasets.ROI), \
		"No dataset or ROI instance given to projection function."

	if outlabel is None:
		outlabel = data.label

	ax_index = data.get_axis_index(axis_id)
	ax = data.get_axis(ax_index)

	sumlist = range(data.dimensions)
	sumlist.remove(ax_index)
	sumtup = tuple(sumlist)

	dfields = []
	if data_id:
		dlabels = data_id
	else:
		dlabels = data.dlabels

	for label in dlabels:
		df = data.get_datafield(label)
		sumdat = df.sum(sumtup)
		outfield = datasets.DataArray(sumdat, label=df.get_label(), plotlabel="projected " + df.get_plotlabel())
		dfields.append(outfield)

	return datasets.DataSet(outlabel, dfields, ax)

def project_2d(data, axis1_id=0, axis2_id=0, data_id=None, outlabel=None):
	"""
	Plots a projection of the data onto one axis. Therefore, it sums the values over all the other axes.

	:param data: The DataSet or ROI to plot.

	:param axis1_id: An identifier of the first axis to project onto.

	:param axis2_id: An identifier of the second axis to project onto.

	:param data_id: Optional: An identifier of the dataarray to take data from. If not given, all DataArrays of the
	Set are projected.

	:param outlabel: String, optional: A label to assign to the projected DataSet. Default: Label of the original
	DataSet.

	:return: A dataset instance with the projected data.
	"""
	assert isinstance(data, datasets.DataSet) or isinstance(data, datasets.ROI), \
		"No dataset or ROI instance given to projection function."

	if outlabel is None:
		outlabel = data.label

	ax1_index = data.get_axis_index(axis1_id)
	ax1 = data.get_axis(ax1_index)
	ax2_index = data.get_axis_index(axis2_id)
	ax2 = data.get_axis(ax2_index)
	if ax1_index < ax2_index:
		axes = [ax1,ax2]
	elif ax1_index > ax2_index:
		axes = [ax2,ax1]
	else:
		raise IndexError("Attempted 2D projection over the same axis given twice.")

	sumlist = range(data.dimensions)
	sumlist.remove(ax1_index)
	sumlist.remove(ax2_index)
	sumtup = tuple(sumlist)

	dfields = []
	if data_id:
		dlabels = data_id
	else:
		dlabels = data.dlabels

	for label in dlabels:
		df = data.get_datafield(label)
		sumdat = df.sum(sumtup)
		outfield = datasets.DataArray(sumdat, label=df.get_label(), plotlabel="projected " + df.get_plotlabel())
		dfields.append(outfield)

	return datasets.DataSet(outlabel, dfields, axes)
