"""
This file provides scripts for common plotting applications working on DataSets as in snomtools.data.datasets.
They use matplotlib.
"""

import snomtools.data.datasets
import matplotlib.patches
import numpy
import os.path


def project_1d(data, plot_dest, axis_id=0, data_id=0, **kwargs):
	"""
	Plots a projection of the data onto one axis. Therefore, it sums the values over all the other axes.

	:param data: The DataSet or ROI to plot.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param axis_id: An identifier of the axis to project onto.

	:param data_id: An identifier of the dataarray to take data from.

	:param kwargs: Keyword arguments for the plot() method of the plot object.

	:return:
	"""
	assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
		"No dataset or ROI instance given to plot function."

	ax_index = data.get_axis_index(axis_id)
	ax = data.get_axis(ax_index)

	sumlist = range(data.dimensions)
	sumlist.remove(ax_index)
	sumtup = tuple(sumlist)
	plotdat = data.get_datafield(data_id).sum(sumtup)

	assert (plotdat.shape == ax.shape), "Plot data shapes don't match."

	plot_dest.plot(ax.get_data(), plotdat, **kwargs)


def project_2d(data, plot_dest, axis_vert=0, axis_hori=1, data_id=0, **kwargs):
	"""
	Plots a projection of the data onto two axes as a pseudocolor 2d map. Therefore, it sums the values over all the
	other axes. Using the pcolor function for matplotlib ensures correct representation of data on nonlinear grids.

	:param data: The DataSet (or ROI) to plot.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param axis_vert: An identifier of the first axis to project onto. This will be the vertical axis in the plot.

	:param axis_hori: An identifier of the second axis to project onto. This will be the horizontal axis in the plot.

	:param data_id: An identifier of the dataarray to take data from.

	:param kwargs: Keyword arguments for the plot() method of the plot object.

	:return:
	"""
	assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
		"No dataset or ROI instance given to plot function."

	axv_index = data.get_axis_index(axis_vert)
	axv = data.get_axis(axv_index)
	axh_index = data.get_axis_index(axis_hori)
	axh = data.get_axis(axh_index)

	sumlist = range(data.dimensions)
	sumlist.remove(axv_index)
	sumlist.remove(axh_index)
	sumtup = tuple(sumlist)
	dat = data.get_datafield(data_id).sum(sumtup)
	if axv_index > axh_index:  # transpose if axes are not in array-like order
		plotdat = dat.T
	else:
		plotdat = dat

	H, V = data.meshgrid([axh_index, axv_index])

	assert (V.shape == plotdat.shape), "2D plot data doesn't fit to axis mesh..."

	plot_dest.pcolormesh(numpy.array(H), numpy.array(V), numpy.array(plotdat))
	# Flip axis to have correct origin (array-index-like): upper left instead of lower left:
	plot_dest.invert_yaxis()


def mark_roi_1d(roi, plot_dest, axis_id=0, **kwargs):
	"""
	Marks a ROI in a 2D plot with a box representing the ROI limits.

	:param roi: The ROI instance.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param axis_id: An identifier of the axis along which the plot is.

	:param kwargs: Keyword arguments for redirection to matplotlib.axes.axvspan(). Specifies the style to be drawn.
	Default will be a grey (black colored transparent alpha=0.2) area.

	:return:
	"""
	assert isinstance(roi, snomtools.data.datasets.ROI), \
		"No ROI instance given to mark function."

	# Set default style kwargs for rectangle if not explicitly given:
	if not kwargs:
		kwargs['fc'] = 'k' # black as face color
		kwargs['alpha'] = 0.2 # transparent
		kwargs['fill'] = True # filled (default)

	lims = roi.get_limits(axis_id, raw=True)

	plot_dest.axvspan(lims[0],lims[1],**kwargs)


def mark_roi_2d(roi, plot_dest, axis_vert=0, axis_hori=1, **kwargs):
	"""
	Marks a ROI in a 2D plot with a box representing the ROI limits.

	:param roi: The ROI instance.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param axis_vert: An identifier of the vertical axis in the plot.

	:param axis_hori: An identifier of the horizontal axis in the plot.

	:param kwargs: Keyword arguments for redirection to the Rectangle methor of matplotlib.patches.

	:return:
	"""
	assert isinstance(roi, snomtools.data.datasets.ROI), \
		"No ROI instance given to mark function."

	# Set fill kwarg for rectangle if not explicitly given:
	if not kwargs.has_key('fill'):
		kwargs['fill'] = False

	xlims = roi.get_limits(axis_hori, raw=True)
	ylims = roi.get_limits(axis_vert, raw=True)

	rectangle = matplotlib.patches.Rectangle((xlims[0], ylims[0]), xlims[1] - xlims[0], ylims[1] - ylims[0], **kwargs)
	plot_dest.add_patch(rectangle)
