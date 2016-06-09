"""
This file provides scripts for common plotting applications working on DataSets as in snomtools.data.datasets.
They use matplotlib.
"""

import snomtools.data.datasets
import snomtools.plots.setupmatplotlib as plt
import numpy
import os.path


def project_1d(data, plot_dest, axis_id=0, data_id=0, **kwargs):
	"""
	Plots a projection of the data onto one axis. Therefore, it sums the values over all the other axes.

	:param data: The DataSet to plot.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param axis_id: An identifier of the axis to project onto.

	:param data_id: An identifier of the dataarray to take data from.

	:param kwargs: Keyword arguments for the plot() method of the plot object.

	:return:
	"""
	assert isinstance(data, snomtools.data.datasets.DataSet), "No dataset instance given to plot function."

	ax_index = data.get_axis_index(axis_id)
	ax = data.get_axis(ax_index)

	sumlist = range(len(data.axes))
	sumlist.remove(ax_index)
	sumtup = tuple(sumlist)
	plotdat = data.get_datafield(data_id).sum(sumtup)

	assert (plotdat.shape == ax.shape), "Plot data shapes don't match."

	plot_dest.plot(ax.get_data(), plotdat, **kwargs)
