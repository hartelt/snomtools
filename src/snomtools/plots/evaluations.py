"""
This file provides scripts for common evaluated data as defined in the classes in the evaluation package.
They use matplotlib.
"""

import numpy


def plot_powerlaw(powerlaw, plot_dest, resolution=1000, legend_loc='lower right'):
	"""
	Plots a powerlaw, meaning the intensity values and corresponding linear fit in the double logarithmic plot.

	:param powerlaw: A snomtools.evaluation.peem.Powerlaw instance.

	:param plot_dest: A matplotlib plot object (like a plot or subplot) to plot into.

	:param resolution: int: The number of points to be calculated for plotting the line representing the fit.

	:param legend_loc: The parameter to be forwarded to the matplotlib.legend method. If set to 'None' or anything
	that evaluates to False, no legend is drawn.

	:return:
	"""
	# import snomtools.evaluation.peem
	# assert isinstance(powerlaw, snomtools.evaluation.peem.Powerlaw), \
	# 	"ERROR: No Powerlaw instance given to plot_powerlaw."

	xforfunc = numpy.linspace(powerlaw.data.get_axis(0).min(), powerlaw.data.get_axis(0).max(), resolution)
	plot_dest.plot(powerlaw.data.get_axis(0).get_data(),
				   powerlaw.data.get_datafield(0).get_data(),
				   'o', label="Counts")
	plot_dest.plot(xforfunc, powerlaw.y(xforfunc), '-', label="Fit with " + str(powerlaw.poly))
	plot_dest.set_xscale("log")
	plot_dest.set_yscale("log")
	if legend_loc:
		if not (legend_loc in ["None", "none"]):
			plot_dest.legend(loc=legend_loc)
