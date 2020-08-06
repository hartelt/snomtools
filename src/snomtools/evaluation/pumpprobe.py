"""
This file provides data evaluation scripts for time-resolved pump-probe data.
For further info about data structures, see:
data.imports.tiff.py
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.calcs.conversions
import snomtools.data.datasets
import snomtools.calcs.units as u

__author__ = 'hartelt'


def time_scale_axis(delay_axis, unit='fs', unitplotlabel=None):
	"""
	Time scaling calculated for the delay axis of a DataSet, Assures time delays instead of spacial delays.

	:param delay_axis: The Axis instance containing the pulse delays as data.

	:param unit: A valid time unit string. The unit to convert the axis to.

	:param unitplotlabel: The plotlabel corresponding to the specified unit. Will be tried to cast in LaTeX siunitx
		notation if not specified.

	:return: A new Axis instance with time scaling in the specified unit, that can replace the delay axis in the
		DataSet.
	"""
	assert isinstance(delay_axis, snomtools.data.datasets.Axis), "ERROR: no Axis instance given to time_scale_axis()"
	delaydata = delay_axis.get_data()
	scaled_data = delaydata.to(unit, 'light')
	if unitplotlabel is None:
		if u.same_unit(unit, 'fs'):
			unitplotlabel = "\\si{\\femto\\second}"
		elif u.same_unit(unit, 'as'):
			unitplotlabel = "\\si{\\atto\\second}"
		elif u.same_unit(unit, 'ps'):
			unitplotlabel = "\\si{\\pico\\second}"
		elif u.same_unit(unit, 'ns'):
			unitplotlabel = "\\si{\\nano\\second}"
		elif u.same_unit(unit, 's'):
			unitplotlabel = "\\si{\\second}"
	return snomtools.data.datasets.Axis(scaled_data, label='delay', plotlabel="Pulse Delay / " + unitplotlabel)


def delay_apply_timescale(data, unit='fs', unitplotlabel=None):
	"""
	Applies an time scaling to a pump-probe DataSet. The delay axis will be replaced with a new delay axis in time
	units.

	:param data: The DataSet instance of the data to normalize.

	:param unit: A valid time unit string: The unit to convert the axis to.

	:return: The modified dataset.
	"""
	assert isinstance(data, snomtools.data.datasets.DataSet), "ERROR: No DataSet given or imported."
	time_axis = time_scale_axis(data.get_axis('delay'), unit, unitplotlabel)
	data.replace_axis('delay', time_axis)
	return data
