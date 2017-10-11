__author__ = 'hartelt'
'''
This file provides miscellaneous fitting scripts for data.
For furter info about data structures, see:
data.datasets.py
'''

import snomtools.data.datasets
import snomtools.calcs.units as u
import snomtools.data.tools
import numpy


def fit_xy_linear(xdata, ydata):
	"""
	A simple linear fit to data given as x and y values. Fits y = m*x + c and returns c tuple of (m, c), where m and
	c are quantities according to the physical dimensions of data. Numerical data is assumed as dimensionless.

	:param xdata: DataArray or Quantity or numpy array: The x values.

	:param ydata: DataArray or Quantity or numpy array: The y values.

	:return:tuple: (m, c)
	"""
	if isinstance(xdata, snomtools.data.datasets.DataArray):
		xdata = xdata.get_data()
	else:
		xdata = u.to_ureg(xdata)
	if isinstance(ydata, snomtools.data.datasets.DataArray):
		ydata = ydata.get_data()
	else:
		ydata = u.to_ureg(ydata)
	xdata = snomtools.data.tools.assure_1D(xdata)
	ydata = snomtools.data.tools.assure_1D(ydata)

	m, c = numpy.polyfit(xdata.magnitude, ydata.magnitude, deg=1, full=False)

	one_xunit = u.to_ureg(str(xdata.units))
	one_yunit = u.to_ureg(str(ydata.units))
	m = u.to_ureg(m, one_yunit / one_xunit)
	c = u.to_ureg(c, one_xunit)
	return m, c
