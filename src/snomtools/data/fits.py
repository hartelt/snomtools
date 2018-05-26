"""
This file provides miscellaneous fitting scripts for data.
For furter info about data structures, see:
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.data.datasets
import snomtools.calcs.units as u
import snomtools.data.tools
import numpy as np

__author__ = 'hartelt'


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

	m, c = np.polyfit(xdata.magnitude, ydata.magnitude, deg=1, full=False)

	one_xunit = u.to_ureg(str(xdata.units))
	one_yunit = u.to_ureg(str(ydata.units))
	m = u.to_ureg(m, one_yunit / one_xunit)
	c = u.to_ureg(c, one_yunit)
	return m, c


def gaussian(x, x_0, sigma, A, C):
	"""
	A Gauss function of the form gaussian(x) = A * exp(-(x-x_0)**2 / 2 sigma**2) + C
	All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
	assumed.

	:param x: (Same unit as x.) The variable x.

	:param x_0: The center of the gaussian.

	:param sigma: (Same unit as x.) The width (standard deviation) of the gaussian. Relates to FWHM by:
		FWHM = 2 sqrt(2 ln 2) sigma

	:param A: (Same unit as C.) The amplitude of the gaussian bell relative to background.

	:param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

	:return: (Same unit as A and C.) The result of the gaussian function.
	"""
	return A * np.exp(-(x - x_0) ** 2 / (2 * sigma ** 2)) + C


def lorentzian(x, x_0, gamma, A, C):
	"""
	A Lorentz function of the form lorentzian(x) = A * ( gamma**2 / ( (x - x_0)**2 + gamma**2 ) ) + C
	All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
	assumed.

	:param x: (Same unit as x.) The variable x.

	:param x_0: The center (peak position) of the distribution.

	:param gamma: (Same unit as x.) The scale parameter. Relates to FWHM by:
		FWHM = 2 * gamma

	:param A: (Same unit as C.) The amplitude of the peak relative to background.

	:param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

	:return: (Same unit as A and C.) The result of the gaussian function.
	"""
	return A * (gamma ** 2 / ((x - x_0) ** 2 + gamma ** 2)) + C


# TODO: Implement this similarly to Powerlaw class in evaluation.peem:
class gauss_fit(object):
	"""
	A Gauss Fit of given data with benefits.
	"""

	def __init__(self, data=None, guess=None, keepdata=True, normalize=False):
		raise NotImplementedError()
