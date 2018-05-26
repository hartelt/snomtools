"""
This file provides miscellaneous fitting scripts for data.
For furter info about data structures, see:
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.data.datasets as ds
import snomtools.calcs.units as u
import snomtools.data.tools
import numpy as np
from scipy.optimize import curve_fit

__author__ = 'hartelt'


def fit_xy_linear(xdata, ydata):
	"""
	A simple linear fit to data given as x and y values. Fits y = m*x + c and returns c tuple of (m, c), where m and
	c are quantities according to the physical dimensions of data. Numerical data is assumed as dimensionless.

	:param xdata: DataArray or Quantity or numpy array: The x values.

	:param ydata: DataArray or Quantity or numpy array: The y values.

	:return:tuple: (m, c)
	"""
	if isinstance(xdata, ds.DataArray):
		xdata = xdata.get_data()
	else:
		xdata = u.to_ureg(xdata)
	if isinstance(ydata, ds.DataArray):
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

	:param x: The variable x.

	:param x_0: (Same unit as x.) The center of the gaussian.

	:param sigma: (Same unit as x.) The width (standard deviation) of the gaussian. Relates to FWHM by:
		FWHM = 2 sqrt(2 ln 2) sigma

	:param A: (Same unit as C.) The amplitude of the gaussian bell relative to background.

	:param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

	:return: (Same unit as A and C.) The result of the gaussian function.
	"""
	return A * np.exp(-(x - x_0) ** 2 / (2 * sigma ** 2)) + C


class Gauss_Fit(object):
	"""
	A Gauss Fit of given data with benefits.
	"""

	def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
		if data:
			self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
			xunit = self.data.get_axis(0).get_unit()
			yunit = self.data.get_datafield(0).get_unit()

			if normalize:
				take_data = 0
			else:
				take_data = 1

			self.coeffs, self.accuracy = self.fit_gaussian(self.data.get_axis(0).get_data(),
														   self.data.get_datafield(take_data).get_data(),
														   guess)
			self.x_0_unit = xunit
			self.sigma_unit = xunit
			self.A_unit = yunit
			self.C_unit = yunit
			if not keepdata:
				self.data = None

	@property
	def x_0(self):
		return u.to_ureg(self.coeffs[0], self.x_0_unit)

	@x_0.setter
	def x_0(self, newvalue):
		newvalue = u.to_ureg(newvalue, self.x_0_unit)
		self.coeffs[0] = newvalue.magnitude

	@property
	def sigma(self):
		return u.to_ureg(self.coeffs[1], self.sigma_unit)

	@property
	def A(self):
		return u.to_ureg(self.coeffs[2], self.A_unit)

	@property
	def C(self):
		return u.to_ureg(self.coeffs[3], self.C_unit)

	@property
	def FWHM(self):
		return 2 * np.sqrt(2 * np.log(2)) * self.sigma

	@classmethod
	def from_coeffs(cls, coeffs):
		new_instance = cls()
		new_instance.coeffs = coeffs
		return new_instance

	@classmethod
	def from_xy(cls, xdata, ydata, guess=None):
		new_instance = cls()
		new_instance.coeffs, new_instance.accuracy = cls.fit_gaussian(xdata, ydata, guess)
		return new_instance

	def gaussian(self, x):
		"""
		The Gaussian function corresponding to the fit values of the Gauss_Fit instance.

		:param x: The value for which to evaluate the gaussian. (Quantity or numerical in correct unit).

		:return: The value of the gaussian function at the value x. Returned as Quantity in whichever unit the fit
			data was given.
		"""
		x = u.to_ureg(x, self.x_0_unit)
		return gaussian(x, self.x_0, self.sigma, self.A, self.C)

	@staticmethod
	def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
		"""
		Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
		and projects the chosen datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the data.

		:param data_id: Identifier of the DataField to use. By default, the first DataField is used.

		:param axis_id: Identifier of the axis to use. By default, the first Axis is used.

		:param label: string: label for the produced DataSet

		:return: 1D-DataSet with projected Intensity Data and Power Axis.
		"""
		assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
			"ERROR: No dataset or ROI instance given to fit data extraction."
		xaxis = data.get_axis(axis_id)
		data_full = data.get_datafield(data_id)
		xaxis_index = data.get_axis_index(axis_id)
		data_projected = data_full.project_nd(xaxis_index)
		data_projected = ds.DataArray(data_projected, label='projected data')
		# Normalize by scaling to 1:
		data_projected_norm = data_projected / data_projected.max()
		data_projected_norm.set_label("projected data normalized")
		# Initialize the DataSet containing only the projected powerlaw data;
		return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

	@staticmethod
	def extract_data_raw(data, data_id=0, axis_id=0):
		"""
		Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
		and projects the chosen datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the data.

		:param data_id: Identifier of the DataField to use. By default, the first DataField is used.

		:param axis_id: Identifier of the axis to use. By default, the first Axis is used.

		:return: xdata, ydata: tuple of quantities with the projected data.
		"""
		assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
			"ERROR: No dataset or ROI instance given to fit data extraction."
		xaxis = data.get_axis(axis_id)
		data_full = data.get_datafield(data_id)
		xaxis_index = data.get_axis_index(axis_id)
		return xaxis.get_data(), data_full.project_nd(xaxis_index)

	@staticmethod
	def fit_gaussian(xdata, ydata, guess=None):
		"""
		This function fits a gauss function to data. Uses numpy.optimize.curve_fit under the hood.

		:param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

		:param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

		:param guess: optional: A tuple of start parameters (x_0, sigma, A, C) as defined in gaussian method.

		:return: The coefficients and uncertainties of the fitted gaussian (x_0, sigma, A, C), as defined in gaussian
			method.
		"""
		xdata = u.to_ureg(xdata)
		ydata = u.to_ureg(ydata)
		if guess is None:
			guess = (np.mean(xdata), (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
		# to assure the guess is represented in the correct units:
		xunit = xdata.units
		yunit = ydata.units
		unitslist = [xunit, xunit, yunit, yunit]
		guesslist = []
		for guesselement, guessunit in zip(guess, unitslist):
			guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
		guess = tuple(guesslist)
		return curve_fit(gaussian, xdata.magnitude, ydata.magnitude, guess)


def lorentzian(x, x_0, gamma, A, C):
	"""
	A Lorentz function of the form lorentzian(x) = A * ( gamma**2 / ( (x - x_0)**2 + gamma**2 ) ) + C
	All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
	assumed.

	:param x: The variable x.

	:param x_0: (Same unit as x.) The center (peak position) of the distribution.

	:param gamma: (Same unit as x.) The scale parameter. Relates to FWHM by:
		FWHM = 2 * gamma

	:param A: (Same unit as C.) The amplitude of the peak relative to background.

	:param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

	:return: (Same unit as A and C.) The result of the gaussian function.
	"""
	return A * (gamma ** 2 / ((x - x_0) ** 2 + gamma ** 2)) + C


class Lorentz_Fit(object):
	"""
	A Lorentz Fit of given data with benefits.
	"""

	def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
		if data:
			self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
			xunit = self.data.get_axis(0).get_unit()
			yunit = self.data.get_datafield(0).get_unit()

			if normalize:
				take_data = 0
			else:
				take_data = 1

			self.coeffs, self.accuracy = self.fit_lorentzian(self.data.get_axis(0).get_data(),
															 self.data.get_datafield(take_data).get_data(),
															 guess)
			self.x_0_unit = xunit
			self.gamma_unit = xunit
			self.A_unit = yunit
			self.C_unit = yunit
			if not keepdata:
				self.data = None

	@property
	def x_0(self):
		return u.to_ureg(self.coeffs[0], self.x_0_unit)

	@x_0.setter
	def x_0(self, newvalue):
		newvalue = u.to_ureg(newvalue, self.x_0_unit)
		self.coeffs[0] = newvalue.magnitude

	@property
	def gamma(self):
		return u.to_ureg(self.coeffs[1], self.gamma_unit)

	@property
	def A(self):
		return u.to_ureg(self.coeffs[2], self.A_unit)

	@property
	def C(self):
		return u.to_ureg(self.coeffs[3], self.C_unit)

	@property
	def FWHM(self):
		return 2 * self.gamma

	@classmethod
	def from_coeffs(cls, coeffs):
		new_instance = cls()
		new_instance.coeffs = coeffs
		return new_instance

	@classmethod
	def from_xy(cls, xdata, ydata, guess=None):
		new_instance = cls()
		new_instance.coeffs, new_instance.accuracy = cls.fit_lorentzian(xdata, ydata, guess)
		return new_instance

	def lorentzian(self, x):
		"""
		The Lorentz function corresponding to the fit values of the Lorentz_Fit instance.

		:param x: The value for which to evaluate the lorentzian. (Quantity or numerical in correct unit).

		:return: The value of the lorentzian function at the value x. Returned as Quantity in whichever unit the fit
			data was given.
		"""
		x = u.to_ureg(x, self.x_0_unit)
		return lorentzian(x, self.x_0, self.gamma, self.A, self.C)

	@staticmethod
	def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
		"""
		Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
		and projects the chosen datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the data.

		:param data_id: Identifier of the DataField to use. By default, the first DataField is used.

		:param axis_id: Identifier of the axis to use. By default, the first Axis is used.

		:param label: string: label for the produced DataSet

		:return: 1D-DataSet with projected Intensity Data and Power Axis.
		"""
		assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
			"ERROR: No dataset or ROI instance given to fit data extraction."
		xaxis = data.get_axis(axis_id)
		data_full = data.get_datafield(data_id)
		xaxis_index = data.get_axis_index(axis_id)
		data_projected = data_full.project_nd(xaxis_index)
		data_projected = ds.DataArray(data_projected, label='projected data')
		# Normalize by scaling to 1:
		data_projected_norm = data_projected / data_projected.max()
		data_projected_norm.set_label("projected data normalized")
		# Initialize the DataSet containing only the projected powerlaw data;
		return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

	@staticmethod
	def extract_data_raw(data, data_id=0, axis_id=0):
		"""
		Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
		and projects the chosen datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the data.

		:param data_id: Identifier of the DataField to use. By default, the first DataField is used.

		:param axis_id: Identifier of the axis to use. By default, the first Axis is used.

		:return: xdata, ydata: tuple of quantities with the projected data.
		"""
		assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
			"ERROR: No dataset or ROI instance given to fit data extraction."
		xaxis = data.get_axis(axis_id)
		data_full = data.get_datafield(data_id)
		xaxis_index = data.get_axis_index(axis_id)
		return xaxis.get_data(), data_full.project_nd(xaxis_index)

	@staticmethod
	def fit_lorentzian(xdata, ydata, guess=None):
		"""
		This function fits a lorentz function to data. Uses numpy.optimize.curve_fit under the hood.

		:param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

		:param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

		:param guess: optional: A tuple of start parameters (x_0, gamma, A, C) as defined in lorentzian method.

		:return: The coefficients and uncertainties of the fitted gaussian (x_0, gamma, A, C), as defined in lorentzian
			method.
		"""
		xdata = u.to_ureg(xdata)
		ydata = u.to_ureg(ydata)
		if guess is None:
			guess = (np.mean(xdata), (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
		# to assure the guess is represented in the correct units:
		xunit = xdata.units
		yunit = ydata.units
		unitslist = [xunit, xunit, yunit, yunit]
		guesslist = []
		for guesselement, guessunit in zip(guess, unitslist):
			guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
		guess = tuple(guesslist)
		return curve_fit(lorentzian, xdata.magnitude, ydata.magnitude, guess)
