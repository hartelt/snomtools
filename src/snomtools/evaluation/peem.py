__author__ = 'hartelt'
'''
This file provides data evaluation scripts for PEEM data.
For furter info about data structures, see:
data.imports.tiff.py
data.datasets.py
'''

import snomtools.calcs.units as u
import snomtools.data.datasets
import snomtools.data.imports.tiff
import os.path
import numpy


class Powerlaw:
	"""
	A powerlaw.
	"""

	# TODO: Test this.

	def __init__(self, data=None):
		if data:
			powers, intensities = self.extract_data(data)
			self.coeffs = self.fit_powerlaw(powers, intensities)
			self.poly = numpy.poly1d(self.coeffs)

	@classmethod
	def from_coeffs(cls, coeffs):
		pl = cls()
		pl.coeffs = coeffs
		pl.poly = numpy.poly1d(pl.coeffs)
		return pl

	@classmethod
	def from_xy(cls, powers, intensities):
		pl = cls()
		pl.coeffs = cls.fit_powerlaw(powers, intensities)
		pl.poly = numpy.poly1d(pl.coeffs)
		return pl

	@classmethod
	def from_folder_camera(cls, folderpath, pattern="mW", powerunit=None, powerunitlabel=None):
		"""
		Reads a powerlaw data from a folder with snomtools.data.imports.tiff.powerlaw_folder_peem_camera() (see that
		method for details on parameters) and evaluates a powerlaw.

		:return: The Powerlaw instance.
		"""
		data = snomtools.data.imports.tiff.powerlaw_folder_peem_camera(folderpath, pattern, powerunit, powerunitlabel)
		return cls(data)

	def extract_data(self, data, data_id=0, axis_id=None):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
		to a Power in its physical dimension is taken.

		:return: powers, intensities: tuple of quantities with the projected data.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet), \
			"ERROR: No dataset instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis)
		# DONE: Project data onto power axis. To be implemented in datasets.py
		return power_axis.get_data(), count_data.project_nd(power_axis_index)

	@staticmethod
	def fit_powerlaw(powers, intensities):
		"""
		This function fits a powerlaw to data.

		:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.

		:param intensities: Quantity or array: The corresponding intensity values to powers.

		:return: The powerlaw coefficients of the fitted polynom.
		"""
		if u.is_quantity(powers):
			assert u.same_dimension(powers, "watts")
			powers = u.to_ureg(powers)
		else:
			powers = u.to_ureg(powers, 'mW')
		intensities = u.to_ureg(intensities)

		return numpy.polyfit(numpy.log(powers.magnitude), numpy.log(intensities.magnitude), deg=1, full=False)

	def y(self, x, logx=False):
		if logx:
			return numpy.exp(self.poly(x))
		else:
			return numpy.exp(self.poly(numpy.log(x)))

	def logy(self, x, logx=False):
		if logx:
			return self.poly(numpy.log(x))
		else:
			return self.poly(x)


def fit_powerlaw(powers, intensities):
	"""
	Shadows Powerlaw.fit_powerlaw. This function fits a powerlaw to data and returns the result as a Powerlaw instance.

	:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.

	:param intensities: Quantity or array: The corresponding intensity values to powers.

	:return: A Powerlaw instance.
	"""
	coeffs = Powerlaw.fit_powerlaw(powers, intensities)
	return Powerlaw.from_coeffs(coeffs)
