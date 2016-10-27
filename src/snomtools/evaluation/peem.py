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
		return pl

	@classmethod
	def from_xy(cls, powers, intensities):
		pl = cls()
		pl.coeffs = cls.fit_powerlaw(powers, intensities)
		return pl

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
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis)
		# TODO: Project data onto power axis. To be implemented in datasets.py
		return power_axis.get_data(), projected_data

	def fit_powerlaw(powers, intensities):
		"""
		This function fits a powerlaw to data.

		:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.

		:param intensities: Quantity or array: The corresponding intensity values to powers.

		:return: A Powerlaw instance.
		"""
		powers = u.to_ureg(powers, 'mW')
		intensities = u.to_ureg(intensities, 'counts')

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
		This function fits a powerlaw to data.
		:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.
		:param intensities: Quantity or array: The corresponding intensity values to powers.
		:return: A Powerlaw instance.
		"""
	powers = u.to_ureg(powers, 'mW')
	intensities = u.to_ureg(intensities, 'counts')

	coeffs = numpy.polyfit(numpy.log(powers.magnitude), numpy.log(intensities.magnitude), deg=1, full=False)
	return Powerlaw(coeffs)
