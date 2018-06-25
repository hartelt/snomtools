"""
This file provides data evaluation scripts for PEEM data.
For furter info about data structures, see:
data.imports.tiff.py
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.calcs.units as u
import snomtools.data.datasets
import snomtools.data.imports.tiff
import numpy
import scipy.optimize

__author__ = 'hartelt'


class Powerlaw_loglinear:
	"""
	A powerlaw.
	"""
	def __init__(self, data=None, keepdata=True, normalize=True):
		if data:
			if normalize:
				self.takedata = 0
			else:
				self.takedata = 1
			self.powerunit = data.get_axis(0).get_unit()
			self.countsunit = data.get_datafield(self.takedata).get_unit()
			if keepdata:
				self.data = self.extract_data(data)
				self.coeffs = self.fit_powerlaw(self.data.get_axis(0).get_data(),
												self.data.get_datafield(self.takedata).get_data())
			else:
				self.data = None
				powers, intensities = self.extract_data_raw(data)
				self.coeffs = self.fit_powerlaw(powers, intensities)
			self.poly = numpy.poly1d(self.coeffs)

	@property
	def exponent(self):
		return u.to_ureg(self.coeffs[0])

	@property
	def offset(self):
		return u.to_ureg(self.coeffs[1], self.countsunit)

	@property
	def powers(self):
		if self.data:
			return self.data.get_axis(0).data
		else:
			raise ValueError("No data present.")

	@property
	def counts(self):
		if self.data:
			return self.data.get_datafield(self.takedata).get_data()
		else:
			raise ValueError("No data present.")

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

	@staticmethod
	def extract_data_raw(data, data_id=0, axis_id=None):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
			to a Power in its physical dimension is taken.

		:return: powers, intensities: tuple of quantities with the projected data.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis.get_label())
		# DONE: Project data onto power axis. To be implemented in datasets.py
		return power_axis.get_data(), count_data.project_nd(power_axis_index)

	@staticmethod
	def extract_data(data, data_id=0, axis_id=None, label="powerlaw"):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
			to a Power in its physical dimension is taken.

		:param label: string: label for the produced DataSet

		:return: 1D-DataSet with projected Intensity Data and Power Axis.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis.get_label())
		count_data_projected = count_data.project_nd(power_axis_index)
		count_data_projected = snomtools.data.datasets.DataArray(count_data_projected, label='counts')
		# Normalize by subtracting dark counts:
		count_data_projected_norm = count_data_projected - count_data_projected.min()
		count_data_projected_norm.set_label("counts_normalized")
		# Initialize the DataSet containing only the projected powerlaw data;
		return snomtools.data.datasets.DataSet(label, [count_data_projected_norm, count_data_projected], [power_axis])

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

		# Do the fit with numpy.ma functions to ignore invalid numbers like log(0)
		# (these occur often when using dark count subtraction)
		return numpy.ma.polyfit(numpy.ma.log(powers.magnitude), numpy.ma.log(intensities.magnitude), deg=1, full=False)

	def y(self, x, logx=False):
		if logx:
			return numpy.exp(self.poly(x))
		else:
			return numpy.exp(self.poly(numpy.log(x)))

	def logy(self, x, logx=False):
		if logx:
			return self.poly(x)
		else:
			return self.poly(numpy.log(x))

	def texlabel(self):
		# return r"%\\prop P^{{{0:.2f}}}% Offset %{1}%".format(self.exponent.magnitude, self.offset.magnitude)
		return "Powerlaw {0:.2f}".format(self.exponent.magnitude)


class Powerlaw:
	"""
	A powerlaw.
	"""
	def __init__(self, data=None, guess=None, keepdata=True, normalize=False):
		if data:
			if normalize:
				self.takedata = 0
			else:
				self.takedata = 1
			self.data = self.extract_data(data)
			self.coeffs, self.covariance = self.fit_powerlaw(self.data.get_axis(0).get_data(),
															 self.data.get_datafield(self.takedata).get_data(),
															 guess)
			self.powerunit = self.data.get_axis(0).get_unit()
			self.countsunit = self.data.get_datafield(self.takedata).get_unit()
			if not keepdata:
				self.data = None

	@property
	def amplitude(self):
		return u.to_ureg(self.coeffs[0], self.countsunit)

	@property
	def exponent(self):
		return u.to_ureg(self.coeffs[1])

	@property
	def offset(self):
		return u.to_ureg(self.coeffs[2], self.countsunit)

	@property
	def powers(self):
		if self.data:
			return self.data.get_axis(0).data
		else:
			raise ValueError("No data present.")

	@property
	def counts(self):
		if self.data:
			return self.data.get_datafield(self.takedata).get_data()
		else:
			raise ValueError("No data present.")

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

	@classmethod
	def from_folder_camera(cls, folderpath, pattern="mW", powerunit=None, powerunitlabel=None):
		"""
		Reads a powerlaw data from a folder with snomtools.data.imports.tiff.powerlaw_folder_peem_camera() (see that
		method for details on parameters) and evaluates a powerlaw.

		:return: The Powerlaw instance.
		"""
		data = snomtools.data.imports.tiff.powerlaw_folder_peem_camera(folderpath, pattern, powerunit, powerunitlabel)
		return cls(data)

	@staticmethod
	def extract_data_raw(data, data_id=0, axis_id=None):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
			to a Power in its physical dimension is taken.

		:return: powers, intensities: tuple of quantities with the projected data.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis.get_label())
		# DONE: Project data onto power axis. To be implemented in datasets.py
		return power_axis.get_data(), count_data.project_nd(power_axis_index)

	@staticmethod
	def extract_data(data, data_id=0, axis_id=None, label="powerlaw"):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
			to a Power in its physical dimension is taken.

		:param label: string: label for the produced DataSet

		:return: 1D-DataSet with projected Intensity Data and Power Axis.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis.get_label())
		count_data_projected = count_data.project_nd(power_axis_index)
		count_data_projected = snomtools.data.datasets.DataArray(count_data_projected, label='counts')
		# Normalize by subtracting dark counts:
		count_data_projected_norm = count_data_projected - count_data_projected.min()
		count_data_projected_norm.set_label("counts_normalized")
		# Initialize the DataSet containing only the projected powerlaw data;
		return snomtools.data.datasets.DataSet(label, [count_data_projected_norm, count_data_projected], [power_axis])

	@staticmethod
	def fitfunction(P, amplitude, exponent, offset):
		return amplitude * P ** exponent + offset

	@staticmethod
	def fit_powerlaw(powers, intensities, guess=None):
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

		if guess is None:
			guess_amplitude = 1.
			guess_exponent = 2.
			guess_offset = min(intensities).magnitude
		else:
			assert len(guess) == 3
			guess_amplitude = u.to_ureg(guess[0], str(powers.units)).magnitude
			guess_exponent = u.to_ureg(guess[1], 'dimensionless').magnitude
			guess_offset = u.to_ureg(guess[2], str(powers.units)).magnitude
		guess = (guess_amplitude, guess_exponent, guess_offset)

		return scipy.optimize.curve_fit(Powerlaw.fitfunction, powers.magnitude, intensities.magnitude, guess)

	def y(self, x, logx=False):
		x = u.to_ureg(x, self.powerunit).magnitude
		if logx:
			x = numpy.exp(x)
		return self.amplitude * x ** self.exponent + self.offset

	def logy(self, x, logx=False):
		return numpy.log(self.y(x, logx=logx))

	def __str__(self):
		return "{0} * Power/{1}^{2:.2f} + {3}".format(self.amplitude, self.powerunit, self.exponent, self.offset)

	def texlabel(self):
		# return r"%\\prop P^{{{0:.2f}}}% Offset %{1}%".format(self.exponent.magnitude, self.offset.magnitude)
		return "Powerlaw {0:.2f}".format(self.exponent.magnitude)


def fit_powerlaw(powers, intensities):
	"""
	Shadows Powerlaw.fit_powerlaw. This function fits a powerlaw to data and returns the result as a Powerlaw instance.

	:param powers: A quantity or array of powers. If no quantity, milliwatts are assumed.

	:param intensities: Quantity or array: The corresponding intensity values to powers.

	:return: A Powerlaw instance.
	"""
	coeffs = Powerlaw.fit_powerlaw(powers, intensities)
	return Powerlaw.from_coeffs(coeffs)


if __name__ == '__main__':  # Just for testing.
	print("testing...")

	# powerfolder = "Powerlaw_befCS"
	powerfolder = "Powerlaw"
	# powerfolder = "/home/hartelt/Promotion/Auswertung/2016/06_Juni/20160623_Circles"
	powerdata = snomtools.data.imports.tiff.powerlaw_folder_peem_camera(powerfolder)
	# powerdata = snomtools.data.imports.tiff.powerlaw_folder_peem_dld(powerfolder)
	# roilimits = {'x': [400, 600], 'y': [400, 600], 'power': [u.ureg("13 mW"), None]}
	roilimits = {'x': [400, 600], 'y': [400, 600]}
	# roilimits = {'y': [470, 550], 'x': [580, 660]}
	plroi = snomtools.data.datasets.ROI(powerdata, roilimits)
	testpl = Powerlaw(plroi)

	# picturefilename = "Powerlaw_befCS/48,2mW.tif"
	picturefilename = "Powerlaw/117mW.tif"
	# picturefilename = "/home/hartelt/Promotion/Auswertung/2016/06_Juni/20160623_Circles/01-147mW.tif"
	picturedata = snomtools.data.imports.tiff.peem_camera_read(picturefilename)

	test_plot = True
	if test_plot:
		import snomtools.plots.setupmatplotlib as plt
		import snomtools.plots.datasets
		import os

		fig = plt.figure((12, 12), 1200)
		ax = fig.add_subplot(111)
		ax.cla()
		vert = 'y'
		hori = 'x'
		ax.autoscale(tight=True)
		ax.set_aspect('equal')
		snomtools.plots.datasets.project_2d(picturedata, ax, axis_vert=vert, axis_hori=hori, data_id='counts')
		snomtools.plots.datasets.mark_roi_2d(plroi, ax, axis_vert=vert, axis_hori=hori, ec="w")
		plt.savefig(filename="test.png", figures_path=os.getcwd(), transparent=False)

		fig.clf()
		ax = fig.add_subplot(111)
		ax.cla()

		# ax.invert_yaxis()
		# xforfunc = numpy.linspace(testpl.data.get_axis(0).min(), testpl.data.get_axis(0).max(), 1000)
		# ax.plot(testpl.data.get_axis(0).get_data(),
		# 		testpl.data.get_datafield(0).get_data(),
		# 		'o', label="Counts in Slice")
		# ax.plot(xforfunc, testpl.y(xforfunc), '-', label="Fit with " + str(testpl.poly))
		# ax.set_xscale("log")
		# ax.set_yscale("log")

		import snomtools.plots.evaluations

		snomtools.plots.evaluations.plot_powerlaw(testpl, ax, legend_loc=False)
		plt.legend(loc="lower right")
		fig.savefig(filename="testpowerlaw.png", figures_path=os.getcwd(), transparent=False)

	print("...done.")
