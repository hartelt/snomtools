__author__ = 'hartelt'
"""
This file contains the base class for datasets, which are n-dimensional sets of physical values, in which every
dimension itself has a physical meaning. This might for example be a 3D-array of count rates, in which the x- y- and
z-dimensions represent the position on a sample (x, y in micrometers) and a time delay (z = t in femtoseconds).
"""

import snomtools.calcs.units as u
import numpy


class DataArray:
	"""
	A data array that holds additional metadata.
	"""

	def __init__(self, dataarray, unit=None, label=None, plotlabel=None):
		self.data = u.to_ureg(dataarray, unit)
		self.label = str(label)
		self.plotlabel = str(plotlabel)

	def get_data(self):
		return self.data

	def set_data(self, newdata, unit=None):
		self.data = u.to_ureg(newdata, unit)

	def get_unit(self):
		return str(self.data.units)

	def set_unit(self, unitstr):
		self.data = u.to_ureg(self.data, unitstr)

	def get_label(self):
		return self.label

	def set_label(self, newlabel):
		self.label = str(newlabel)

	def get_plotlabel(self):
		return self.plotlabel

	def set_plotlabel(self, newlabel):
		self.plotlabel = str(newlabel)

	def __pos__(self):
		return Axis(self.data, label=self.label, plotlabel=self.plotlabel)

	def __neg__(self):
		return Axis(-self.data, label=self.label, plotlabel=self.plotlabel)

	def __abs__(self):
		return Axis(abs(self.data), label=self.label, plotlabel=self.plotlabel)

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return Axis(self.data + other, label=self.label, plotlabel=self.plotlabel)

	def __sub__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return Axis(self.data - other, label=self.label, plotlabel=self.plotlabel)

	def __mul__(self, other):
		other = u.to_ureg(other)
		return Axis(self.data * other, label=self.label, plotlabel=self.plotlabel)

	def __div__(self, other):
		other = u.to_ureg(other)
		return Axis(self.data / other, label=self.label, plotlabel=self.plotlabel)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return Axis(self.data // other, label=self.label, plotlabel=self.plotlabel)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return Axis(self.data ** other, label=self.label, plotlabel=self.plotlabel)

	def __array__(self):  # to numpy array
		return numpy.array(self.data)

	def __iter__(self):
		return iter(self.data)

	def __len__(self):  # len of data array
		return len(self.data)

	def __str__(self):
		out = "DataArray"
		if self.label:
			out += ": " + self.label
		out += " with shape " + str(self.data.shape)
		# out += '\n' + str(self.data)
		return out

	def __repr__(self):
		out = 'DataArray('
		out += str(self.data)
		out += ', '
		if self.label:
			out += 'label=' + self.label
			out += ', '
		if self.plotlabel:
			out += 'plotlabel=' + self.plotlabel
			out += ', '
		out = out.rstrip(', ')
		out += ')'
		return out


class Axis(DataArray):
	"""
	An axis is a data array that holds the data for an axis of a dataset.
	"""

	def __init__(self, dataarray, unit=None, label=None, plotlabel=None):
		DataArray.__init__(self, dataarray, unit=unit, label=label, plotlabel=plotlabel)
		if len(self.data.shape) != 1:  # The given array is not 1D
			print("ERROR: Axis must be initialized with 1D array-like object.")
			raise ValueError('dimension mismatch')

	def __str__(self):
		out = "Axis"
		if self.label:
			out += ": " + self.label
		out += " with shape " + str(self.data.shape)
		# out += '\n' + str(self.data)
		return out

	def __repr__(self):
		out = 'Axis('
		out += str(self.data)
		out += ', '
		if self.label:
			out += 'label=' + self.label
			out += ', '
		if self.plotlabel:
			out += 'plotlabel=' + self.plotlabel
			out += ', '
		out = out.rstrip(', ')
		out += ')'
		return out


if False:  # just for testing
	testarray = numpy.arange(10).reshape((2,5))
	testaxis = Axis(testarray, 'meter')
	print(testaxis)
	testaxis.set_unit('mm')
	print(testaxis.get_unit())
	test2 = testaxis ** 2 * u.ureg('J')
	print test2.get_data()