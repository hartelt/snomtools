__author__ = 'hartelt'
"""
This file contains the base class for datasets.
"""

import snomtools.calcs.units as u
import numpy
import os
# import h5py
from termcolor import colored#, cprint


class DataArray:
	"""
	A data array that holds additional metadata.
	"""

	def __init__(self, dataarray, unit=None, label=None, plotlabel=None):
		self.data = u.to_ureg(dataarray, unit)
		self.label = str(label)
		self.plotlabel = str(plotlabel)
		self.shape = self.data.shape

	def get_data(self):
		return self.data

	def set_data(self, newdata, unit=None):
		self.data = u.to_ureg(newdata, unit)
		self.shape = self.data.shape

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
		return self.__class__(self.data, label=self.label, plotlabel=self.plotlabel)

	def __neg__(self):
		return self.__class__(-self.data, label=self.label, plotlabel=self.plotlabel)

	def __abs__(self):
		return self.__class__(abs(self.data), label=self.label, plotlabel=self.plotlabel)

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return self.__class__(self.data + other, label=self.label, plotlabel=self.plotlabel)

	def __sub__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return self.__class__(self.data - other, label=self.label, plotlabel=self.plotlabel)

	def __mul__(self, other):
		other = u.to_ureg(other)
		return self.__class__(self.data * other, label=self.label, plotlabel=self.plotlabel)

	def __div__(self, other):
		other = u.to_ureg(other)
		return self.__class__(self.data / other, label=self.label, plotlabel=self.plotlabel)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return self.__class__(self.data // other, label=self.label, plotlabel=self.plotlabel)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return self.__class__(self.data ** other, label=self.label, plotlabel=self.plotlabel)

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
		assert (len(self.data.shape) == 1), "Axis not initialized with 1D array-like object."

	@classmethod
	def from_dataarray(cls, da):
		return cls(da.data, label=da.label, plotlabel=da.plotlabel)

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


class DataSet:
	"""
	A data set is a collection of data arrays combined to have a physical meaning. These are n-dimensional
	sets of physical values, in which every dimension itself has a physical meaning. This might for example be a
	3D-array of count rates, in which the x- y- and z-dimensions represent the position on a sample (x,
	y in micrometers) and a time delay (z = t in femtoseconds).
	"""

	def __init__(self, label="", datafields=(), axes=(), plotconf=()):
		self.label = label
		# check data format and convert it do correct DataArray and Axis objects before assigning it to members:
		self.datafields = []
		for field in datafields:  # Fill datafield list with correctly formatted datafield objects.
			if isinstance(field, DataArray):
				self.datafields.append(field)
			elif u.is_quantity(field):
				self.datafields.append(DataArray(field))
			elif type(field) == str:
				self.datafields.append(DataArray(numpy.array(eval(field))))
			else:
				self.datafields.append(DataArray(numpy.array(field)))

		self.axes = []
		for ax in axes:  # Fill axes list with correctly formatted axes objects.
			if isinstance(ax, Axis):
				self.axes.append(ax)
			elif isinstance(ax, DataArray):
				self.axes.append(Axis.from_dataarray(ax))
			elif u.is_quantity(ax):
				self.axes.append(Axis(ax))
			elif type(ax) == str:
				self.axes.append(Axis(numpy.array(eval(ax))))
			else:
				self.axes.append(Axis(numpy.array(ax)))

		self.check_data_consistency()

		if type(plotconf) == str:
			self.plotconf = dict(eval(plotconf))
		else:
			self.plotconf = dict(plotconf)

	@classmethod
	def from_h5file(cls, path):
		path = os.path.abspath(path)
		filename = os.path.basename(path)
		dataset = cls(filename)
		dataset.loadh5(path)
		return dataset

	def check_data_consistency(self):
		for field in self.datafields:  # Check if all datafields have the same shape.
			assert (field.shape == self.datafields[0].shape), "Dataset datafields with different shapes."
		if self.datafields:  # If we have data at all...
			# Check if we have the same number of axes as the number of datafield dimensions:
			assert len(self.datafields[0].shape) == len(self.axes), "Number of axes don't fit to data dimensionality."
		for i in range(len(self.axes)):  # Check if axes have the same lengths as the corresponding datafield dimension.
			# print "Axis "+str(i)+":"
			# print("axis len: "+str(len(self.axes[i])))
			# print("data len: "+str(self.datafields[0].shape[i]))
			assert (len(self.axes[i]) == self.datafields[0].shape[i]), "Axes lenghts don't fit to data dimensions."

	def saveh5(self, path):
		pass

	def loadh5(self, path):
		pass

	def __del__(self):
		pass


if False:  # just for testing
	print(colored('Testing...', 'green'))
	testarray = numpy.arange(0, 20, 2.)
	testaxis = DataArray(testarray, 'meter')
	testaxis2 = testaxis / 2.
	X, Y = numpy.meshgrid(testaxis, testaxis2)
	# testaxis = DataArray(testarray[testarray<5], 'meter')
	testdata = numpy.sin((X + Y) * u.ureg('rad')) * u.ureg('counts')
	#print(testdata)
	testdataset = DataSet("test", [testdata], [testaxis, testaxis2])