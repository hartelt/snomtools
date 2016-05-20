__author__ = 'hartelt'
"""
This file contains the base class for datasets.
"""

import snomtools.calcs.units as u
import numpy
import os
import h5py
import h5tools
import re
from termcolor import colored, cprint


class DataArray:
	"""
	A data array that holds additional metadata.
	"""

	def __init__(self, data, unit=None, label=None, plotlabel=None):
		"""
		Guess what, this is an initializer. It differences between input formats, which should be clear from the
		parameter doc and the comments in the code.
		:param data: required. If this is already a DataArray instance, it is just copied by default, so if the
		other parameters are set, the contents of the instance are overwritten.
		:param unit: Request a unit for the data to be in. If the data holds a unit (quantity), it must be the same
		dimensionality. If it doesn't (array-like), it is assumed to be in the given unit! Default is None for
		unchanged in the first case and dimensionless in the second.
		:param label: A short identifier label that should be meaningful to get the physical context of the data.
		:param plotlabel: A label for the data that should be plotted in a diagram.
		:return:
		"""
		if isinstance(data, DataArray):  # If the data already comes in a DataArray, just take it.
			self.data = data.get_data()
			if unit:  # If a unit is explicitly requested anyway, make sure we set it.
				self.set_unit(unit)
			if label:  # Same with label, else take the one that's already with the data
				self.label = label
			else:
				self.label = data.get_label()
			if plotlabel:  # Same as with label.
				self.plotlabel = label
			else:
				self.plotlabel = data.get_plotlabel()
			# A DataArray contains everything we need, so we should be done here!
		else:  # We DON'T have everything contained in data, so we need to process it seperately.
			if u.is_quantity(data):  # Kind of the same as above, just for the data itself.
				self.data = data
				if unit:  # If a unit is explicitly requested anyway, make sure we set it.
					self.set_unit(unit)
			elif type(data) == str:  # If it's a string, try to evaluate it as an array.
				self.data = u.to_ureg(numpy.array(eval(data)), unit)
			else:  # If it's none of the above, it hopefully is an array-like. So let's try to cast it.
				self.data = u.to_ureg(numpy.array(data), unit)
			self.label = str(label)
			self.plotlabel = str(plotlabel)

	@classmethod
	def from_h5(cls, h5source):
		"""
		This method initializes a DataArray from a HDF5 source.
		:param h5source: The HDF5 source to read from. This is generally the subgroup for the DataArray.
		:return: The initialized DataArray.
		"""
		out = cls([])
		out.load_from_h5(h5source)
		return out

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.
		:param item: The name to get the corresponding attribute.
		:return: The attribute corresponding to the given name.
		"""
		if item == "shape":
			return self.data.shape
		if item == "units":
			return self.data.units
		raise AttributeError("Attribute \'{0}\' of DataArray instance cannot be resolved.".format(item))

	def get_data(self):
		return self.data

	def get_data_raw(self):
		return self.data.magnitude

	def set_data(self, newdata, unit=None):
		self.data = u.to_ureg(newdata, unit)

	def swapaxes(self, axis1, axis2):
		"""
		Swaps two axes of the data. Uses numpy.swapaxes.
		:param axis1: int: First axis.
		:param axis2: int: Second axis.
		:return: The data after the transformation.
		"""
		self.data = u.to_ureg(self.get_data_raw().swapaxes(axis1, axis2), self.get_unit())
		return self.data

	def get_unit(self):
		return str(self.data.units)

	def set_unit(self, unitstr):
		"""
		Set the unit of the dataarray as specified.
		Warning: The plotlabel typically includes a unit, so this might get invalid!
		:param unitstr: A valid unit string.
		:return: Nothing.
		"""
		self.data = u.to_ureg(self.data, unitstr)

	def to(self, unitstr):
		"""
		Returns a copy of the dataarray with the unit set as specified. For compatibility with pint quantity.
		Warning: The plotlabel typically includes a unit, so this might get invalid!
		:param unitstr: A valid unit string.
		:return: The dataset copy with the specified unit.
		"""
		return self.__class__(self.data.to(unitstr), label=self.label, plotlabel=self.plotlabel)

	def get_label(self):
		return self.label

	def set_label(self, newlabel):
		self.label = str(newlabel)

	def get_plotlabel(self):
		return self.plotlabel

	def set_plotlabel(self, newlabel):
		self.plotlabel = str(newlabel)

	def store_to_h5(self, h5dest, subgrp_name=None):
		"""
		Stores the DataArray into a HDF5 file. It will create a subgroup and store the data there in a unified format.
		:param h5dest: The destination. This is a HDF5 file or subgroup.
		:param subgrp_name: Optional. The name for the subgroup that is created to store the data in, Default is the
		label of the DataArray.
		:return The subgroup that was created and holds the data.
		"""
		if not subgrp_name:
			subgrp_name = self.label
		grp = h5dest.create_group(subgrp_name)
		grp.create_dataset("data", data=self.get_data_raw())
		grp.create_dataset("unit", data=self.get_unit())
		grp.create_dataset("label", data=self.get_label())
		grp.create_dataset("plotlabel", data=self.get_plotlabel())
		return grp

	def load_from_h5(self, h5source):
		"""
		Loads the data from a HDF5 source.
		:param h5source: The source to read from. This is the subgroup of the DataArray.
		"""
		self.set_data(numpy.array(h5source["data"]), h5source["unit"][()])
		self.set_label(h5source["label"][()])
		self.set_plotlabel(h5source["plotlabel"][()])

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

	def __getitem__(self, key):
		"""
		To allow adressing parts or elements of the DataArray with [], including slicing as in numpy. This just
		forwards to the underlying __getitem__ method of the data object.
		:param key: The key which is given as adressed in dataarray[key]. Can be an integer or a slice object.
		:return: The sliced data as returned by self.data[key].
		"""
		return self.__class__(self.data[key], label=self.label, plotlabel=self.plotlabel)

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

	def __init__(self, data, unit=None, label=None, plotlabel=None):
		"""
		So far, an Axis is the same as a DaraArray, with the exception that it is one-dimensional. Therefore this
		method uses the __init__ of the parent class and parameters are exactly as there.
		:param data:
		:param unit:
		:param label:
		:param plotlabel:
		:return:
		"""
		DataArray.__init__(self, data, unit=unit, label=label, plotlabel=plotlabel)
		self.assure_1D()
		assert (len(self.data.shape) == 1), "Axis not initialized with 1D array-like object."

	@classmethod
	def from_dataarray(cls, da):
		"""
		Initializes an Axis instance from a DataArray instance (which is mostly, but not completely the same).
		:param da: An instance of the DataArray class.
		:return: The new initialized instance of Axis.
		"""
		return cls(da.data, label=da.label, plotlabel=da.plotlabel)

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.
		:param item: The name to get the corresponding attribute.
		:return: The attribute corresponding to the given name.
		"""
		if item == "shape":
			return self.data.shape
		if item == "units":
			return self.data.units
		raise AttributeError("Attribute \'{0}\' of Axis instance cannot be resolved.".format(item))

	def assure_1D(self):
		"""
		Makes sure the data is onedimensional. Try a consistent conversion, if that fails raise error.
		"""
		if len(self.data.shape) == 1:  # Array itself is 1D
			return
		# Else we have a moredimensional array. Try to flatten it:
		flatarray = self.data.flatten()
		if not (len(flatarray) in self.data.shape):  # an invalid conversion.
			raise ArithmeticError("Non-1D convertable data array in Axis.")
		else:
			self.data = flatarray

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


class ROI():
	"""
	A Region of Interest: This is a way to define a (rectangular) mask in the multidimensional DataSet,
	and then interfacing the generated ROI, as you would the overlying DataSet.
	Consequently, the ROI shall be implemented to behave as a DataSet.
	"""

	def __init__(self, dataset, limitlist=None):
		"""
		The constructor. Translates the given limits to the corresponding array indices and stores them as instance
		variables. Each set of limits must be given as 2-tuples of the form (start,stop), where start and stop can be
		a valid axis array index, a quantity with a value on the axis which will be cast as the next nearest index, or
		None for no limitation.

		:param dataset: The DataSet instance to work on. A ROI only makes sense on a specific dataset...

		:param limitlist: A list or dict containing the limits for each axis. If a dict is given, the keys must be
		valid identifiers of an axis of the DataSet (see DataSet.get_axis()). If a list is given, each entry must
		contain the limit set for one axis in the same order as in the axes list of the DataSet, Limit sets must
		2-tuples as described above.

		:return: The constructed ROI instance.
		"""
		self.dataset = dataset


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
			moep = DataArray(field)
			self.datafields.append(moep)
		# The previous line is in place of the following part that was moved to the DataArray init.
		# if isinstance(field, DataArray):
		# self.datafields.append(field)
		# elif u.is_quantity(field):
		# 	self.datafields.append(DataArray(field))
		# elif type(field) == str:
		# 	self.datafields.append(DataArray(numpy.array(eval(field))))
		# else:
		# 	self.datafields.append(DataArray(numpy.array(field)))

		self.axes = []
		for ax in axes:  # Fill axes list with correctly formatted axes objects.
			moep = Axis(ax)
			self.axes.append(moep)
		# The previous line is in place of the following part that was moved to the DataArray init.
		# if isinstance(ax, Axis):
		# self.axes.append(ax)
		# elif isinstance(ax, DataArray):
		# 	self.axes.append(Axis.from_dataarray(ax))
		# elif u.is_quantity(ax):
		# 	self.axes.append(Axis(ax))
		# elif type(ax) == str:
		# 	self.axes.append(Axis(numpy.array(eval(ax))))
		# else:
		# 	self.axes.append(Axis(numpy.array(ax)))

		self.check_data_consistency()

		if type(plotconf) == str:
			self.plotconf = dict(eval(plotconf))
		else:
			self.plotconf = dict(plotconf)

	@classmethod
	def from_h5file(cls, path):
		"""
		Initializes a new DataSet from an existing HDF5 file. The file must be structured in accordance to the
		saveh5() and loadh5() methods in this class. Uses loadh5 under the hood!
		:param path: The (absolute or relative) path of the HDF5 file to read.
		:return: The initialized DataSet
		"""
		path = os.path.abspath(path)
		# Initalize empty DataSet with the filename as label:
		filename = os.path.basename(path)
		dataset = cls(filename)
		# Load data:
		dataset.loadh5(path)
		return dataset

	@classmethod
	def from_textfile(cls, path, **kwargs):
		"""
		Initializes a new DataSet from an existing text file. The file must contain the data in a column structure
		and can contain additional metadata in comment lines.
		Uses load_textfile() under the hood! See doc of this method for more details!
		:param path: The (absolute or relative) path of the text file to read.
		:param kwargs: Keyword arguments for load_textfile and the underlying numpy.loadtxt(). See there
		documentation for specifics,
		:return: The initialized DataSet
		"""
		path = os.path.abspath(path)
		# Initalize empty DataSet with the filename as label:
		filename = os.path.basename(path)
		dataset = cls(filename)
		# Load data:
		dataset.load_textfile(path, **kwargs)
		return dataset

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.
		:param item: The name to get the corresponding attribute.
		:return: The attribute corresponding to the given name.
		"""
		if item == "alldata":
			return self.datafields + self.axes
		elif item == "labels":
			labels = []
			for e in self.alldata:
				labels.append(e.get_label())
			return labels
		elif item in self.labels:
			for darray in self.alldata:
				if item == darray.get_label():
					return darray
		# TODO: address xyz.
		raise AttributeError("Name \'{0}\' in DataSet object cannot be resolved!".format(item))

	def add_datafield(self, data, unit=None, label=None, plotlabel=None):
		"""
		Initalizes a datafield and adds it to the list. All parameters have to be given like the __init__ of DataSets
		expects them.
		:param data:
		:param unit:
		:param label:
		:param plotlabel:
		:return:
		"""
		self.datafields.append(DataArray(data, unit, label, plotlabel))

	def get_datafield(self, label_or_index):
		"""
		Tries to assign a DataField to a given parameter, that can be an integer as an index in the
		datafields list or a label string. Raises exceptions if there is no matching field.
		:param label_or_index: Identifier of the DataField
		:return: The corresponding DataField.
		"""
		try:
			return self.datafields[label_or_index]
		except TypeError:
			if label_or_index in self.labels:
				return self.__getattr__(label_or_index)
			else:
				raise AttributeError("DataField not found.")

	def get_datafield_index(self, label_or_index):
		try:
			self.datafields[label_or_index]  # If this works it is an int (or int-like) and addressable as it is.
			return label_or_index
		except TypeError:
			if label_or_index in self.labels:
				return self.datafields.index(self.__getattr__(label_or_index))
			else:
				raise AttributeError("DataField not found.")

	def add_axis(self, data, unit=None, label=None, plotlabel=None):
		"""
		Initalizes a datafield and adds it to the list. All parameters have to be given like the __init__ of Axis
		expects them.
		:param data:
		:param unit:
		:param label:
		:param plotlabel:
		:return:
		"""
		self.axes.append(Axis(data, unit, label, plotlabel))

	def get_axis(self, label_or_index):
		"""
		Tries to assign an Axis to a given parameter, that can be an integer as an index in the
		axes list or a label string. Raises exceptions if there is no matching element.
		:param label_or_index: Identifier of the Axis.
		:return: The corresponding Axis.
		"""
		try:
			return self.axes[label_or_index]
		except TypeError:
			if label_or_index in self.labels:
				return self.__getattr__(label_or_index)
			else:
				raise AttributeError("Axis not found.")

	def get_axis_index(self, label_or_index):
		try:
			self.axes[label_or_index]  # If this works it is an int (or int-like) and addressable as it is.
			return label_or_index
		except TypeError:
			if label_or_index in self.labels:
				return self.axes.index(self.__getattr__(label_or_index))
			else:
				raise AttributeError("Axis not found.")

	def meshgrid(self, axes=None):
		"""
		This function returns coordinate arrays corresponding to the axes. See numpy.meshgrid. If axes are not
		constricted by axes argument, the output arrays will have the same shape as the data arrays of the dataset.
		Uses snomtools.calcs.units.meshgrid() under the hood to preserve units.
		:param axes: optional: a list of axes identifiers to be included. If none is given, all axes are included.
		:return: A tuple of coordinate arrays.
		"""
		if axes:
			# Assemble axes:
			list_of_axes = []
			for identifier in axes:
				list_of_axes.append(self.get_axis(identifier))
			# Build grid:
			return u.meshgrid(*list_of_axes)
		else:
			return u.meshgrid(*self.axes)

	def swapaxis(self, axis1, axis2):
		"""
		Interchanges the place of two axes.
		:param axis1: The first axis, addressed by its label or index.
		:param axis2: The second axis, addressed by its label or index.
		:return: Nothing.
		"""
		# Assure numerical indices:
		axis1 = self.get_axis_index(axis1)
		axis2 = self.get_axis_index(axis2)

		# Swap data axes and order of axes list:
		for field in self.datafields:
			field.swapaxes(axis1, axis2)
		self.axes[axis2], self.axes[axis1] = self.axes[axis1], self.axes[axis2]

		# Assure we did nothing wrong:
		self.check_data_consistency()

	def check_data_consistency(self):
		"""
		Self test method which checks the dimensionality and shapes of the axes and datafields. Raises
		AssertionError if one of the tests fails.
		:return: True if the test is successful.
		"""
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
		return True

	def check_label_uniqueness(self, newlabel=None):
		"""
		Checks if the labels of the DataSets datafields and axes are unique. This is important to call them directly
		by their labels.
		:param newlabel: If given, this method checks the viability of a new label to add to the DataSet.
		:return: True if test is successful.
		"""
		if newlabel:  # we check if a new label would be viable:
			return not (newlabel in self.labels)
		else:  # selfcheck without newlabel:
			assert (len(self.labels) == len(set(self.labels))), "DataSet data array and axes labels not unique."
			return True

	def saveh5(self, path):
		path = os.path.abspath(path)
		outfile = h5py.File(path, 'w')
		datafieldgrp = outfile.create_group("datafields")
		for i in range(len(self.datafields)):
			grp = self.datafields[i].store_to_h5(datafieldgrp)
			grp.create_dataset("index", data=i)
		axesgrp = outfile.create_group("axes")
		for i in range(len(self.axes)):
			grp = self.axes[i].store_to_h5(axesgrp)
			grp.create_dataset("index", data=i)
		outfile.create_dataset("label", data=self.label)
		plotconfgrp = outfile.create_group("plotconf")
		h5tools.store_dictionary(self.plotconf, plotconfgrp)
		outfile.close()

	def loadh5(self, path):
		path = os.path.abspath(path)
		infile = h5py.File(path, 'r')
		self.label = str(infile["label"][()])
		datafieldgrp = infile["datafields"]
		self.datafields = [None for i in range(len(datafieldgrp))]
		for datafield in datafieldgrp:
			index = datafieldgrp[datafield]['index'][()]
			self.datafields[index] = (DataArray.from_h5(datafieldgrp[datafield]))
		axesgrp = infile["axes"]
		self.axes = [None for i in range(len(axesgrp))]
		for axis in axesgrp:
			index = axesgrp[axis]['index'][()]
			self.axes[index] = (Axis.from_h5(axesgrp[axis]))
		self.plotconf = h5tools.load_dictionary(infile['plotconf'])
		self.check_data_consistency()

	def load_textfile(self, path, axis=0, comments='#', delimiter=None, unitsplitter="[-\/ ]+", labelline=0,
					  unitsline=0, **kwargs):
		"""
		Loads the contents of a textfile to the dataset instance. The text files are two-dimensional arrays of lines
		and n columns, so it can hold up to one axis and (n-1) or n DataFields. See axis parameter for information on
		how to set axis. Data consistency is checked at the end, so shapes of data and axis arrays must fit (as
		always),
		Uses numpy.loadtxt() under the hood!
		Defaults fit for gnuplot friendly files. Tries to cast the heading comment line(s) as labels and units for
		the data fields.
		:param path: The (relative or absolute) path of the text file to read.
		:param axis: Integer that specifies the column of the axis in the data file OR a whole axis instance of the
		correct shape OR None for unchanged axes of the DataSet. Default is 0 for the first column of the text file.
		:param comments: the character used to indicate the start of a comment line
		:param delimiter: character used to separate values. By default, this is any whitespace.
		:param unitsplitter: Regular expression to split comment columns in labels and units. Default is "[-\/ ]+",
		which matches combinations of the chars '-'. '/' and ' '.
		:param labelline: Index (starting with 0) of the comment line in which the labels are stored. (Default 0)
		:param unitsline: Index (starting with 0) of the comment line in which the units are stored. (Default 0) If
		this is different from labelline, it is assumed that ONLY the unit is in that line.
		:param kwargs: Keyword arguments as used for the numpy.loadtxt() method
		:return: Nothing
		"""
		# Normalize path:
		path = os.path.abspath(path)
		# Load data from text file:
		datacontent = numpy.loadtxt(path, comments=comments, delimiter=delimiter, **kwargs)
		# All columns contain data by default. This can change if there is an Axis:
		datacolumns = range(datacontent.shape[1])

		# Handle comment lines which hold metadata like labels and units of the data columns:
		commentsentries = []  # The list which will hold the strings of the comment lines.
		labels = ["" for i in datacolumns]  # The list which will hold the label for each data column.
		units = [None for i in datacolumns]  # The list which will hold the unit string for each data column.
		textfile = open(path, 'r')
		try:
			for line in textfile:
				line = line.strip()
				if line.startswith(comments):
					commentsentries.append(line.strip(comments).strip().split(delimiter))
		finally:
			textfile.close()

		# Check if relevant comments lines have the correct number of columns:
		lines_not_ok = []
		for comments_line_i in {labelline, unitsline}:
			if len(commentsentries[comments_line_i]) != len(datacolumns):
				lines_not_ok.append(comments_line_i)
		if lines_not_ok:  # The list is not empty.
			print(colored("WARNING: Comment line(s) {0} in textfile {1} has wrong number of columns. "
						  "No metadata can be read.".format(lines_not_ok, path), 'yellow'))
		else:  # There is a corresponding column in the comment line to each data line.
			if labelline == unitsline:  # Labels and units in same line. We need to extract units, rest are labels:
				for column in datacolumns:
					for part in re.split(unitsplitter, commentsentries[unitsline][column]):
						if u.is_valid_unit(part):
							units[column] = part
						else:
							labels[column] += part
			else:  # Two different lines for units and labels.
				for column in datacolumns:
					unit = commentsentries[unitsline][column]
					if u.is_valid_unit(unit):
						units[column] = unit
					else:
						print(colored("WARNING: Invalid unit string '{2}' in unit line {0} in textfile {1}".format(
							unitsline, path, unit), 'yellow'))
					labels[column] = commentsentries[labelline][column]

		# If we should handle axis:
		if not (axis is None):
			if type(axis) == int:  # Column number was given.
				datacolumns.remove(axis)  # Column contains axis and not data.
				self.axes = [Axis(datacontent[:, axis], unit=units[axis], label=labels[axis])]  # Initialize axis
			elif type(axis) == Axis:  # Complete axis was given.
				self.axes = [axis]
			elif type(axis) == DataArray:  # DataArray was given for axis.
				self.axes = [Axis.from_dataarray(axis)]
			else:  # We don't know what was given... Let's try to cast it as an axis.
				try:
					self.axes = [Axis(axis)]
				except Exception as e:
					print colored("ERROR! Axis initialization in load_textfile failed.", "red")
					raise e

		# Write the remaining data to datafields:
		self.datafields = []  # Reset datafields
		for i in datacolumns:  # Initialize new datafields
			self.datafields.append(DataArray(datacontent[:, i], unit=units[i], label=labels[i]))

		self.check_label_uniqueness()
		return self.check_data_consistency()

	def __del__(self):
		pass


if True:  # just for testing
	print colored('Testing...', 'yellow'),
	testarray = numpy.arange(0, 10, 2.)
	testaxis = DataArray(testarray, 'meter', label="xaxis")
	testaxis2 = testaxis / 2.
	testaxis2.set_label("yaxis")
	X, Y = numpy.meshgrid(testaxis, testaxis2)
	# testaxis = DataArray(testarray[testarray<5], 'meter')
	testdata = DataArray(numpy.sin((X + Y) * u.ureg('rad')) * u.ureg('counts'), label="testdaten", plotlabel="pl")
	# print(testdata)
	pc = {'a': 1.0, 'b': "moep", 'c': 3, 'de': "eins/zwo"}
	print(pc)
	testdataset = DataSet("test", [testdata], [testaxis, testaxis2], plotconf=pc)
	print("Store...")
	testdataset.saveh5('test.hdf5')
	unsliced = testdataset.testdaten
	sliced = testdataset.testdaten[0:3:, 0:3:]

	print("Load...")
	newdataset = DataSet.from_h5file('test.hdf5')

	print("Load textfile...")
	newestdataset = DataSet.from_textfile('test2.txt', comments='#', delimiter='\t', unitsline=1)
	print(newestdataset)
	print("Store...")
	newestdataset.saveh5("test2.hdf5")

	print(newestdataset.labels)
	# print newestdataset.get_axis(0)

	cprint("OK", 'green')
