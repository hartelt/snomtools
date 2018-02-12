"""
This file contains the base class for datasets.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.calcs.units as u
import numpy
import os
import h5py
from snomtools.data import h5tools
import re
import tempfile
from six import string_types, integer_types
import scipy.ndimage

__author__ = 'hartelt'


class Data_Handler_H5(u.Quantity):
	"""
	A Data Handler, emulating a Quantity.
	This "H5 mode" handler keeps the data in h5py objects, but provides access to data as if it were a quantity.
	"""

	def __new__(cls, data=None, unit=None, shape=None, h5target=None,
				chunks=True, compression="gzip", compression_opts=4, chunk_cache_mem_size=None):
		"""
		Initializes and returns a new instance. __new__ is used instead of __init__ because pint Quantity does so,
		and the method is overwritten.

		:param data: The data to store.

		:param unit: A valid units string for the unit to convert the data to if necessary.

		:param shape: If no data is given, data of given shape are initialized with zeroes.

		:param h5target: If None or True, temporary file mode is enabled, Data is kept on temp
			files, which are cleaned up in __del__.
		:type h5target: h5py Group/File

		:param chunks: (See h5py docs. Chunks are good in most big data cases!)

		:param compression: (See h5py docs. Compression is good in most cases!)

		:param compression_opts: (See h5py docs. Compression is good in most cases!)

		:param chunk_cache_mem_size: Set custom chunk cache memory size for temp files. Default is set in h5tools.

		:return: The initialized instance.
		"""
		# TODO: Handle Datatypes. Sort compression opts for initializing from existing h5 data.
		if not chunks:
			compression = None
			compression_opts = None
		if h5target is None or h5target is True:
			temp_dir = tempfile.mkdtemp(prefix="snomtools_H5_tempspace-")
			# temp_dir = os.getcwd() # upper line can be replaced by this for debugging.
			temp_file_path = os.path.join(temp_dir, "snomtools_H5_tempspace.hdf5")
			temp_file = h5tools.File(temp_file_path, 'w', chunk_cache_mem_size=chunk_cache_mem_size)
			h5target = temp_file
		else:
			temp_file = None
			temp_dir = None

		if isinstance(data, cls):
			if chunks is True:
				# To improve performance. If the data has a specific chunk size, take that instead of
				# letting h5py guess it.
				chunks = data.chunks
				if compression == 'gzip' and compression_opts == 4:
					# Also, if compression is default, rather use data options.
					compression = data.compression
					compression_opts = data.compression_opts
			if not unit is None and not data.units == u.to_ureg(1, unit).units:
				data = data.to(unit)
			inst = object.__new__(cls)
			inst.__used = False
			inst.__handling = None
			inst.compression = compression
			inst.compression_opts = compression_opts
			if data.h5target is h5target:
				# Data already there.
				pass
			else:
				# We got h5 already, so copying on h5 level is faster because of compression.
				h5tools.clear_name(h5target, "data")
				data.h5target.copy("data", h5target)
				h5tools.clear_name(h5target, "unit")
				data.h5target.copy("unit", h5target)
			inst.ds_data = h5target["data"]
			inst.ds_unit = h5target["unit"]
			inst.h5target = h5target
			inst.temp_file = temp_file
			inst.temp_dir = temp_dir
			return inst
		elif data is not None:
			inst = object.__new__(cls)
			inst.__used = False
			inst.__handling = None
			inst.compression = compression
			inst.compression_opts = compression_opts
			compiled_data = u.to_ureg(data, unit)
			if (not hasattr(compiled_data, 'shape')) or compiled_data.shape == ():
				# Scalar data. No chunking or compression supported.
				chunks = False
				compression = None
				compression_opts = None
			h5tools.clear_name(h5target, "data")
			h5tools.clear_name(h5target, "unit")
			inst.ds_data = h5target.create_dataset("data", data=compiled_data.magnitude, chunks=chunks,
												   compression=compression,
												   compression_opts=compression_opts)
			inst.ds_unit = h5target.create_dataset("unit", data=str(compiled_data.units))
			inst.h5target = h5target
			inst.temp_file = temp_file
			inst.temp_dir = temp_dir
			return inst
		elif shape is not None:
			inst = object.__new__(cls)
			inst.__used = False
			inst.__handling = None
			inst.compression = compression
			inst.compression_opts = compression_opts
			if shape == ():  # Scalar data. No chunking or compression supported.
				chunks = False
				compression = None
				compression_opts = None
			h5tools.clear_name(h5target, "data")
			h5tools.clear_name(h5target, "unit")
			inst.ds_data = h5target.create_dataset("data", shape, chunks=chunks, compression=compression,
												   compression_opts=compression_opts)
			inst.ds_unit = h5target.create_dataset("unit", data=u.normalize_unitstr(unit))
			inst.h5target = h5target
			inst.temp_file = temp_file
			inst.temp_dir = temp_dir
			return inst
		elif not temp_file:
			inst = object.__new__(cls)
			inst.__used = False
			inst.__handling = None
			inst.ds_data = h5target["data"]
			inst.ds_unit = h5target["unit"]
			inst.h5target = h5target
			inst.temp_file = temp_file
			inst.temp_dir = temp_dir
			inst.compression = compression
			inst.compression_opts = compression_opts
			return inst
		else:
			raise ValueError("Initialized Data_Handler_np with wrong parameters.")

	# TODO: overwrite __getattr__ to avoid invoking _magnitude for performance reasons (all data loaded to RAM).

	def _get__magnitude(self):
		if self.ds_data.shape:  # array-like
			return self.ds_data[:]
		else:  # scalar
			return self.ds_data[()]

	def _set__magnitude(self, val):
		if numpy.asarray(val).shape == self.shape:  # Same shape, so just overwrite everything in place.
			if self.shape:  # array-like
				self.ds_data[:] = val
			else:  # scalar
				self.ds_data[()] = val
		else:  # Different shape, so generate new h5 dataset.
			del self.h5target["data"]
			if hasattr(val, '__len__'):  # Sequence... so non-scalar data.
				chunks = self.chunks
				compression = self.compression
				compression_opts = self.compression_opts
			else:  # Scalar data. No chunking or compression supported.
				chunks = False
				compression = None
				compression_opts = None
			self.ds_data = self.h5target.create_dataset("data", data=val,
														chunks=chunks,
														compression=compression,
														compression_opts=compression_opts)

	_magnitude = property(_get__magnitude, _set__magnitude, None, "The _magnitude property for Quantity emulation.")

	def _get__units(self):
		return u.unit_from_str(self.ds_unit[()])._units

	def _set__units(self, val):
		self.ds_unit[()] = str(u.Quantity(1., val).units)

	_units = property(_get__units, _set__units, None, "The _units property for Quantity emulation.")

	@property
	def dimensionality(self):
		return u.to_ureg(1, self.units).dimensionality

	@property
	def q(self):
		"""
		The corresponding quantity.

		:return: The data, converted to a quantity.
		"""
		return u.to_ureg(self.magnitude, self.units)

	@property
	def temp_file_path(self):
		if not self.temp_file is None:
			return os.path.join(self.temp_dir, self.temp_file.filename)
		else:
			return None

	@property
	def shape(self):
		return self.ds_data.shape

	@property
	def dtype(self):
		return self.ds_data.dtype

	@property
	def chunks(self):
		return self.ds_data.chunks

	def __getitem__(self, key):
		return self.__class__(self.ds_data[key], self._units)

	def __setitem__(self, key, value):
		"""
		This method provides write access to indexed elements of the data. It directly writes to the h5 dataset
		without invoking _magnitude, thereby it avoids loading all data into ram.

		:param key: Index or slice (numpy style as usual) of data to address.

		:param value: Data to write in addressed elements. Input units will be converted
			to Data_Handler units (error if not possible). Numeric data (non-Quantities) are assumed as dimensionless (
			pint-style).

		.. warning::
			Value must fit into RAM. Setting bigger-than-RAM slices at a time is not supported (yet).

		:return: Nothing.
		"""
		# The following line could be replaced with
		# value = u.to_ureg(value).to(self.units)
		# without changing any functionality. But calling to_ureg twice is more efficient because unneccesary calling
		#  of value.to(self.units), which always generates a copy is avoided if possible.
		value = u.to_ureg(u.to_ureg(value), self.units)
		self.ds_data[key] = value.magnitude

	def flush(self):
		"""
		Flushes the HDF5 buffer to disk. This always concerns the whole H5 file, so the Data_Handler resides on a
		subgroup, all other datasets on that file are also flushed.
		:return: nothing
		"""
		self.h5target.file.flush()

	def get_unit(self):
		return str(self.units)

	def set_unit(self, unitstr):
		"""
		Set the unit of the Quantity as specified.

		:param unitstr: A valid unit string.

		:return: Nothing.
		"""
		self.ito(unitstr)

	def to(self, unit, *contexts, **ctx_kwargs):
		"""
		A more performant version of pints Quantity's to, that avoids unneccesary calling of magnitude, and copies on
		HDF5 level if possible.

		:param unit: A valid unit string or

		:param contexts: See pint._Quantity.to().

		:param ctx_kwargs: See pint._Quantity.to().

		:return:
		"""
		if self.units == u.to_ureg(1, unit).units:
			return self.__class__(self)
		return super(Data_Handler_H5, self).to(unit, *contexts, **ctx_kwargs)

	def ito(self, unit, *contexts, **ctx_kwargs):
		"""
		A more performant version of pints Quantity's ito, that avoids unneccesary calling of magnitude.

		:param unit: A valid unit string or

		:param contexts: See pint._Quantity.to().

		:param ctx_kwargs: See pint._Quantity.to().

		:return:
		"""
		if u.same_unit(self, u.to_ureg(1, unit)):  # Nothing to do.
			pass
		else:
			super(Data_Handler_H5, self).ito(unit, *contexts, **ctx_kwargs)

	def get_nearest_index(self, value):
		"""
		Get the index of the value in the DataArray nearest to a given value.

		:param value: A value to look for in the array.

		:return: The index tuple of the array entry nearest to the given value.
		"""
		# TODO: Adapt and optimize memory performance.
		value = u.to_ureg(value, unit=self.get_unit())
		idx_flat = (abs(self - value)).argmin()
		idx_tup = numpy.unravel_index(idx_flat, self.shape)
		return idx_tup

	def get_nearest_value(self, value):
		"""
		Like get_nearest_index, but return the value instead of the index.

		:param value: value: A value to look for in the array.

		:return: The value in the array nearest to the given one.
		"""
		return self[self.get_nearest_index(value)]

	def sum_raw(self, axis=None, dtype=None, out=None, keepdims=False):
		"""
		As sum(), only on bare numpy array instead of Quantity. See sum() for details.
		:return: ndarray
		An array with the same shape as a, with the specified axes removed. If a is a 0-d array, or if axis is None, a
		scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		return self.magnitude.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

	def sum(self, axis=None, dtype=None, out=None, keepdims=False, h5target=None):
		"""
		Behaves as the sum() function of a numpy array.
		See: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html

		:param axis: None or int or tuple of ints, optional
			Axis or axes along which a sum is performed. The default (axis = None) is perform a sum over all the dimensions
			of the input array. axis may be negative, in which case it counts from the last to the first axis.
			New in version 1.7.0.:
			If this is a tuple of ints, a sum is performed on multiple axes, instead of a single axis or all the axes as
			before.

		:param dtype: dtype, optional
			The type of the returned array and of the accumulator in which the elements are summed. By default, the dtype
			of a is used. An exception is when a has an integer type with less precision than the default platform integer.
			In that case, the default platform integer is used instead.

		:param out: ndarray, optional
			Array into which the output is placed. By default, a new array is created. If out is given, it must be of the
			appropriate shape (the shape of a with axis removed, i.e., numpy.delete(a.shape, axis)). Its type is preserved.
			See doc.ufuncs (Section Output arguments) for more details.

		:param keepdims: bool, optional
			If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
			option, the result will broadcast correctly against the original arr.

		:return: ndarray Quantity
			An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a
			scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		# TODO: Handle datatypes.
		inshape = self.shape
		if axis is None:
			axis = tuple(range(len(inshape)))
		try:
			if len(axis) == 1:  # If we have a sequence of len 1, we sum over only 1 axis.
				axis = axis[0]
				single_axis_flag = True
			else:
				single_axis_flag = False
		except TypeError:
			# axis has no len, so it is propably an integer already. Just go on...
			single_axis_flag = True

		if single_axis_flag:  # Only one axis to sum over.
			if keepdims:
				outshape = list(inshape)
				outshape[axis] = 1
				outshape = tuple(outshape)
			else:
				outshape = tuple(numpy.delete(inshape, axis))
			if out:
				assert out.shape == outshape, "Wrong shape of given destination."
				outdata = out
			else:
				outdata = self.__class__(shape=outshape, unit=self.get_unit(), h5target=h5target)
			for i in range(inshape[axis]):
				slicebase = [numpy.s_[:] for j in range(len(inshape) - 1)]
				slicebase.insert(axis, i)
				if outdata.shape == ():  # Scalar
					outdata.ds_data[()] += self.ds_data[tuple(slicebase)]
				else:
					outdata.ds_data[:] += self.ds_data[tuple(slicebase)]
			return outdata
		else:  # We still have a list or tuple of several axes to sum over.
			axis = numpy.array(sorted(axis))
			axisnow = axis[0]
			if keepdims:  # Axes positions stay as they are. Prepare sum tuple for rest of summations.
				axisrest = tuple(axis[1:])
			else:  # Sum erases axis number axis[0], rest of axis ids to sum over is shifted by -1
				axisrest = tuple(axis[1:] - 1)
			# Perform summation over axisnow and recursively sum over rest:
			return self.sum(axisnow, dtype, out, keepdims, h5target=None).sum(axisrest, dtype, out, keepdims, h5target)

	def absmax(self):
		return abs(self).max()

	def absmin(self):
		return abs(self).min()

	def shift(self, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None, h5target=None):
		"""
		Shifts the complete data with scipy.ndimage.interpolation.shift.
		In difference to that method, it does not need an input, but works on the instance data.
		Also, the defaults are different.

		.. warning::
			For shifting, the whole data is loaded into RAM. Sequential shifting might be implemented later.

		See: :func:`scipy.ndimage.interpolation.shift` for full documentation of parameters.

		:param output: The array in which to place the output, or the dtype of the returned array.
			If :code:`False` is given, the instance data is overwritten.
		:type output: ndarray *or* dtype *or* :code:`False`, *optional*

		:param h5target: The h5target to in case a new Data_Handler_H5 is generated.

		:returns: The shifted data. If output is given as a parameter or :code:`False`, None is returned.
		:rtype: Data_Handler_np *or* None
		"""
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False

		if output == False:
			self._magnitude = scipy.ndimage.interpolation.shift(self.magnitude, shift, None, order, mode, cval,
																prefilter)
			return None
		elif isinstance(output, numpy.ndarray):
			scipy.ndimage.interpolation.shift(self.magnitude, shift, output, order, mode, cval, prefilter)
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_H5(scipy.ndimage.interpolation.shift(self.magnitude, shift, output, order, mode, cval,
																	 prefilter),
								   self.units, h5target=h5target)

	def shift_slice(self, slice_, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None,
					h5target=None):
		"""
		Shifts a certain slice of the data with scipy.ndimage.interpolation.shift.
		See: :func:`scipy.ndimage.interpolation.shift` for full documentation of parameters.

		:param slice_: A selection of a subset of the data, typically a tuple of ints or slices. Can be generated
			easily with	:func:`numpy.s_` or builtin method :func:`slice`.
		:type slice_: slice **or** tuple(slice) **or** *(tuple of) castable*.

		:param output: The array in which to place the output, or the dtype of the returned array.
			If :code:`False` is given, the slice of the instance data is overwritten.
		:type output: ndarray *or* dtype *or* :code:`False`, *optional*

		:param h5target: The h5target to in case a new Data_Handler_H5 is generated.

		:returns: The shifted data. If output is given as a parameter or :code:`False`, None is returned.
		:rtype: Data_Handler_np *or* None
		"""
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False

		if output == False:
			self.ds_data[slice_] = scipy.ndimage.interpolation.shift(self.ds_data[slice_], shift, None, order, mode,
																	 cval, prefilter)
			return None
		elif isinstance(output, numpy.ndarray):
			scipy.ndimage.interpolation.shift(self.ds_data[slice_], shift, output, order, mode, cval, prefilter)
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_H5(
				scipy.ndimage.interpolation.shift(self.ds_data[slice_], shift, output, order, mode, cval,
												  prefilter),
				self.units, h5target=h5target)

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return super(Data_Handler_H5, self).__add__(other)

	def __sub__(self, other):
		other = u.to_ureg(other, self.units)
		return super(Data_Handler_H5, self).__sub__(other)

	def __mul__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_H5, self).__mul__(other)

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		return super(Data_Handler_H5, self).__truediv__(other)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_H5, self).__floordiv__(other)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return super(Data_Handler_H5, self).__pow__(other)

	def __array__(self):
		return self.magnitude

	def __repr__(self):
		return "<Data_Handler_H5 on {0} with shape {1}>".format(repr(self.h5target), self.shape)

	def __del__(self):
		if not (self.temp_file is None):
			file_to_remove = self.temp_file_path
			self.temp_file.close()
			os.remove(file_to_remove)
			try:
				os.rmdir(self.temp_dir)
			except OSError as e:
				print("WARNING: Data_Handler_H5 could not remove tempdir. Propably not empty.")
				print(e)

	@classmethod
	def stack(cls, tostack, axis=0, unit=None, h5target=None):
		"""
		Stacks a sequence of given Data_Handlers (or castables) along a new axis.

		:param tostack: Sequence of Data_Handlers (or castables), each must be of the same shape and unit.

		:param axis: int, optional: The axis in the result array along which the input arrays are stacked.

		:param unit: optional: A valid unit string to convert the stack to. Default is the unit of the first element
			of the sequence.

		:param h5target: optional: A h5target to work on. See __new__

		:return: stacked Data_Handler
		"""
		if unit is None:
			unit = str(tostack[0].units)
		inshape = tostack[0].shape
		for e in tostack:
			assert e.shape == inshape, "Data_Handler_H5.stack got elements of varying shape."
		shapelist = list(inshape)
		shapelist.insert(axis, len(tostack))
		outshape = tuple(shapelist)
		inst = cls(shape=outshape, unit=unit, h5target=h5target)
		for i in range(len(tostack)):
			slicebase = [numpy.s_[:] for j in range(len(inshape))]
			slicebase.insert(axis, i)
			inst[tuple(slicebase)] = tostack[i]
		return inst


class Data_Handler_np(u.Quantity):
	"""
	A Data Handler, emulating a Quantity.
	This "numpy mode" handler actually just mainly shadows the attributes of the Quantity that is held and should be
	use as such a Quantity. Of cause this Quantity itself shadows most of the attributes of the underlying numpy
	array and can therefore be used as one in many contexts.
	"""

	def __new__(cls, data=None, unit=None, shape=None):
		if data is not None:
			compiled_data = u.to_ureg(data, unit)
			return super(Data_Handler_np, cls).__new__(cls, compiled_data.magnitude, compiled_data.units)
		elif shape is not None:
			return cls.__new__(cls, numpy.zeros(shape=shape), unit)
		else:
			raise ValueError("Initialized Data_Handler_np with wrong parameters.")

	def __setitem__(self, key, value):
		if not isinstance(value, self.__class__):
			value = self.__class__(value)
		super(Data_Handler_np, self).__setitem__(key, value)

	def get_unit(self):
		return str(self.units)

	def set_unit(self, unitstr):
		"""
		Set the unit of the Quantity as specified.

		:param unitstr: A valid unit string.

		:return: Nothing.
		"""
		self.ito(unitstr)

	@property
	def q(self):
		"""
		The corresponding quantity.

		:return: The data, converted to a quantity.
		"""
		return u.to_ureg(self.magnitude, self.units)

	def get_nearest_index(self, value):
		"""
		Get the index of the value in the DataArray nearest to a given value.

		:param value: A value to look for in the array.

		:return: The index tuple of the array entry nearest to the given value.
		"""
		value = u.to_ureg(value, unit=self.get_unit())
		idx_flat = (numpy.abs(self - value)).argmin()
		idx_tup = numpy.unravel_index(idx_flat, self.shape)
		return idx_tup

	def get_nearest_value(self, value):
		"""
		Like get_nearest_index, but return the value instead of the index.

		:param value: value: A value to look for in the array.

		:return: The value in the array nearest to the given one.
		"""
		return self[self.get_nearest_index(value)]

	def sum_raw(self, axis=None, dtype=None, out=None, keepdims=False):
		"""
		As sum(), only on bare numpy array instead of Quantity. See sum() for details.
		:return: ndarray
		An array with the same shape as a, with the specified axes removed. If a is a 0-d array, or if axis is None, a
		scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		return self.magnitude.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

	def absmax(self):
		return abs(self).max()

	def absmin(self):
		return abs(self).min()

	def shift(self, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None):
		"""
		Shifts the complete data with scipy.ndimage.interpolation.shift.
		In difference to that method, it does not need an input, but works on the instance data.
		Also, the defaults are different.

		See: :func:`scipy.ndimage.interpolation.shift` for full documentation of parameters.

		:param output: The array in which to place the output, or the dtype of the returned array.
			If :code:`False` is given, the instance data is overwritten.
		:type output: ndarray *or* dtype *or* :code:`False`, *optional*

		:returns: The shifted data. If output is given as a parameter or :code:`False`, None is returned.
		:rtype: Data_Handler_np *or* None
		"""
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False

		if output == False:
			self.magnitude = scipy.ndimage.interpolation.shift(self.magnitude, shift, None, order, mode, cval,
															   prefilter)
			return None
		elif isinstance(output, numpy.ndarray):
			scipy.ndimage.interpolation.shift(self.magnitude, shift, output, order, mode, cval, prefilter)
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_np(scipy.ndimage.interpolation.shift(self.magnitude, shift, output, order, mode, cval,
																	 prefilter),
								   self.units)

	def shift_slice(self, slice_, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None):
		"""
		Shifts a certain slice of the data with scipy.ndimage.interpolation.shift.
		See: :func:`scipy.ndimage.interpolation.shift` for full documentation of parameters.

		:param slice_: A selection of a subset of the data, typically a tuple of ints or slices. Can be generated
			easily with	:func:`numpy.s_` or builtin method :func:`slice`.
		:type slice_: slice **or** tuple(slice) **or** *(tuple of) castable*.

		:param output: The array in which to place the output, or the dtype of the returned array.
			If :code:`False` is given, the slice of the instance data is overwritten.
		:type output: ndarray *or* dtype *or* :code:`False`, *optional*

		:returns: The shifted data. If output is given as a parameter or :code:`False`, None is returned.
		:rtype: Data_Handler_np *or* None
		"""
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False

		if output == False:
			self.magnitude[slice_] = scipy.ndimage.interpolation.shift(self.magnitude[slice_], shift, None, order, mode,
																	   cval, prefilter)
			return None
		elif isinstance(output, numpy.ndarray):
			scipy.ndimage.interpolation.shift(self.magnitude[slice_], shift, output, order, mode, cval, prefilter)
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_np(
				scipy.ndimage.interpolation.shift(self.magnitude[slice_], shift, output, order, mode, cval,
												  prefilter),
				self.units)

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return super(Data_Handler_np, self).__add__(other)

	def __sub__(self, other):
		other = u.to_ureg(other, self.units)
		return super(Data_Handler_np, self).__sub__(other)

	def __mul__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__mul__(other)

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__truediv__(other)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__floordiv__(other)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return super(Data_Handler_np, self).__pow__(other)

	def __repr__(self):
		return "<Data_Handler_np(" + super(Data_Handler_np, self).__repr__() + ")>"

	def __del__(self):
		pass

	@classmethod
	def stack(cls, tostack, axis=0, unit=None):
		"""
		Stacks a sequence of given Data_Handlers (or castables) along a new axis.

		:param tostack: Sequence of Data_Handlers (or castables), each must be of the same shape and dimensionality.

		:param axis: int, optional: The axis in the result array along which the input arrays are stacked.

		:param unit: optional: A valid unit string to convert the stack to. Default is the unit of the first element
			of the sequence.

		:return: stacked Data_Handler
		"""
		if unit is None:
			unit = str(tostack[0].units)
		return cls(numpy.stack(u.magnitudes(u.as_ureg_quantities(tostack, unit)), axis), unit)


class DataArray(object):
	"""
	A data array that holds additional metadata.
	"""

	def __init__(self, data, unit=None, label=None, plotlabel=None, h5target=None,
				 chunks=True, compression="gzip", compression_opts=4, chunk_cache_mem_size=None):
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
		self.chunks = chunks
		self.compression = compression
		self.compression_opts = compression_opts
		self.chunk_cache_mem_size = chunk_cache_mem_size
		if isinstance(h5target, h5py.Group):
			self.h5target = h5target
			self.own_h5file = False
		elif isinstance(h5target, string_types):
			self.h5target = h5tools.File(h5target, chunk_cache_mem_size=self.chunk_cache_mem_size)
			self.own_h5file = True
		elif h5target:  # True but no designated target means temp file mode.
			self.h5target = True
			self.own_h5file = False
		else:  # Numpy mode.
			self.h5target = None
			self.own_h5file = False

		if isinstance(data, DataArray):  # If the data already comes in a DataArray, just take it.
			if chunks == True:  # Instead of defaults, take data options
				self.chunks = data.chunks
			if compression == 'gzip' and compression_opts == 4:  # Instead of defaults, take data options
				self.compression = data.compression
				self.compression_opts = data.compression_opts
			self.data = data.get_data()
			if unit:  # If a unit is explicitly requested anyway, make sure we set it.
				self.set_unit(unit)
			if label:  # Same with label, else take the one that's already with the data
				self.label = label
			else:
				self.label = data.get_label()
			if plotlabel:  # Same as with label.
				self.plotlabel = plotlabel
			else:
				self.plotlabel = data.get_plotlabel()
			# A DataArray contains everything we need, so we should be done here!
		elif isinstance(data, h5py.Group):  # If a HDF5 Group was given, load data directly.
			self.load_from_h5(data)
		else:  # We DON'T have everything contained in data, so we need to process it seperately.
			if data is None:
				self._data = None  # No data. Initialize empty instance.
			elif u.is_quantity(data):  # Kind of the same as above, just for the data itself.
				self.data = data
				if unit:  # If a unit is explicitly requested anyway, make sure we set it.
					self.set_unit(unit)
			elif type(data) == str:  # If it's a string, try to evaluate it as an array.
				self.data = u.to_ureg(numpy.array(eval(data)), unit)
			elif type(data) == numpy.ndarray:  # If it is an ndarray, just make a quantity with it.
				self.data = u.to_ureg(data, unit)
			else:  # If it's none of the above, it hopefully is an array-like. So let's try to cast it.
				self.data = u.to_ureg(numpy.array(data), unit)
			self.label = str(label)
			self.plotlabel = str(plotlabel)

	@classmethod
	def from_h5(cls, h5source, h5target=None):
		"""
		This method initializes a DataArray from a HDF5 source.

		:param h5source: The HDF5 source to read from. This is generally the subgroup for the DataArray.

		:param h5target: Optional. The HDF5 target to work on, if on-disk h5 mode is desired.

		:return: The initialized DataArray.
		"""
		assert isinstance(h5source, h5py.Group), "DataArray.from_h5 requires h5py group as source."
		if h5target:
			assert isinstance(h5target, h5py.Group), "DataArray.from_h5 requires h5py group as target."
		out = cls(None, h5target=h5target)
		out.load_from_h5(h5source)
		return out

	@classmethod
	def in_h5(cls, h5source):
		"""
		This method initializes a DataArray from a HDF5 source, which then works on the same H5 File. This is
		identical to calling DataArray.from_h5(h5source, h5target=h5source), which is actually done here.

		:param h5source: The HDF5 source to read from. This is generally the subgroup for the DataArray.

		:return: The initialized DataArray.
		"""
		return cls.from_h5(h5source, h5source)

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

		:param item: The name to get the corresponding attribute.

		:return: The attribute corresponding to the given name.
		"""
		raise AttributeError("Attribute \'{0}\' of DataArray instance cannot be resolved.".format(item))

	def get_data(self):
		# print "data property getter"
		return self._data

	def set_data(self, newdata, unit=None):
		self.data = u.to_ureg(newdata, unit)

	def _set_data(self, val):
		# print "data property setter"
		if self.h5target:
			if isinstance(self.h5target, h5py.Group):  # initialize H5 data in h5target group
				self._data = Data_Handler_H5(val, h5target=self.h5target, chunks=self.chunks,
											 compression=self.compression, compression_opts=self.compression_opts)
			else:  # no group given but h5target==True, so work in h5 tempfile mode.
				self._data = Data_Handler_H5(val, chunks=self.chunks,
											 compression=self.compression, compression_opts=self.compression_opts,
											 chunk_cache_mem_size=self.chunk_cache_mem_size)
		else:
			self._data = Data_Handler_np(val)

	def del_data(self):
		print('WARNING: Trying to delete data from DataArray.')

	data = property(get_data, _set_data, del_data, "The data property for the DataArray.")

	@property
	def rawdata(self):
		"""
		Get the data as a raw numpy array, not a quantity. Forwards to get_data_raw().

		:return: ndarray: The data as a numpy array.
		"""
		return self.get_data_raw()

	@property
	def shape(self):
		return self._data.shape

	@property
	def units(self):
		return self._data.units

	def get_data_raw(self):
		"""
		Get the data as a raw numpy array, not a quantity.

		:return: ndarray: The data as a numpy array.
		"""
		return self.data.magnitude

	def swapaxes(self, axis1, axis2):
		"""
		Swaps two axes of the data. Uses numpy.swapaxes.

		:param axis1: int: First axis.

		:param axis2: int: Second axis.

		:return: The data after the transformation.
		"""
		self.data = u.to_ureg(self.get_data_raw().swapaxes(axis1, axis2), self.get_unit())
		# TODO: This is not feasible for big data in H5 handlers. Look for more efficient implementation.
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
		self.data.set_unit(unitstr)

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

	def store_to_h5(self, h5dest, subgrp_name=None, chunks=True, compression="gzip", compression_opts=4):
		"""
		Stores the DataArray into a HDF5 file. It will create a new subgroup and store the data there in a unified
		format.

		:param h5dest: The destination. This is a HDF5 file or subgroup.

		:param subgrp_name: Optional. The name for the subgroup that is created to store the data in, Default is the
			label of the DataArray.

		:return The subgroup that was created and holds the data.
		"""
		if subgrp_name is None:
			subgrp_name = self.label
		grp = h5dest.require_group(subgrp_name)
		self.write_to_h5(grp, chunks=chunks, compression=compression, compression_opts=compression_opts)
		return grp

	def write_to_h5(self, h5dest=None, chunks=True, compression="gzip", compression_opts=4):
		"""
		Writes the DataArray into a HDF5 group. It will store the data there in a unified
		format. It will overwrite any dataset in the given group that is named with any of the unified names.

		:param h5dest: The destination. This is a HDF5 file or subgroup. Can be None if the DataSet has a h5target (
			works on a h5 group in file mode), in that case, self.h5target will be taken.

		:return The h5 group (or file) that holds the written data.
		"""
		if h5dest is None and self.h5target:
			h5dest = self.h5target
		if not chunks:
			compression = None
			compression_opts = None

		if h5dest == self.h5target:
			self._data.flush()
		else:
			if self.h5target:
				# We are in h5 mode, so copying on h5 level is faster because of compression.
				h5tools.clear_name(h5dest, "data")
				self.data.h5target.copy(self.data.ds_data.name, h5dest)
			else:
				h5tools.write_dataset(h5dest, "data", data=self.get_data_raw(), chunks=chunks, compression=compression,
									  compression_opts=compression_opts)
			h5tools.write_dataset(h5dest, "unit", self.get_unit())
		h5tools.write_dataset(h5dest, "label", self.get_label())
		h5tools.write_dataset(h5dest, "plotlabel", self.get_plotlabel())
		return h5dest

	def load_from_h5(self, h5source):
		"""
		Loads the data from a HDF5 source.

		:param h5source: The source to read from. This is the subgroup of the DataArray.
		"""
		if self.h5target == h5source:  # We already work on the h5source. Just initialize handler.
			self._data = Data_Handler_H5(h5target=self.h5target)
		elif isinstance(self.h5target, h5py.Group):
			# We work on a h5target, but not h5source. Copy h5source and initialize handler. This should be much more
			# performant than reading data and storing them again, because of compression.
			for h5set in h5source.keys():
				h5source.copy(h5set, self.h5target)
			self._data = Data_Handler_H5(h5target=self.h5target)
		else:
			self.set_data(numpy.array(h5source["data"]), h5source["unit"][()])
		self.set_label(h5source["label"][()])
		self.set_plotlabel(h5source["plotlabel"][()])

	def flush(self):
		"""
		Flushes HDF5 buffers to disk. This only makes sense in h5 disk mode and in non-tempfile mode.

		:return: Nothing.
		"""
		if isinstance(self.h5target, h5py.Group):
			self.write_to_h5()
		else:
			print("WARNING: DataSet cannot flush without working on valid HDF5 file.")

	def get_nearest_index(self, value):
		"""
		Get the index of the value in the DataArray nearest to a given value.

		:param value: A value to look for in the array.

		:return: The index tuple of the array entry nearest to the given value.
		"""
		return self.data.get_nearest_index(value)

	def get_nearest_value(self, value):
		"""
		Like get_nearest_index, but return the value instead of the index.

		:param value: value: A value to look for in the array.

		:return: The value in the array nearest to the given one.
		"""
		return self.data.get_nearest_value(value)

	def sum(self, axis=None, dtype=None, out=None, keepdims=False):
		"""
		Behaves as the sum() function of a numpy array.
		See: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html

		:param axis: None or int or tuple of ints, optional
			Axis or axes along which a sum is performed. The default (axis = None) is perform a sum over all the dimensions
			of the input array. axis may be negative, in which case it counts from the last to the first axis.
			New in version 1.7.0.:
			If this is a tuple of ints, a sum is performed on multiple axes, instead of a single axis or all the axes as
			before.

		:param dtype: dtype, optional
			The type of the returned array and of the accumulator in which the elements are summed. By default, the dtype
			of a is used. An exception is when a has an integer type with less precision than the default platform integer.
			In that case, the default platform integer is used instead.

		:param out: ndarray, optional
			Array into which the output is placed. By default, a new array is created. If out is given, it must be of the
			appropriate shape (the shape of a with axis removed, i.e., numpy.delete(a.shape, axis)). Its type is preserved.
			See doc.ufuncs (Section Output arguments) for more details.

		:param keepdims: bool, optional
			If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
			option, the result will broadcast correctly against the original arr.

		:return: ndarray Quantity
			An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a
			scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		return self.data.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

	def sum_raw(self, axis=None, dtype=None, out=None, keepdims=False):
		"""
		As sum(), only on bare numpy array instead of Quantity. See sum() for details.
		:return: ndarray
		An array with the same shape as a, with the specified axes removed. If a is a 0-d array, or if axis is None, a
		scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		return self.data.sum_raw(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

	def project_nd(self, *args):
		"""
		Projects the datafield onto the given axes. Uses sum() method, but adresses axes to keep instead of axes to
		sum over.

		:param args: Integer indices for the axes to project onto.

		:return: ndarray quantity: The projected data.
		"""
		sumlist = list(range(len(self.shape)))  # initialize list of axes to sum over
		for arg in args:
			assert (type(arg) == int), "ERROR: Invalid type. Axis index must be integer."
			sumlist.remove(arg)
		sumtup = tuple(sumlist)
		if len(sumtup):
			return self.sum(sumtup)
		else:
			return self

	def shift(self, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None):
		"""
		Shifts the complete data. This is just calling the underlying methods implemented in
			:func:`Data_Handler_H5.shift` and :func:`Data_Handler_np.shift`
			See those methods for documentation or under-the-hood-used :func:`scipy.ndimage.interpolation.shift` for
			full documentation of parameters.
		"""
		return self.data.shift(shift, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

	def shift_slice(self, slice_, shift, output=None, order=0, mode='constant', cval=numpy.nan, prefilter=None):
		"""
		Shifts a certain slice of the data. This is just calling the underlying methods implemented in
			:func:`Data_Handler_H5.shift_slice` and :func:`Data_Handler_np.shift_slice`
			See those methods for documentation or under-the-hood-used :func:`scipy.ndimage.interpolation.shift` for
			full documentation of parameters.
		"""
		return self.data.shift_slice(slice_, shift, output=output, order=order, mode=mode, cval=cval,
									 prefilter=prefilter)

	def max(self):
		return self.data.max()

	def min(self):
		return self.data.min()

	def absmax(self):
		return self.data.absmax()

	def absmin(self):
		return self.data.absmin()

	def mean(self):
		return self.data.mean()

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

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		return self.__class__(self.data / other, label=self.label, plotlabel=self.plotlabel)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return self.__class__(self.data // other, label=self.label, plotlabel=self.plotlabel)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return self.__class__(self.data ** other, label=self.label, plotlabel=self.plotlabel)

	def __array__(self):  # to numpy array
		return self.data.magnitude

	def __iter__(self):
		return iter(self.data)

	def __getitem__(self, key):
		"""
		To allow adressing parts or elements of the DataArray with [], including slicing as in numpy. This just
		forwards to the underlying __getitem__ method of the data object.

		:param key: The key which is given as adressed in dataarray[key]. Can be an integer or a slice object.

		:return: The sliced data as returned by self.data[key].
		"""
		# if an int is used as an index in a 1D dataarray, a single element is adressed:
		if isinstance(key, int) and len(self.data.shape) == 1:
			return self.data[key]
		# if tuple of ints with length of the number of dimensions of the data is given, we have a single element again:
		elif type(key) == tuple and len(key) == len(self.data.shape):
			if all(isinstance(key[i], int) for i in range(len(key))):
				assert self.data[key].shape == ()  # a single element must have shape () because it is 0D
				return self.data[key]
		# else a part of the array is adressed (even if all dimensions of len 1, this will still be an dataarray...
		return self.__class__(self.data[key], label=self.label, plotlabel=self.plotlabel)

	def __setitem__(self, key, value):
		"""
		To allow adressing parts or elements of the DataArray with [], including slicing as in numpy. This just
		forwards to the underlying __setitem__ method of the data object.

		:param key: the key in the []

		:param value: the value to set it to

		:return:
		"""
		self.data[key] = value

	def __len__(self):  # len of data array
		if self.data is not None:
			return len(self.data)
		else:
			return None

	def __str__(self):
		out = "DataArray"
		if self.label:
			out += ": " + self.label
		out += " with shape " + str(self.data.shape)
		# out += '\n' + str(self.data)
		return out

	def __repr__(self):
		out = 'DataArray('
		out += repr(self.data)
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

	def __del__(self):
		if self.own_h5file:
			self.h5target.close()
		elif self.h5target is True:  # Temp file mode
			del self._data

	@classmethod
	def stack(cls, datastack, axis=0, unit=None, label=None, plotlabel=None, h5target=None):
		"""
		Stacks a sequence of DataArrays to a new DataArray.
		See numpy.stack, as this method is used.

		:param datastack: sequence of DataArrays: The Data to be stacked.

		:param axis: int, optional: The axis in the result array along which the input arrays are stacked.

		:param unit: string, optional: The unit for the stacked DataArray. All data must be convertible to that unit. If
			not given, the unit of the first DataArray in the input stack is used.

		:param label: string, optional: The label for the new DataSet. If not given, the label of the first DataArray in
			the input stack is used.

		:param plotlabel: string, optional: The plotlabel for the new DataArray. If not given, the label of the first
			DataArray in the input stack is used.

		:return: The stacked DataArray.
		"""
		for da in datastack:
			assert (isinstance(da, DataArray)), "ERROR: Non-DataArray object given to stack_DataArrays"
		if unit is None:
			unit = datastack[0].get_unit()
		if label is None:
			label = datastack[0].get_label()
		if plotlabel is None:
			plotlabel = datastack[0].get_plotlabel()
		onlydata = [da.get_data() for da in datastack]
		if h5target:
			stacked_data = Data_Handler_H5.stack(onlydata, axis, h5target=h5target)
		else:
			stacked_data = Data_Handler_np.stack(onlydata, axis)
		return cls(stacked_data, unit=unit, label=label, plotlabel=plotlabel, h5target=h5target)


class Axis(DataArray):
	"""
	An axis is a data array that holds the data for an axis of a dataset.
	"""

	def __init__(self, data, unit=None, label=None, plotlabel=None, h5target=None):
		"""
		So far, an Axis is the same as a DaraArray, with the exception that it is one-dimensional. Therefore this
		method uses the __init__ of the parent class and parameters are exactly as there.

		:param data:

		:param unit:

		:param label:

		:param plotlabel:

		:return:
		"""
		DataArray.__init__(self, data, unit=unit, label=label, plotlabel=plotlabel, h5target=h5target)
		if not data is None:  # check if data makes sense if instance is not empty.
			self.assure_1D()
			assert (len(self.data.shape) == 1), "Axis not initialized with 1D array-like object."

	@classmethod
	def from_dataarray(cls, da):
		"""
		Initializes an Axis instance from a DataArray instance (which is mostly, but not completely the same).

		:param da: An instance of the DataArray class.

		:return: The new initialized instance of Axis.
		"""
		return cls(da.data, label=da.label, plotlabel=da.plotlabel, h5target=da.h5target)

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

		:param item: The name to get the corresponding attribute.

		:return: The attribute corresponding to the given name.
		"""
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

	def get_index_searchsort(self, values):
		"""
		Assuming the axis elements are sorted (which should be the case, the way all is implemented atm), for every
		value given, there is a corresponding index in the array, where the value could be inserted while maintaining
		order of the array.
		This function searches these indexes by using the numpy.searchsorted function.

		:param values: array_like: Values to insert into the axis.

		:return: array of ints: The corresponding indexes with the same shape as values.
		"""
		return numpy.searchsorted(self.data, values)

	def scale_linear(self, scaling=1., offset=None, unit=None, label=None, plotlabel=None):
		"""
		Transforms the Axis by scaling it with a linear factor and optionally shifting it by an offset.
		:param scaling: The scaling factor.
		:param offset: The offset. Must have the same dimension as the scaled axis.
		:param unit: Specifies the output unit for the Axis, Must evaluate a unit with same dimension as the scaled axis.
		:return: The transformed Axis.
		"""
		points_scaled = scaling * self.data
		if offset:
			points_scaled = points_scaled + offset
		if unit:
			points_scaled = u.to_ureg(points_scaled, unit)
		self.data = points_scaled

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


class ROI(object):
	"""
	A Region of Interest: This is a way to define a (rectangular) mask in the multidimensional DataSet,
	and then interfacing the generated ROI, as you would the overlying DataSet.
	Consequently, the ROI shall be implemented to behave as a DataSet.
	"""

	def __init__(self, dataset, limitlist=None, by_index=False, label="", plotlabel=""):
		"""
		The constructor. Translates the given limits to the corresponding array indices and stores them as instance
		variables. Each set of limits must be given as 2-tuples of the form (start,stop), where start and stop can be a
		quantity with a value on the axis which will be cast as the next nearest index, a valid axis array index
		(see by_index), or None for no limitation.

		:param dataset: The DataSet instance to work on. A ROI only makes sense on a specific dataset...

		:param limitlist: A list or dict containing the limits for each axis. If a dict is given, the keys must be
			valid identifiers of an axis of the DataSet (see DataSet.get_axis()). If a list is given, each entry must
			contain the limit set for one axis in the same order as in the axes list of the DataSet, Limit sets must
			2-tuples as described above.

		:param by_index: Bool flag to interpret limits not as positions, but as array indices.

		:param label: String: A label describing the ROI.

		:param plotlabel: String: A label describing the ROI for plots.

		:return:
		"""
		self.dataset = dataset
		self.label = label
		self.plotlabel = plotlabel
		# create empty limit list:
		self.limits = [[None, None] for i in range(len(self.dataset.axes))]
		# iterate over keys in the limit list if given:
		if limitlist:
			self.set_limits_all(limitlist, by_index)

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

		:param item: The name to get the corresponding attribute.

		:return: The attribute corresponding to the given name.
		"""
		if item == "alldata":
			return self.dataset.datafields + self.dataset.axes
		elif item == "axlabels":
			labels = []
			for e in self.dataset.axes:
				labels.append(e.get_label())
			return labels
		elif item == "dlabels":
			labels = []
			for e in self.dataset.datafields:
				labels.append(e.get_label())
			return labels
		elif item == "labels":
			return self.dlabels + self.axlabels
		elif item == "dimensions":
			return len(self.dataset.axes)
		elif item == "shape":
			return self.get_datafield(0).shape
		elif item in self.labels:
			for darray in self.alldata:
				if item == darray.get_label():
					lim = self.get_slice(item)
					return darray[lim]
		# TODO: address xyz.
		raise AttributeError("Name \'{0}\' in ROI object cannot be resolved!".format(item))

	def set_limits_all(self, limitlist, by_index=False):
		"""
		Each set of limits must be given as 2-tuples of the form (start,stop), where start and stop can be a quantity
		with a value on the axis which will be cast as the next nearest index, a valid axis array index (see by_index),
		or None for an unchanged limit.

		:param limitlist: A list or dict containing the limits for each axis. If a dict is given, the keys must be
			valid identifiers of an axis of the DataSet (see DataSet.get_axis()). If a list is given, each entry must
			contain the limit set for one axis in the same order as in the axes list of the DataSet, Limit sets must
			2-tuples as described above.

		:param by_index: Bool flag to interpret limits not as positions, but as array indices.

		:return: Nothing
		"""
		if type(limitlist) == list:
			for i, lims in enumerate(limitlist):
				self.set_limits(i, lims, by_index=by_index)
		elif type(limitlist) == dict:
			for key in limitlist:
				# key is a dict key or the correct index, so get_axis_index works...
				keyindex = self.dataset.get_axis_index(key)
				# get the corresponding axis:
				ax = self.dataset.axes[keyindex]
				# get the axis index corresponding to the limit and store it:
				if limitlist[key][0]:
					if by_index:
						left_limit_index = limitlist[key][0]
					else:
						left_limit_index = ax.get_nearest_index(limitlist[key][0])
						# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
						assert (len(left_limit_index) == 1), "Index tuple for axis must have one element!"
						left_limit_index = left_limit_index[0]
					# Try to address the place of the index:
					try:
						ax.get_data()[left_limit_index]
					except TypeError as e:
						print("ERROR: ROI index not int as in indexing.")
						raise e
					except IndexError as e:
						print("ERROR: ROI index not valid (typically out of bounds).")
						raise e
					self.limits[keyindex][0] = left_limit_index
				if limitlist[key][1]:
					if by_index:
						right_limit_index = limitlist[key][1]
					else:
						right_limit_index = ax.get_nearest_index(limitlist[key][1])
						# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
						assert (len(right_limit_index) == 1), "Index tuple for axis must have one element!"
						right_limit_index = right_limit_index[0]
					# Try to address the place of the index:
					try:
						ax.get_data()[right_limit_index]
					except TypeError as e:
						print("ERROR: ROI index not int as in indexing.")
						raise e
					except IndexError as e:
						print("ERROR: ROI index not valid (typically out of bounds).")
						raise e
					self.limits[keyindex][1] = right_limit_index
		else:
			raise TypeError("ROI set_limits_all input not a list or dict.")

	def set_limits(self, key, values, by_index=False):
		"""
		Sets both limits for one axis:

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:param values: Tuple or list of length 2 with the new values for the limits.

		:param by_index: Bool flag to interpret limits not as positions, but as array indices.

		:return:
		"""
		assert (len(values) == 2), "Invalid set for values."
		keyindex = self.dataset.get_axis_index(key)
		ax = self.dataset.axes[keyindex]
		if by_index:
			left_limit_index = values[0]
			right_limit_index = values[1]
		else:
			left_limit_index = ax.get_nearest_index(values[0])
			# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
			assert (len(left_limit_index) == 1), "Index tuple for axis must have one element!"
			left_limit_index = left_limit_index[0]
			right_limit_index = ax.get_nearest_index(values[1])
			# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
			assert (len(right_limit_index) == 1), "Index tuple for axis must have one element!"
			right_limit_index = right_limit_index[0]
		# Try to address the place of the index:
		try:
			ax.get_data()[left_limit_index]
			ax.get_data()[right_limit_index]
		except TypeError as e:
			print("ERROR: ROI index not int as in indexing.")
			raise e
		except IndexError as e:
			print("ERROR: ROI index not valid (typically out of bounds).")
			raise e
		self.limits[keyindex][0] = left_limit_index
		self.limits[keyindex][1] = right_limit_index

	def unset_limits(self, key):
		"""
		Unset both limits for one axis:

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:return:
		"""
		keyindex = self.dataset.get_axis_index(key)
		self.limits[keyindex] = [None, None]

	def set_limit_left(self, key, value, by_index=False):
		"""
		Sets the left limit for one axis.

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:param value: The new value for the left limit.

		:param by_index: Bool flag to interpret limits not as positions, but as array indices.

		:return:
		"""
		keyindex = self.dataset.get_axis_index(key)
		ax = self.dataset.axes[keyindex]
		if by_index:
			left_limit_index = value
		else:
			left_limit_index = ax.get_nearest_index(value)
			# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
			assert (len(left_limit_index) == 1), "Index tuple for axis must have one element!"
			left_limit_index = left_limit_index[0]
		# Try to address the place of the index:
		try:
			ax.get_data()[left_limit_index]
		except TypeError as e:
			print("ERROR: ROI index not int as in indexing.")
			raise e
		except IndexError as e:
			print("ERROR: ROI index not valid (typically out of bounds).")
			raise e
		self.limits[keyindex][0] = left_limit_index

	def unset_limit_left(self, key):
		"""
		Unset left limit for one axis.

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:return:
		"""
		keyindex = self.dataset.get_axis_index(key)
		self.limits[keyindex][0] = None

	def set_limit_right(self, key, value, by_index=False):
		"""
		Sets the right limit for one axis:

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:param value: Tuple or list of length 2 with the new values for the limits.

		:param by_index: Bool flag to interpret limits not as positions, but as array indices.

		:return:
		"""
		keyindex = self.dataset.get_axis_index(key)
		ax = self.dataset.axes[keyindex]
		if by_index:
			right_limit_index = value
		else:
			right_limit_index = ax.get_nearest_index(value)
			# get_nearest_index returns an index tuple, which has only 1 element for the 1D axis.
			assert (len(right_limit_index) == 1), "Index tuple for axis must have one element!"
			right_limit_index = right_limit_index[0]
		try:
			ax.get_data()[right_limit_index]
		except TypeError as e:
			print("ERROR: ROI index not int as in indexing.")
			raise e
		except IndexError as e:
			print("ERROR: ROI index not valid (typically out of bounds).")
			raise e
		self.limits[keyindex][1] = right_limit_index

	def unset_limit_right(self, key):
		"""
		Unset right limit for one axis.

		:param key: A valid identifier of an axis of the DataSet (see DataSet.get_axis())

		:return:
		"""
		keyindex = self.dataset.get_axis_index(key)
		self.limits[keyindex][1] = None

	def get_slice(self, data_key=None):
		"""
		Creates a slice object (or tuple of them) out of the limits of the ROI This can be used directly in the [] for
		adressing the part of the arrays corresponding to the ROI.
		See:
		http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.s_.html
		https://docs.python.org/2/c-api/slice.html

		:param data_key: A valid identifier of an axis or datafield of the DataSet (see DataSet.get_axis()) If given,
			return a slice applicable to an axis of the dataset instead of the whole data array. If the identifier
			corresponds to a datafield, the whole slice as with None will be returned.

		:return: The create slice object.
		"""
		if (data_key is None) or (data_key in self.dlabels):
			# Generate slice tuple for all axes:
			slicelist = []
			for i in range(len(self.limits)):
				slicelist.append(self.get_slice(i))  # recursive call for single axis slice element.
			return tuple(slicelist)
		else:
			# Generate slice for single axis:
			axis_index = self.get_axis_index(data_key)
			llim = self.limits[axis_index][0]
			rlim = self.limits[axis_index][1]
			if rlim is None:  # there is no right limit, so the left limit is llim, which can be a number or None:
				return numpy.s_[llim:None]
			elif llim is None:  # if there is a right but no left limit, the slice index (works excluding) is rlim+1:
				return numpy.s_[None:rlim + 1]
			elif llim <= rlim:  # both limits given, standard order:
				return numpy.s_[llim:rlim + 1]
			else:  # both limits given, reverse order:
				if rlim == 0:  # 0 as index is only adressable like follows, b/c excluding:
					return numpy.s_[llim::-1]
				else:  # rlim must be shifted like above, b/c excluding:
					return numpy.s_[llim:rlim - 1:-1]

	def get_limits(self, data_key=None, by_index=False, raw=False):
		"""
		Returns the limits that define the ROI. Normally, the physical positions of the limits are given,
		see by_index. The limits of the axes are given as lists of length 2 (2-lists).

		:param data_key: A valid identifier of an axis or datafield of the DataSet (see DataSet.get_axis()) If given,
			return the limits of an axis of the dataset instead of the whole data array. If the identifier
			corresponds to a datafield, the whole limit list as with None will be returned.

		:param by_index: Boolean flag: If True, return the limit axis indices instead of physical positions on the
			axis.

		:param raw: Boolean flag: If True, return floats instead of quantities.

		:return: A 2-list or list of 2-lists as specified before.
		"""
		if (data_key is None) or (data_key in self.dlabels):
			if by_index:
				return self.limits
			else:
				limlist = []
				for i in range(len(self.limits)):
					ax = self.get_axis(i)
					if raw:
						limlist.append([ax[0].magnitude, ax[-1].magnitude])
					else:
						limlist.append([ax[0], ax[-1]])
				return limlist
		else:
			ax_index = self.get_axis_index(data_key)
			if by_index:
				return self.limits[ax_index]
			else:
				ax = self.get_axis(data_key)
				if raw:
					return [ax[0].magnitude, ax[-1].magnitude]
				else:
					return [ax[0], ax[-1]]

	def get_datafield(self, label_or_index):
		"""
		Tries to assign a DataField to a given parameter, that can be an integer as an index in the
		datafields list or a label string. Raises exceptions if there is no matching field.
		Uses the underlying method of the DataSet.

		:param label_or_index: Identifier of the DataField

		:return: The corresponding DataField.
		"""
		return self.dataset.get_datafield(label_or_index)[self.get_slice()]

	def get_datafield_index(self, label_or_index):
		return self.dataset.get_datafield_index(label_or_index)

	def get_datafield_normalized(self, label_or_index, method="maximum"):
		"""
		Like get_datafield, but returns a normalized DataArray.

		:param label_or_index: Identifier of the DataField

		:param method: Method to normalize with: Valid options:
			* "maximum", "max" (default): divide every value by the maximum value in the set
			* "mean": divide every value by the average value in the set
			* "minimum", "min": divide every value by the minimum value in the set
			* "absolute maximum", "absmax": divide every value by the maximum absolute value in the set
			* "absolute minimum", "absmin": divide every value by the minimum absolute value in the set

		:return: The normalized DataArray instance.
		"""
		ds = self.get_datafield(label_or_index)
		if method in ["maximum", "max"]:
			return ds / ds.max()
		elif method in ["minimum", "min"]:
			return ds / ds.min()
		elif method in ["mean"]:
			return ds / ds.mean()
		elif method in ["absolute maximum", "max"]:
			return ds / ds.absmax()
		elif method in ["absolute minimum", "min"]:
			return ds / ds.absmin()
		else:
			print("WARNING: Normalization method not valid. Returning unnormalized data.")
			return ds
		# TODO: Testing of this method.

	def get_datafield_by_dimension(self, unit):
		"""
		Returns the first datafield that corresponds to a given unit in its physical dimensionality.

		:param unit: Quantity or anything that can be cast as quantity by the UnitRegistry.

		:return: the datafield
		"""
		return self.dataset.get_datafield_by_dimension(unit)[self.get_slice()]

	def get_axis(self, label_or_index):
		"""
		Tries to assign an Axis to a given parameter, that can be an integer as an index in the
		axes list or a label string. Raises exceptions if there is no matching element.
		Uses the underlying method of the DataSet.

		:param label_or_index: Identifier of the Axis.

		:return: The corresponding Axis.
		"""
		return self.dataset.get_axis(label_or_index)[self.get_slice(label_or_index)]

	def get_axis_index(self, label_or_index):
		return self.dataset.get_axis_index(label_or_index)

	def get_axis_by_dimension(self, unit):
		"""
		Returns the first axis that corresponds to a given unit in its physical dimensionality.

		:param unit: Quantity or anything that can be cast as quantity by the UnitRegistry.

		:return: the Axis
		"""
		ax = self.dataset.get_axis_by_dimension(unit)
		return ax[self.get_slice(ax.label)]

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
			# Assemble axes:
			list_of_axes = []
			for identifier in range(len(self.dataset.axes)):
				list_of_axes.append(self.get_axis(identifier))
			# Build grid:
			return u.meshgrid(*list_of_axes)

	def project_nd(self, *args, **kwargs):
		"""
		Projects the ROI onto the given axes. Uses the DataSet.project_nd() method for every datset and returns a
		new ROI with the projected DataFields and the chosen axes.

		:param args: Valid identifiers for the axes to project onto.

		:return: DataSet: Projected DataSet.
		"""
		if 'h5target' in kwargs:
			h5target = kwargs['h5target']
		else:
			h5target = None
		indexlist = sorted([self.get_axis_index(arg) for arg in args])
		newdataset = DataSet(datafields=[self.dataset.datafields[i].project_nd(*indexlist) for i in indexlist],
							 axes=[self.dataset.axes[i] for i in indexlist], h5target=h5target)
		return self.__class__(newdataset,
							  limitlist=[self.get_limits(by_index=True)[i] for i in indexlist],
							  by_index=True)


class DataSet(object):
	"""
	A data set is a collection of data arrays combined to have a physical meaning. These are n-dimensional
	sets of physical values, in which every dimension itself has a physical meaning. This might for example be a
	3D-array of count rates, in which the x- y- and z-dimensions represent the position on a sample (x,
	y in micrometers) and a time delay (z = t in femtoseconds).
	"""

	# FIXME: check for unique axis and datafield identifiers.

	def __init__(self, label="", datafields=(), axes=(), plotconf=(), h5target=None, chunk_cache_mem_size=None):
		if isinstance(h5target, h5py.Group):
			self.h5target = h5target
			self.own_h5file = False
			self.datafieldgrp = self.h5target.require_group("datafields")
			self.axesgrp = self.h5target.require_group("axes")
		elif isinstance(h5target, string_types):
			self.h5target = h5tools.File(h5target, chunk_cache_mem_size=chunk_cache_mem_size)
			self.own_h5file = True
			self.datafieldgrp = self.h5target.require_group("datafields")
			self.axesgrp = self.h5target.require_group("axes")
		elif h5target:  # True but no designated target means temp file mode.
			self.h5target = True
			self.own_h5file = False
			self.datafieldgrp = True
			self.axesgrp = True
		else:  # Numpy mode.
			self.h5target = None
			self.own_h5file = False
			self.datafieldgrp = None
			self.axesgrp = None

		self.label = label
		# check data format and convert it do correct DataArray and Axis objects before assigning it to members:
		self.datafields = []
		for field in datafields:  # Fill datafield list with correctly formatted datafield objects.
			if self.h5target is True:  # Temp h5 mode
				moep = DataArray(field, h5target=True)
			elif self.h5target:  # Proper h5 file mode
				moep = DataArray(field, h5target=True)
				grp = moep.store_to_h5(self.datafieldgrp)
				moep = DataArray.in_h5(grp)
			else:  # Numpy mode
				moep = DataArray(field)
			self.datafields.append(moep)
		self.axes = []
		for ax in axes:  # Fill axes list with correctly formatted axes objects.
			if self.h5target is True:  # Temp h5 mode
				moep = Axis(ax, h5target=True)
			elif self.h5target:  # Proper h5 file mode
				moep = Axis(ax, h5target=True)
				grp = moep.store_to_h5(self.axesgrp)
				moep = Axis.in_h5(grp)
			else:  # Numpy mode
				moep = Axis(ax)
			self.axes.append(moep)
		self.check_data_consistency()

		if type(plotconf) == str:
			self.plotconf = dict(eval(plotconf))
		else:
			self.plotconf = dict(plotconf)

	@classmethod
	def from_h5file(cls, path, h5target=None, chunk_cache_mem_size=None):
		"""
		Initializes a new DataSet from an existing HDF5 file. The file must be structured in accordance to the
		saveh5() and loadh5() methods in this class. Uses loadh5 under the hood!
		This method is kept for backwards compatibility, while newer from_h5 method expands its function.

		:param path: The (absolute or relative) path of the HDF5 file to read.

		:return: The initialized DataSet
		"""
		return cls.from_h5(path, h5target=h5target)

	@classmethod
	def from_h5(cls, h5source, h5target=None, chunk_cache_mem_size=None):
		"""
		Initializes a new DataSet from an existing HDF5 source. The file must be structured in accordance to the
		saveh5() and loadh5() methods in this class. Uses loadh5 under the hood!

		:param h5source: The (absolute or relative) path of the HDF5 file to read, or an existing h5py Group/File of
			the base of	the Dataset.

		:return: The initialized DataSet
		"""
		dataset = cls(repr(h5source), h5target=h5target)
		if isinstance(h5source, string_types):
			path = os.path.abspath(h5source)
			h5source = h5tools.File(path, chunk_cache_mem_size=chunk_cache_mem_size)
		# Load data:
		dataset.loadh5(h5source)
		return dataset

	@classmethod
	def in_h5(cls, h5group):
		"""
		Opens a DataSet from a h5 source, which then works on the source (in-place). This is forwards to
		 from_h5(h5group, h5group).

		:param h5group: The (absolute or relative) path of the HDF5 file to read, or an existing h5py Group/File of
			the base of the Dataset.

		:return: The generated instance.
		"""
		return cls.from_h5(h5group, h5group)

	@classmethod
	def from_textfile(cls, path, h5target=None, **kwargs):
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
		dataset = cls(filename, h5target=h5target)
		# Load data:
		dataset.load_textfile(path, **kwargs)
		return dataset

	# TODO: Property all the things in __getattr__!
	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

		:param item: The name to get the corresponding attribute.

		:return: The attribute corresponding to the given name.
		"""
		if item == "alldata":
			return self.datafields + self.axes
		elif item == "axlabels":
			labels = []
			for e in self.axes:
				labels.append(e.get_label())
			return labels
		elif item == "dlabels":
			labels = []
			for e in self.datafields:
				labels.append(e.get_label())
			return labels
		elif item == "labels":
			return self.dlabels + self.axlabels
		elif item == "dimensions":
			return len(self.axes)
		elif item == "shape":
			if self.datafields:
				return self.datafields[0].shape
			else:
				return ()
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
		if self.h5target is True:  # Temp h5 mode
			moep = DataArray(data, unit, label, plotlabel, h5target=True)
		elif self.h5target:  # Proper h5 file mode
			moep = DataArray(data, unit, label, plotlabel, h5target=True)
			grp = moep.store_to_h5(self.datafieldgrp)
			moep = DataArray.in_h5(grp)
		else:  # Numpy mode
			moep = DataArray(data, unit, label, plotlabel)
		self.datafields.append(moep)

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

	def get_datafield_normalized(self, label_or_index, method="maximum"):
		"""
		Like get_datafield, but returns a normalized DataArray.

		:param label_or_index: Identifier of the DataField

		:param method: Method to normalize with: Valid options:
			* "maximum", "max" (default): divide every value by the maximum value in the set
			* "mean": divide every value by the average value in the set
			* "minimum", "min": divide every value by the minimum value in the set
			* "absolute maximum", "absmax": divide every value by the maximum absolute value in the set
			* "absolute minimum", "absmin": divide every value by the minimum absolute value in the set

		:return: The normalized DataArray instance.
		"""
		ds = self.get_datafield(label_or_index)
		if method in ["maximum", "max"]:
			return ds / ds.max()
		elif method in ["minimum", "min"]:
			return ds / ds.min()
		elif method in ["mean"]:
			return ds / ds.mean()
		elif method in ["absolute maximum", "max"]:
			return ds / ds.absmax()
		elif method in ["absolute minimum", "min"]:
			return ds / ds.absmin()
		else:
			print("WARNING: Normalization method not valid. Returning unnormalized data.")
			return ds
		# TODO: Testing of this method.

	def get_datafield_by_dimension(self, unit):
		"""
		Returns the first datafield that corresponds to a given unit in its physical dimensionality.

		:param unit: Quantity or anything that can be cast as quantity by the UnitRegistry.

		:return: the datafield
		"""
		for df in self.datafields:
			if u.same_dimension(df.get_data(), unit):
				return df
		raise ValueError("No Axis with dimensionsality found.")

	def replace_datafield(self, datafield_id, new_datafield):
		"""
		Replaces a datafield of the dataset with another. For this to make sense, the new datafield must describe the
		same coordinates in the dataset as the old one: Obviously, the shape must therefore be the same.

		:param datafield_id: Identifier of the Axis to replace. Must be a valid identifier as in get_datafield_index
			and get_datafield.

		:param new_datafield: Axis: An Axis instance that shall be put in place of the old one.

		:return:
		"""
		old_datafield_index = self.get_datafield_index(datafield_id)
		old_datafield = self.get_datafield(datafield_id)
		assert isinstance(new_datafield, DataArray), "ERROR in replace_datafield: Datafield can only be replaced with " \
													 "DataArray instance."
		assert (old_datafield.shape == new_datafield.shape), "ERROR in replace_datafield: New Datafield must have " \
															 "same shape as old one."
		self.datafields[old_datafield_index] = new_datafield

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
		if self.h5target is True:  # Temp h5 mode
			moep = Axis(data, unit, label, plotlabel, h5target=True)
		elif self.h5target:  # Proper h5 file mode
			moep = Axis(data, unit, label, plotlabel, h5target=True)
			grp = moep.store_to_h5(self.axesgrp)
			moep = Axis.in_h5(grp)
		else:  # Numpy mode
			moep = Axis(data, unit, label, plotlabel)
		self.axes.append(moep)

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

	def get_axis_by_dimension(self, unit):
		"""
		Returns the first axis that corresponds to a given unit in its physical dimensionality.

		:param unit: Quantity or anything that can be cast as quantity by the UnitRegistry.

		:return: the Axis
		"""
		for ax in self.axes:
			if u.same_dimension(ax.get_data(), unit):
				return ax
		raise ValueError("No Axis with dimensionsality found.")

	def replace_axis(self, axis_id, new_axis):
		"""
		Replaces an axis of the dataset with another. For this to make sense, the new axis must describe the same
		coordinates in the dataset as the old one: Obviously, the shape must therefore be the same.

		:param axis_id: Identifier of the Axis to replace. Must be a valid identifier as in get_axis_index and get_axis.

		:param new_axis: Axis: An Axis instance that shall be put in place of the old one.

		:return:
		"""
		old_axis_index = self.get_axis_index(axis_id)
		old_axis = self.get_axis(axis_id)
		assert isinstance(new_axis, Axis), "ERROR in replace_axis: Axis can only be replaced with Axis instance."
		assert (old_axis.shape == new_axis.shape), "ERROR in replace_axis: New Axis must have same shape as old one."
		self.axes[old_axis_index] = new_axis

	def get_plotconf(self):
		return self.plotconf

	def get_label(self):
		return self.label

	def set_label(self, newlabel):
		self.label = newlabel

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

	def project_nd(self, *args, **kwargs):
		"""
		Projects the datafield onto the given axes. Uses the DataSet.project_nd() method for every datset and returns a
		new DataSet with the projected DataFields and the chosen axes.

		:param args: Valid identifiers for the axes to project onto.

		:return: DataSet: Projected DataSet.
		"""
		if 'h5target' in kwargs:
			h5target = kwargs['h5target']
		else:
			h5target = None
		indexlist = sorted([self.get_axis_index(arg) for arg in args])
		return self.__class__(
			datafields=[self.datafields[i].project_nd(*indexlist) for i in range(len(self.datafields))],
			axes=[self.axes[i] for i in indexlist],
			h5target=h5target)

	def bin(self, bin_size=()):
		"""
		To be implemented (imported from Ben's script)
		:param bin_size:
		:return:
		"""
		raise NotImplementedError()

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

	def saveh5(self, h5dest=None):
		"""
		Saves the Dataset to a HDF5 destination in a unified format.

		:param h5dest: String or h5py Group/File: The destination to write to.
		 
		:return: Nothing.
		"""
		if h5dest is None:
			h5dest = self.h5target
		if isinstance(h5dest, string_types):
			path = os.path.abspath(h5dest)
			h5dest = h5tools.File(path, 'w')
		else:
			path = False
		assert isinstance(h5dest, h5py.Group), "DataSet.saveh5 needs h5 group or destination path as argument!"

		# TODO: Store snomtools version that data was saved with!
		datafieldgrp = h5dest.require_group("datafields")
		for i in range(len(self.datafields)):
			grp = self.datafields[i].store_to_h5(datafieldgrp)
			h5tools.write_dataset(grp, "index", i)
		axesgrp = h5dest.require_group("axes")
		for i in range(len(self.axes)):
			grp = self.axes[i].store_to_h5(axesgrp)
			h5tools.write_dataset(grp, "index", i)
		h5tools.write_dataset(h5dest, "label", self.label)
		plotconfgrp = h5dest.require_group("plotconf")
		h5tools.store_dictionary(self.plotconf, plotconfgrp)
		h5dest.file.flush()
		if path:  # We got a path and wrote in new h5 file, so we'll close that file.
			h5dest.close()

	def loadh5(self, h5source):
		if isinstance(h5source, string_types):
			path = os.path.abspath(h5source)
			h5source = h5tools.File(path, 'r')
		else:
			path = False
		assert isinstance(h5source, h5py.Group), \
			"DataSet.saveh5 needs h5 group or destination path as argument if no instance h5target is set."

		self.label = str(h5source["label"][()])
		datafieldgrp = h5source["datafields"]
		self.datafields = [None for i in range(len(datafieldgrp))]
		for datafield in datafieldgrp:
			index = datafieldgrp[datafield]['index'][()]
			if h5source == self.h5target:
				dest = datafieldgrp[datafield]
			elif self.h5target is True:
				dest = True
			elif self.h5target:
				dest = self.datafieldgrp.require_group(datafield)
			else:
				dest = None
			self.datafields[index] = (DataArray.from_h5(datafieldgrp[datafield], h5target=dest))
		axesgrp = h5source["axes"]
		self.axes = [None for i in range(len(axesgrp))]
		for axis in axesgrp:
			index = axesgrp[axis]['index'][()]
			if h5source == self.h5target:
				dest = axesgrp[axis]
			elif self.h5target is True:
				dest = True
			elif self.h5target:
				dest = self.axesgrp.require_group(axis)
			else:
				dest = None
			self.axes[index] = (Axis.from_h5(axesgrp[axis], h5target=dest))
		self.plotconf = h5tools.load_dictionary(h5source['plotconf'])
		self.check_data_consistency()
		if path:  # We got a path and read from opened h5 file, so we'll close that file.
			h5source.close()

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
		datacolumns = list(range(datacontent.shape[1]))

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
			print(("WARNING: Comment line(s) {0} in textfile {1} has wrong number of columns. "
				   "No metadata can be read.".format(lines_not_ok, path)))
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
						print(("WARNING: Invalid unit string '{2}' in unit line {0} in textfile {1}".format(
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
					print(("ERROR! Axis initialization in load_textfile failed.", "red"))
					raise e

		# Write the remaining data to datafields:
		self.datafields = []  # Reset datafields
		for i in datacolumns:  # Initialize new datafields
			self.add_datafield(datacontent[:, i], unit=units[i], label=labels[i])

		self.check_label_uniqueness()
		return self.check_data_consistency()

	def __del__(self):
		if self.own_h5file:
			self.h5target.close()

	@classmethod
	def stack(cls, datastack, new_axis, axis=0, label=None, plotconf=None, h5target=None):
		"""
		Stacks a sequence of DataSets to a new DataSet.
		Therefore it stacks the DataArrays with stack_DataArrays() and inserts a new Axis.

		:param datastack: sequence of DataSets: The Data to be stacked.

		:param new_axis: Axis or castable as Axis: The new axis to be inserted for the dimension along which the data is
			stacked.

		:param axis: int, optional: The axis in the result array along which the input arrays are stacked.

		:param label: string, optional: The label for the new DataSet. If not given, the label of the first DataArray in
			the input stack is used.

		:param plotconf: The plot configuration to be used for the new DataSet. If not given, the configuration of the
			first DataSet in the input stack is used.

		:return: The stacked DataSet.
		"""
		# Check if input data types are ok and cast defaults if necessary:
		for ds in datastack:
			assert (isinstance(ds, DataSet)), "ERROR: Non-DataSet object given to stack_DataSets"
		new_axis = Axis(new_axis)
		if label is None:
			label = datastack[0].get_label()
		if plotconf is None:
			plotconf = datastack[0].get_plotconf()

		# Check if data is compatible: All DataSets must have same dimensions and number of datafields:
		for ds in datastack:
			assert (
				ds.shape == datastack[0].shape), "ERROR: DataSets of inconsistent dimensions given to stack_DataSets"
			assert (len(ds.datafields) == len(datastack[0].datafields)), "ERROR: DataSets with different number of " \
																		 "datafields given to stack_DataSets"

		# Initialize new DataSet:
		stack = DataSet(label=label, plotconf=plotconf, h5target=h5target)

		# Build axes list by taking the axes from the first element of the stack and inserting the new one.
		# This makes sense because for stacking to be meaningful, the axes sets of the stack need to be identical in
		# their physical meaning.
		axes = datastack[0].axes
		# Case-like due to different indexing of python's builtin insert method and numpy's stack:
		if axis == -1:  # last element
			axes.append(new_axis)
		elif axis < -1:  # n'th to last element (count from back)
			axes.insert(axis + 1, new_axis)
		else:  # normal count from front
			axes.insert(axis, new_axis)
		stack.axes = axes

		# Stack the datafields all the DataSets and add them to the stacked Set:
		for i in range(len(datastack[0].datafields)):
			dfstack = [ds.get_datafield(i) for ds in datastack]
			if h5target:
				stack.add_datafield(stack_DataArrays(dfstack, axis=axis, h5target=True))
			else:
				stack.add_datafield(stack_DataArrays(dfstack, axis=axis))

		stack.check_data_consistency()
		return stack


def stack_DataArrays(datastack, axis=0, unit=None, label=None, plotlabel=None, h5target=None):
	"""
	Stacks a sequence of DataArrays to a new DataArray.
	See DataArray.stack.
	"""
	return DataArray.stack(datastack, axis=axis, unit=unit, label=label, plotlabel=plotlabel, h5target=h5target)


def stack_DataSets(datastack, new_axis, axis=0, label=None, plotconf=None, h5target=None):
	"""
	Stacks a sequence of DataSets to a new DataSet.
	See DataSet.stack.
	"""
	return DataSet.stack(datastack, new_axis, axis=0, label=None, plotconf=None, h5target=h5target)


if __name__ == "__main__":  # just for testing
	print('Testing...')
	testarray = numpy.arange(0, 10, 2.)
	testaxis = DataArray(testarray, 'meter', label="xaxis")
	testaxis2 = testaxis / 2.
	testaxis2.set_label("yaxis")
	X, Y = numpy.meshgrid(testaxis, testaxis2)
	# testaxis = DataArray(testarray[testarray<5], 'meter')
	testdata = DataArray(numpy.sin((X + Y) * u.ureg('rad')) * u.ureg('counts'), label="testdaten", plotlabel="pl")
	# testdatastacklist = [testdata * i for i in range(3)]

	pc = {'a': 1.0, 'b': "moep", 'c': 3, 'de': "eins/zwo"}

	testdataset = DataSet("test", [testdata], [testaxis, testaxis2], plotconf=pc, h5target='test.hdf5')

	testroi = ROI(testdataset)
	testroi.set_limits('xaxis', (2, 4))

	testdataset.saveh5()

	del testdataset

	testdataset2 = DataSet.from_h5file('test.hdf5')
	testdataset2.saveh5("exampledata.hdf5")

	testdataset3 = DataSet.from_textfile('test2.txt', unitsline=1, h5target="test3.hdf5")
	testdataset3.saveh5()

	testh5 = h5tools.File('test.hdf5')

	test_dataarray = True
	# noinspection PyPackageRequirements
	if test_dataarray:
		moep = DataArray(testaxis.data, label="test", h5target=testh5)
		moep2 = moep + moep
		moep2 = moep - moep
		moep2 = moep * moep
		moep2 = moep / moep
		# moep2 = moep // moep # FIXME: to_ureg(DataArray) for newer versions of pint.
		moep2 = moep ** 2.
		moep.absmax()
		moep.absmin()
		moep.mean()
		moep.sum()
		moep.sum_raw()
		# works till here
		moep.get_nearest_value(2.)
		moep.set_unit('mm')
		del moep

		bigfuckindata = Data_Handler_H5(unit='km', shape=(1000, 1000))
		moep = DataArray(bigfuckindata, label="test", h5target=testh5)

		# testdataset.saveh5('test.hdf5')
		moep.write_to_h5()
		del moep
		testh5.close()

		testh5 = h5tools.File('test.hdf5')
		moep3 = DataArray(testh5, h5target=testh5)

		dhs = [Data_Handler_H5(numpy.arange(5), 'meter') for i in range(3)]
		dhs.append(u.to_ureg(numpy.arange(5), 'millimeter'))
		stacktest = DataArray(Data_Handler_np.stack(dhs, unit='millimeter', axis=1))

		dhs = [stacktest for i in range(10)]
		stacktest = DataArray.stack(dhs, h5target=True)
		# stackh5 = h5tools.File("stacktest.hdf5")
		# stacktest.store_to_h5(stackh5)
		# stackh5.close()
		del stacktest

	test_sum = True
	if test_sum:
		h5 = h5tools.File("test4.hdf5")
		mediumfuckindata = Data_Handler_H5(numpy.ones((100, 100, 20)), unit="m/s")
		sum1 = mediumfuckindata.sum(0)
		sum2 = mediumfuckindata.sum(0, keepdims=True)
		sum3 = sum1.sum(0)
		sum4 = sum2.sum(1, keepdims=True)
		sum5 = mediumfuckindata.sum((0, 1))
		sum6 = mediumfuckindata.sum((0, 1), keepdims=True)
		sum7 = mediumfuckindata.sum((0, 2), h5target=h5)
		sum8 = mediumfuckindata.sum()

	test_manyfiles = False
	if test_manyfiles:
		h5files = []
		for i in range(1000):
			h5files.append(h5tools.File("ZZZZ{0:04d}.hdf5".format(i)))

		print("writing data...")
		for f in h5files:
			h5tools.write_dataset(f, "data", data=numpy.ones((100, 100, 20)),
								  chunks=True,
								  compression="gzip",
								  compression_opts=4)

		print("closing...")
		for f in h5files:
			f.close()

	print("OK")
