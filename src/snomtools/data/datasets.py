"""
This file contains the base class for datasets.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types
import numpy
import os
import h5py
import re
import scipy.ndimage
import datetime
import warnings
import sys
import itertools
import snomtools.calcs.units as u
from snomtools.data import h5tools
from snomtools import __package__, __version__
from snomtools.data.tools import full_slice, broadcast_shape, broadcast_indices, reversed_slice

__author__ = 'Michael Hartelt'

if '-v' in sys.argv:
	verbose = True
else:
	verbose = False


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
			temp_file = h5tools.Tempfile(chunk_cache_mem_size=chunk_cache_mem_size)
			temp_dir = temp_file.temp_dir
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
		elif isinstance(data, h5py.Dataset):
			inst = object.__new__(cls)
			inst.__used = False
			inst.__handling = None
			inst.compression = compression
			inst.compression_opts = compression_opts
			unit = u.to_ureg(1, unit).units
			h5tools.clear_name(h5target, "data")
			# h5target.copy(data, h5target) # breaks in h5py 2.2.1, propably because of bug therein.
			data.file.copy(data.name, h5target)
			inst.ds_data = h5target["data"]
			h5tools.clear_name(h5target, "unit")
			inst.ds_unit = h5target.create_dataset("unit", data=str(unit))
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
		return u.unit_from_str(h5tools.read_as_str(self.ds_unit))._units

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
	def dims(self):
		return len(self.shape)

	@property
	def dtype(self):
		return self.ds_data.dtype

	@property
	def chunks(self):
		return self.ds_data.chunks

	def __getitem__(self, key):
		# Find out if there is backwards addressed elements in the selection:
		to_reverse = self.find_backwards_slices(key)
		# If not, just read the data and return it as a new instance:
		if not any(to_reverse):
			return self.__class__(self.ds_data[key], self._units)
		# If there is, we need to address the corresponding elements in forward direction, read it, and flip the result:
		key = full_slice(key, self.dims)
		readkeylist = []
		for i, key_element in enumerate(key):
			if to_reverse[i]:
				readkeylist.append(reversed_slice(key_element, self.shape[i]))
			else:
				readkeylist.append(key_element)
		read_data = self.ds_data[tuple(readkeylist)]
		ordered_data = read_data[tuple([numpy.s_[::-1] if flip else numpy.s_[:] for flip in to_reverse])]
		return self.__class__(ordered_data, self._units)

	def get_slice_q(self, key):
		# Find out if there is backwards addressed elements in the selection:
		to_reverse = self.find_backwards_slices(key)
		# If not, just read the data and return it as a new instance:
		if not any(to_reverse):
			return u.to_ureg(self.ds_data[key], self._units)
		# If there is, we need to address the corresponding elements in forward direction, read it, and flip the result:
		key = full_slice(key, self.dims)
		readkeylist = []
		for i, key_element in enumerate(key):
			if to_reverse[i]:
				readkeylist.append(reversed_slice(key_element, self.shape[i]))
			else:
				readkeylist.append(key_element)
		read_data = self.ds_data[tuple(readkeylist)]
		ordered_data = read_data[tuple([numpy.s_[::-1] if flip else numpy.s_[:] for flip in to_reverse])]
		return u.to_ureg(ordered_data, self._units)

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
		# FIXME: This behaves differently, as in throws exception, than numpy with negative step (reverse array).

		# The following line could be replaced with
		# value = u.to_ureg(value).to(self.units)
		# without changing any functionality. But calling to_ureg twice is more efficient because unneccesary calling
		#  of value.to(self.units), which always generates a copy, is avoided if possible.
		value = u.to_ureg(u.to_ureg(value), self.units)

		# Find out if there is backwards addressed elements in the selection:
		to_reverse = self.find_backwards_slices(key)
		# If not, just read the data and return it as a new instance:
		if not any(to_reverse):
			self.ds_data[key] = value.magnitude
			return
		# If there is, we need to address the corresponding elements in forward direction, read it, and flip the result:
		key = full_slice(key, self.dims)
		writekeylist = []
		for i, key_element in enumerate(key):
			if to_reverse[i]:
				writekeylist.append(reversed_slice(key_element, self.shape[i]))
			else:
				writekeylist.append(key_element)
		write_key = tuple(writekeylist)
		raise NotImplementedError("This functionality is not finished yet!")

	def find_backwards_slices(self, s):
		"""
		Analyzes a selection on the data for backwards (step < 0) slices.

		:param s: The selection slice or tuple of slices and ints to analize.

		:return: A list of bools of `len == self.dims` with `True` for every backwards element, else `False`.
		:rtype: list(bool)
		"""
		s = full_slice(s, self.dims)
		l = []
		for element in s:
			if isinstance(element, slice) and element.step is not None and element.step < 0:
				l.append(True)
			else:
				l.append(False)
		return l

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

	def sum(self, axis=None, dtype=None, out=None, keepdims=False, h5target=None, ignorenan=False):
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

		:param ignorenan: bool, optional
			If this is set to True, nans in the array will be ignored and set to zero when summing them up.

		:return: ndarray Quantity
			An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a
			scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		# TODO: Handle datatypes.
		# TODO: Autodetect appropriate chunk size for better performance.
		# TODO: printing progress when verbose option is set
		inshape = self.shape
		if axis is None:
			axis = tuple(range(len(inshape)))
		try:
			if len(axis) == 1:  # If we have a sequence of len 1, we sum over only 1 axis.
				axis = axis[0]
				single_axis_flag = True
			elif len(axis) == 0:
				# An empty tuple... so we have nothing to do and return a copy of self or write own data to out.
				if out is None:
					return self.__class__(self, h5target=h5target)
				else:
					assert out.shape == self.shape, "Wrong shape of given destination."
					if out.shape == ():  # Scalar
						out.ds_data[()] = self.ds_data[()]
					else:
						out.ds_data[:] = self.ds_data[:]
					return out
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
					if ignorenan:
						outdata.ds_data[()] += numpy.ma.fix_invalid(self.ds_data[tuple(slicebase)], fill_value=0.)
					else:
						outdata.ds_data[()] += self.ds_data[tuple(slicebase)]
				else:
					if ignorenan:
						outdata.ds_data[:] += numpy.ma.fix_invalid(self.ds_data[tuple(slicebase)], fill_value=0.)
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
			return self.sum(axisnow, dtype, out, keepdims, h5target=None, ignorenan=ignorenan).sum(axisrest, dtype, out,
																								   keepdims, h5target,
																								   ignorenan=ignorenan)

	# TODO: Overwrite max, min, mean to avoid working on magnitude and breaking memory.
	# TODO: Improve nanmax, nanmin, nanmean, absmax, nanabsmax, absmin, nanabsmin to avoid working on magnitude and breaking memory.

	def nanmax(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmax(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def nanmin(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmin(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def nanmean(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmean(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def absmax(self, axis=None, keepdims=None):
		return abs(self).max(axis=axis, keepdims=keepdims)

	def nanabsmax(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmax(abs(self), axis=axis, keepdims=keepdims), unit=self.get_unit())

	def absmin(self, axis=None, keepdims=None):
		return abs(self).min(axis=axis, keepdims=keepdims)

	def nanabsmin(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmin(abs(self), axis=axis, keepdims=keepdims), unit=self.get_unit())

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

		if output is False:
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

		# TODO: DOCS!!!

		:param output: The array in which to place the output, or the dtype of the returned array.
			If :code:`False` is given, the slice of the instance data is overwritten.
		:type output: ndarray *or* dtype *or* :code:`False`, *optional*

		:param h5target: The h5target to in case a new Data_Handler_H5 is generated.

		:returns: The shifted data. If output is given as a parameter or :code:`False`, None is returned.
		:rtype: Data_Handler_H5 *or* None
		"""
		# TODO: Optimize performance by not loading full data along shifted axes.
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False
		slice_ = full_slice(slice_, len(self.shape))

		try:
			test = float(shift)
			expanded_slice = full_slice(numpy.s_[:], len(self.shape))
			recover_slice = slice_
			shift_dimensioncorrected = shift
		except TypeError:  # Shift is a sequence with shifts for each dimension
			assert len(shift) == len(slice_), "Propably invalid shift argument."
			expanded_slice = []
			recover_slice = []
			shift_dimensioncorrected = []
			for shift_element, slice_element in zip(shift, slice_):
				if shift_element != 0:
					expanded_slice.append(numpy.s_[:])
					recover_slice.append(slice_element)
					shift_dimensioncorrected.append(shift_element)
				else:
					expanded_slice.append(slice_element)
					if isinstance(slice_element, slice):  # If we don't end up with one less dimension.
						recover_slice.append(numpy.s_[:])
						shift_dimensioncorrected.append(0.)
			expanded_slice = tuple(expanded_slice)  # Part of the data we need for shifting.
			recover_slice = tuple(recover_slice)  # Part of the shifted data we wanted to address.
			shift_dimensioncorrected = tuple(shift_dimensioncorrected)

		if output is False:
			self.ds_data[slice_] = \
				scipy.ndimage.interpolation.shift(self.ds_data[expanded_slice], shift_dimensioncorrected, None, order,
												  mode, cval, prefilter)[recover_slice]
			return None
		elif isinstance(output, numpy.ndarray):
			output[:] = \
				scipy.ndimage.interpolation.shift(self.ds_data[expanded_slice], shift_dimensioncorrected, None, order,
												  mode, cval, prefilter)[recover_slice]
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_H5(
				scipy.ndimage.interpolation.shift(self.ds_data[expanded_slice], shift_dimensioncorrected, output, order,
												  mode, cval, prefilter)[recover_slice], self.units, h5target=h5target)

	# TODO: Implement rotate_slice similar to shift_slice by using scipy.ndimage.interpolation.rotate

	# FIXME: Iterators for scalar data seems to freeze system.

	def iterchunkslices(self, dim=None, dims=None):
		"""
		Iterator, which returns slice objects which address the data chunk-wise. This can be used wo very efficiently
		perform operations on the data since chunk-wise is the fastest way to access the data in the HDF5 file.

		:param int dim: Do this only for the first :code:`dim` dimensions. Used for recursively calling the generator.
			If not given, all dimensions are used, so this defaults to :code:`dim = len(self.shape)-1`

		:param dims: Iterate chunk-wise only for the dimension in dims. Full-slices [:] are given for all others.
		:type dims: sequence of ints

		:return: A tuple of slice objects of length :code:`dim`
		"""
		if dim is None:
			dim = len(self.shape) - 1
		if dims is None:
			dims = list(range(len(self.shape)))

		if dim == 0:  # Break condition
			if 0 in dims:
				csize_dim = self.chunks[dim]
				start = 0
				while start < self.shape[dim]:
					stop = start + csize_dim
					if stop >= self.shape[dim]:
						stop = None
					yield (slice(start, stop),)
					start = start + csize_dim
			else:
				yield (numpy.s_[:],)
		else:  # Generate slices for dim and attach them to all slices for dim-1 recursively
			if dim in dims:
				csize_dim = self.chunks[dim]
				start = 0
				while start < self.shape[dim]:
					stop = start + csize_dim
					if stop >= self.shape[dim]:
						stop = None
					for slicetuple_before in self.iterchunkslices(dim - 1, dims):
						yield slicetuple_before + (slice(start, stop),)
					start = start + csize_dim
			else:
				for slicetuple_before in self.iterchunkslices(dim - 1, dims):
					yield slicetuple_before + (numpy.s_[:],)

	def iterchunks(self, dims=None):
		"""
		Iterator, which returns the data of the chunks, chunk-wise. It returns the data as Quantities, because they are
		small, so it will be much faster to keep them in RAM.

		:param dims: Iterate chunk-wise only for the dimension in dims. Full-slices [:] are given for all others.
		:type dims: sequence of ints

		:return: The data in the chunk.
		:rtype: pint.Quantity
		"""
		for chunkslice in self.iterchunkslices(dims=dims):
			yield u.to_ureg(self.ds_data[chunkslice], self._units)

	def iterlineslices(self):
		"""
		Iterator, which provides slices corresponding to a line-wise iteration over the data.

		:return: Slice tuple.
		"""
		if self.shape == (): # If self is scalar, return iterator with only () as slice, else iteration breaks.
			return iter([()])
		iterlist = [range(i) for i in self.shape]
		if iterlist:
			iterlist.pop()
		iterlist.append([numpy.s_[:]])
		return itertools.product(*iterlist)

	def iterlines(self):
		"""
		Iterator, which yields the data line-wise. It returns the data as Quantities, because they are small, so it
		will be much faster to keep them in RAM.

		:return: The data of the current line.
		:rtype: pint.Quantity
		"""
		for lineslice in self.iterlineslices():
			yield u.to_ureg(self.ds_data[lineslice], self._units)

	def iterflatslices(self):
		"""
		Iterator, which provides index tuples corresponding to a flat iteration over the array.

		:return: Tuple of ints of length corresponding to the data dimensions.
		"""
		iterlist = [range(i) for i in self.shape]
		return itertools.product(*iterlist)

	def iterflat(self):
		"""
		Iterator, which yields the single data elements, as flattened (1D) iteration. It returns the data as
		Quantities, because they are small (scalar), so it will be much faster to keep them in RAM.

		:return: The data of the current point.
		:rtype: pint.Quantity
		"""
		for indextuple in self.iterflatslices():
			yield u.to_ureg(self.ds_data[indextuple], self._units)

	def iterfastslices(self):
		"""
		Iterator, which returns slice objects which iterate over the data as fast as possible according to memory order.
		This means chunk-wise for chunked data, line-wise for unchunked data. This provides the fastest way of accessing
		the data sequentially.

		:return: An iterator of slices.
		"""
		if self.chunks:
			# Get the fastest possible iteration, by choosing for which axis to iterate chunk-wise or take the full
			# dimension. The optimal selection will be the fastest one fitting into the chunk_cache of the hdf5 file.
			if isinstance(self.h5target, h5tools.File):
				cachesize = self.h5target.get_chunk_cache_mem_size()
			else:
				cachesize = 1 * 1024 ** 2  # Default h5py files have 1 MB cache size.
			# optimal number of elements to select at once is the cache size divided by the byte size of the elements:
			select_elements = cachesize // self.ds_data.dtype.itemsize
			# Try every combination, select the best:
			best_elements = 0
			best_selection = None
			for selection in itertools.product([False, True], repeat=self.dims):
				elements = numpy.where(selection, self.shape, self.chunks).prod()
				if elements <= select_elements:
					if elements > best_elements:
						best_elements = elements
						best_selection = selection
			# Return the optimal iterator, dims being the dimensions to iterate over chunk-wise:
			if best_selection is not None:
				dims = [i for i in range(self.dims) if not best_selection[i]]
			else:
				dims = None
			return self.iterchunkslices(dims=dims)
		else:
			return self.iterlineslices()

	def iterfast(self):
		"""
		Iterator, which yields the data, accessing it in the fastest way to iterate over them, see :func:iterfastslices.
		It returns the data as Quantities, because they are small, so it will be much faster to keep them in RAM.

		:return: The data of the current slice.
		:rtype: pint.Quantity
		"""
		for slice_ in self.iterfastslices():
			yield u.to_ureg(self.ds_data[slice_], self._units)

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newdh = self.__class__(shape=self.shape, unit=self.get_unit())
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata + other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			newdh = self.__class__(shape=self.shape, unit=self.get_unit())
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata + other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			newshape = broadcast_shape(self.shape, other.shape)
			if self.chunks:
				newchunks = h5tools.probe_chunksize(newshape)
				# Use at least one line worth of chunks as buffer for performance:
				min_cache_size = numpy.prod(newchunks, dtype=numpy.int64) // newchunks[-1] * newshape[-1] \
								 * 4  # 32bit floats require 4 bytes.
				use_cache_size = min_cache_size + 16 * 1024 ** 2  # Add 16 MB just to be sure.
				newdh = self.__class__(shape=newshape, unit=self.get_unit(), chunk_cache_mem_size=use_cache_size)
			else:
				newdh = self.__class__(shape=newshape, unit=self.get_unit(), chunks=False)
			# Because we have different shapes and potentially different chunking, we need to iterate element-wise:
			# TODO: This is still extremely unefficient due to loads of read-write on H5. Needs better iteration order.
			for ind_self, ind_other, ind_out in broadcast_indices(self.shape, other.shape):
				newdh[ind_out] = u.to_ureg(self.ds_data[ind_self], self._units) + other[ind_other]
			return newdh

	def __iadd__(self, other):
		other = u.to_ureg(other, self.get_unit())
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] += other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] += other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			newshape = broadcast_shape(self.shape, other.shape)
			# Because we have different shapes and potentially different chunking, we need to iterate element-wise:
			# TODO: This is still extremely unefficient due to loads of read-write on H5. Needs better iteration order.
			for ind_self, ind_other, ind_out in broadcast_indices(self.shape, other.shape):
				self.ds_data[ind_out] += other[ind_other].magnitude
			return self

	def __sub__(self, other):
		other = u.to_ureg(other, self.get_unit())
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newdh = self.__class__(shape=self.shape, unit=self.get_unit())
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata - other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			newdh = self.__class__(shape=self.shape, unit=self.get_unit())
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata - other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__sub__(other)

	def __isub__(self, other):
		other = u.to_ureg(other, self.get_unit())
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] -= other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] -= other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__isub__(other)

	def __mul__(self, other):
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newunit = str((other * u.to_ureg(1., self.get_unit())).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata * other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			newunit = str((u.to_ureg(1., str(other.units)) * u.to_ureg(1., self.get_unit())).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata * other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__mul__(other)

	def __imul__(self, other):
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			self._units = str((other * u.to_ureg(1., self.get_unit())).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] *= other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			self._units = str((u.to_ureg(1., str(other.units)) * u.to_ureg(1., self.get_unit())).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] *= other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__imul__(other)

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newunit = str((u.to_ureg(1., self.get_unit()) / other).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata / other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			newunit = str((u.to_ureg(1., str(other.units)) / u.to_ureg(1., self.get_unit())).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata / other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__truediv__(other)

	def __itruediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			self._units = str((u.to_ureg(1., self.get_unit()) / other).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] /= other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			self._units = str((u.to_ureg(1., str(other.units)) / u.to_ureg(1., self.get_unit())).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] /= other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__itruediv__(other)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newunit = str((u.to_ureg(1., self.get_unit()) // other).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata // other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			newunit = str((u.to_ureg(1., str(other.units)) // u.to_ureg(1., self.get_unit())).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata // other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__floordiv__(other)

	def __ifloordiv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			self._units = str((u.to_ureg(1., self.get_unit()) // other).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] //= other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			self._units = str((u.to_ureg(1., str(other.units)) // u.to_ureg(1., self.get_unit())).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] //= other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__ifloordiv__(other)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			newunit = str((u.to_ureg(1., self.get_unit()) ** other).units)
			newdh = self.__class__(shape=self.shape, unit=newunit)
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata ** other
			return newdh
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			assert self.dimensionless(), "Quantity array exponents are only allowed if the base is dimensionless"
			newdh = self.__class__(shape=self.shape, unit="dimensionless")
			for slice_, owndata in zip(self.iterfastslices(), self.iterfast()):
				newdh[slice_] = owndata ** other.get_slice_q(slice_)
			return newdh
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__pow__(other)

	def __ipow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		if not hasattr(other, 'shape') or other.shape == ():
			# If other is scalar, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use.
			assert numpy.isscalar(other.magnitude), "Input seemed scalar but isn't."
			self._units = str((u.to_ureg(1., self.get_unit()) ** other).units)
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] **= other.magnitude
			return self
		elif other.shape == self.shape:
			# If other has the same shape, the shape doesn't change and we can do everything chunk-wise with better
			# performance and memory use. For this, we need a Data_Handler to use get_slice_q instead of potentially
			# slower getitem.
			if not isinstance(other, (self.__class__, Data_Handler_np)):
				other = self.__class__(other)
			assert self.dimensionless(), "Quantity array exponents are only allowed if the base is dimensionless"
			for slice_ in self.iterfastslices():
				self.ds_data[slice_] **= other.get_slice_q(slice_).magnitude
			return self
		else:
			# Else we need the numpy broadcasting magic to an array of different shape.
			# TODO: Implement this memory-efficiently with broadcasting.
			# The following line is a fallback which will break for big data due to using magnitudes.
			return super(Data_Handler_H5, self).__ipow__(other)

	def __array__(self):
		return self.magnitude

	def __repr__(self):
		return "<Data_Handler_H5 on {0} with shape {1}>".format(repr(self.h5target), self.shape)

	def __del__(self):
		if not (self.temp_file is None):
			del self.temp_file

	@classmethod
	def add_multiple(cls, *toadd, **kwargs):
		"""
		Adds two or more arrays (Quantities, Data_Handlers, ...) of the same shape up and return the result as a new
		Data_Handler_H5.

		:param toadd: The elements to add. All must have the same shape.

		:param kwargs: kwargs for the new Data_Handler, see :func:`~Data_Handler_H5.__new__`.

		:return: The summed up data.
		:rtype: Data_Handler_H5
		"""
		toadd = list(toadd)
		importunit = kwargs.pop('unit', None)
		if importunit is None:
			toadd[0] = u.to_ureg(toadd[0])
			importunit = str(toadd[0].units)
		for i, element in enumerate(toadd):
			if isinstance(toadd[i], cls):
				toadd[i] = u.to_ureg(element, importunit)
			else:
				toadd[i] = cls(element, importunit)
			assert toadd[i].shape == toadd[0].shape, "Elements of different shape given."
		# Initialize new Data_Handler:
		newdh = cls(shape=toadd[0].shape, unit=importunit, **kwargs)
		for slice_ in newdh.iterfastslices():
			slicedata = toadd[0].get_slice_q(slice_)
			for element in toadd[1:]:
				slicedata += element.get_slice_q(slice_)
			newdh[slice_] = slicedata
		return newdh

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

		# Find optimized buffer size:
		chunk_size = h5tools.probe_chunksize(outshape)
		min_cache_size = chunk_size[axis] * numpy.prod(inshape) * 4  # 32bit floats require 4 bytes.
		use_cache_size = min_cache_size + 64 * 1024 ** 2  # Add 64 MB just to be sure.

		inst = cls(shape=outshape, unit=unit, h5target=h5target, chunk_cache_mem_size=use_cache_size)
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

	def get_slice_q(self, key):
		return u.to_ureg(self.magnitude[key], str(self.units))

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

	def sum(self, axis=None, dtype=None, out=None, keepdims=False, h5target=None, ignorenan=False):
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

		:param ignorenan: bool, optional
			If this is set to True, nans in the array will be ignored and set to zero when summing them up.

		:return: ndarray Quantity
			An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a
			scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		if ignorenan:
			return self.__class__(
				numpy.ma.fix_invalid(self.magnitude, fill_value=0.).sum(axis=axis, dtype=dtype, out=out,
																		keepdims=keepdims), unit=self.get_unit())
		else:
			return self.__class__(self.magnitude.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims),
								  unit=self.get_unit())

	def nanmax(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmax(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def nanmin(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmin(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def nanmean(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmean(self.magnitude, axis=axis, keepdims=keepdims), unit=self.get_unit())

	def absmax(self, axis=None, keepdims=None):
		return abs(self).max(axis=axis, keepdims=keepdims)

	def nanabsmax(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmax(abs(self), axis=axis, keepdims=keepdims), unit=self.get_unit())

	def absmin(self, axis=None, keepdims=None):
		return abs(self).min(axis=axis, keepdims=keepdims)

	def nanabsmin(self, axis=None, keepdims=None):
		return u.to_ureg(numpy.nanmin(abs(self), axis=axis, keepdims=keepdims), unit=self.get_unit())

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

		if output is False:
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
		# TODO: Optimize performance by not loading full data along shifted axes.
		if prefilter is None:  # if not explicitly set, determine neccesity of prefiltering
			if order > 0:  # if interpolation is required, spline prefilter is neccesary.
				prefilter = True
			else:
				prefilter = False

		slice_ = full_slice(slice_, len(self.shape))

		try:
			test = float(shift)
			expanded_slice = full_slice(numpy.s_[:], len(self.shape))
			recover_slice = slice_
			shift_dimensioncorrected = shift
		except TypeError:  # Shift is a sequence with shifts for each dimension
			assert len(shift) == len(slice_), "Propably invalid shift argument."
			expanded_slice = []
			recover_slice = []
			shift_dimensioncorrected = []
			for shift_element, slice_element in zip(shift, slice_):
				if shift_element != 0:
					expanded_slice.append(numpy.s_[:])
					recover_slice.append(slice_element)
					shift_dimensioncorrected.append(shift_element)
				else:
					expanded_slice.append(slice_element)
					if isinstance(slice_element, slice):  # If we don't end up with one less dimension.
						recover_slice.append(numpy.s_[:])
						shift_dimensioncorrected.append(0.)
			expanded_slice = tuple(expanded_slice)  # Part of the data we need for shifting.
			recover_slice = tuple(recover_slice)  # Part of the shifted data we wanted to address.
			shift_dimensioncorrected = tuple(shift_dimensioncorrected)

		if output is False:
			self.magnitude[slice_] = \
				scipy.ndimage.interpolation.shift(self.magnitude[expanded_slice], shift_dimensioncorrected, None, order,
												  mode, cval, prefilter)[recover_slice]
			return None
		elif isinstance(output, numpy.ndarray):
			output[:] = \
				scipy.ndimage.interpolation.shift(self.magnitude[expanded_slice], shift_dimensioncorrected, None, order,
												  mode, cval, prefilter)[recover_slice]
			return None
		else:
			assert (output is None) or isinstance(output, type), "Invalid output argument given."
			return Data_Handler_np(
				scipy.ndimage.interpolation.shift(self.magnitude[expanded_slice], shift_dimensioncorrected, output,
												  order, mode, cval, prefilter)[recover_slice], self.units)

	def iterlineslices(self):
		"""
		Iterator, which provides slices corresponding to a line-wise iteration over the data.

		:return: Slice tuple.
		"""
		if self.shape == (): # If self is scalar, return iterator with only () as slice, else iteration breaks.
			return iter([()])
		iterlist = [range(i) for i in self.shape]
		iterlist.pop()
		iterlist.append([numpy.s_[:]])
		return itertools.product(*iterlist)

	def iterlines(self):
		"""
		Iterator, which yields the data line-wise.

		:return: The data of the current line.
		:rtype: pint.Quantity
		"""
		for lineslice in self.iterlineslices():
			yield self[lineslice].q

	def iterflatslices(self):
		"""
		Iterator, which provides index tuples corresponding to a flat iteration over the array.

		:return: Tuple of ints of length corresponding to the data dimensions.
		"""
		iterlist = [range(i) for i in self.shape]
		return itertools.product(*iterlist)

	def iterflat(self):
		"""
		Iterator, which yields the single data elements, as flattened (1D) iteration. It returns the data as
		Quantities, because they are small (scalar), so it will be much faster to keep them in RAM.

		:return: The data of the current point.
		:rtype: pint.Quantity
		"""
		for indextuple in self.iterflatslices():
			yield self[indextuple].q

	def iterfastslices(self):
		"""
		Iterator, which returns slice objects which iterate over the data as fast as possible according to memory order.
		This means line-wise for data stored in numpy (C) order. This method is kept as analog for compatibility to
		Data_Handler_H5.

		:return: Line iterator.
		"""
		return self.iterlineslices()

	def iterfast(self):
		"""
		Iterator, which yields the data, accessing it in the fastest way to iterate over them, see :func:iterfastslices.

		:return: The data of the current slice.
		:rtype: pint.Quantity
		"""
		for slice_ in self.iterfastslices():
			yield self[slice_].q

	def __add__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return super(Data_Handler_np, self).__add__(other)

	def __iadd__(self, other):
		other = u.to_ureg(other, self.get_unit())
		return super(Data_Handler_np, self).__iadd__(other)

	def __sub__(self, other):
		other = u.to_ureg(other, self.units)
		return super(Data_Handler_np, self).__sub__(other)

	def __isub__(self, other):
		other = u.to_ureg(other, self.units)
		return super(Data_Handler_np, self).__isub__(other)

	def __mul__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__mul__(other)

	def __imul__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__imul__(other)

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__truediv__(other)

	def __itruediv__(self, other):
		"""
		This replaces __idiv__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__itruediv__(other)

	def __floordiv__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__floordiv__(other)

	def __ifloordiv__(self, other):
		other = u.to_ureg(other)
		return super(Data_Handler_np, self).__ifloordiv__(other)

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return super(Data_Handler_np, self).__pow__(other)

	def __ipow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return super(Data_Handler_np, self).__ipow__(other)

	def __repr__(self):
		return "<Data_Handler_np(" + super(Data_Handler_np, self).__repr__() + ")>"

	def __del__(self):
		pass

	@classmethod
	def add_multiple(cls, *toadd, **kwargs):
		"""
		Adds two or more arrays (Quantities, Data_Handlers, ...) and return the result as a new Data_Handler_np.

		:param toadd: The elements to add.

		:param kwargs: kwargs for the new Data_Handler, see :func:`~Data_Handler_np.__new__`. Only :code:`unit` makes
			sense here.

		:return: The summed up data.
		:rtype: Data_Handler_np
		"""
		toadd = list(toadd)
		importunit = kwargs.pop('unit', None)
		if kwargs:
			raise TypeError('Unexpected **kwargs: %r' % kwargs)
		if importunit is None:
			toadd[0] = u.to_ureg(toadd[0])
			importunit = str(toadd[0].units)

		newdh = cls(shape=toadd[0].shape, unit=importunit)
		for element in toadd:
			newdh += element
		return newdh

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

		:param h5target: Optional. The HDF5 target to work on, if on-disk h5 mode is desired. :code:`True` can be given
			to enable temp file mode.

		:return: The initialized DataArray.
		"""
		assert isinstance(h5source, h5py.Group), "DataArray.from_h5 requires h5py group as source."
		if h5target:
			assert isinstance(h5target, h5py.Group) or h5target is True, \
				"DataArray.from_h5 requires h5py group as target, or True to use temp file."
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
		warnings.warn("Trying to delete data from DataArray.")

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
	def dims(self):
		return len(self.shape)

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

		.. warning::
			The plotlabel typically includes a unit, so this might get invalid!

		:param unitstr: A valid unit string.

		:return: Nothing.
		"""
		self.data.set_unit(unitstr)

	def to(self, unitstr):
		"""
		Returns a copy of the dataarray with the unit set as specified. For compatibility with pint quantity.

		.. warning::
			The plotlabel typically includes a unit, so this might get invalid!

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
				h5tools.clear_name(self.h5target, h5set)
				h5source.copy(h5set, self.h5target)
			self._data = Data_Handler_H5(h5target=self.h5target)
		elif self.h5target is True:
			self._data = Data_Handler_H5(h5source["data"], unit=h5tools.read_as_str(h5source["unit"]), h5target=True)
			self.label = h5tools.read_as_str(h5source["label"])
			self.plotlabel = h5tools.read_as_str(h5source["plotlabel"])
		else:
			self.set_data(numpy.array(h5source["data"]), h5tools.read_as_str(h5source["unit"]))
		self.set_label(h5tools.read_as_str(h5source["label"]))
		self.set_plotlabel(h5tools.read_as_str(h5source["plotlabel"]))

	def flush(self):
		"""
		Flushes HDF5 buffers to disk. This only makes sense in h5 disk mode and in non-tempfile mode.

		:return: Nothing.
		"""
		if isinstance(self.h5target, h5py.Group):
			self.write_to_h5()
		else:
			warnings.warn("DataSet cannot flush without working on valid HDF5 file.")

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

	def sum(self, axis=None, dtype=None, out=None, keepdims=False, ignorenan=False):
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
		return self.data.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims, ignorenan=ignorenan)

	def sum_raw(self, axis=None, dtype=None, out=None, keepdims=False):
		"""
		As sum(), only on bare numpy array instead of Quantity. See sum() for details.
		:return: ndarray
		An array with the same shape as a, with the specified axes removed. If a is a 0-d array, or if axis is None, a
		scalar is returned. If an output array is specified, a reference to out is returned.
		"""
		return self.data.sum_raw(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

	def project_nd(self, *args, **kwargs):
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
			return self.sum(sumtup, **kwargs)
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

	def max(self, axis=None, keepdims=False, ignorenan=False):
		if ignorenan:
			return self.data.nanmax(axis=axis, keepdims=keepdims)
		else:
			return self.data.max(axis=axis, keepdims=keepdims)

	def min(self, axis=None, keepdims=False, ignorenan=False):
		if ignorenan:
			return self.data.nanmin(axis=axis, keepdims=keepdims)
		else:
			return self.data.min(axis=axis, keepdims=keepdims)

	def absmax(self, axis=None, keepdims=False, ignorenan=False):
		if ignorenan:
			return self.data.nanabsmax(axis=axis, keepdims=keepdims)
		else:
			return self.data.absmax(axis=axis, keepdims=keepdims)

	def absmin(self, axis=None, keepdims=False, ignorenan=False):
		if ignorenan:
			return self.data.nanabsmin(axis=axis, keepdims=keepdims)
		else:
			return self.data.absmin(axis=axis, keepdims=keepdims)

	def mean(self, axis=None, keepdims=False, ignorenan=False):
		if ignorenan:
			return self.data.nanmean(axis=axis, keepdims=keepdims)
		else:
			return self.data.mean(axis=axis, keepdims=keepdims)

	def __pos__(self):
		return self.__class__(self.data, label=self.label, plotlabel=self.plotlabel)

	def __neg__(self):
		return self.__class__(-self.data, label=self.label, plotlabel=self.plotlabel)

	def __abs__(self):
		return self.__class__(abs(self.data), label=self.label, plotlabel=self.plotlabel)

	def __add__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		other = u.to_ureg(other, self.get_unit())
		return self.__class__(self.data + other, label=self.label, plotlabel=self.plotlabel)

	def __iadd__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		self._data += other
		return self

	def __sub__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		other = u.to_ureg(other, self.get_unit())
		return self.__class__(self.data - other, label=self.label, plotlabel=self.plotlabel)

	def __isub__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		self._data -= other
		return self

	def __mul__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		other = u.to_ureg(other)
		return self.__class__(self.data * other, label=self.label, plotlabel=self.plotlabel)

	def __imul__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		self._data *= other
		return self

	def __truediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		if isinstance(other, self.__class__):
			other = other.data
		other = u.to_ureg(other)
		return self.__class__(self.data / other, label=self.label, plotlabel=self.plotlabel)

	def __itruediv__(self, other):
		"""
		This replaces __div__ in Python 3. All divisions are true divisions per default with '/' operator.
		In python 2, this new function is called anyway due to :code:`from __future__ import division`.
		"""
		if isinstance(other, self.__class__):
			other = other.data
		self._data /= other
		return self

	def __floordiv__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		other = u.to_ureg(other)
		return self.__class__(self.data // other, label=self.label, plotlabel=self.plotlabel)

	def __ifloordiv__(self, other):
		if isinstance(other, self.__class__):
			other = other.data
		self._data //= other
		return self

	def __pow__(self, other):
		other = u.to_ureg(other, 'dimensionless')
		return self.__class__(self.data ** other, label=self.label, plotlabel=self.plotlabel)

	def __ipow__(self, other):
		self._data **= other
		return self

	def __array__(self):  # to numpy array
		return self.data.magnitude

	def __iter__(self):
		return iter(self.data)

	# noinspection PyTypeChecker,PyUnresolvedReferences
	def __getitem__(self, key):
		"""
		To allow adressing parts or elements of the DataArray with [], including slicing as in numpy. This just
		forwards to the underlying __getitem__ method of the data object.

		:param key: The key which is given as adressed in dataarray[key].
		:type key: slice **or** int **or** tuples thereof

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

	@classmethod
	def add(cls, to_add, unit=None, label=None, plotlabel=None, h5target=None):
		"""
		Adds a sequence of two or more DataArrays up to a new DataArray. All of them must have the same shape.

		:param to_add: sequence of DataArrays: The Data to be stacked.

		:param unit: string, optional: The unit for the stacked DataArray. All data must be convertible to that unit. If
			not given, the unit of the first DataArray in the input stack is used.

		:param label: string, optional: The label for the new DataSet. If not given, the label of the first DataArray in
			the input stack is used.

		:param plotlabel: string, optional: The plotlabel for the new DataArray. If not given, the label of the first
			DataArray in the input stack is used.

		:return: The stacked DataArray.
		"""
		for da in to_add:
			assert (isinstance(da, DataArray)), "ERROR: Non-DataArray object given to stack_DataArrays"
			assert da.shape == to_add[0].shape, "Tried to add DataArrays of different shape."
		if unit is None:
			unit = to_add[0].get_unit()
		if label is None:
			label = to_add[0].get_label()
		if plotlabel is None:
			plotlabel = to_add[0].get_plotlabel()
		onlydata = [da.get_data() for da in to_add]
		if h5target:
			sum_data = Data_Handler_H5.add_multiple(*onlydata, unit=unit, h5target=h5target)
		else:
			sum_data = Data_Handler_np.add_multiple(*onlydata, unit=unit)
		return cls(sum_data, label=label, plotlabel=plotlabel, h5target=h5target)


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

	def value_floatindex(self, idx):
		"""
		Assuming the axis elements are sorted, this returns an approximated value between two points,
		addressed by a float number, with a simple linear approximation.

		:param idx: float: An number between 0 and len(Axis)-1.
		If a number out of this range is given, the outer limit value of the axes is returned.

		:return: Corresponding linearly approximated value.
		"""
		if idx <= 0:
			return self.data.q[0]
		if idx >= len(self)-1:
			return self.data.q[-1]
		idx_int = int(idx)
		left_weight = 1 - (idx - idx_int)
		right_weight = idx - idx_int
		# Return approximated values, use quantities rather than Data_Handlers, to avoid spamming scalar temp h5s:
		return left_weight * self.data.q[idx_int] + right_weight * self.data.q[idx_int + 1]

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
		self.set_label(label)
		self.set_plotlabel(plotlabel)

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
		if item in self.labels:
			for darray in self.alldata:
				if item == darray.get_label():
					lim = self.get_slice(item)
					return darray[lim]
		raise AttributeError("Name \'{0}\' in ROI object cannot be resolved!".format(item))

	@property
	def axlabels(self):
		return self.dataset.axlabels

	@property
	def dlabels(self):
		return self.dataset.dlabels

	@property
	def labels(self):
		return self.dataset.labels

	@property
	def alldata(self):
		return self.dataset.alldata

	@property
	def dimensions(self):
		return self.dataset.dimensions

	@property
	def shape(self):
		return self.get_datafield(0).shape

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
			warnings.warn("Normalization method not valid. Returning unnormalized data.")
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
		Projects the ROI onto the given axes. Uses the DataSet.project_nd() method for the addressed region and returns
		a new DataSet with the projected DataFields and the chosen axes sections.

		:param args: Valid identifiers for the axes to project onto.

		:return: DataSet: Projected DataSet.
		"""
		if 'h5target' in kwargs:
			h5target = kwargs['h5target']
		else:
			h5target = None
		indexlist = sorted([self.get_axis_index(arg) for arg in args])
		newdataset = DataSet(datafields=[self.get_datafield(i).project_nd(*indexlist)
										 for i in range(len(self.dataset.datafields))],
							 axes=[self.get_axis(i) for i in indexlist], h5target=h5target)
		return newdataset

	def get_DataSet(self, label=None, plotconf=None, h5target=None, chunk_cache_mem_size=None):
		"""
		Initialize a new DataSet containing the data of the RoI.

		:param label: A label for the new DataSet. Default: The label of the ROI, or if that is empty a generated label.

		:param plotconf: A plotconf for the new DataSet. Default: The plotconf of the DataSet the RoI was defined on.

		:param h5target: Optional. A h5target of the new DataSet, if h5 mode is desired.

		:param chunk_cache_mem_size: If a h5target is given, a chunk cache size can be specified.

		:return: The DataSet of the RoI Region.
		:rtype: DataSet
		"""
		# TODO: Test me!
		if label is None:
			if self.label:
				label = self.label
			else:
				label = "RoI of DataSet '{0:s}'".format(self.dataset.label)
		if plotconf is None:
			plotconf = self.dataset.plotconf
		newds = DataSet(label, plotconf=plotconf, h5target=h5target, chunk_cache_mem_size=chunk_cache_mem_size)
		for i in range(len(self.dataset.datafields)):
			newds.add_datafield(self.get_datafield(i))
		for i in range(len(self.dataset.axes)):
			newds.add_axis(self.get_axis(i))
		newds.check_data_consistency()
		return newds

	def saveh5(self, h5dest):
		# TODO: Test me!
		newds = self.get_DataSet(h5target=h5dest)
		newds.saveh5()


class DataSet(object):
	"""
	A data set is a collection of data arrays combined to have a physical meaning. These are n-dimensional
	sets of physical values, in which every dimension itself has a physical meaning. This might for example be a
	3D-array of count rates, in which the x- y- and z-dimensions represent the position on a sample (x,
	y in micrometers) and a time delay (z = t in femtoseconds).
	"""

	# TODO: Handle 'synonyms' of axes (Several axis per data dimension.)
	# TODO: Handle metadata!

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
		self.check_label_uniqueness()

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
		return cls.from_h5(path, h5target=h5target, chunk_cache_mem_size=chunk_cache_mem_size)

	@classmethod
	def from_h5(cls, h5source, h5target=None, chunk_cache_mem_size=None):
		"""
		Initializes a new DataSet from an existing HDF5 source. The file must be structured in accordance to the
		saveh5() and loadh5() methods in this class. Uses loadh5 under the hood!

		:param h5source: The (absolute or relative) path of the HDF5 file to read, or an existing h5py Group/File of
			the base of	the Dataset.

		:return: The initialized DataSet
		"""
		dataset = cls(repr(h5source), h5target=h5target, chunk_cache_mem_size=chunk_cache_mem_size)
		if isinstance(h5source, string_types):
			sourcepath = os.path.normcase(os.path.abspath(h5source))
			# FIXME: This breaks for h5target=True (Temp file mode) when trying to access dataset.h5target.filename:
			if h5target and sourcepath == os.path.normcase(os.path.abspath(dataset.h5target.filename)):
				# The source file was already opened and is used as the h5target of the new dataset. This happens for
				# example when using in_h5. So avoid opening the file twice and just use the one we have.
				dataset.loadh5(dataset.h5target)
			else:  # We need to open the source file, read from it, and close it afterwards.
				h5source = h5tools.File(sourcepath, chunk_cache_mem_size=chunk_cache_mem_size)
				dataset.loadh5(h5source)
				h5source.close()
		else:  # We have a h5py Group to read, so just do it:
			dataset.loadh5(h5source)
		return dataset

	@classmethod
	def in_h5(cls, h5group, chunk_cache_mem_size=None):
		"""
		Opens a DataSet from a h5 source, which then works on the source (in-place). This is forwards to
		 from_h5(h5group, h5group).

		:param h5group: The (absolute or relative) path of the HDF5 file to read, or an existing h5py Group/File of
			the base of the Dataset.

		:return: The generated instance.
		"""
		return cls.from_h5(h5group, h5group, chunk_cache_mem_size=chunk_cache_mem_size)

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

	def __getattr__(self, item):
		"""
		This method provides dynamical naming in instances. It is called any time an attribute of the intstance is
		not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

		:param item: The name to get the corresponding attribute.

		:return: The attribute corresponding to the given name.
		"""
		if item in self.labels:
			for darray in self.alldata:
				if item == darray.get_label():
					return darray
		raise AttributeError("Name \'{0}\' in DataSet object cannot be resolved!".format(item))

	@property
	def axlabels(self):
		return [a.get_label() for a in self.axes]

	@property
	def dlabels(self):
		return [d.get_label() for d in self.datafields]

	@property
	def labels(self):
		return self.dlabels + self.axlabels

	@property
	def alldata(self):
		return self.datafields + self.axes

	@property
	def dimensions(self):
		return len(self.axes)

	@property
	def shape(self):
		if self.datafields:
			return self.datafields[0].shape
		else:
			return ()

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
		assert self.check_label_uniqueness(moep.label), "Cannot add datafield. Label already exists!"
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
			warnings.warn("Normalization method not valid. Returning unnormalized data.")
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
		assert self.check_label_uniqueness(new_datafield.label) or old_datafield.label == new_datafield.label, \
			"Cannot add datafield. Label already exists!"
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
		assert self.check_label_uniqueness(moep.label), "Cannot add axis. Label already exists!"
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
		assert self.check_label_uniqueness(new_axis.label) or old_axis.label == new_axis.label, \
			"Cannot add axis. Label already exists!"
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
		# TODO: Implement binning.
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
			if isinstance(self.h5target, h5py.File) and (path == os.path.abspath(self.h5target.filename)):
				# own h5target was explicitly (redundantly) requested, so just take it instead of making a new file.
				h5dest = self.h5target
				path = False
			else:
				h5dest = h5tools.File(path, 'w')
		else:
			path = False
		assert isinstance(h5dest, h5py.Group), "DataSet.saveh5 needs h5 group or destination path as argument!"

		h5tools.write_dataset(h5dest, "version", __package__ + " " + __version__)
		h5tools.write_dataset(h5dest, "savedate", datetime.datetime.now().isoformat())
		datafieldgrp = h5dest.require_group("datafields")
		for i in range(len(self.datafields)):
			grp = self.datafields[i].store_to_h5(datafieldgrp)
			h5tools.write_dataset(grp, "index", i)
		h5tools.clean_group(datafieldgrp, self.dlabels)  # Remove old entries from h5 file.
		axesgrp = h5dest.require_group("axes")
		for i in range(len(self.axes)):
			grp = self.axes[i].store_to_h5(axesgrp)
			h5tools.write_dataset(grp, "index", i)
		h5tools.clean_group(axesgrp, self.axlabels)  # Remove old entries from h5 file.
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
		h5tools.check_version(h5source)
		self.label = h5tools.read_as_str(h5source["label"])
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
			try:
				index = axesgrp[axis]['index'][()]
			except KeyError as e:
				warnings.warn("Axis Group '{0}' without key in H5 structure... ignoring".format(axis), UserWarning)
				self.axes.remove(None)
				continue
			if h5source == self.h5target:
				dest = axesgrp[axis]
			elif self.h5target is True:
				dest = True
			elif self.h5target:
				dest = self.axesgrp.require_group(axis)
			else:
				dest = None
			if self.axes[index] is not None:
				warnings.warn(
					"Axis {0} occurs more than once in H5 file! Overwriting with Axis '{1}'".format(index, axis))
				# If one element is overwritten, there was one too much initialized... remove one None element:
				self.axes.remove(None)
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
			warnings.warn("Comment line(s) {0} in textfile {1} has wrong number of columns. "
						  "No metadata can be read.".format(lines_not_ok, path))
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
						warnings.warn("Invalid unit string '{2}' in unit line {0} in textfile {1}"
									  "".format(unitsline, path, unit))
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
					ds.shape == datastack[
				0].shape), "ERROR: DataSets of inconsistent dimensions given to stack_DataSets"
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

	@classmethod
	def add(cls, to_add, label=None, plotconf=None, h5target=None):
		"""
		Adds up a sequence of two or more DataSets to a new DataSet. All of them must have the same shape and axes.

		:param to_add: sequence of DataSets: The Data to be stacked.

		:param string label: *optional* The label for the new DataSet. If not given, the label of the first DataArray in
			the input stack is used.

		:param plotconf: The plot configuration to be used for the new DataSet. If not given, the configuration of the
			first DataSet in the input stack is used.

		:return: The stacked DataSet.
		:rtype: DataSet
		"""
		# TODO: Test me!
		# Check if input data types are ok and cast defaults if necessary:
		for ds in to_add:
			assert (isinstance(ds, DataSet)), "ERROR: Non-DataSet object given to stack_DataSets"
		if label is None:
			label = to_add[0].get_label()
		if plotconf is None:
			plotconf = to_add[0].get_plotconf()

		# Check if data is compatible: All DataSets must have same dimensions and number of datafields and axes:
		for ds in to_add:
			assert (ds.shape == to_add[0].shape), \
				"ERROR: DataSets of inconsistent dimensions given to add"
			assert (len(ds.datafields) == len(to_add[0].datafields)), "ERROR: DataSets with different number of " \
																	  "datafields to add up."
			for i, axis in enumerate(ds.axes):
				# Test at least if all corresponding axes have the same units. Testing every single value of each axis
				# would be a bit overkill.
				assert axis.units == to_add[0].axes[i].units, "DataSets have axes with different units."
				assert axis.label == to_add[0].axes[i].label, "DataSets have axes with different labels."

		# Initialize new DataSet:
		sum_dataset = DataSet(label=label, plotconf=plotconf, h5target=h5target)
		sum_dataset.axes = to_add[0].axes

		# Stack the datafields all the DataSets and add them to the stacked Set:
		for i in range(len(to_add[0].datafields)):
			dflist = [ds.get_datafield(i) for ds in to_add]
			if h5target:
				sum_dataset.add_datafield(DataArray.add(dflist, h5target=True))
			else:
				sum_dataset.add_datafield(DataArray.add(dflist))

		sum_dataset.check_data_consistency()
		return sum_dataset


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
	return DataSet.stack(datastack, new_axis, axis=axis, label=label, plotconf=plotconf, h5target=h5target)


if __name__ == "__main__":  # just for testing
	print("snomtools version " + __version__)
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

	testdataset.replace_axis('xaxis', Axis(testarray, 'second', label="newaxis"))

	testdataset.saveh5()

	del testdataset

	testdataset2 = DataSet.in_h5('test.hdf5')
	testdataset2.saveh5("exampledata.hdf5")

	testdataset3 = DataSet.from_textfile('test2.txt', unitsline=1, h5target="test3.hdf5")
	testdataset3.saveh5("test3.hdf5")

	testdataset4 = DataSet.add([testdataset3, testdataset3], h5target="test4.hdf5")
	testdataset4.saveh5()

	testh5 = h5tools.File('test.hdf5')

	test_dataarray = False
	# noinspection PyPackageRequirements
	if test_dataarray:
		moep = DataArray(testaxis.data, label="test", h5target=testh5)
		moep2 = moep + moep
		moep2 = moep - moep
		moep2 = moep * moep
		moep2 = moep / moep
		# moep2 = moep // moep
		# FIXME: For newer versions of pint, truediv seems to only work with dimensionless quantities as denominator.
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

	test_sum = False
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

	test_bigdata_operations = False
	if test_bigdata_operations:
		bigfuckindata = Data_Handler_H5(unit='km', shape=(1000, 1000, 50), chunk_cache_mem_size=500 * 1024 ** 2)
		import time

		# start_time = time.time()
		# bigplusline = bigfuckindata + numpy.ones(10)
		# print("Plus line of ones took {0:.2f} seconds".format(time.time() - start_time))

		start_time = time.time()
		bigplus = bigfuckindata + 1
		print("Plus 1 took {0:.2f} seconds".format(time.time() - start_time))
		start_time = time.time()
		bigplusplus = bigplus + bigplus
		print("data plus data took {0:.2f} seconds".format(time.time() - start_time))
		start_time = time.time()
		bigminus = bigfuckindata - 1
		print("Minus 1 took {0:.2f} seconds".format(time.time() - start_time))
		start_time = time.time()
		bigtimes = bigfuckindata * 2
		print("Times 2 took {0:.2f} seconds".format(time.time() - start_time))
		start_time = time.time()
		bigdiv = bigtimes / 2
		print("Divided by 2 took {0:.2f} seconds".format(time.time() - start_time))
		start_time = time.time()
		bigfloordiv = bigtimes // u.to_ureg("2 km")
		print("Truediv by 2 took {0:.2f} seconds".format(time.time() - start_time))

		datalist = [bigplus, bigplusplus, bigminus, bigtimes, bigdiv]

		start_time = time.time()
		bigmultipleadd = Data_Handler_H5.add_multiple(*datalist)
		print("Adding 5 arrays took {0:.2f} seconds".format(time.time() - start_time))

		if False:
			bignumpy = numpy.zeros(shape=(1000, 1000, 1000), dtype=numpy.float32)
			start_time = time.time()
			bignumpyplus = bignumpy + 1
			print("Numpy plus 1 took {0:.2f} seconds".format(time.time() - start_time))

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
