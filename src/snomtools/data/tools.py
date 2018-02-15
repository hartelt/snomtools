"""
This file provides miscellaneous tools for data manipulation.
For furter info about data structures, see:
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types
import numpy as np

__author__ = 'Michael Hartelt'


def assure_1D(data):
	"""
	Makes sure the data is onedimensional. Try a consistent conversion, if that fails raise error.

	:param data: Quantity or numpy array.

	:returns: The flattened data or the data itself if it was already 1D.
	"""
	if len(data.shape) == 1:  # Array itself is 1D
		return data
	# Else we have a moredimensional array. Try to flatten it:
	flatarray = data.flatten()
	if not (len(flatarray) in data.shape):  # an invalid conversion.
		raise ArithmeticError("Non-1D convertable data array in Axis.")
	else:
		return flatarray


def full_slice(slice_, len_=None):
	"""
	Generate a full slice tuple from a general slice tuple, which can have entries for only some of the first
	dimensions.

	:param slice_: The incomplete slice, as generated with numpy.s_[something].
	:type slice_: tuple **or** slice **or** int

	:param int len_: The length of the full slice tuple (number of dimensions). If not given, input is only returned as
		tuple.

	:return: The complete slice.
	:rtype: tuple
	"""
	try:
		slice_ = tuple(np.s_[slice_])
	except TypeError:
		slice_ = (np.s_[slice_],)
	if len_ is not None:
		missing = tuple([np.s_[:] for i in range(len_ - len(slice_))])
		slice_ = slice_ + missing
	return slice_


def sliced_shape(slice_, shape_):
	"""
	Calculate the shape one would get by slicing an array of shape :code:`shape_` with a slice :code:`slice_`.

	.. note:: This is propably very inefficient, because an 1D-Array is initialized and sliced for each dimension.

	:param slice_: A slice, as generated with numpy.s_[something].

	:param tuple shape_: The shape of the hypothetical array to be sliced.

	:return: The resulting shape.
	:rtype: tuple
	"""
	full_slice_ = full_slice(slice_, len(shape_))
	sizes = []
	for i in range(len(shape_)):
		arr = np.empty(shape_[i])
		subarr = arr[full_slice_[i]]
		try:
			sizes.append(len(subarr))
		except TypeError:
			pass
	return tuple(sizes)


def slice_expansion(inslice, expansion, dimensions=None):
	"""
	Extends a given slice in both directions by given values.

	:param inslice: The slice to be extended, as produced with numpy.s_[something]
	:type inslice: tuple **or** slice **or** int

	:param expansion: The value or values to expand the slice by. If an int is given, expand all dimensions by that
		value. If a tuple is given, it should contain one int for each dimension, by which the dimension is then
		expanded.
	:type expansion: int **or** tuple(int)

	:param dimensions: (optional) A shape of the data to be addressed by the extended slice. If this is given, a full
		slice with elements for every axis is returned.
	:type dimensions: int

	:return: The expanded slice.
	:rtype: tuple( int / slice )
	"""
	inslice = full_slice(inslice, dimensions)
	try:
		expansion = tuple(expansion)
	except TypeError:
		expansion = tuple([int(expansion) for i in range(len(inslice))])
	assert len(expansion) == len(inslice), "Expansion tuple with invalid length given."
	newslice = []
	for slice_, exp in zip(inslice, expansion):
		if isinstance(slice_, slice):
			start, stop = slice_.start, slice_.stop
			if start is not None:
				start -= exp
			if stop is not None:
				stop += exp
			newslice.append(np.s_[start: stop: slice_.step])
		else:  # Scalar, int-like
			newslice.append(np.s_[slice_ - exp: slice_ + exp: None])
	return tuple(newslice)


def slice_overhang(slice_, shape_):
	"""
	Calculate the overhang of an expanded slice over the edge of the data shape.

	.. warning::
		Slices with step statements are not supported yet.

	:param slice_: The extended full tuple of slice objects (and ints).
	:type slice_: tuple

	:param tuple shape_: The shape tuple of the addressed array-like.

	:return: A tuple of len(shape_) 2-tuples of ints. containing the overhang in the form (negative,positive).
	:rtype: tuple
	"""
	assert len(slice_) == len(shape_), "Invalid argument lengths."
	overhangs = []
	for sl, sh in zip(slice_, shape_):
		if isinstance(sl, slice):
			start, stop, step = sl.start, sl.stop, sl.step
			if step is None:
				if start is not None:
					left = - min(start, 0)
				else:
					left = 0
				if stop is not None:
					right = max(stop - sh, 0)
				else:
					right = 0
			else:
				raise NotImplementedError()
			overhangs.append((left, right))
		else:  # scalar, int-like
			overhangs.append((0, 0))
	return tuple(overhangs)


def iterfy(iterable):
	"""
	Makes sure a given object can be iterated over: If an iterable is given, it just returns it. If a non-iterable
		e.g. a single value is given, it returns a list containing the input as a single element.
	"""
	if isinstance(iterable, string_types):
		iterable = [iterable]
	try:
		iter(iterable)
	except TypeError:
		iterable = [iterable]
	return iterable


def find_next_prime(N):
	"""
	Find next prime >= N

	This is modified from the _find_next_prime function in the h5py_cache package. Copyright (c) 2016 Mike Boyle,
	under MIT license.

	:param N: A number.

	:return: The next prime number >= N.
	"""

	def is_prime(n):
		if n % 2 == 0:
			return False
		i = 3
		while i * i <= n:
			if n % i:
				i += 2
			else:
				return False
		return True

	if N < 3:
		return 2
	if N % 2 == 0:
		N += 1
	for n in range(N, 2 * N, 2):
		if is_prime(n):
			return n
	raise AssertionError("Failed to find a prime number between {0} and {1}...".format(N, 2 * N))
