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
from numpy.lib.stride_tricks import as_strided

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


def reversed_slice(s, len_):
	"""
	Reverses a slice selection on a sequence of length len_, addressing the same elements in reverse order.

	:param slice s: The slice object to reverse.

	:param int len_: The length of the sequence on which the reverse selection should apply.

	:returns: A slice object, addressing the same elements in reverse order.
	:rtype: slice
	"""
	assert isinstance(s, slice)
	instart, instop, instep = s.indices(len_)

	m = (instop - instart) % instep or instep

	if instep > 0 and instart - m < 0:
		outstop = None
	else:
		outstop = instart - m
	if instep < 0 and instop - m > len_:
		outstart = None
	else:
		outstart = instop - m

	return slice(outstart, outstop, -instep)


def sliced_shape(slice_, shape_):
	"""
	Calculate the shape one would get by slicing an array of shape :code:`shape_` with a slice :code:`slice_`.

	.. note:: This is propably very inefficient, because an 1D-Array is initialized and sliced for each dimension.

	:param slice_: A slice, as generated with numpy.s_[something].

	:param tuple shape_: The shape of the hypothetical array to be sliced.

	:return: The resulting shape.
	:rtype: tuple
	"""
	dummy = dummy_array(shape_)
	return dummy[slice_].shape


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


def dummy_array(shape_):
	"""
	Dummy array of a given shape, meaning an array that needs virtually no memory, by just referencing every element of
	the array to one :code:`0` in memory.

	:param tuple shape_: A tuple of ints, corresponding to the required shape.

	:return: An array view of the required shape.
	:rtype: numpy.ndarray
	"""
	x = np.array([0])
	return as_strided(x, shape=shape_, strides=[0] * len(shape_), writeable=False)


def broadcast_shape(*shapes):
	"""
	Given a set of array shapes, return the shape of the output when arrays of those
	shapes are broadcast together

	:param shapes: One or more shapes (tuples of ints) representing the shape of the arrays to be broadcasted.

	:returns: The shape of the array that is generated, when arrays of the input shapes are broadcast together.
	:rtype: tuple(int)
	"""
	max_nim = max(len(s) for s in shapes)
	equal_len_shapes = np.array([(1,) * (max_nim - len(s)) + s for s in shapes])
	max_dim_shapes = np.max(equal_len_shapes, axis=0)
	assert np.all(np.bitwise_or(equal_len_shapes == 1, equal_len_shapes == max_dim_shapes[None, :])), \
		'Shapes %s are not broadcastable together' % (shapes,)
	return tuple(max_dim_shapes)


def broadcast_indices(*shapes):
	"""
	Given a set of shapes of arrays that you could broadcast together, return an iterator that returns a len(shapes)+1
	tuple of the indices of each input array and their corresponding index in the output array.

	:param shapes: One or more shapes (tuples of ints) representing the shape of the arrays to be broadcasted.

	:returns: broadcast_shape_iterator: An iterator that for every iterations gives a len(shapes)+1 tuple of the
		indices of each input array and their corresponding index in the output array, in the order
		:code:`in1, in2, ... , out`.
		The indices are given as tuples of ints that can directly be used to address the elements of the arrays as in
		:code:`arr1[in1]`.
	:rtype: tuple(tuple(int))
	"""
	output_shape = broadcast_shape(*shapes)
	base_iter = np.ndindex(output_shape)

	def broadcast_shape_iterator():
		for out_ix in base_iter:
			in_ixs = tuple(tuple(0 if s[i] == 1 else ix for i, ix in enumerate(out_ix[-len(s):])) for s in shapes)
			yield in_ixs + (out_ix,)

	return broadcast_shape_iterator()


if __name__ == '__main__':
	import numpy

	a = numpy.arange(30)
	s = numpy.s_[::-1]
	print(s)
	print(reversed_slice(s, len(a)))
	print(a[s])
	print(a[reversed_slice(s, len(a))])
	print("done")
