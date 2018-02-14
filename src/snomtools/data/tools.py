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

__author__ = 'hartelt'


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


def full_slice(slice_, len_):
	"""
	Generate a full slice tuple from a general slice tuple, which can have entries for only some of the first
	dimensions.

	:param slice_: The incomplete slice, as generated with numpy.s_[something].
	:type slice_: tuple **or** slice **or** int

	:param int len_: The length of the full slice tuple (number of dimensions).

	:return: The complete slice.
	:rtype: tuple
	"""
	try:
		slice_ = tuple(np.s_[slice_])
	except TypeError:
		slice_ = (np.s_[slice_],)
	missing = tuple([np.s_[:] for i in range(len_ - len(slice_))])
	return slice_ + missing


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
