__author__ = 'hartelt'
'''
This file provides miscellaneous tools for data manipulation.
For furter info about data structures, see:
data.datasets.py
'''

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