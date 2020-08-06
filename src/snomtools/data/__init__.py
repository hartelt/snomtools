"""
This module provides data processing scripts. There will be a base class which shall be used for all physical
datasets, as well as tools for storing, loading and basic processing of data (e.g. normalization). More complex data
evaluation and calculation shall be done elsewhere, using the here defined data structures.
For storage, the preferred format will be HDF5 using the python package h5py. See:
https://www.hdfgroup.org/HDF5/doc/index.html
http://docs.h5py.org/en/latest/index.html

"""
__author__ = 'Michael Hartelt'

#just for testing:
if False:
	import h5py
	h5py.run_tests()