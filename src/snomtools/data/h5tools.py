__author__ = 'hartelt'
"""
This script provides some simple tools for the storage in h5 files.
"""


def store_dictionary(dict_to_store, h5target):
	for key in dict_to_store.keys():
		assert (not ('/' in key)), "Special group separation char '/' used as key."
		h5target.create_dataset(key, data=dict_to_store[key])


def load_dictionary(h5source):
	outdict = {}
	for key in iter(h5source):
		key = unicode(key)
		outdict[key] = h5source[key][()]
	return outdict