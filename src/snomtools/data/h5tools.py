__author__ = 'hartelt'
"""
This script provides some simple tools for the storage in h5 files.
"""
import h5py
import h5py_cache


# Set default cache size for h5py-cache files. h5py-default is 1024**2 (1 MB)
chunk_cache_mem_size_default = 1024 * 1024 ** 2  # 1 GB


def File(*args, **kwargs):
	"""
	Initializes a h5py_cache File object as documented in h5py_cache.File and h5py.File. Uses the value
	chunk_cache_mem_size_default as defined above as buffer size if not given otherwise explicitly.

	:return: A h5py.File object with the chosen buffer settings.
	"""
	key = "chunk_cache_mem_size"
	if not key in kwargs:
		kwargs[key] = chunk_cache_mem_size_default
	return h5py_cache.File(*args, **kwargs)


def store_dictionary(dict_to_store, h5target):
	for key in dict_to_store.keys():
		assert (not ('/' in key)), "Special group separation char '/' used as key."
		write_dataset(h5target, key, dict_to_store[key])


def load_dictionary(h5source):
	outdict = {}
	for key in iter(h5source):
		key = unicode(key)
		outdict[key] = h5source[key][()]
	return outdict


def write_dataset(h5dest, name, data, **kwargs):
	"""
	Writes a HDF5 dataset to a destination, deleting any data of the same name that are already there first.

	:param dest: The HDF5 destination to write to.

	:param name: The name of the dataset to be written

	:param data: The data to write.

	:param kwargs: kwargs to be forwarded to create_dataset.

	:return: The written dataset.
	"""
	clear_name(h5dest, name)
	h5dest.create_dataset(name, data=data, **kwargs)


def clear_name(h5dest, name):
	"""
	Removes an entry of a HDF5 group if it exists, thereby clearing the namespace for creating a new dataset.

	:param h5dest: The h5py group in which to clear the entry.

	:param name: String: The name to clear.

	:return: nothing
	"""
	assert isinstance(h5dest, h5py.Group)
	if h5dest.get(name):
		del h5dest[name]
