"""
This script provides some simple tools for the storage in h5 files.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import psutil
import warnings
import tempfile
import os.path
import sys
import numpy
from snomtools import __package__, __version__
from snomtools.data.tools import find_next_prime

__author__ = 'Michael Hartelt'

# Set default cache size for h5py-cache files. h5py-default is 1024**2 (1 MB)
chunk_cache_mem_size_default = 16 * 1024 ** 2  # 16 MB
chunk_cache_mem_size_tempdefault = 8 * 1024 ** 2  # 8 MB


class File(h5py.File):
	"""
	A h5py.File object with the additional functionality of h5py_cache.File of setting buffer sizes. Uses the value
	chunk_cache_mem_size_default as defined above as buffer size if not given otherwise explicitly.
	"""

	# TODO: Control method for cache size of existing file.

	def __init__(self, name, mode='a', chunk_cache_mem_size=None, w0=0.75, n_cache_chunks=None,
				 **kwargs):
		"""
		The constructor. Apart from calling the parent constructor. It uses code from the h5py_cache package
		(Copyright (c) 2016 Mike Boyle, under MIT license)
		to set the buffer settings for the file to be opened.

		:param str name:

		:param str mode:

		:param **kwargs : dict (as keywords)
			Standard h5py.File arguments, passed to its constructor

		:param int chunk_cache_mem_size:
			Number of bytes to use for the chunk cache.  Defaults to 1024**2 (1MB), which
			is also the default for h5py.File -- though it cannot be changed through the
			standard interface.

		:param float w0: float between 0.0 and 1.0
			Eviction parameter.  Defaults to 0.75.  "If the application will access the
			same data more than once, w0 should be set closer to 0, and if the application
			does not, w0 should be set closer to 1."
			<https://www.hdfgroup.org/HDF5/doc/Advanced/Chunking/>

		:param int n_cache_chunks: int
			Number of chunks to be kept in cache at a time.  Defaults to the (smallest
			integer greater than) the square root of the number of elements that can fit
			into memory.  This is just used for the number of slots (nslots) maintained
			in the cache metadata, so it can be set larger than needed with little cost.
		"""
		# Get default cache size if needed:
		if chunk_cache_mem_size is None:
			chunk_cache_mem_size = chunk_cache_mem_size_default
		# Check if required buffer size is available, and reduce if necessary:
		mem_free = psutil.virtual_memory().available
		if chunk_cache_mem_size >= mem_free:
			mem_use = mem_free - (32 * 1024 ** 2)
			warning_message = (("Required buffer size of {0:d} MB exceeds free memory. \
					  Reducing to {1:d} MB.".format(chunk_cache_mem_size / 1024 ** 2, mem_use / 1024 ** 2)))
			warning_message += "\n Performance might be worse than expected!"
			warnings.warn(warning_message)
			chunk_cache_mem_size = mem_use

		# From h5py_cache.File:
		name = name.encode(sys.getfilesystemencoding())
		open(name, mode).close()  # Just make sure the file exists
		if mode in [m + b for m in ['w', 'w+', 'r+', 'a', 'a+'] for b in ['', 'b']]:
			mode = h5py.h5f.ACC_RDWR
		else:
			mode = h5py.h5f.ACC_RDONLY
		if 'dtype' in kwargs:
			bytes_per_object = numpy.dtype(kwargs['dtype']).itemsize
		else:
			bytes_per_object = numpy.dtype(numpy.float).itemsize  # assume float as most likely
		if n_cache_chunks is None:
			n_cache_chunks = int(numpy.ceil(numpy.sqrt(chunk_cache_mem_size / bytes_per_object)))
		nslots = find_next_prime(100 * n_cache_chunks)
		propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
		settings = list(propfaid.get_cache())
		settings[1:] = (nslots, chunk_cache_mem_size, w0)
		propfaid.set_cache(*settings)

		h5py.File.__init__(self, h5py.h5f.open(name, flags=mode, fapl=propfaid), **kwargs)

	def __del__(self):
		self.__exit__()


class Tempfile(File):
	""" A temporary h5 file with adjustable buffer size."""

	def __init__(self, **kwargs):
		# Make temporary space for file, with tempfile module:
		temp_dir = tempfile.mkdtemp(prefix="snomtools_H5_tempspace-")
		# temp_dir = os.getcwd() # upper line can be replaced by this for debugging.
		temp_file_path = os.path.join(temp_dir, "snomtools_H5_tempspace.hdf5")

		# Handle cache size in kwargs:
		key = "chunk_cache_mem_size"
		if not key in kwargs or kwargs[key] is None:
			kwargs[key] = chunk_cache_mem_size_tempdefault

		# Call parent constructor:
		File.__init__(self, temp_file_path, 'w', **kwargs)

		# Save paths to clean up later:
		self.temp_dir = temp_dir
		self.temp_file_path = temp_file_path

	def __del__(self):
		"""
		Clean up the temporary file.

		"""
		file_to_remove = self.temp_file_path
		self.__exit__()
		os.remove(file_to_remove)
		try:
			os.rmdir(self.temp_dir)
		except OSError as e:
			warnings.warn("Tempfile could not remove tempdir. Propably not empty.")
			print(e)


# TODO: Handle different data types including DataSets in dictionaries.
def store_dictionary(dict_to_store, h5target):
	for key in list(dict_to_store.keys()):
		assert (not ('/' in key)), "Special group separation char '/' used as key."
		write_dataset(h5target, key, dict_to_store[key])


def load_dictionary(h5source):
	outdict = {}
	for key in iter(h5source):
		key = str(key)
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


def read_as_str(h5source):
	"""
	Reads data from a h5 dataset and returns it as string. This is helpful for python2/3 support, to guarantee reading
	a string and not a bytes type. Additionally, it can be used to cast any data stored in h5 into str.

	:param h5source: The dataset to read.
	:type h5source: h5py.Dataset

	:return: The data, converted to str.
	:rtype: str
	"""
	assert isinstance(h5source, h5py.Dataset), "No h5 dataset given."
	data = h5source[()]
	if isinstance(data, bytes):
		data = data.decode()
	return str(data)


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


def check_version(h5root):
	"""
	Checks if the version information in h5root fits the running version of the package. If not, warnings are issued
	accordingly.

	:param h5root: A h5py entity containing version information as a string dataset named 'version'.
	:type h5root: h5py.Group

	:returns: :code:`True` if same version is detected, :code:`False` if not.
	:rtype: bool
	"""
	try:
		version_str = read_as_str(h5root['version'])
		packagename, version = version_str.split()
		if packagename != __package__:
			warnings.warn("Accessed H5 entity {0} was not written with {1}".format(h5root, __package__))
			return False
		if version != __version__:
			warnings.warn(
				"Accessed H5 entity {0} was not written with {1} {2}".format(h5root, __package__, __version__))
			return False
	except Exception as e:
		warnings.warn("No compatible version string in accessed H5 entity {0}".format(h5root))
		return False
	return True


def probe_chunksize(shape, compression="gzip", compression_opts=4):
	"""
	Probe the chunk size that would be guessed by the h5py driver.

	:param tuple shape: A shape tuple.

	:param str compression: Compression mode.

	:param int compression_opts: Compression options.

	:return: The guessed chunk size.
	:rtype: tuple(int)
	"""
	h5target = Tempfile()
	ds = h5target.create_dataset("data", shape, chunks=True, compression=compression, compression_opts=compression_opts)
	chunk_size = ds.chunks
	del h5target
	return chunk_size


if __name__ == "__main__":
	testfile = File('test.hdf5')
	testfile.flush()
	testfile.close()

	cs = probe_chunksize((10, 10, 10))
	print("done")
