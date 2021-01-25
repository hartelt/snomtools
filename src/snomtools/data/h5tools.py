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
from packaging import version
from snomtools import __package__, __version__
from snomtools.data.tools import find_next_prime
import snomtools.calcs.units as u

__author__ = 'Michael Hartelt'

H5PY_OUTDATED = version.parse(h5py.__version__) < version.parse('2.9.0')
if H5PY_OUTDATED:
    warnings.warn("You seem to be using an older version of h5py. Please update to h5py>=2.9.0!")

# Set default cache size for h5py-cache files. h5py-default is 1024**2 (1 MB)
chunk_cache_mem_size_default = 16 * 1024 ** 2  # 16 MB
chunk_cache_mem_size_tempdefault = 8 * 1024 ** 2  # 8 MB


class File(h5py.File):
    """
    A h5py.File object with the additional functionality of h5py_cache.File of setting buffer sizes. Uses the value
    :code:`chunk_cache_mem_size_default` as defined above as buffer size if not given otherwise explicitly.
    """

    def __init__(self, name, mode='a',
                 chunk_cache_mem_size=None, w0=0.75, n_cache_chunks=None,
                 **kwargs):
        """
        Constructs a h5py.File object with some additional functionality.
        A custom-size chunk cache is used, given as a parameter, with a global default.
        The available system memory is checked and the cache size is reduced if not enough memory is available,
        in this case a warning is thrown.
        Code from the h5py_cache package (Copyright (c) 2016 Mike Boyle, under MIT license)
        was used in earlier versions to set access parameters for low-level h5 file to be opened.
        This functionality was merged to the main h5py package in version 2.9, so the native usage is prefered.
        Therefore h5py_cache code is only used to optimize some cache access parameters for performance, now.

        The naming of the parameters is kept to the old snomtools scheme for backwards compatibility,
        but the new h5py-style parameters `rdcc_nbytes`, `rdcc_w0` and `rdcc_nslots` can also be used as kwargs,
        overwriting the snomtools-style parameters accordingly.

        See https://docs.h5py.org/en/2.9.0rc1/high/file.html# for the h5py documentation.

        :param str name: Name of file (bytes or str),
            or an instance of h5f.FileID to bind to an existing file identifier,
            or a file-like object.

        :param str mode: Mode in which to open file; one of (`'w'`, `'r'`, `'r+'`, `'a'`, `'w-'`).

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
        # Parse new h5py kwargs to enable h5py-like behaviour:
        chunk_cache_mem_size = kwargs.pop('rdcc_nbytes', chunk_cache_mem_size)
        w0 = kwargs.pop('rdcc_w0', w0)
        n_cache_chunks = kwargs.pop('rdcc_nslots', n_cache_chunks)

        # Get default cache size if needed:
        if chunk_cache_mem_size is None:
            chunk_cache_mem_size = chunk_cache_mem_size_default
        # Check if required buffer size is available, and reduce if necessary:
        mem_free = psutil.virtual_memory().available
        if chunk_cache_mem_size >= mem_free:
            mem_use = mem_free - (32 * 1024 ** 2)
            warning_message = (("Required buffer size of {0:f} MB exceeds free memory. \
					  Reducing to {1:f} MB.".format(chunk_cache_mem_size / 1024 ** 2, mem_use / 1024 ** 2)))
            warning_message += "\n Performance might be worse than expected!"
            warnings.warn(warning_message)
            chunk_cache_mem_size = mem_use

        # Chunk cache address table optimization from h5py_cache.File:
        if 'dtype' in kwargs:
            bytes_per_object = numpy.dtype(kwargs['dtype']).itemsize
        else:
            bytes_per_object = numpy.dtype(numpy.float).itemsize  # assume float as most likely
        if n_cache_chunks is None:
            n_cache_chunks = int(numpy.ceil(numpy.sqrt(chunk_cache_mem_size / bytes_per_object)))
        nslots = find_next_prime(100 * n_cache_chunks)

        # Initialize file with h5py:
        h5py.File.__init__(self, name, mode,
                           rdcc_nbytes=chunk_cache_mem_size, rdcc_w0=w0, rdcc_nslots=nslots,
                           **kwargs)

    def get_PropFAID(self):
        """
        Retrieve a copy of the file access property list which manages access to this file.

        :returns: File access properties.
        :rtype: h5py.h5p.PropFAID
        """
        return self.id.get_access_plist()

    def get_cache_params(self):
        """
        Gets the caching parameters from the File Access Properties.
        See :func:`File.__init__` for details about those parameters.

        :return: A tuple of length 4 containing the cache parameters (0, nslots, chunk_cache_mem_size, w0)
        :rtype: tuple
        """
        return self.get_PropFAID().get_cache()

    def get_chunk_cache_mem_size(self):
        """
        Gets the chunk_cache size from the File Access Properties out of the low-level API.

        :return: The chunk cache size in bytes.
        :rtype: int
        """
        return self.get_cache_params()[2]

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
            warnings.warn("Tempfile could not remove tempdir. Probably not empty.")
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


def store_quantity(h5grp, name, q, chunks=True, compression="gzip", compression_opts=4):
    """
    Stores a pint quantity in an open HDF5 file. Because a quantity consists of a magnitude and a unit, a group
    containing the two is created instead of just creating a dataset.

    :param h5grp: The h5 group to store the data in.
    :type h5grp: h5py.Group

    :param str name: The name (key) under which the data is stored.

    :param q: The Quantity to store.
    :type q: Quantity

    :param bool chunks: (Optional) Bool flag to enable chunked data storage.

    :param str compression: (Optional) Specify compression mode for chunked data storage.

    :param compression_opts: (Optional) Define compression options according to used compression mode.
    """
    if not chunks:
        compression = None
        compression_opts = None
    assert isinstance(h5grp, h5py.Group)
    assert isinstance(q, u.Quantity)
    if not hasattr(q, 'shape') or q.shape == ():
        chunks = False
        compression = None
        compression_opts = None
    clear_name(h5grp, name)
    qgrp = h5grp.create_group(name)
    qgrp.create_dataset("data", data=q.magnitude, chunks=chunks, compression=compression,
                        compression_opts=compression_opts)
    qgrp.create_dataset("unit", data=str(q.units))


def load_quantity(h5grp, name=None):
    """
    Loads a pint quantity from a group in an open HDF5 file.

    :param h5grp: The h5 group to store the data in.
    :type h5grp: h5py.Group

    :param str name: (Optional) If given, data is not read directly from h5grp, but from a member with key :code:`name`.

    :return: The read Quantity.
    :rtype: Quantity
    """
    assert isinstance(h5grp, h5py.Group)
    if name is not None:
        h5grp = h5grp[name]
    try:
        magnitude = h5grp['data'][:]
    except ValueError as e:
        magnitude = h5grp['data'][()]
    return u.to_ureg(magnitude, h5grp['unit'][()])


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


def read_dataset(h5source):
    """
    Reads a full dataset, which can be scalar or array-like.

    :param h5source: The dataset to read.
    :type h5source: h5py.Dataset

    :return: The data read from the dataset. The type depends on what was stored there.
    """
    assert isinstance(h5source, h5py.Dataset), "No h5py dataset given."
    if h5source.shape == ():
        return h5source[()]
    else:
        return h5source[:]


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


def clean_group(h5group, validitems):
    """
    Deletes all items in an h5 group that are not listed in validitems.

    :param h5group: The h5py group in which to clean entries.
    :type h5group: h5py.Group

    :param validitems: Valid identifiers of items to keep.
    :type validitems: list of strings
    """
    assert isinstance(h5group, h5py.Group)
    for key in h5group.keys():
        if key not in validitems:
            del h5group[key]


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


def probe_chunksize(shape, dtype=numpy.float32, compression="gzip", compression_opts=4):
    """
    Probe the chunk size that would be guessed by the h5py driver.

    :param tuple shape: A shape tuple.

    :param dtype: The data type.

    :param str compression: Compression mode.

    :param int compression_opts: Compression options.

    :return: The guessed chunk size.
    :rtype: tuple(int)
    """
    h5target = Tempfile()
    ds = h5target.create_dataset("data", shape, dtype=dtype,
                                 chunks=True, compression=compression, compression_opts=compression_opts)
    chunk_size = ds.chunks
    del h5target
    return chunk_size


if __name__ == "__main__":
    testfile = File('test.hdf5')
    cc_size = testfile.get_chunk_cache_mem_size()

    testarray = numpy.arange(9).reshape((3, 3))
    testquantity = u.to_ureg(testarray, 'meter')
    store_quantity(testfile, 'moep', testquantity)

    loadtest = load_quantity(testfile, 'moep')

    testfile.flush()
    testfile.close()

    cs = probe_chunksize((10, 10, 10))
    print("done")
