# coding=utf-8
"""
This script is a small extension to the already existing script that allows the importing of jpeg files
that were generated by generic cameras. This iteration allows also the import of images of different than jpeg file
types. At it's current version this script supports additionally .bmp files

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.data.datasets as ds
import snomtools.calcs.units as u
from snomtools.data.h5tools import probe_chunksize
import numpy as np
import sys
import os
import datetime
import imageio
import dateutil.parser as dparser

__author__ = 'Michael Hartelt, Martin Mitkov'

if '-v' in sys.argv or __name__ == "__main__":
	verbose = True
else:
	verbose = False


def is_image_file(filename):
	"""

	:param str filename: The filename.

	:rtype: bool
	"""
	return os.path.splitext(filename)[1] in [".jpg", ".jpeg", ".bmp"]


def read_image(filepath):
	"""
	Reads a generic image in the one of the implemented file extensions. The 2D image dimensions are interpreted as x and y.
	Reads only greyscale, if a color (RGB or RGBA) image is given, it will be converted to greyscale.

	:param filepath: String: The (absolute or relative) path of input file.

	:return: The dataset instance generated from the image file.
	"""
	# Translate input path to absolute path:
	filepath = os.path.abspath(filepath)
	filebase = os.path.basename(filepath)

	# Read tif file to numpy array. Axes will be (x, y):
	indata = imageio.imread(filepath, as_gray=True)

	# Initialize data for dataset:
	dataarray = ds.DataArray(indata, unit='dimensionless', label='brightness', plotlabel='Brightness')

	# Careful about orientation! This is like a matrix:
	# rows go first and are numbered in vertical direction -> Y
	# columns go last and are numbered in horizontal direction -> X
	yaxis = ds.Axis(np.arange(0, indata.shape[0]), unit='pixel', label='y', plotlabel='y')
	xaxis = ds.Axis(np.arange(0, indata.shape[1]), unit='pixel', label='x', plotlabel='x')

	# Return dataset:
	return ds.DataSet(label=filebase, datafields=[dataarray], axes=[yaxis, xaxis])


def timelog_folder(folderpath, timeunit='s', timeunitlabel=None,
				   timeformat=None, prefix="", postfix="",
				   h5target=True):
	"""
	:param folderpath: The (relative or absolute) path of the folders containing the measurement series.

	:param: timeunit: Set unit (dimension) of the sequence of images. If not explicitly stated this parameter will have
			dimension of second

	:param: timeunitlabel: Set a label of your given timeunit.If no explicit value is given to this parameter it assumes
			the same value as the time unit

	:param: timefomrat: State the time format of the targeted file(s) (e.g. %d%m%Y). If no format is explicitly stated
			the script will try to guess the time format

	:param: prefix: part of the filename that is BEFORE the timeformat

	:param: postfix: part of the file name that is AFTER the timeformat

	:param: h5target: optional, set a h5 group or path on which the data is being saved on (NOT SURE)

	:return: The dataset containing the images stacked along a time axis.

	"""
	if timeunitlabel is None:
		timeunitlabel = timeunit

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the image files:
	timefiles = {}
	for filename in filter(is_image_file, os.listdir(folderpath)):
		# Strip extension, prefix, postfix:
		timestring = os.path.splitext(filename)[0]
		timestring = timestring.lstrip(prefix)
		timestring = timestring.rstrip(postfix)

		if timeformat:  # If format is given, parse accordingly:
			timestring = timestring.strip()
			imgtime = datetime.datetime.strptime(timestring, timeformat)
		else:  # Else try to parse as best as guessable:
			imgtime = dparser.parse(filename, fuzzy=True)
		timefiles[imgtime] = filename

	# Build time axis:
	axlist = []
	starttime = min(timefiles.keys())
	for imgtime in iter(sorted(timefiles.keys())):
		axlist.append((imgtime - starttime).total_seconds())
	times = u.to_ureg(axlist, 'second').to(timeunit)
	pl = 'Time / ' + timeunitlabel  # Plot label for power axis.
	timeaxis = ds.Axis(times, label='time', plotlabel=pl)

	# ----------------------Create dataset------------------------
	# Test data size:
	sample_data = read_image(os.path.join(folderpath, timefiles[list(timefiles.keys())[0]]))
	axlist = [timeaxis] + sample_data.axes
	newshape = timeaxis.shape + sample_data.shape
	# Build the data-structure that the loaded data gets filled into
	if h5target:
		chunks = True
		compression = 'gzip'
		compression_opts = 4

		# Probe HDF5 initialization to optimize buffer size:
		if chunks is True:  # Default is auto chunk alignment, so we need to probe.
			chunk_size = probe_chunksize(shape=newshape, compression=compression, compression_opts=compression_opts)
		else:
			chunk_size = chunks
		min_cache_size = chunk_size[0] * np.prod(sample_data.shape) * 4  # 32bit floats require 4 bytes.
		use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 64 MB just to be sure.

		# Initialize full DataSet with zeroes:
		dataspace = ds.Data_Handler_H5(unit=sample_data.get_datafield(0).get_unit(),
									   shape=newshape, chunks=chunks,
									   compression=compression,
									   compression_opts=compression_opts,
									   chunk_cache_mem_size=use_cache_size)
		dataarray = ds.DataArray(dataspace,
								 label=sample_data.get_datafield(0).get_label(),
								 plotlabel=sample_data.get_datafield(0).get_plotlabel(),
								 h5target=dataspace.h5target,
								 chunks=chunks,
								 compression=compression, compression_opts=compression_opts,
								 chunk_cache_mem_size=use_cache_size)
		dataset = ds.DataSet("Powerlaw " + folderpath, [dataarray], axlist, h5target=h5target,
							 chunk_cache_mem_size=use_cache_size)
	else:
		# In-memory data processing without h5 files.
		dataspace = u.to_ureg(np.zeros(newshape), sample_data.datafields[0].get_unit())
		dataarray = ds.DataArray(dataspace,
								 label=sample_data.get_datafield(0).get_label(),
								 plotlabel=sample_data.get_datafield(0).get_plotlabel(),
								 h5target=None)
		dataset = ds.DataSet("Powerlaw " + folderpath, [dataarray], axlist, h5target=h5target)
	dataarray = dataset.get_datafield(0)

	# ----------------------Fill dataset------------------------
	# Fill in data from imported tiffs:
	slicebase = tuple([np.s_[:] for j in range(len(sample_data.shape))])

	if verbose:
		import time
		print("Reading Time Series Folder of shape: ", dataset.shape)
		if h5target:
			print("... generating chunks of shape: ", dataset.get_datafield(0).data.ds_data.chunks)
			print("... using cache size {0:d} MB".format(use_cache_size // 1024 ** 2))
		else:
			print("... in memory")
		start_time = time.time()
	for i, imgtime in zip(list(range(len(timefiles))), iter(sorted(timefiles.keys()))):
		islice = (i,) + slicebase
		# Import image:
		idata = read_image(os.path.join(folderpath, timefiles[imgtime]))

		# Check data consistency:
		assert idata.shape == sample_data.shape, "Trying to combine scan data with different shape."
		for ax1, ax2 in zip(idata.axes, sample_data.axes):
			assert ax1.units == ax2.units, "Trying to combine scan data with different axis dimensionality."
		assert idata.get_datafield(0).units == sample_data.get_datafield(0).units, \
			"Trying to combine scan data with different data dimensionality."

		# Write data:
		dataarray[islice] = idata.get_datafield(0).data
		if verbose:
			tpf = ((time.time() - start_time) / float(i + 1))
			etr = tpf * (dataset.shape[0] - i + 1)
			print("image {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(i, dataset.shape[0], etr, tpf))

	return dataset
