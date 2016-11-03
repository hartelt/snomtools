# coding=utf-8
__author__ = 'hartelt'
"""
This scripts imports tiff files, as generated for example by Terra and the PEEM Camera Software. The methods defined
here will read those files and return the data as a DataSet instances. 3D tiff stacks shall be supported.
"""

import snomtools.data.datasets
import os
import numpy
import tifffile
import re
import snomtools.calcs.units as u


def is_tif(filename):
	"""
	Checks if a filename is a tifffile.

	:param filename: string: The filename.

	:return: Boolean.
	"""
	return os.path.splitext(filename)[1] in [".tiff", ".tif"]


def search_tag(tif, tag_id):
	"""
	Searches for a tag in all pages of a tiff file and returns the first match as

	:param tif: An open TiffFile. See tifffile.TiffFile.

	:param tag_id: String: The ID of the tag to search for.

	:return: The tag, object, instance of tifffile.TiffTag.
	"""
	for page in tif:
		for tag in page.tags.values():
			if tag.name == tag_id:
				return tag
	print("WARNING: Tiff tag not found.")
	return None


def peem_dld_read(filepath):
	"""
	Reads a tif file as generated by Terra when using the DLD. Therefore, the 3D tif dimensions are interpreted as
	time-channel, x and y, with the first two time channels being the sum and the error image, which will be ignored.

	:param filepath: String: The (absolute or relative) path of input file.

	:return: The dataset instance generated from the tif file.
	"""
	# Translate input path to absolute path:
	filepath = os.path.abspath(filepath)
	filebase = os.path.basename(filepath)

	# Read tif file to numpy array. Axes will be (timechannel, x, y):
	infile = tifffile.TiffFile(filepath)
	indata = infile.asarray()

	# Read time binning metadata from tags:
	roi_and_bin_id = "41010"  # as defined by Christian Schneider #define TIFFTAG_ROI_AND_BIN 41010
	tag = search_tag(infile, roi_and_bin_id)
	# roi_and_bin_list = tag.value
	T, St, Tbin = int(tag.value[2]), int(tag.value[5]), int(tag.value[8])
	infile.close()

	# Remove sum and error image:
	realdata = numpy.delete(indata, [0, 1], axis=0)

	# Initialize data for dataset:
	dataarray = snomtools.data.datasets.DataArray(realdata, unit='count', label='counts', plotlabel='Counts')
	if tag:
		# The following commented lines won't work because of Terras irreproducible channel assignment and saving...
		# assert (realdata.shape[0] == round(St / float(Tbin))), \
		# 	"ERROR: Tifffile metadata time binning does not fit to data size."
		# uplim = T+(round(St/float(Tbin)))*Tbin # upper limit calculation
		# So just take the data dimensions... and pray the channels start at the set T value:
		taxis = snomtools.data.datasets.Axis([T + i * Tbin for i in range(realdata.shape[0])], label='channel',
											 plotlabel='Time Channel')
	else:
		taxis = snomtools.data.datasets.Axis(numpy.arange(0, realdata.shape[0]), label='channel',
											 plotlabel='Time Channel')
	# Careful about orientation! This is like a matrix:
	# rows go first and are numbered in vertical direction -> Y
	# columns go last and are numbered in horizontal direction -> X
	xaxis = snomtools.data.datasets.Axis(numpy.arange(0, realdata.shape[1]), unit='pixel', label='y', plotlabel='y')
	yaxis = snomtools.data.datasets.Axis(numpy.arange(0, realdata.shape[2]), unit='pixel', label='x', plotlabel='x')

	# Return dataset:
	return snomtools.data.datasets.DataSet(label=filebase, datafields=[dataarray], axes=[taxis, xaxis, yaxis])


def peem_camera_read(filepath):
	"""
	Reads a tif file as generated by the Camera Software (PCO Camware) when using the Camera. Therefore, the 2D tif
	dimensions are interpreted as x and y.

	:param filepath: String: The (absolute or relative) path of input file.

	:return: The dataset instance generated from the tif file.
	"""
	# Translate input path to absolute path:
	filepath = os.path.abspath(filepath)
	filebase = os.path.basename(filepath)

	# Read tif file to numpy array. Axes will be (timechannel, x, y):
	infile = tifffile.TiffFile(filepath)
	indata = infile.asarray()

	# Initialize data for dataset:
	dataarray = snomtools.data.datasets.DataArray(indata, unit='count', label='counts', plotlabel='Counts')

	# Careful about orientation! This is like a matrix:
	# rows go first and are numbered in vertical direction -> Y
	# columns go last and are numbered in horizontal direction -> X
	xaxis = snomtools.data.datasets.Axis(numpy.arange(0, indata.shape[0]), unit='pixel', label='y', plotlabel='y')
	yaxis = snomtools.data.datasets.Axis(numpy.arange(0, indata.shape[1]), unit='pixel', label='x', plotlabel='x')

	# Return dataset:
	return snomtools.data.datasets.DataSet(label=filebase, datafields=[dataarray], axes=[xaxis, yaxis])


def powerlaw_folder_peem_camera(folderpath, pattern="mW", powerunit=None, powerunitlabel=None):
	"""

	:param folderpath: The (relative or absolute) path of the folders containing the powerlaw measurement series.

	:param pattern: string: A pattern the powers in the filenames are named with. For example in the default case
	"mW", the filename containing '50,2mW' or '50.2mW' or '50.2 mW' would accord to a power of 50.2 milliwatts. The
	power units for the axis quantities are also cast from this pattern if not explicitly given with powerunit.

	:param powerunit: A valid unit string that will be cast as the unit for the power axis values. If not given,
	the pattern parameter will be cast as unit.

	:param powerunitlabel: string: Will be used as the unit for the power axis plotlabel. Can be for example a LaTeX
	siunitx command. If not given, the powerunit parameter will be used.

	:return: The dataset containing the images stacked along a power axis.
	"""
	if powerunit is None:
		powerunit = pattern
	if powerunitlabel is None:
		powerunitlabel = powerunit
	pat = re.compile('(\d*[,|.]?\d+)\s?' + pattern)

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the tif files of the powerlaw:
	powerfiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		found = re.search(pat, filename)
		if found:
			power = float(found.group(1).replace(',', '.'))
			powerfiles[power] = filename

	axlist = []
	datastack = []
	for power in iter(sorted(powerfiles.iterkeys())):
		datastack.append(peem_camera_read(os.path.join(folderpath, powerfiles[power])))
		axlist.append(power)
	powers = u.to_ureg(axlist, powerunit)

	pl = 'Power / ' + powerunitlabel  # Plot label for power axis.
	poweraxis = snomtools.data.datasets.Axis(powers, label='power', plotlabel=pl)

	return snomtools.data.datasets.stack_DataSets(datastack, poweraxis, axis=-1, label="Powerlaw " + folderpath)


def powerlaw_folder_peem_dld(folderpath, pattern="mW", powerunit=None, powerunitlabel=None):
	"""

	:param folderpath: The (relative or absolute) path of the folders containing the powerlaw measurement series.

	:param pattern: string: A pattern the powers in the filenames are named with. For example in the default case
	"mW", the filename containing '50,2mW' or '50.2mW' or '50.2 mW' would accord to a power of 50.2 milliwatts. The
	power units for the axis quantities are also cast from this pattern if not explicitly given with powerunit.

	:param powerunit: A valid unit string that will be cast as the unit for the power axis values. If not given,
	the pattern parameter will be cast as unit.

	:param powerunitlabel: string: Will be used as the unit for the power axis plotlabel. Can be for example a LaTeX
	siunitx command. If not given, the powerunit parameter will be used.

	:return: The dataset containing the images stacked along a power axis.
	"""
	if powerunit is None:
		powerunit = pattern
	if powerunitlabel is None:
		powerunitlabel = powerunit
	pat = re.compile('(\d*[,|.]?\d+)\s?' + pattern)

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the tif files of the powerlaw:
	powerfiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		found = re.search(pat, filename)
		if found:
			power = float(found.group(1).replace(',', '.'))
			powerfiles[power] = filename

	axlist = []
	datastack = []
	for power in iter(sorted(powerfiles.iterkeys())):
		datastack.append(peem_dld_read(os.path.join(folderpath, powerfiles[power])))
		axlist.append(power)
	powers = u.to_ureg(axlist, powerunit)

	pl = 'Power / ' + powerunitlabel  # Plot label for power axis.
	poweraxis = snomtools.data.datasets.Axis(powers, label='power', plotlabel=pl)

	return snomtools.data.datasets.stack_DataSets(datastack, poweraxis, axis=-1, label="Powerlaw " + folderpath)


if False:  # Just for testing...
	test_camera_read = False
	if test_camera_read:
		filename = "14_800nm_Micha_crosspol_ppol320_t-80fs_50µm.tif"
		testdata = peem_camera_read(filename)
		outname = filename.replace('.tif', '.hdf5')
		testdata.saveh5(outname)

	test_plot = False
	if test_plot:
		import snomtools.plots.setupmatplotlib as plt
		import snomtools.plots.datasets

		fig = plt.figure((12, 12), 1200)
		ax = fig.add_subplot(111)
		ax.cla()
		vert = 'y'
		hori = 'x'
		ax.autoscale(tight=True)
		ax.set_aspect('equal')
		snomtools.plots.datasets.project_2d(testdata, ax, axis_vert=vert, axis_hori=hori, data_id='counts')
		plt.savefig(filename="test.png", figures_path=os.getcwd(), transparent=False)

	test_powerlaw = True
	if test_powerlaw:
		plfolder = "Powerlaw"
		pldata = powerlaw_folder_peem_camera(plfolder, powerunitlabel='\\SI{\\milli\\watt}')

	print('done.')
