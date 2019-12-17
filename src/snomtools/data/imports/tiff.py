# coding=utf-8
"""
This scripts imports tiff files, as generated for example by Terra and the PEEM Camera Software. The methods defined
here will read those files and return the data as a DataSet instances. 3D tiff stacks shall be supported.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.data.datasets
import os
import numpy
import tifffile
import re
import warnings
import sys
import snomtools.calcs.units as u
from snomtools.data.h5tools import probe_chunksize

__author__ = 'Michael Hartelt'

if '-v' in sys.argv or __name__ == "__main__":
	verbose = True
else:
	verbose = False

terra_tag_ids = {
	"peem_settings": "41000",
	"roi_and_bin": "41010",
	"exposure_time": "41020",
	"usercomment": "41030",
	"excitation": "41040",
	"date": "41041",
	"time": "41042",
	"author": "41043",
	"probe": "41044",
	"delaystage": "41045",
	"data_device": "41046",
	"delay_ist": "41050",
	"delay_soll": "41051",
	"devices": "41052",
	"device_values": "41053",
	"version": "41055",
	"peem_ini": "41060",
	"artist": "Artist"
}
terra_tag_descriptions = {
	"peem_settings": "The PEEM Settings, saved in an Array of numerical values.",
	"roi_and_bin": "ROI and Binnings set for the data acquisition.",
	"exposure_time": "The exposure time set for the integration of the single PEEM image in milliseconds.",
	"usercomment": "A comment describing the measurement, set freely by the PEEM user.",
	"excitation": "The light source used for the photoemission measured with PEEM.",
	"date": "The date of the measurement, formatted DD.MM.YYYY",
	"time": "The time of the start of the measurement, formatted HH:MM:SS (24h)",
	"author": "The operator of the experiment that logged into TERRA.",
	"probe": "The name of the sample that was measured.",
	"delaystage": "The delay stage used for the scan in which the image was taken.",
	"data_device": "The device used for detecting the image.",
	"delay_ist": "The delay value for the image, as set in the delay list.",
	"delay_soll": "The actual delay value for the image, which can vary due to the step resolution of the delay stage.",
	"devices": "(?)",
	"device_values": "(?)",
	"version": "A version number. Propably of the TERRA software (?)",
	"peem_ini": "The PEEM ini file read from the PEEM control software buffer, which contains all nominal and actual PEEM settings.",
	"artist": "The image artist and copyright owner of the image."
}


# TODO: Handle keeping of sum images in metadata for terra import.
# TODO: Save metadata from Terra tifftags.

def is_tif(filename):
	"""
	Checks if a filename is a tifffile.

	:param str filename: The filename.

	:rtype: bool
	"""
	return os.path.splitext(filename)[1] in [".tiff", ".tif"]


def search_tag(tif, tag_id):
	"""
	Searches for a tag in all pages of a tiff file and returns the first match as

	:param TiffFile tif: An open TiffFile. See tifffile.TiffFile.

	:param str tag_id: The ID of the tag to search for.

	:return: The tag object.
	:rtype: tifffile.TiffTag
	"""
	try:  # For older versions of tifffile, TiffFile objects are iterable and pages can be adressed directly.
		for page in tif:
			for tag in list(page.tags.values()):
				if tag.name == tag_id:
					return tag
	except TypeError as e:  # In newer versions of tifffile, tags are stored in a dict.
		return tif.pages._keyframe.tags[tag_id]
	warnings.warn("Tiff tag not found.")
	return None


def peem_dld_read(filepath, mode="terra"):
	"""
	Reads a time-resolved dld dataset. Shadows the different readin functions for the different measurement programs
	and returns the generated DataSets.

	:param filepath: String: The (absolute or relative) path of the input file or folder.

	:param mode: String: The readin mode. Valid options are "terra" for a tiff generated with Terra (default),
		"pne" for a folder of tiffs generated with the Focus ProNanoESCA Software.

	:return: The generated DataSet.
	"""
	if mode == "terra":
		return peem_dld_read_terra(filepath)
	elif mode == "pne":
		raise NotImplementedError("ProNanoESCA DLD tiff readin not yet implemented.")
	# TODO: Implement Focus ProNanoESCA readin.
	elif mode == "dldgui":
		raise NotImplementedError("dldgui tiff readin not yet implemented.")
	# TODO: Implement dldgui readin.
	else:
		raise ValueError("Unrecognized readin mode given to peem_dld_read")


def peem_dld_read_terra(filepath):
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
	indata = infile.asarray(numpy.s_[2:])  # slices 0,1 contain sum and error image, so we don't need those.

	# Read time binning metadata from tags:
	roi_and_bin_id = "41010"  # as defined by Christian Schneider #define TIFFTAG_ROI_AND_BIN 41010
	tag = search_tag(infile, roi_and_bin_id)
	# roi_and_bin_list = tag.value
	T, St, Tbin = int(tag.value[2]), int(tag.value[5]), int(tag.value[8])
	infile.close()

	# Initialize data for dataset:
	dataarray = snomtools.data.datasets.DataArray(indata, unit='count', label='counts', plotlabel='Counts')
	if tag:
		# The following commented lines won't work because of Terras irreproducible channel assignment and saving...
		# assert (realdata.shape[0] == round(St / float(Tbin))), \
		# 	"ERROR: Tifffile metadata time binning does not fit to data size."
		# uplim = T+(round(St/float(Tbin)))*Tbin # upper limit calculation
		# So just take the data dimensions... and pray the channels start at the set T value:
		taxis = snomtools.data.datasets.Axis([T + i * Tbin for i in range(indata.shape[0])], label='channel',
											 plotlabel='Time Channel')
	else:
		taxis = snomtools.data.datasets.Axis(numpy.arange(0, indata.shape[0]), label='channel',
											 plotlabel='Time Channel')
	# Careful about orientation! This is like a matrix:
	# rows go first and are numbered in vertical direction -> Y
	# columns go last and are numbered in horizontal direction -> X
	xaxis = snomtools.data.datasets.Axis(numpy.arange(0, indata.shape[1]), unit='pixel', label='y', plotlabel='y')
	yaxis = snomtools.data.datasets.Axis(numpy.arange(0, indata.shape[2]), unit='pixel', label='x', plotlabel='x')

	# Return dataset:
	return snomtools.data.datasets.DataSet(label=filebase, datafields=[dataarray], axes=[taxis, xaxis, yaxis])


def peem_dld_read_terra_sumimage(filepath):
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
	sumdata = infile.asarray(0)
	infile.close()

	# Initialize data for dataset:
	sumdataarray = snomtools.data.datasets.DataArray(sumdata, unit='count', label='counts', plotlabel='Counts')
	# Careful about orientation! This is like a matrix:
	# rows go first and are numbered in vertical direction -> Y
	# columns go last and are numbered in horizontal direction -> X
	xaxis = snomtools.data.datasets.Axis(numpy.arange(0, sumdata.shape[0]), unit='pixel', label='y', plotlabel='y')
	yaxis = snomtools.data.datasets.Axis(numpy.arange(0, sumdata.shape[1]), unit='pixel', label='x', plotlabel='x')

	# Build dataset:
	ds_sum = snomtools.data.datasets.DataSet(label="sumimage " + filebase, datafields=[sumdataarray],
											 axes=[xaxis, yaxis])
	return ds_sum


def peem_camera_read(filepath, mode="camware"):
	"""
	Reads a PEEM image dataset aquired with the camera. Shadows the different readin functions for the different
	measurement programs and returns the generated DataSets.

	:param filepath: String: The (absolute or relative) path of the input file.

	:param mode: String: The readin mode. Valid options are "terra" for a tiff generated with Terra,
		"camware" for a tiff generated with the PCO CamWare Software (default) or "pne" for a tiff generated with the
		Focus ProNanoESCA Software.

	:return: The generated DataSet.
	"""
	if mode == "camware":
		return peem_camera_read_camware(filepath)
	if mode == "terra":
		return (filepath)
	elif mode == "pne":
		raise NotImplementedError("ProNanoESCA DLD tiff readin not yet implemented.")
	# TODO: Implement Focus ProNanoESCA readin.
	else:
		raise ValueError("Unrecognized readin mode given to peem_dld_read")


def peem_camera_read_camware(filepath):
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


def peem_camera_read_terra(filepath):
	"""
	Reads a PEEM image aquired with the Camera and saved with Terra. For now, this just shadows the
	peem_camera_read_camware function, because they work the same way, this can be changed in the future,
	maybe to read addidional metadata generated by Terra.

	:param filepath: String: The (absolute or relative) path of the input file.

	:return: The generated DataSet.
	"""
	return peem_camera_read_camware(filepath)


def opo_folder_peem_camera(folderpath, pattern="", waveunit='nm', waveunitlabel='nm'):
	"""

	:param folderpath: The (relative or absolute) path of the folders containing the OPO wavelength measurement series.

	:param pattern: string: A pattern the wavelengths in the filenames are named with. For example in the default case
		"nm", the filename containing '500nm' or '500 nm' would accord to a wavelength of 500nm. The
		OPO wavelength units for the axis quantities are also cast from this pattern if not explicitly given with waveunit.

	:param waveunit: A valid unit string that will be cast as the unit for the OPO wavelength axis values. If not given,
		the pattern parameter will be cast as unit.

	:param waveunitlabel: string: Will be used as the unit for the power axis plotlabel. Can be for example a LaTeX
		siunitx command. If not given, the waveunit parameter will be used.

	:return: The dataset containing the images stacked along a power axis.
	"""
	if waveunit is None:
		waveunit = pattern
	if waveunitlabel is None:
		waveunitlabel = waveunit
	pat = re.compile('(\d*[,|.]?\d+)\s?' + pattern)

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the tif files of the OPO series:
	wavefiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		found = re.search(pat, filename)
		if found:
			wave = float(found.group(1).replace(',', '.'))
			wavefiles[wave] = filename

	axlist = []
	datastack = []
	for wave in iter(sorted(wavefiles.keys())):
		datastack.append(peem_camera_read(os.path.join(folderpath, wavefiles[wave])))
		axlist.append(wave)
	waves = u.to_ureg(axlist, waveunit)

	pl = 'Wavelength / ' + waveunitlabel  # Plot label for wavelength axis.
	waveaxis = snomtools.data.datasets.Axis(waves, label='wave', plotlabel=pl)

	return snomtools.data.datasets.stack_DataSets(datastack, waveaxis, axis=-1,
												  label="OPO Wavelength scan " + folderpath)


def powerlaw_folder_peem_camera(folderpath, pattern="mW", powerunit=None, powerunitlabel=None, decimal=None):
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
	if decimal is None:
		pat = re.compile('(\d*[,|.]?\d+)\s?' + pattern)
	else:
		pat = re.compile('(\d*[{0:s}]?\d+)\s?'.format(decimal) + pattern)

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the tif files of the powerlaw:
	powerfiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		found = re.search(pat, filename)
		if found:
			if decimal is None:
				power = float(found.group(1).replace(',', '.'))
			else:
				power = float(found.group(1).replace(decimal, '.'))
			powerfiles[power] = filename

	axlist = []
	datastack = []
	for power in iter(sorted(powerfiles.keys())):
		datastack.append(peem_camera_read(os.path.join(folderpath, powerfiles[power])))
		axlist.append(power)
	powers = u.to_ureg(axlist, powerunit)

	pl = 'Power / ' + powerunitlabel  # Plot label for power axis.
	poweraxis = snomtools.data.datasets.Axis(powers, label='power', plotlabel=pl)

	return snomtools.data.datasets.stack_DataSets(datastack, poweraxis, axis=-1, label="Powerlaw " + folderpath)


def powerlaw_folder_peem_dld(folderpath, pattern="mW", powerunit=None, powerunitlabel=None, h5target=False,
							 sum_only=False,
							 norm_to_exptime=False):
	"""

	:param folderpath: The (relative or absolute) path of the folders containing the powerlaw measurement series.

	:param pattern: string: A pattern the powers in the filenames are named with. For example in the default case
		"mW", the filename containing '50,2mW' or '50.2mW' or '50.2 mW' would accord to a power of 50.2 milliwatts. The
		power units for the axis quantities are also cast from this pattern if not explicitly given with powerunit.

	:param powerunit: A valid unit string that will be cast as the unit for the power axis values. If not given,
		the pattern parameter will be cast as unit.

	:param powerunitlabel: string: Will be used as the unit for the power axis plotlabel. Can be for example a LaTeX
		siunitx command. If not given, the powerunit parameter will be used.

	:param h5target: The HDF5 target to write to.
	:type h5target: str **or** h5py.Group **or** True, *optional*

	:param sum_only: If True, only sum images will be read instead of full energy resolved data. *default: False*

	:param norm_to_exptime: If True, counts will be divided by exposure time in seconds. The exposure time will be
		taken out of the filename. This helps to make powerlaws with measurements with diffrent exposure times. 

	:type sum_only: bool, *optional*

	:return: The dataset containing the images stacked along a power axis.
	"""
	# ----------------------Create Axis------------------------
	if powerunit is None:
		powerunit = pattern
	if powerunitlabel is None:
		powerunitlabel = powerunit
	pat = re.compile('(\d*[,|.]?\d+)\s?' + pattern)

	if norm_to_exptime:
		# Defining regex for getting aquisition time
		tunitm = "m"
		tunits = "s"
		tunith = "h"
		tunm = re.compile('(\d*[,|.]?\d+)\s?' + tunitm + '[^a-hj-zA-Z]')
		tuns = re.compile('(\d*[,|.]?\d+)\s?' + tunits)
		tunh = re.compile('(\d*[,|.]?\d+)\s?' + tunith)

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for the tif files of the powerlaw:
	powerfiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		foundp = re.search(pat, filename)
		if foundp:
			power = float(foundp.group(1).replace(',', '.'))

			if norm_to_exptime:
				# calculate aquisition time in seconds and add it to powerfiles list as norming factor
				foundm = re.search(tunm, filename)
				founds = re.search(tuns, filename)
				foundh = re.search(tunh, filename)
				exp_time = 0
				if foundh:
					hour = float(foundh.group(1).replace(',', '.'))
					exp_time = 60 * 60 * hour
				if foundm:
					minute = float(foundm.group(1).replace(',', '.'))

					exp_time = exp_time + 60 * minute

				if founds:
					second = float(founds.group(1).replace(',', '.'))

					exp_time = exp_time + second
				if exp_time == 0:
					print('WARNING in norm_to_exptime. Exptime 0 detected for power ' + str(power))
				powerfiles[power] = [filename, u.to_ureg(exp_time, 's')]
			else:
				powerfiles[power] = [filename]

	# Generate power axis:
	pl = 'Power / ' + powerunitlabel
	axlist = []
	for powerstep in iter(sorted(powerfiles.keys())):
		axlist.append(powerstep)
	powers = u.to_ureg(axlist, powerunit)
	poweraxis = snomtools.data.datasets.Axis(powers, label='power', plotlabel=pl)

	if sum_only:
		sample_data = peem_dld_read_terra_sumimage(os.path.join(folderpath, powerfiles[list(powerfiles.keys())[0]][
			0]))
		if norm_to_exptime:
			sample_data.datafields[0] = sample_data.datafields[0] / u.to_ureg(1,'s')
	else:
		sample_data = peem_dld_read_terra(os.path.join(folderpath, powerfiles[list(powerfiles.keys())[0]][
			0]))
		if norm_to_exptime:
			sample_data.datafields[0] = sample_data.datafields[0] / u.to_ureg(1,'s')

	# ----------------------Create dataset------------------------
	# Test data size:
	axlist = [poweraxis] + sample_data.axes
	newshape = poweraxis.shape + sample_data.shape
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
		min_cache_size = chunk_size[0] * numpy.prod(sample_data.shape) * 4  # 32bit floats require 4 bytes.
		use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 64 MB just to be sure.

		# Initialize full DataSet with zeroes:
		dataspace = snomtools.data.datasets.Data_Handler_H5(unit=sample_data.get_datafield(0).get_unit(),
															shape=newshape, chunks=chunks,
															compression=compression, compression_opts=compression_opts,
															chunk_cache_mem_size=use_cache_size)
		dataarray = snomtools.data.datasets.DataArray(dataspace,
													  label=sample_data.get_datafield(0).get_label(),
													  plotlabel=sample_data.get_datafield(0).get_plotlabel(),
													  h5target=dataspace.h5target,
													  chunks=chunks,
													  compression=compression, compression_opts=compression_opts,
													  chunk_cache_mem_size=use_cache_size)
		dataset = snomtools.data.datasets.DataSet("Powerlaw " + folderpath, [dataarray], axlist, h5target=h5target,
												  chunk_cache_mem_size=use_cache_size)
	else:
		# In-memory data processing without h5 files.
		dataspace = numpy.zeros(newshape)
		dataarray = snomtools.data.datasets.DataArray(dataspace,
													  label=sample_data.get_datafield(0).get_label(),
													  plotlabel=sample_data.get_datafield(0).get_plotlabel(),
													  h5target=None)
		dataset = snomtools.data.datasets.DataSet("Powerlaw " + folderpath, [dataarray], axlist, h5target=h5target)

	dataarray = dataset.get_datafield(0)

	# ----------------------Fill dataset------------------------
	# Fill in data from imported tiffs:
	slicebase = tuple([numpy.s_[:] for j in range(len(sample_data.shape))])

	if verbose:
		import time
		print("Reading Powerlaw Folder of shape: ", dataset.shape)
		if h5target:
			print("... generating chunks of shape: ", dataset.get_datafield(0).data.ds_data.chunks)
			print("... using cache size {0:d} MB".format(use_cache_size // 1024 ** 2))
		else:
			print("... in memory")
		start_time = time.time()
	for i, power in zip(list(range(len(powerfiles))), iter(sorted(powerfiles.keys()))):
		islice = (i,) + slicebase
		# Import tiff:
		if sum_only:
			idata = peem_dld_read_terra_sumimage(os.path.join(folderpath, powerfiles[power][0]))
		else:
			idata = peem_dld_read_terra(os.path.join(folderpath, powerfiles[power][0]))

		if norm_to_exptime:
			idata.get_datafield(0).data = idata.get_datafield(0).data / powerfiles[power][1]

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
			print("tiff {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(i, dataset.shape[0], etr, tpf))

	return dataset


def tr_folder_peem_camera_terra(folderpath, delayunit="um", delayfactor=0.2, delayunitlabel=None,
								h5target=True, **kwargs):
	"""
	Imports a Terra time scan folder that was measured by scanning the time steps in an interferometer while using
	the Camera.

	:param str folderpath: The path of the folder containing the scan data. Example: "/path/to/measurement/1. Durchlauf"

	:param str delayunit: A valid unit string, according to the physical dimension that was scanned over. Typically,
		this is:
		 	* :code:`"um" Micrometers for the normal Interferometer.
		 	* :code:`"as"` Attoseconds for the phase resolved interferometer.

	:param float delayfactor: A factor that the numbers in the filenames need to be multiplied with to get the
		real value of the scan point in units of delayunit. Typically for a time resolved measurement, this is
		:code:`0.2` due to a decimal in the Terra file names, plus pulse delays are twice the stage position difference.

	:param str delayunitlabel: A label for the delay axis. For example :code:`"\\si{\\atto\\second}"` if it's a PSI
		measurement and plotting will be done with TeX typesetting.

	:param h5target: The HDF5 target to write to.
	:type h5target: str **or** h5py.Group **or** True, *optional*

	:return: Imported DataSet.
	:rtype: DataSet
	"""
	if len(kwargs):
		warnings.warn("Unrecognized (propably depreciated) keyword args used in tr_folder_peem_dld_terra!",
					  DeprecationWarning)

	if delayunitlabel is None:
		delayunitlabel = delayunit
	pl = 'Pulse Delay / ' + delayunitlabel  # Plot label for time axis
	return measurement_folder_peem_terra(folderpath, "camera", "D", delayunit, delayfactor, "delay", pl, h5target)


def tr_psi_folder_peem_camera_terra(folderpath, h5target=True):
	"""
	Convenience shortcut method for PSI scans with Camera. Calls tr_folder_peem_camera_terra with correct parameters.
	See: :func:`tr_folder_peem_camera_terra`
	"""
	return tr_folder_peem_camera_terra(folderpath, 'as', 0.2, "\\si{\\atto\\second}", h5target)


def tr_normal_folder_peem_camera_terra(folderpath, h5target=True):
	"""
	Convenience shortcut method for normal interferometer scans with Camera. Calls tr_folder_peem_camera_terra with
	correct parameters.
	See: :func:`tr_folder_peem_camera_terra`
	"""
	return tr_folder_peem_camera_terra(folderpath, 'um', 0.2, "\\si{\\micro\\meter}", h5target)


def tr_folder_peem_dld_terra(folderpath, delayunit="um", delayfactor=0.2, delayunitlabel=None,
							 h5target=True, sum_only=False, **kwargs):
	"""
	Imports a Terra time scan folder that was measured by scanning the time steps in an interferometer while using
	the Delaylinedetector (DLD).

	:param str folderpath: The path of the folder containing the scan data. Example: "/path/to/measurement/1. Durchlauf"

	:param str delayunit: A valid unit string, according to the physical dimension that was scanned over. Typically,
		this is:
		 	* :code:`"um" Micrometers for the normal Interferometer.
		 	* :code:`"as"` Attoseconds for the phase resolved interferometer.

	:param float delayfactor: A factor that the numbers in the filenames need to be multiplied with to get the
		real value of the scan point in units of delayunit. Typically for a time resolved measurement, this is
		:code:`0.2` due to a decimal in the Terra file names, plus pulse delays are twice the stage position difference.

	:param str delayunitlabel: A label for the delay axis. For example :code:`"\\si{\\atto\\second}"` if it's a PSI
		measurement and plotting will be done with TeX typesetting.

	:param h5target: The HDF5 target to write to.
	:type h5target: str **or** h5py.Group **or** True, *optional*

	:param sum_only: If True, only sum images will be read instead of full energy resolved data. *default: False*
	:type sum_only: bool, *optional*

	:return: Imported DataSet.
	:rtype: DataSet
	"""
	if len(kwargs):
		warnings.warn("Unrecognized (propably depreciated) keyword args used in tr_folder_peem_dld_terra!",
					  DeprecationWarning)

	if delayunitlabel is None:
		delayunitlabel = delayunit
	pl = 'Pulse Delay / ' + delayunitlabel  # Plot label for time axis
	if sum_only:
		return measurement_folder_peem_terra(folderpath, "dld-sum", "D", delayunit, delayfactor, "delay", pl, h5target)
	else:
		return measurement_folder_peem_terra(folderpath, "dld", "D", delayunit, delayfactor, "delay", pl, h5target)


def tr_psi_folder_peem_dld_terra(folderpath, h5target=True, sum_only=False):
	"""
	Convenience shortcut method for PSI scans with the DLD. Calls tr_folder_peem_camera_terra with correct parameters.
	See: :func:`tr_folder_peem_dld_terra`
	"""
	return tr_folder_peem_dld_terra(folderpath, 'as', 0.2, "\\si{\\atto\\second}", h5target, sum_only=sum_only)


def tr_normal_folder_peem_dld_terra(folderpath, h5target=True, sum_only=False):
	"""
	Convenience shortcut method for normal interferometer scans with DLD. Calls tr_folder_peem_camera_terra with
	correct parameters.
	See: :func:`tr_folder_peem_dld_terra`
	"""
	return tr_folder_peem_dld_terra(folderpath, 'um', 0.2, "\\si{\\micro\\meter}", h5target, sum_only=sum_only)


def rotationmount_folder_peem_camera_terra(folderpath, h5target=True):
	"""
	Convenience shortcut method for rotation mount scans with camera. Calls measurement_folder_peem_terra with
	correct parameters.
	See: :func:`measurement_folder_peem_terra`
	"""
	pl = 'Rotation Mount Angle / \\si{\\degree}'  # Plot label for time axis
	return measurement_folder_peem_terra(folderpath, "camera", "R", "deg", 0.1, "angle", pl, h5target)


def rotationmount_folder_peem_dld_terra(folderpath, h5target=True, sum_only=False):
	"""
	Convenience shortcut method for rotation mount scans with DLD. Calls measurement_folder_peem_terra with
	correct parameters.
	See: :func:`measurement_folder_peem_terra`
	"""
	pl = 'Rotation Mount Angle / \\si{\\degree}'  # Plot label for rotation angle axis.
	if sum_only:
		return measurement_folder_peem_terra(folderpath, "dld-sum", "R", "deg", 0.1, "angle", pl, h5target)
	else:
		return measurement_folder_peem_terra(folderpath, "dld", "R", "deg", 0.1, "angle", pl, h5target)


def dummy_folder_peem_camera_terra(folderpath, h5target=True):
	"""
	Convenience shortcut method for dummy device scans with camera. Calls measurement_folder_peem_terra with
	correct parameters.
	See: :func:`measurement_folder_peem_terra`
	"""
	pl = 'Dummy Index'  # Plot label for time axis
	return measurement_folder_peem_terra(folderpath, "camera", "N", "", 1, "dummyaxis", pl, h5target)


def dummy_folder_peem_dld_terra(folderpath, h5target=True, sum_only=False):
	"""
	Convenience shortcut method for dummy device scans with dld. Calls measurement_folder_peem_terra with
	correct parameters.
	See: :func:`measurement_folder_peem_terra`
	"""
	pl = 'Dummy Index'  # Plot label for time axis
	if sum_only:
		return measurement_folder_peem_terra(folderpath, "dld-sum", "N", "", 1, "dummyaxis", pl, h5target)
	else:
		return measurement_folder_peem_terra(folderpath, "dld", "N", "", 1, "dummyaxis", pl, h5target)


def measurement_folder_peem_terra(folderpath, detector="dld", pattern="D", scanunit="um", scanfactor=1,
								  scanaxislabel="scanaxis", scanaxispl=None, h5target=True):
	"""
	The base method for importing terra scan folders. Covers all scan possibilities, so far only in 1D scans.

	:param str folderpath: The path of the folder containing the scan data. Example: "/path/to/measurement/1. Durchlauf"

	:param str detector: Read mode corresponding to the used detector.
		Valid inputs:
			* :code:`"dld"` for reading the energy-resolved data out of dld tiffs.
			* :code:`"dld-sum"` for reading the sum image out of dld tiffs.
			* :code:`"camera"`

	:param str pattern: The pattern in the filenames that indicates the scan enumeration and is followed by the number
		indicating the position, according to the device used for the scan. Terra uses:
			* :code:`"D"` for a delay stage
			* :code:`"R"` for the rotation mount
			* :code:`"N"` for a dummy device.

	:param str scanunit: A valid unit string, according to the physical dimension that was scanned over.

	:param float scanfactor: A factor that the numbers in the filenames need to be multiplied with to get the
		real value of the scan point in units of scanunit.
		This would be for example:
			* :code:`0.2` (because of stage position) with delayunit :code:`"um"` for normal Interferometer because \
			one decimal is in filenames.
			* :code:`0.2` (because of stage position and strange factor 10 in filenames) with delayunit :code:`"as"` \
			for PR interferometer

	:param str scanaxislabel: A label for the axis of the scan.

	:param str scanaxispl: A plot label for the axis of the scan.

	:param h5target: The HDF5 target to write to.
	:type h5target: str **or** h5py.Group **or** True, *optional*

	:return: Imported DataSet.
	:rtype: DataSet
	"""
	assert detector in ["dld", "dld-sum", "camera"], "Invalid detector mode."
	if scanaxispl is None:
		scanaxispl = 'Scan / ' + scanunit

	# Compile regex for file detection:
	pat = re.compile(pattern + "(-?\d*).tif")

	# Translate input path to absolute path:
	folderpath = os.path.abspath(folderpath)

	# Inspect the given folder for time step files:
	scanfiles = {}
	for filename in filter(is_tif, os.listdir(folderpath)):
		found = re.search(pat, filename)
		if found:
			scanstep = float(found.group(1))
			scanfiles[scanstep] = filename

	# Generate delay axis:
	axlist = []
	for scanstep in iter(sorted(scanfiles.keys())):
		axlist.append(scanstep)
	scanvalues = u.to_ureg(numpy.array(axlist) * scanfactor, scanunit)
	scanaxis = snomtools.data.datasets.Axis(scanvalues, label=scanaxislabel, plotlabel=scanaxispl)

	# Test data size:
	if detector == "dld":
		sample_data = peem_dld_read_terra(os.path.join(folderpath, scanfiles[list(scanfiles.keys())[0]]))
	elif detector == "dld-sum":
		sample_data = peem_dld_read_terra_sumimage(os.path.join(folderpath, scanfiles[list(scanfiles.keys())[0]]))
	else:
		sample_data = peem_camera_read_terra(os.path.join(folderpath, scanfiles[list(scanfiles.keys())[0]]))
	axlist = [scanaxis] + sample_data.axes
	newshape = scanaxis.shape + sample_data.shape

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
		min_cache_size = chunk_size[0] * numpy.prod(sample_data.shape) * 4  # 32bit floats require 4 bytes.
		use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 64 MB just to be sure.

		# Initialize full DataSet with zeroes:
		dataspace = snomtools.data.datasets.Data_Handler_H5(unit=sample_data.get_datafield(0).get_unit(),
															shape=newshape, chunks=chunks,
															compression=compression, compression_opts=compression_opts,
															chunk_cache_mem_size=use_cache_size)
		dataarray = snomtools.data.datasets.DataArray(dataspace,
													  label=sample_data.get_datafield(0).get_label(),
													  plotlabel=sample_data.get_datafield(0).get_plotlabel(),
													  h5target=dataspace.h5target,
													  chunks=chunks,
													  compression=compression, compression_opts=compression_opts,
													  chunk_cache_mem_size=use_cache_size)
		dataset = snomtools.data.datasets.DataSet("Terra Scan " + folderpath, [dataarray], axlist, h5target=h5target,
												  chunk_cache_mem_size=use_cache_size)
	else:
		# In-memory data processing without h5 files.
		dataspace = numpy.zeros(newshape)
		dataarray = snomtools.data.datasets.DataArray(dataspace,
													  label=sample_data.get_datafield(0).get_label(),
													  plotlabel=sample_data.get_datafield(0).get_plotlabel(),
													  h5target=None)
		dataset = snomtools.data.datasets.DataSet("Terra Scan " + folderpath, [dataarray], axlist, h5target=h5target)

	dataarray = dataset.get_datafield(0)

	# Fill in data from imported tiffs:
	slicebase = tuple([numpy.s_[:] for j in range(len(sample_data.shape))])

	if verbose:
		import time
		print("Reading Terra Scan Folder of shape: ", dataset.shape)
		if h5target:
			print("... generating chunks of shape: ", dataset.get_datafield(0).data.ds_data.chunks)
			print("... using cache size {0:d} MB".format(use_cache_size // 1024 ** 2))
		else:
			print("... in memory")
		start_time = time.time()

	for i, scanstep in zip(list(range(len(scanfiles))), iter(sorted(scanfiles.keys()))):
		islice = (i,) + slicebase
		# Import tiff:
		if detector == "dld":
			idata = peem_dld_read_terra(os.path.join(folderpath, scanfiles[scanstep]))
		elif detector == "dld-sum":
			idata = peem_dld_read_terra_sumimage(os.path.join(folderpath, scanfiles[scanstep]))
		else:
			idata = peem_camera_read_terra(os.path.join(folderpath, scanfiles[scanstep]))
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
			print("tiff {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(i, dataset.shape[0], etr, tpf))

	return dataset


# if True:  # Just for testing...
if __name__ == "__main__":
	testdata = None

	test_camera_read = False
	if test_camera_read:
		testfilename = "14_800nm_Micha_crosspol_ppol320_t-80fs_50µm.tif"
		testdata = peem_camera_read(testfilename)
		outname = testfilename.replace('.tif', '.hdf5')
		testdata.saveh5(outname)

	test_plot = False
	if test_plot and testdata:
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

	test_powerlaw = False
	if test_powerlaw:
		plfolder = "Powerlaw"
		pldata = powerlaw_folder_peem_camera(plfolder, powerunitlabel='\\SI{\\milli\\watt}')

	test_timeresolved = False
	if test_timeresolved:
		trfolder = "terra-dummy-dld"
		trdata = dummy_folder_peem_dld_terra(trfolder, h5target=trfolder + '.hdf5')
		trdata.saveh5()

		trfolder = "terra-rotationmount-dld"
		trdata = rotationmount_folder_peem_dld_terra(trfolder, h5target=trfolder + '.hdf5')
		trdata.saveh5()

		trfolder = "terra-tr-psi-camera"
		trdata = tr_psi_folder_peem_camera_terra(trfolder, h5target=trfolder + '.hdf5')
		trdata.saveh5()

		trfolder = "terra-tr-psi-dld"
		trdata = tr_psi_folder_peem_dld_terra(trfolder, h5target=trfolder + '.hdf5')
		trdata.saveh5()

		trfolder = "terra-tr-normal-dld"
		trdata = tr_normal_folder_peem_dld_terra(trfolder, h5target=trfolder + '.hdf5')
		trdata.saveh5()

		trfolder = "terra-tr-normal-dld"
		trdata = tr_normal_folder_peem_dld_terra(trfolder, h5target=trfolder + '_sum.hdf5', sum_only=True)
		trdata.saveh5()

	test_opo_measurement = True
	if test_opo_measurement:
		trfolder = "D:/Messdaten/2018/20181205 a-SiH on ZnO/01 OPO NI"
		trdata = opo_folder_peem_camera(trfolder)

	print('done.')
