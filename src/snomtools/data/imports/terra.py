# coding=utf-8
"""
This scripts imports the file formats defined by the Terra measurement software.
The Tiff files that are saved by terra can be imported by the methods provided in snomtools.data.imports.tiff, imported
here as tiff for convenience.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy
import sys
import snomtools.data.datasets as ds
import snomtools.data.imports.tiff as tiff

__author__ = 'Michael Hartelt'

if '-v' in sys.argv or __name__ == "__main__":
	verbose = True
else:
	verbose = False


def hist_asc(source, T_start=None, T_bin=1, tif_probe=None):
	"""
	Reads an DLD energy channel histogram, saved in a file with the extension ".hist.asc".

	:param str source: The path of the source file.

	:param int T_start: The start channel of the chosen time binning. By default, the first channel containing counts
		is taken.

	:param int T_bin: The binning of the chosen time binning.

	:param str tif_probe: A tif that was saved at the same time (or with the same settings) as the histogram to read,
		typically when executing "save all" in Terra. This deactivates *T_start* and *T_bin* and reads the binning from
		the	tags in the tiff file instead.

	:return: The imported data.
	:rtype: snomtools.data.datasets.DataSet
	"""
	filepath = os.path.abspath(source)
	filebase = os.path.basename(filepath)

	if tif_probe is not None:
		# Read tif probe file:
		infile = tiff.tifffile.TiffFile(tif_probe)

		# Read time binning metadata from tags:
		roi_and_bin_id = "41010"  # as defined by Christian Schneider #define TIFFTAG_ROI_AND_BIN 41010
		tag = tiff.search_tag(infile, roi_and_bin_id)
		# roi_and_bin_list = tag.value
		T_start, St, T_bin = int(tag.value[2]), int(tag.value[5]), int(tag.value[8])
		infile.close()

	# Read the "HistoXplusY" column from the .asc file to an array:
	count_data = numpy.loadtxt(filepath, dtype=int, skiprows=1, usecols=2)
	# Trim the trailing zeroes:
	count_data = numpy.trim_zeros(count_data, 'b')

	# If no start channel is given, guess it by taking the first non-zero entry, taking the binning into account.
	if not tif_probe and T_start is None:
		start_index = numpy.nonzero(count_data)[0][0]
		T_start = start_index * T_bin

	# Trim the leading zeroes:
	count_data = numpy.trim_zeros(count_data)

	# Initialize Channel axis and Count DataArray
	taxis = ds.Axis([T_start + i * T_bin for i in range(count_data.shape[0])], label='channel',
					plotlabel='Time Channel')
	dataarray = ds.DataArray(count_data, unit='count', label='counts', plotlabel='Counts')

	# Return DataSet:
	return ds.DataSet(label=filebase, datafields=[dataarray], axes=[taxis])


def convert_comma(s):
	# The function that converts the string with comma as decimal seperator to float
	s = s.strip().replace(',', '.')
	return float(s)


def load_maxima_map(file):
	data = []
	with open(file, 'r') as f:
		for l in f:
			numberstr = l.split('\t')
			data.append([convert_comma(s) for s in numberstr if s != ''])
	return data
