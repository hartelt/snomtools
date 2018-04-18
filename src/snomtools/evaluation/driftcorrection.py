"""
This file provides driftkorrection for array stacks. It generates the drift vectors via
Crosscorrelation-Methods provided by the OpenCV library

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import cv2 as cv
import numpy as np
import snomtools.data.datasets
import snomtools.data.h5tools
from snomtools.data.tools import iterfy, full_slice

__author__ = 'Benjamin Frisch'

if '-v' in sys.argv:
	verbose = True
else:
	verbose = False


class Drift(object):
	# TODO: Implement usage of more than one DataArray in the DataSet.

	def __init__(self, data=None, precalculated_drift=None, template=None, stackAxisID=None, yAxisID=None, xAxisID=None,
				 subpixel=True, method='cv.TM_CCOEFF_NORMED', template_origin=None, interpolation_order=None):
		"""
		Calculates the correlation of a given 2D template with all slices in a n-D dataset which gets projected onto the
		three axes stackAxis, yAxis, xAxis.
		Different methods and subpixel accuracy are available.

		:param data: n-D Dataset to be driftcorrected.
		:type data: snomtools.data.datasets.Dataset **or** snomtools.data.datasets.ROI

		:param template: A template dataset to match to. This is typically a subset of ROI of :code:`data`.
		:type template: snomtools.data.datasets.Dataset **or** snomtools.data.datasets.ROI

		:param stackAxisID: Axis, along which the template_matching is calculated

		:param yAxisID: ID of the first axis of the image, i.e. y

		:param xAxisID: ID of the second axis of the image, i.e. x

		:param subpixel: Generate subpixel accurate drift vectors

		:param method: Method to calculate the Correlation between template and data. Possible methods:
			'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
			'cv.TM_SQDIFF_NORMED'

		:param template_origin: An origin for the relative drift vectors in the form (y_pixel, x_pixel). If :code:`None`
			is given (the default), the first detected drift vector along stackAxis is used.
		:type template_origin: tuple(int **or** float) of len==2 **or** None

		:param int interpolation_order: An order for the interpolation for the calculation of driftcorrected data.
			See: :func:`scipy.ndimage.interpolation.shift` for details.
		"""
		if data:
			if stackAxisID is None:
				self.dstackAxisID = data.get_axis_index('delay')
			else:
				self.dstackAxisID = data.get_axis_index(stackAxisID)
			if yAxisID is None:
				self.dyAxisID = data.get_axis_index('y')
			else:
				self.dyAxisID = data.get_axis_index(yAxisID)
			if xAxisID is None:
				self.dxAxisID = data.get_axis_index('x')
			else:
				self.dxAxisID = data.get_axis_index(xAxisID)

			# read or guess template
			if template:
				if yAxisID is None:
					tyAxisID = template.get_axis_index('y')
				else:
					tyAxisID = template.get_axis_index(yAxisID)
				if xAxisID is None:
					txAxisID = template.get_axis_index('x')
				else:
					txAxisID = template.get_axis_index(xAxisID)
				self.template = self.extract_templatedata(template, tyAxisID, txAxisID)
			else:
				self.template = self.guess_templatedata(data, self.dyAxisID, self.dxAxisID)

			stackAxisID = data.get_axis_index(stackAxisID)

			# check for external drift vectors
			if precalculated_drift:
				assert len(precalculated_drift) == data.shape[
					self.dstackAxisID], "Number of driftvectors unequal to stack dimension of data"
				assert len(precalculated_drift[0]) == 2, "Driftvector has not dimension 2"
				self.drift = precalculated_drift
			else:
				# process data towards 3d array
				if verbose:
					print("Projecting 3D data...", end=None)
				self.data3D = self.extract_3Ddata(data, self.dstackAxisID, self.dyAxisID, self.dxAxisID)
				if verbose:
					print("...done")

				# for layers along stackAxisID find drift:
				self.drift = self.template_matching_stack(self.data3D.get_datafield(0), self.template, stackAxisID,
														  method=method, subpixel=subpixel)
		else:
			if precalculated_drift:
				assert len(precalculated_drift[0]) == 2, "Driftvector has not dimension 2"
				self.drift = precalculated_drift
			else:
				self.drift = None

		if template_origin is None:
			if self.drift is not None:
				self.template_origin = self.drift[0]
			else:
				self.template_origin = None
		else:
			template_origin = tuple(template_origin)
			assert len(template_origin) == 2, "template_origin has invalid length."
			self.template_origin = template_origin

		self.data = data
		self.subpixel = subpixel
		if interpolation_order is None:
			if self.subpixel:
				self.interpolation_order = 1
			else:
				self.interpolation_order = 0
		else:
			self.interpolation_order = interpolation_order

	@property
	def drift_relative(self):
		return self.as_relative_vectors(self.drift)

	def as_relative_vectors(self, vectors):
		for vec in vectors:
			yield self.relative_vector(vec)

	def relative_vector(self, vector):
		o_y, o_x = self.template_origin
		d_y, d_x = vector
		return (d_y - o_y, d_x - o_x)

	def generate_shiftvector(self, stack_index):
		"""
		Generates the full shift vector according to the shape of self.data (minus the stackAxis) out of the 2d drift
			vectors at a given index along the stackAxis.

		:param int stack_index: An index along the stackAxis

		:return: The 1D array of shift values of length len(self.data)-1.
		:rtype: np.ndarray
		"""
		# Initialize empty shiftvector according to the number of dimensions of the data:
		arr = np.zeros(len(self.data.shape))
		# Get the drift at the index position as a numpy array:
		# drift = np.array(list(self.drift_relative)[stack_index])
		drift = np.array(self.relative_vector(self.drift[stack_index]))
		# Put the negated drift in the corresponding shiftvector places:
		np.put(arr, [self.dyAxisID, self.dxAxisID], -drift)
		return arr

	def __getitem__(self, sel):
		# Get full addressed slice from selection.
		full_selection = full_slice(sel, len(self.data.shape))
		slicebase_wo_stackaxis = np.delete(full_selection, self.dstackAxisID)

		shifted_slice_list = []
		# Iterate over all selected elements along dstackAxis:
		for i in iterfy(np.arange(self.data.shape[self.dstackAxisID])[full_selection[self.dstackAxisID]]):
			# Generate full slice of data to shift, by inserting i into slicebase:
			subset_slice = tuple(np.insert(slicebase_wo_stackaxis, self.dstackAxisID, i))
			# Get shiftvector for the stack element i:
			shift = self.generate_shiftvector(i)
			# Get the shifted data from the Data_Handler method:
			shifted_data = self.data.get_datafield(0).data.shift_slice(subset_slice, shift,
																	   order=self.interpolation_order)
			# Attach data to list:
			shifted_slice_list.append(shifted_data)
		if len(shifted_slice_list) < 2:  # We shifted only a single slice along the stackAxis:
			return shifted_slice_list[0]
		else:  # We shifted several slices, so we have to stack them together again.
			return shifted_slice_list[0].__class__.stack(shifted_slice_list)

	def corrected_data(self, h5target=None):
		"""Return the full driftcorrected dataset."""

		oldda = self.data.get_datafield(0)
		if h5target:
			# Probe HDF5 initialization to optimize buffer size:
			chunk_size = snomtools.data.h5tools.probe_chunksize(shape=self.data.shape)
			min_cache_size = np.prod(self.data.shape, dtype=np.int64) // self.data.shape[self.dstackAxisID] * \
							 chunk_size[
								 self.dstackAxisID] * 4  # 32bit floats require 4 bytes.
			use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 128 MB just to be sure.
			# Initialize data handler to write to:
			dh = snomtools.data.datasets.Data_Handler_H5(unit=str(self.data.datafields[0].units), shape=self.data.shape,
														 chunk_cache_mem_size=use_cache_size)

			# Calculate driftcorrected data and write it to dh:
			if verbose:
				import time
				start_time = time.time()
				print(str(start_time))
				print("Calculating {0} driftcorrected slices...".format(self.data.shape[self.dstackAxisID]))
			# Get full slice for all the data:
			full_selection = full_slice(np.s_[:], len(self.data.shape))
			slicebase_wo_stackaxis = np.delete(full_selection, self.dstackAxisID)
			# Iterate over all elements along dstackAxis:
			for i in range(self.data.shape[self.dstackAxisID]):
				# Generate full slice of data to shift, by inserting i into slicebase:
				subset_slice = tuple(np.insert(slicebase_wo_stackaxis, self.dstackAxisID, i))
				# Get shiftvector for the stack element i:
				shift = self.generate_shiftvector(i)
				if verbose:
					step_starttime = time.time()
				# Get the shifted data from the Data_Handler method:
				shifted_data = self.data.get_datafield(0).data.shift_slice(subset_slice, shift,
																		   order=self.interpolation_order)
				if verbose:
					print('interpolation done in {0:.2f} s'.format(time.time() - step_starttime))
					step_starttime = time.time()
				# Write shifted data to corresponding place in dh:
				dh[subset_slice] = shifted_data
				if verbose:
					print('data written in {0:.2f} s'.format(time.time() - step_starttime))
					tpf = ((time.time() - start_time) / float(i + 1))
					etr = tpf * (self.data.shape[self.dstackAxisID] - i + 1)
					print("Slice {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(i, self.data.shape[
						self.dstackAxisID], etr, tpf))

			# Initialize DataArray with data from dh:
			newda = snomtools.data.datasets.DataArray(dh, label=oldda.label, plotlabel=oldda.plotlabel,
													  h5target=dh.h5target)
		else:
			newda = snomtools.data.datasets.DataArray(self[:], label=oldda.label, plotlabel=oldda.plotlabel)
		# Put all the shifted data and old axes together to new DataSet:
		newds = snomtools.data.datasets.DataSet(self.data.label + " driftcorrected", (newda,), self.data.axes,
												self.data.plotconf, h5target=h5target)
		return newds

	@classmethod
	def template_matching_stack(cls, data, template, stackAxisID, method='cv.TM_CCOEFF_NORMED', subpixel=True,
								threshold=(0, 0.1)):
		"""
		Passes the data of a 3D array along the stackAxis in form of 2D data to the template_matching function

		:param data: 3D dataset

		:param template: 2D template which is used to calculate the correlation

		:param stackAxisID: Axis along which the 2D data is extracted and passed for template_matching

		:param method: Method to calculate the Correlation between template and data. Possible methods: 'cv.TM_CCOEFF',
			'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		:param subpixel: Generate subpixel accurate drift vectors

		:return: List of tuples containing the coordinates of best correlation corrected for values below threshold
		"""
		driftlist = []

		if verbose:
			import time
			start_time = time.time()
			print(str(start_time))
			print("Calculating {0} driftvectors...".format(data.shape[stackAxisID]))

		for i in range(data.shape[stackAxisID]):
			slicebase = [np.s_[:], np.s_[:]]
			slicebase.insert(stackAxisID, i)
			slice_ = tuple(slicebase)
			driftlist.append(cls.template_matching((data.data[slice_]), template, method, subpixel))

			if verbose:
				tpf = ((time.time() - start_time) / float(i + 1))
				etr = tpf * (data.shape[stackAxisID] - i + 1)
				print("vector {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(i, data.shape[stackAxisID], etr,
																					   tpf))

		indexList = cls.findindex(threshold[0], threshold[1], [result[1] for result in driftlist])
		driftlist_corrected = cls.cleanList(indexList, [xydata[0] for xydata in driftlist])
		return driftlist_corrected

	@staticmethod
	def template_matching(data_to_match, template, method='cv.TM_CCOEFF_NORMED', subpixel=True):
		"""
		Uses the openCV matchTemplate function to calculate a correlation between a 2D template array and the 2D data
		array.
		The coordinates of the maximum (or minimum for SQDIFF methods) of the calculated correlation are returned.

		:param data_to_match: 2D dataset

		:param template: 2D template which is used to calculate the correlation

		:param method: Method to calculate the Correlation between template and data. Possible methods: 'cv.TM_CCOEFF',
			'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		:param subpixel: Generate subpixel accurate drift vectors

		:return: Returns the coordinate (y,x) of the correlation maximum referenced to the top left corner of the
			template and the value of the correlation at that point.
		"""
		method = eval(method)

		data_to_match = np.float32(np.array(data_to_match))
		template = np.float32(np.array(template))

		res = cv.matchTemplate(data_to_match, template, method)

		if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
			xCorrValue = res.min()
			min_loc = np.unravel_index(res.argmin(), res.shape)
			if subpixel:
				top_left = Drift.subpixel_peak(min_loc, res)
			else:
				top_left = min_loc
		else:
			xCorrValue = res.max()
			max_loc = np.unravel_index(res.argmax(), res.shape)

			if subpixel:
				top_left = Drift.subpixel_peak(max_loc, res)
			else:
				top_left = max_loc

		return top_left, xCorrValue

	@staticmethod
	def subpixel_peak(max_var, results):
		"""
		Extrapolates the position of a maximum in a 2D array by logarithmical evaluation of it's surrounding values.
		I.e. for x: max_new=max + ( log(f(x-1)) - log(f(x+1)) ) / ( 2*log(f(x-1)) + 2*log(f(x+1)) - 4*log(f(x)) )

		:param max_var: tuple with position of the maximum

		:param results: array with the values in which the subpixel maximum is to be found. Has to contain at least the
			neighbouring elements of max_var.

		:return: tuple of subpixel accurate maximum position
		"""
		y = max_var[0]
		x = max_var[1]
		try:
			y_sub = y \
					+ (np.log(results[y - 1, x]) - np.log(results[y + 1, x])) \
					  / \
					  (2 * np.log(results[y - 1, x]) + 2 * np.log(results[y + 1, x]) - 4 * np.log(results[y, x]))
			x_sub = x + \
					(np.log(results[y, x - 1]) - np.log(results[y, x + 1])) \
					/ \
					(2 * np.log(results[y, x - 1]) - 4 * np.log(results[y, x]) + 2 * np.log(results[y, x + 1]))
		except (IndexError):
			y_sub = y
			x_sub = x
			print('Warning: Subpixel ignored once. Index out of bounds')
		return (y_sub, x_sub)

	@staticmethod
	def extract_3Ddata(data, stackAxisID, yAxisID, xAxisID):
		"""
		Projects the data of a n-D dataset onto the 3D system of stackAxis, yAxis, xAxis.

		:param data:

		:param stackAxisID: Defines the stack axis that is not projected and can be use for template_matching

		:param yAxisID: ID of the first axis of the image, i.e. y

		:param xAxisID: ID of the second axis of the image, i.e. x

		:return: Dataset projected on the defined axes while keeping the stackAxis
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet), \
			"ERROR: No dataset or ROI instance given to extract_3Ddata."
		return data.project_nd(stackAxisID, yAxisID, xAxisID)

	@staticmethod
	def extract_templatedata(data, yAxisID, xAxisID):
		"""
		Transforms the data of a template for further processing in the template_matching method.

		:param data: Raw dataset as quantity with axisID's

		:param yAxisID: ID of the first axis, i.e. y

		:param xAxisID: ID of the second axis, i.e. x

		:return: Data of the whole template projected onto the two defined axes.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to extract_templatedata."

		yAxisID = data.get_axis_index(yAxisID)
		xAxisID = data.get_axis_index(xAxisID)

		return data.project_nd(yAxisID, xAxisID).get_datafield(0)

	@staticmethod
	def guess_templatedata(data, yAxisID, xAxisID):
		"""
		Generates a default template for the template_matching method. The template is generated in the center between
			2/5 and 3/5 of the image.

		:param data: Any dataset as quantity with axisID's

		:param yAxisID: ID of the first axis, i.e. y

		:param xAxisID: ID of the second axis, i.e. x

		:return: Projected ROI of the 2/5 to 3/5 field.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to guess_template."

		yAxisID = data.get_axis_index(yAxisID)
		xAxisID = data.get_axis_index(xAxisID)
		fieldshape = (data.shape[yAxisID], data.shape[xAxisID])
		yl, yr, xl, xr = fieldshape[0] * 2 // 5, fieldshape[0] * 3 // 5, fieldshape[1] * 2 // 5, fieldshape[1] * 3 // 5
		limitlist = {yAxisID: (yl, yr), xAxisID: (xl, xr)}
		roi = snomtools.data.datasets.ROI(data, limitlist, by_index=True)
		return roi.project_nd(yAxisID, xAxisID).get_datafield(0)

	@staticmethod
	def findindex(lowerlim, upperlim, inputlist):
		"""Returns List indices for all elements between lower and upper limit"""
		return [n for n, item in enumerate(inputlist) if lowerlim < item and item < upperlim]

	@staticmethod
	def cleanList(indexes, inputlist):
		"""Substitutes list[i] with [i-1] for all i in indexes"""
		for i in indexes:
			try:
				inputlist[i] = inputlist[i - 1]
			except (IndexError):
				j = i + 1
				while j in indexes:
					j = j + 1
				inputlist[i] = inputlist[j]
			except:
				print('Warning: cleanlist failed for Object ' + str(i))
				pass

		return inputlist


class Terra_maxmap(object):
	def __init__(self, data=None, precalculated_map=None, energyAxisID=None, yAxisID=None, xAxisID=None,
				 method=None, interpolation_order=None, use_meandrift=True):

		if energyAxisID is None:
			self.deAxisID = data.get_axis_index('energy')
		else:
			self.deAxisID = data.get_axis_index(energyAxisID)
		if yAxisID is None:
			self.dyAxisID = data.get_axis_index('y')
		else:
			self.dyAxisID = data.get_axis_index(yAxisID)
		if xAxisID is None:
			self.dxAxisID = data.get_axis_index('x')
		else:
			self.dxAxisID = data.get_axis_index(xAxisID)

		energyAxisID = data.get_axis_index(energyAxisID)

		# check for external drift vectors
		if precalculated_map:
			assert len(precalculated_drift) == data.shape[
				(self.dyAxisID, self.dxAxisID)], "Number of energy shiftvectors unequal to xy dimension of data"
			self.drift = precalculated_drift
		else:
			self.method = method
			self.drift = None
			print('No internal maxima map calculation implemented. Please load precalculated map')
		if use_meandrift:
			# ToDo: check if this clutters
			# calculating the mean Drift for the center half of the image to ignore wrong values at edges
			lower0 = int_(shape(data[:][0])[0] / 4)
			upper0 = int_(shape(data[:][0])[0] * 3 / 4)
			lower1 = int_(shape(data[0][:])[0] / 4)
			upper1 = int_(shape(data[0][:])[0] * 3 / 4)

			self.meanDrift = int_(mean(drift[lower0:upper0][lower1:upper1]))

		self.data = data
		if interpolation_order is None:
			if self.subpixel:
				self.interpolation_order = 1
			else:
				self.interpolation_order = 0
		else:
			self.interpolation_order = interpolation_order

	@property
	def drift_relative(self):
		return self.as_relative_vectors(self.drift)

	def as_relative_vectors(self, vectors):
		for vec in vectors:
			yield self.relative_vector(vec)

	def relative_vector(self, vector):
		o_E = self.meanDrift
		d_E = vector
		return (d_E - o_E)

	def generate_shiftvector(self, stack_index):
		"""
		Generates the full shift vector according to the shape of self.data (minus the stackAxis) out of the 2d drift
			vectors at a given index along the stackAxis.

		:param int stack_index: An index along the stackAxis

		:return: The 1D array of shift values of length len(self.data)-1.
		:rtype: np.ndarray
		"""
		# Initialize empty shiftvector according to the number of dimensions of the data:
		arr = np.zeros(len(self.data.shape))
		# Get the drift at the index position as a numpy array:
		drift = np.array(self.relative_vector(self.drift[stack_index[0], stack_index[1]]))
		# Put the negated drift in the corresponding shiftvector places:
		np.put(arr, [self.dyAxisID, self.dxAxisID], -drift)
		return arr

	def corrected_data(self, h5target=None):
		"""Return the full driftcorrected dataset."""

		oldda = self.data.get_datafield(0)
		if h5target:
			# Probe HDF5 initialization to optimize buffer size:
			chunk_size = snomtools.data.h5tools.probe_chunksize(shape=self.data.shape)
			min_cache_size = np.prod(self.data.shape, dtype=np.int64) // self.data.shape[self.dstackAxisID] * chunk_size[self.dstackAxisID] * 4  # 32bit floats require 4 bytes.
			use_cache_size = min_cache_size + 128 * 1024 ** 2  # Add 128 MB just to be sure.
			# Initialize data handler to write to:
			dh = snomtools.data.datasets.Data_Handler_H5(unit=str(self.data.datafields[0].units), shape=self.data.shape,
														 chunk_cache_mem_size=use_cache_size)

			# Calculate driftcorrected data and write it to dh:
			if verbose:
				import time
				start_time = time.time()
				print(str(start_time))
				print("Calculating {0} driftcorrected slices...".format(self.data.shape[self.dstackAxisID]))
			# Get full slice for all the data:
			full_selection = full_slice(np.s_[:], len(self.data.shape))
			# Delete x and y Axis
			slicebase_wo_stackaxis = np.delete(np.delete(full_selection, self.dxAxisID), self.dyAxisID)
			# Iterate over all elements along dstackAxis:
			for i in range(self.data.shape[self.dyAxisID]):
				for j in range(self.data.shape[self.dxAxisID]):
					# Generate full slice of data to shift, by inserting i for yAxis and j for xAxis position of the energy pixel into slicebase:
					subset_slice = tuple(np.insert(np.insert(slicebase_wo_stackaxis, self.dyAxisID, i)), self.dxAxisID,
										 j)
					# Get shiftvector for the stack element i:
					shift = self.generate_shiftvector(i, j)
					if verbose:
						step_starttime = time.time()
					# Get the shifted data from the Data_Handler method:
					shifted_data = self.data.get_datafield(0).data.shift_slice(subset_slice, shift,
																			   order=self.interpolation_order)
					if verbose:
						print('interpolation done in {0:.2f} s'.format(time.time() - step_starttime))
						step_starttime = time.time()
					# Write shifted data to corresponding place in dh:
					dh[subset_slice] = shifted_data
					if verbose:
						print('data written in {0:.2f} s'.format(time.time() - step_starttime))
						tpf = ((time.time() - start_time) / float(i + 1))
						etr = tpf * (
						self.data.shape[self.dyAxisID] * self.data.shape[self.dxAxisID] - (i + 1) * (j + 1))
						print("Slice {0:d} / {1:d}, Time/slice {3:.2f}s ETR: {2:.1f}s".format(i, self.data.shape[
							self.dyAxisID] * self.data.shape[self.dxAxisID], etr, tpf))

			# Initialize DataArray with data from dh:
			newda = snomtools.data.datasets.DataArray(dh, label=oldda.label, plotlabel=oldda.plotlabel,
													  h5target=dh.h5target)
		else:
			newda = snomtools.data.datasets.DataArray(self[:], label=oldda.label, plotlabel=oldda.plotlabel)
		# Put all the shifted data and old axes together to new DataSet:
		newds = snomtools.data.datasets.DataSet(self.data.label + " maximacorrected", (newda,), self.data.axes,
												self.data.plotconf, h5target=h5target)
		return newds


if __name__ == '__main__':  # Testing...
	# testfolder = "test/Drifttest/new"

	import snomtools.data.imports.tiff as imp
	import snomtools.data.datasets
	import os

	templatefile = "template.tif"
	template = imp.peem_camera_read_camware(templatefile)

	objects = os.listdir('rawdata/')
	rawdatalist = []
	for i in objects:
		if i.endswith("Durchlauf.hdf5"):
			rawdatalist.append(i)

	for run in rawdatalist:
		data = snomtools.data.datasets.DataSet.from_h5file('rawdata/' + run, h5target=run + '_testdata.hdf5',
														   chunk_cache_mem_size=2048 * 1024 ** 2)

		# data = snomtools.data.datasets.stack_DataSets(data, snomtools.data.datasets.Axis([1, 2, 3], 's', 'faketime'))

		data.saveh5()

		driftfile = ('Summenbilder/' + run.replace('.hdf5', '.txt'))

		precal_drift = np.loadtxt(driftfile)
		precal_drift = [tuple(row) for row in precal_drift]

		drift = Drift(data, precalculated_drift=precal_drift, stackAxisID="delay", template=template, subpixel=True,
					  template_origin=(123, 347))

		# Calculate corrected data:
		correcteddata = drift.corrected_data(h5target='Driftcorrected_external/' + run)

		correcteddata.saveh5()

		print("done.")
