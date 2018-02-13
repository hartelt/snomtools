"""
This file provides driftkorrection for array stacks. It generates the drift vectors via
Crosscorrelation-Methods provided by the OpenCV library

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2 as cv
import numpy as np
import snomtools.data.datasets

__author__ = 'Benjamin Frisch'


class Drift:
	def __init__(self, data=None, template=None, stackAxisID=None, yAxisID=None, xAxisID=None,
				 subpixel=True, method='cv.TM_CCOEFF_NORMED'):
		"""
		Calculates the correlation of a given 2D template with all slices in a n-D dataset which gets projected onto the
		three axes stackAxis, yAxis, xAxis.
		Differend methods and subpixel accuracy are available.

		:param data: n-D dataset

		:param template: 2D template array

		:param stackAxisID: Axis, along which the template_matching is calculated

		:param yAxisID: ID of the first axis of the image, i.e. y

		:param xAxisID: ID of the second axis of the image, i.e. x

		:param subpixel: Generate subpixel accurate drift vectors

		:param method: Method to calculate the Correlation between template and data. Possible methods:
			'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
			'cv.TM_SQDIFF_NORMED'
		"""

		# read axis
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

			# process data towards 3d array
			self.data3D = self.extract_3Ddata(data, self.dstackAxisID, self.dyAxisID, self.dxAxisID)

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
		# for layers along stackAxisID find drift:
		self.drift = self.template_matching_stack(self.data3D.get_datafield(0), self.template, stackAxisID,
												  method=method, subpixel=subpixel)
		self.data = data

	@classmethod
	def template_matching_stack(cls, data, template, stackAxisID, method='cv.TM_CCOEFF_NORMED', subpixel=True):
		"""
		Passes the data of a 3D array along the stackAxis in form of 2D data to the template_matching function

		:param data: 3D dataset

		:param template: 2D template which is used to calculate the correlation

		:param stackAxisID: Axis along which the 2D data is extracted and passed for template_matching

		:param method: Method to calculate the Correlation between template and data. Possible methods: 'cv.TM_CCOEFF',
			'cv.TM_CCOEFF_NORMED' (default), 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		:param subpixel: Generate subpixel accurate drift vectors

		:return: List of tuples containing the coordinates of best correlation
		"""
		driftlist = []
		for i in range(data.shape[stackAxisID]):
			slicebase = [np.s_[:], np.s_[:]]
			slicebase.insert(stackAxisID, i)
			slice_ = tuple(slicebase)
			driftlist.append(cls.template_matching((data.data[slice_]), template, method, subpixel))
		return driftlist

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
			template.
		"""
		method = eval(method)

		data_to_match = np.float32(np.array(data_to_match))
		template = np.float32(np.array(template))

		res = cv.matchTemplate(data_to_match, template, method)

		if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
			min_loc = np.unravel_index(res.argmin(), res.shape)
			if subpixel:
				top_left = Drift.subpixel_peak(min_loc, res)
			else:
				top_left = min_loc
		else:
			max_loc = np.unravel_index(res.argmax(), res.shape)
			if subpixel:
				top_left = Drift.subpixel_peak(max_loc, res)
			else:
				top_left = max_loc

		return top_left

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

		y_sub = y \
				+ (np.log(results[y - 1, x]) - np.log(results[y + 1, x])) \
				/ \
				(2 * np.log(results[y - 1, x]) + 2 * np.log(results[y + 1, x]) - 4 * np.log(results[y, x]))
		x_sub = x + \
				(np.log(results[y, x - 1]) - np.log(results[y, x + 1])) \
				/ \
				(2 * np.log(results[y, x - 1]) - 4 * np.log(results[y, x]) + 2 * np.log(results[y, x + 1]))
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


if __name__ == '__main__':  # Testing...
	testfolder = "test/Drifttest/new"

	import snomtools.data.imports.tiff as imp
	import os.path

	files = ["{0:1d}full.tif".format(i) for i in range(1, 4)]
	templatefile = "template.tif"

	data = [imp.peem_camera_read_camware(os.path.join(testfolder, f)) for f in files]
	template = imp.peem_camera_read_camware(os.path.join(testfolder, templatefile))

	data = snomtools.data.datasets.stack_DataSets(data, snomtools.data.datasets.Axis([1, 2, 3], 's', 'faketime'))

	data.saveh5(os.path.join(testfolder, 'testdata.hdf5'))

	drift = Drift(data, template, stackAxisID="faketime", subpixel=False)
	drift2 = Drift(data, template, stackAxisID="faketime", subpixel=True)

	print("done.")
