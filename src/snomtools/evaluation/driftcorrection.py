__author__ = 'frisch'
''' 
This file provides driftkorrection for any array stack. 
'''

import cv2 as cv
import numpy as np
import snomtools.data.datasets


class Drift:
	def __init__(self, data=None, template=None, stackAxisID=None, yAxisID=None, xAxisID=None,
				 subpixel=True, method='cv.TM_CCOEFF_NORMED'):

		# Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

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
		driftlist = []
		for i in range(data.shape[stackAxisID]):
			slicebase = [np.s_[:], np.s_[:]]
			slicebase.insert(stackAxisID, i)
			slice_ = tuple(slicebase)
			driftlist.append(cls.template_matching((data.data[slice_]), template, method, subpixel))
		return driftlist

	@staticmethod
	def template_matching(data_to_match, template, method='cv.TM_CCOEFF_NORMED', subpixel=True):
		# Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		method = eval(method)

		data_to_match = np.float32(np.array(data_to_match))
		template = np.float32(np.array(template))

		res = cv.matchTemplate(data_to_match, template, method)

		if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
			min_loc = np.unravel_index(res.argmin(),res.shape)
			if subpixel:
				top_left = Drift.subpixel_peak(min_loc, res)
			else:
				top_left = min_loc
		else:
			max_loc = np.unravel_index(res.argmax(),res.shape)
			if subpixel:
				top_left = Drift.subpixel_peak(max_loc, res)
			else:
				top_left = max_loc

		return top_left

	@staticmethod
	def subpixel_peak(max_var, results):
		y = max_var[0]
		x = max_var[1]
		coord = []

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
		assert isinstance(data, snomtools.data.datasets.DataSet), \
			"ERROR: No dataset or ROI instance given to extract_3Ddata."
		return data.project_nd(stackAxisID, yAxisID, xAxisID)

	@staticmethod
	def extract_templatedata(data, yAxisID, xAxisID):
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to extract_templatedata."

		yAxisID = data.get_axis_index(yAxisID)
		xAxisID = data.get_axis_index(xAxisID)

		return data.project_nd(yAxisID, xAxisID).get_datafield(0)

	@staticmethod
	def guess_templatedata(data, yAxisID, xAxisID):
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to guess_template."

		yAxisID = data.get_axis_index(yAxisID)
		xAxisID = data.get_axis_index(xAxisID)
		fieldshape = (data.shape[yAxisID], data.shape[xAxisID])
		yl, yr, xl, xr = fieldshape[0] * 2 / 5, fieldshape[0] * 3 / 5, fieldshape[1] * 2 / 5, fieldshape[1] * 3 / 5
		limitlist = {yAxisID: (yl, yr),xAxisID: (xl, xr)}
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

	drift = Drift(data, template, stackAxisID="faketime", subpixel=True)

	print "done."
