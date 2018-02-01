__author__ = 'frisch'
''' 
This file provides driftkorrection for any array stack. 
'''

import cv2 as cv
import numpy as np
import snomtools.data.datasets

class Drift:
	def __init__(self,data=None, template=None, stackAxisID = None, xAxisID=None, yAxisID=None,
				 subpixel=True, method = 'cv.TM_CCOEFF_NORMED'):

		# Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		#read axis
		if data:
			if stackAxisID is None:
				stackAxisID = data.get_axis_index('delay')
			else:
				stackAxisID = data.get_axis_index(stackAxisID)
			if xAxisID is None:
				xAxisID = data.get_axis_index('x')
			else:
				xAxisID = data.get_axis_index(xAxisID)
			if yAxisID is None:
				yAxisID = data.get_axis_index('y')
			else:
				yAxisID = data.get_axis_index(yAxisID)

			#process data towards 3d array
			self.data = self.extract_3Ddata(data, stackAxisID, xAxisID, yAxisID)

			#read or guess template
			if template:
				self.template = self.extract_templatedata(template,xAxisID,yAxisID)
			else:
				self.template = self.guess_templatedata(data,xAxisID,yAxisID)

			#for layers along stackAxisID find drift:
			self.drift = self.template_matching_stack(self.data,self.template,stackAxisID)

		pass


	@classmethod
	def template_matching_stack(cls, data, template, stackAxisID, method = 'cv.TM_CCOEFF_NORMED' , subpixel = True):
		driftlist=[]
		for i in range(data.shape[stackAxisID]):
			driftlist.append(cls.template_matching((data[i]),template, method,subpixel))
		return driftlist



	@staticmethod
	def template_matching(array, template, method = 'cv.TM_CCOEFF_NORMED' , subpixel='True'):
		#Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

		method = eval(method)

		template = np.float32(template)
		w, h = template.shape[::-1]

		res = cv.matchTemplate(array, template, method)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

		if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
			if subpixel:
				top_left = Drift.subpixel_peak(min_loc, res)
			else:
				top_left = min_loc
		else:
			if subpixel:
				top_left = Drift.subpixel_peak(max_loc, res)
			else:
				top_left = max_loc

		return top_left

	@staticmethod
	def subpixel_peak(max_var, results):
		x = max_var[0]
		y = max_var[1]
		coord = []

		x_sub = x \
				   + (np.log(results[x - 1, y]) - np.log(results[x + 1, y])) \
					 / \
					 (2 * np.log(results[x - 1, y]) + 2 * np.log(results[x + 1, y]) - 4 * np.log(results[x, y]))
		y_sub = y + \
				   (np.log(results[x, y - 1]) - np.log(results[x, y + 1])) \
				   / \
				   (2 * np.log(results[x, y - 1]) - 4 * np.log(results[x, y]) + 2 * np.log(results[x, y + 1]))
		return (x_sub,y_sub)


	@staticmethod
	def extract_3Ddata(data,stackAxisID, xAxisID, yAxisID):
		assert isinstance(data, snomtools.data.datasets.DataSet), \
			"ERROR: No dataset or ROI instance given to extract_3Ddata."
		return data.project_nd(stackAxisID, xAxisID, yAxisID)


	@staticmethod
	def extract_templatedata(data, xAxisID, yAxisID):
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to extract_templatedata."

		xAxisID = data.get_axis_index(xAxisID)
		yAxisID = data.get_axis_index(yAxisID)
		roi = snomtools.data.datasets.ROI(data,limitlist,by_index=True)
		return roi.project_nd(xAxisID,yAxisID)

	@staticmethod
	def guess_templatedata(data,xAxisID,yAxisID):
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to guess_template."

		xAxisID = data.get_axis_index(xAxisID)
		yAxisID = data.get_axis_index(yAxisID)
		fieldshape = (data.shape[xAxisID],data.shape[yAxisID])
		xl, xr, yl, yr = fieldshape[0]*2/5, fieldshape[0]*3/5, fieldshape[0]2/5, fieldshape[0]*3/5 #seems to work
		limitlist = {xAxisID:(xl,xr),yAxisID:(yl,yr)}
		roi = snomtools.data.datasets.ROI(data,limitlist,by_index=True)
		return roi.project_nd(xAxisID,yAxisID)