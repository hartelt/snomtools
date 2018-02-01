__author__ = 'frisch'
''' 
This file provides driftkorrection for any array stack. 
'''

import cv2 as cv
import numpy as np
import snomtools.data.datasets

class Drift:
	def __init__(self,data=None, template=None, stackAxisID = None, subpixel=True, method = 'cv.TM_CCOEFF_NORMED'):
		# Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
		if data:
			if template:

				self.template = self.extract_data(template)
				self.imageplane = self.data.get_axis()

				self.data = self.extract_3Ddata(data,imageplane, stackAxisID)

				#for layers along stackAxisID:
				self.drift = self.template_matching(self.data,self.template)

		# from template get 2D axis


		pass  # dataset zu numpyarray für driftkorrektur von 2D arrays in der dim1_axisID dim2_axisID Ebene entlang stack_axisID Achse

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
	def extract_data(data, data_id=0, axis_id=None, label="powerlaw"):
		"""
		Extracts the powers and intensities out of a dataset. Therefore, it takes the power axis of the input data,
		and projects the datafield onto that axis by summing over all the other axes.

		:param data: Dataset containing the powerlaw data.

		:param data_id: Identifier of the DataField to use.

		:param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
			to a Power in its physical dimension is taken.

		:param label: string: label for the produced DataSet

		:return: 1D-DataSet with projected Intensity Data and Power Axis.
		"""
		assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
			"ERROR: No dataset or ROI instance given to Powerlaw data extraction."
		if axis_id is None:
			power_axis = data.get_axis_by_dimension("watts")
		else:
			power_axis = data.get_axis(axis_id)
		count_data = data.get_datafield(data_id)
		power_axis_index = data.get_axis_index(power_axis.get_label())
		count_data_projected = count_data.project_nd(power_axis_index)
		count_data_projected = snomtools.data.datasets.DataArray(count_data_projected, label='counts')
		# Normalize by subtracting dark counts:
		count_data_projected_norm = count_data_projected - count_data_projected.min()
		count_data_projected_norm.set_label("counts_normalized")
		# Initialize the DataSet containing only the projected powerlaw data;
		return snomtools.data.datasets.DataSet(label, [count_data_projected_norm, count_data_projected], [power_axis])

	@staticmethod
	def extract_3Ddata(data, imageplane, stackAxisID = None, label="powerlaw"):
		pass #return 3D Dataset for template matching
