__author__ = 'frisch'
''' 
This file provides driftkorrection for any array stack. 
'''

import cv2 as cv
import numpy as np


def calculate_drift(dataset, dim1_axisID, dim2_axisID, stack_axisID, template, method = 'cv.TM_CCORR_NORMED', subpixel = 'True'):
	# Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

	pass  # dataset zu numpyarray für driftkorrektur von 2D arrays in der dim1_axisID dim2_axisID Ebene entlang stack_axisID Achse


def template_matching(array, template, method = 3 , subpixel='True'):
	#Methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'


	method = eval(methode)

	template = np.float32(template)
	w, h = template.shape[::-1]

	res = cv.matchTemplate(array, template, method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	if methode in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		if subpixel:
			top_left = subpixel_peak(min_loc, res)
		else:
			top_left = min_loc
	else:
		if subpixel:
			top_left = subpixel_peak(max_loc, res)
		else:
			top_left = max_loc

	return top_left


def subpixel_peak(max_var, results):
	x = max_var[0]
	y = max_var[1]
	coord = []

	temp = x \
			   + (np.log(results[x - 1, y]) - np.log(results[x + 1, y])) \
				 / \
				 (2 * np.log(results[x - 1, y]) + 2 * np.log(results[x + 1, y]) - 4 * np.log(results[x, y]))
	coord.append(temp)
	temp = y + \
			   (np.log(results[x, y - 1]) - np.log(results[x, y + 1])) \
			   / \
			   (2 * np.log(results[x, y - 1]) - 4 * np.log(results[x, y]) + 2 * np.log(results[x, y + 1]))
	coord.append(temp)
	return (coord)
