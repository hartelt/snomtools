__author__ = 'frisch'
''' 
This file provides driftkorrection for any array stack. 
'''

import cv2 as cv
import numpy as np


def do_drift(dataset, axisID, template, methode='cv.TM_CCORR_NORMED', subpixel='True'):
	pass  # dataset zu numpyarray für driftkorrektur entlang axisID Achse als Stack


def template_matching(array, template, methode='cv.TM_CCORR_NORMED', subpixel='True'):
	method = eval(methode)

	template = np.float32(template)
	w, h = template.shape[::-1]

	res = cv.matchTemplate(array, template, method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	if methode in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		if subpixel:
			top_left = subpixel(min_loc)
		else:
			top_left = min_loc
	else:
		if subpixel:
			top_left = subpixel(max_loc)
		else:
			top_left = max_loc

	return top_left


def subpixel(max_var, results):
	x = max_var[0]
	y = max_var[1]
	coord = []

	coord[0] = x \
			   + (np.log(results[x - 1, y]) - np.log(results[x + 1, y])) \
				 / \
				 (2 * np.log(results[x - 1, y]) + 2 * np.log(results[x + 1, y] - 4 * np.log(results[x, y])))

	coord[1] = y + \
			   (np.log(results[x, y - 1]) - np.log(results[x, y + 1])) \
			   / \
			   (2 * np.log(results[x, y - 1]) - 4 * np.log(results[x, y]) + 2 * np.log(results[x, y + 1]))

	return (coord)
