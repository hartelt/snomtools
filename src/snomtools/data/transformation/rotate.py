'''
This script holds transformation functions for arrays.
Rotation and scaling is possible with output of biggest possible rectangle of valid data
'''

import numpy as np
import scipy.ndimage


def rotate_cropped(data, angle):
	'''
	Take data and calls rotate, calculates center square with actual data, crops it
	:param data: raw 2D data array
	:param angle: rotaton angle in deg
	:return: Rotated and cropped image
	'''

	cache = scipy.ndimage.interpolation.rotate(data, angle=angle, reshape=False, output=None,
											   order=1,
											   mode='constant', cval=np.nan, prefilter=False)
	w, h = rotatedRectWithMaxArea(data.shape[0], data.shape[1], np.radians(angle))
	return crop_around_center(cache, w, h)


def crop_around_center(image, width, height):
	"""
	Given a NumPy / OpenCV 2 image, crops it to the given width and height,
	around it's centre point
	"""

	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

	if (width > image_size[0]):
		width = image_size[0]

	if (height > image_size[1]):
		height = image_size[1]

	x1 = int(image_center[0] - width * 0.5) + 1
	x2 = int(image_center[0] + width * 0.5) - 1
	y1 = int(image_center[1] - height * 0.5) + 1
	y2 = int(image_center[1] + height * 0.5) - 1

	return image[y1:y2, x1:x2]


def rotatedRectWithMaxArea(w, h, angle):
	"""
	Given a rectangle of size wxh that has been rotated by 'angle' (in
	radians), computes the width and height of the largest possible
	axis-aligned rectangle (maximal area) within the rotated rectangle.
	np.radians(angle) for deg->rad as input
	Based on Coproc Stackoverflow https://stackoverflow.com/a/16778797/8654672
	"""
	if w <= 0 or h <= 0:
		return 0, 0

	width_is_longer = w >= h
	# side_long, side_short = (w,h) if width_is_longer else (h,w)
	side_long, side_short = (w, h)

	# since the solutions for angle, -angle and 180-angle are all the same,
	# if suffices to look at the first quadrant and the absolute values of sin,cos:

	sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))

	if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:

		# half constrained case: two crop corners touch the longer side,
		#   the other two corners are on the mid-line parallel to the longer line

		x = 0.5 * side_short
		wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)

	else:
		# fully constrained case: crop touches all 4 sides
		cos_2a = cos_a * cos_a - sin_a * sin_a
		wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

	return wr, hr


def scale_data(data, zoomfactor):
	"""
	Scales data with scipy.ndimage.zoom. Keep in mind that this 'interpolates' data onto a new grid, which introduces
	'blurring'.

	:param data: The input array.
	:param zoomfactor: Factor of new vs old data size.
	:return: The scaled array.
	"""
	return scipy.ndimage.zoom(data, zoomfactor, output=None, order=1,
							  mode='constant', cval=0.0, prefilter=False)


def rot_scale_data(data, angle_settings=(0, 0, 0), scale_settings=(1, 0, 0), digits=5, output_dir=None,
				   saveImg=False):
	'''
	Rotate and scale data, save resulting data in array
	:param data:
	:param angle_settings:	tuple (center, resolution, number of variations)
	:param scale_settings:	tuple (center, resolution, number of variations)
	:param digits: number of digits to round the variations to
	:param output_dir: save target
	:param saveImg: save all rotated and scaled images
	:return:
	'''

	angle_centervalue = angle_settings[0]
	angle_res = angle_settings[1]
	angle_variations = angle_settings[2]

	scale_centervalue = scale_settings[0]
	scale_res = scale_settings[1]
	scale_variations = scale_settings[2]

	zeroangle = angle_centervalue - angle_variations / 2 * angle_res
	zeroscale = scale_centervalue - scale_variations / 2 * scale_res

	rot_crop_data = {
		(round(zeroangle + i * angle_res, digits), round(zeroscale + j * scale_res, digits)): scale_data(
			rotate_cropped(data, round(zeroangle + i * angle_res, digits)),
			round(zeroscale + j * scale_res, digits),
			round(zeroangle + i * angle_res, digits), output_dir, saveImg=saveImg)
		for i in range(angle_variations + 1) for j in range(scale_variations + 1)}
	return rot_crop_data


def scale_rotated(data, zoomfactor, angle, debug_dir=None, saveImg=False):
	zoomed_data = scipy.ndimage.zoom(data, zoomfactor, output=None, order=2,
									 mode='constant', cval=0.0, prefilter=False)

	print((angle, zoomfactor))
	if saveImg == True:
		cv.imwrite(debug_dir + 'angle' + str(angle) + 'scale' + str(zoomfactor) + '.tif', np.uint16(zoomed_data))
	return zoomed_data


if __name__ == '__main__':  # Just for testing:
	raw_data = np.zeros((10, 10))
	raw_data[3] = 10

	output_dir = 'None'

	angle_variations = 0  # How many values besides centerpoint (even for centerpoint in variation)
	angle_res = 90
	angle_centervalue = 45

	scale_variations = 2
	scale_res = 0.5
	scale_centervalue = 1

	rf = 8  # rounding of angle and scale to rf = x digits

	zeroangle = angle_centervalue - angle_variations / 2 * angle_res
	zeroscale = scale_centervalue - scale_variations / 2 * scale_res

	# Rotate, crop then scale for all variations
	rot_crop_data = {(round(zeroangle + i * angle_res, rf), round(zeroscale + j * scale_res, rf)):
		scale_rotated(
			rotate_cropped(raw_data, round(zeroangle + i * angle_res, rf)), round(zeroscale + j * scale_res, rf),
			round(zeroangle + i * angle_res, rf), saveImg=False)
		for i in range(angle_variations + 1) for j in range(scale_variations + 1)}

	# Results
	print('Raw Data')
	print(np.round(raw_data))
	# print(raw_data)
	print('Raw-Data has shape: ' + str(raw_data.shape) + '\n')

	print('Only Rotation:')
	print(np.round(scipy.ndimage.interpolation.rotate(raw_data, angle=angle_centervalue, reshape=False, output=None,
													  order=1, mode='constant', cval=np.nan, prefilter=False)))
	print('\n')

	print('Only Scale:')
	data = np.round(scale_data(raw_data, 1.2))
	print(data)
	print('Data has shape: ' + str(data.shape) + '\n')

	print('Altered Data')
	for variations in rot_crop_data:
		print('Angle,Scale')
		print(variations)
		# print(rot_crop_data[variations])
		print(np.round(rot_crop_data[variations]))
		print('Data has shape: ' + str(rot_crop_data[variations].shape) + '\n')

	print('Done')
