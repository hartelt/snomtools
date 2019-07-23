import cv2 as cv
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import tifffile as tf
import os
import re
import math
import snomtools.evaluation.driftcorrection as dm
import snomtools.data.datasets as ds


###Windows stuff

def dircheck(directory):
	parts = directory.split('/')
	target = ""
	for i in parts:
		target = target + i + '/'

		try:
			os.stat(target)
		except:
			os.mkdir(target)


def getfilesOrdered(directory, type='.tif', firstletter='D'):
	objects = os.listdir(directory)
	files = []
	for i in objects:
		if i.endswith(type):
			files.append(i.split(firstletter)[1].split('.')[0])
	files.sort(key=int)
	print('Files found:')
	print(files)
	return (files)


def direct_subdirs(directory):
	a = os.walk(directory)
	return [x[0] + '/' for x in a]


def win_dir(directory):
	# e.g. reference_file = win_dir(r'C:\Users\Benjamin Frisch\Desktop\matching debug') + 'data.tif'
	# change windows back slashes to forward slashes and add one at the end if necessary
	if ((directory[-1] != r'/') and (directory[-1] != r'\\')):
		directory += str(r'/')
		directory = re.sub(r'\\', r'/', directory)
	return directory





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
	w, h = rotatedRectWithMaxArea(data.shape[1], data.shape[0], np.radians(angle))
	return crop_around_center(cache, w, h)


def crop_around_center(image, width, height):
	"""
	Given a NumPy / OpenCV 2 image, crops it to the given width and height,
	around it's centre point
	"""

	image_center = (np.rint(image.shape[0] * 0.5), np.rint(image.shape[1] * 0.5))

	if (width > image.shape[1]):
		width = image.shape[1]

	if (height > image.shape[0]):
		height = image.shape[0]

	y1 = int(np.ceil(image_center[0] - height * 0.5))
	y2 = int(np.floor(image_center[0] + height * 0.5))
	x1 = int(np.ceil(image_center[1] - width * 0.5))
	x2 = int(np.floor(image_center[1] + width * 0.5))


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
		(round(zeroangle + i * angle_res, digits), round(zeroscale + j * scale_res, digits)): scale_rotated(
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

def match_rotation_scale(reference_data, tomatch_data, angle_settings=(0, 0, 0), scale_settings=(1, 0, 0), digits=5,
						 output_dir=None, saveImg=False,
						 saveRes=False):
	'''
	Calculates the correlation of different scales and rotations of tomatch_data against some reference_data
	:param reference_data: 2D dataset, bigger than tomatch_data
	:param tomatch_data: 2D dataset
	:param angle_settings:	tuple (center, resolution, number of variations)
	:param scale_settings:	tuple (center, resolution, number of variations)
	:param digits: number of digits to round the variations to
	:param output_dir: save target
	:param saveImg:	save all rotated and scaled images
	:param saveRes:	save results as TXT with titles
	:return: 	Results array with (angle, scale, ypos, xpos, correlation).
	'''
	rot_crop_data = rot_scale_data(data=tomatch_data, angle_settings=angle_settings,
								   scale_settings=scale_settings, digits=digits, output_dir=output_dir, saveImg=saveImg)
	results = []
	for variations in rot_crop_data:
		template = np.float32(rot_crop_data[variations])
		drift = dm.Drift.template_matching(reference_data, template, method='cv.TM_CCOEFF_NORMED', subpixel=False)
		results.append((variations[0], variations[1], drift[0][0], drift[0][1], drift[1]))
		print('Matched: angle ' + str(variations[0]) + ' scale ' + str(variations[1]) + '\t' + 'Correlation = ' + str(drift[1]) )
	results = np.asarray(results)

	if saveRes == True:
		np.savetxt(output_dir + 'rot_scale_max' + str(results[:, 4].max()) + '.txt', np.asarray(results))

	return results


# ------------------Script start-----------------------------------

### All variables

working_directory = os.getcwd()
# reference_file = win_dir(r'C:\Users\Benjamin Frisch\Desktop\Matching Example\data') + 'raw_data.tif'
reference_file = win_dir(working_directory + '/data') + 'raw_data.tif'
tomatch_file = win_dir(working_directory + '/data') + 'template_10deg.tif'
output_dir = win_dir(working_directory + '/out')
dircheck(output_dir)  # checks if dir exists
dircheck(output_dir+'/variations/')

template_size = 400	#make shure it's smaller than the image
angle_settings = (10,1, 20)  # center, resolution, variations
scale_settings = (0.8, 0.1, 10)	# center, resolution, variations (cannot zoom <0)



#--------------------------------------------

###Read start images

reference_data = cv.imread(reference_file, -1)
tomatch_data = cv.imread(tomatch_file, -1)

### Template match all

reference_data = ds.DataArray(np.float32(reference_data))
tomatch_data_cropped = ds.DataArray(crop_around_center(np.float32(tomatch_data), template_size, template_size))
# Here a 'template_size' part of the image can be cropped out of the center automatically.
# This ensures, that the value of rotation calculated for this template is the same as for the full image.
# You can also use the full image as template by using:
# tomatch_data_cropped = tomatch_data

#plot a rectangle of the template_size into the tomatch_data for visualization of used template
fig0,ax0 = plt.subplots(1)
ax0.imshow(tomatch_data)
rect = patches.Rectangle((int(tomatch_data.shape[1]/2-template_size/2),int(tomatch_data.shape[0]/2-template_size/2)),template_size,template_size,linewidth=1,edgecolor='r',facecolor='none')
ax0.add_patch(rect)
plt.show()



results = match_rotation_scale(reference_data, tomatch_data_cropped, angle_settings=angle_settings,
							   scale_settings=scale_settings, output_dir=output_dir+'/variations/',
							   saveImg=True, saveRes=True)

best_match = results[np.argmax(results[:, 4])]
print('\n')
print('Best match:')
print('Correlation=' + str(best_match[4]) + ' for generated template by angle ' + str(best_match[0]) + ' and scale ' + str(
		best_match[1]) + ' by x,y ' + str(best_match[2]) + ',' + str(best_match[3]))



#--------------------------------------------

### Plotting and data manipulation
shift_to_match = True
if shift_to_match == True:
	print('Saving modified tomatch_file by angle ' + str(best_match[0]) + ' and scale ' + str(best_match[1]))
	tomatch_data_rotated = rotate_cropped(tomatch_data,best_match[0])
	cv.imwrite(output_dir + 'tomatch_rotated_' + str(np.around(best_match[0],decimals=3)) + 'deg.tif', tomatch_data_rotated)
	tomatch_data_rotated_scaled = scipy.ndimage.zoom(tomatch_data_rotated,best_match[1])
	cv.imwrite(output_dir + 'tomatch_rotated_' + str(np.around(best_match[0],decimals=3)) + 'deg_s_' + str(np.around(best_match[1],decimals=3)) + '.tif', tomatch_data_rotated_scaled)

	tomatch_data_rotated_scaled_template = ds.DataArray(crop_around_center(np.float32(tomatch_data_rotated_scaled), template_size, template_size))


	#template position in reference_data
	drift = dm.Drift.template_matching(reference_data, tomatch_data_rotated_scaled_template, method='cv.TM_CCOEFF_NORMED', subpixel=False)

	fig1, ax1 = plt.subplots(1)
	ax1.imshow(reference_data)
	rect = patches.Rectangle((drift[0][1],drift[0][0]),template_size, template_size, linewidth=1, edgecolor='r', facecolor='none')
	ax1.add_patch(rect)
	plt.show()

	#template position in the tomatch_data of correct angle and scale
	template_zero = dm.Drift.template_matching(tomatch_data_rotated_scaled, tomatch_data_rotated_scaled_template, method='cv.TM_CCOEFF_NORMED', subpixel=False)

	fi2g, ax2 = plt.subplots(1)
	ax2.imshow(tomatch_data_rotated_scaled)
	rect = patches.Rectangle((template_zero[0][1],template_zero[0][0]),template_size, template_size, linewidth=1, edgecolor='r', facecolor='none')
	ax2.add_patch(rect)
	plt.show()
	#shiftvector of tomatch_data of correct angle and scale relative to reference_data
	shiftvector = (drift[0][0]-template_zero[0][0], drift[0][1]-template_zero[0][1])

	#shifting the tomatch_data to the correct position in relation to (0,0) of reference_data
	print('Correlation=' + str(drift[1]) + ' Shifting image by y,x ' + str(np.round(shiftvector, 3)))
	tomatch_data_rotated_scaled = ds.DataArray(tomatch_data_rotated_scaled)
	datashifted = ds.DataArray.shift(tomatch_data_rotated_scaled, shiftvector)
	datashifted = np.uint16(datashifted)

	cv.imwrite(output_dir + 'data_rotcrop_shifted_by_yx' + str(np.round(shiftvector, 3)) + '.tif', datashifted)



	print('done')
