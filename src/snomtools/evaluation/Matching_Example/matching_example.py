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
import snomtools.data.transformation.rotate as rot


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



### Functions

#Most of the used funcionts are located in snomtools.data.transformation.rotate.

def match_rotation_scale(reference_data, data_tomatch, angle_settings=(0, 0, 0), scale_settings=(1, 0, 0), digits=5,
						 output_dir=None, saveImg=False,
						 saveRes=False):
	'''
	Calculates the correlation of different scales and rotations of data_tomatch against some reference_data

	All variations of angles and scales given by the settings are generated as data and written to rot_crop_data.

	It is then iterated over all variations and the maximum correlation for the data_tomatch with the reference_data is calculated.

	The function dm.Drift.template_matching calculates the normalized correlation for all positions of the data_tomatch that fit fully in reference_data.
	This is why data_tomatch has to be be smaller then reference_data.
	The maximum correlation and the position of the (0,0) point (upper left corner) of the data_tomatch inside the reference_data is returned.

	This function here returns an array with the correlation values for the respective angles and scales in the format (angle, scale, ypos, xpos, correlation).


	:param reference_data: 2D dataset, bigger than tomatch_data
	:param data_tomatch: 2D dataset
	:param angle_settings:	tuple (center, resolution, number of variations)
	:param scale_settings:	tuple (center, resolution, number of variations)
	:param digits: number of digits to round the variations to
	:param output_dir: save target
	:param saveImg:	save all rotated and scaled images
	:param saveRes:	save results as TXT with titles
	:return: 	Results array with (angle, scale, ypos, xpos, correlation).
	'''
	rot_crop_data = rot.rot_scale_data(data=data_tomatch, angle_settings=angle_settings,
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
tomatch_file = win_dir(working_directory + '/data') + 'template_10deg_s0.5.tif'
output_dir = win_dir(working_directory + '/out')
dircheck(output_dir)  # checks if dir exists
dircheck(output_dir+'/variations/')

template_size = 100	#Size of the image used for correlation calculation. Make shure it's smaller then the reference image
angle_settings = (10,1, 20)  # center, resolution, variations
scale_settings = (1.5, 0.1, 16)	# center, resolution, variations (cannot zoom <0)




#--------------------------------------------

###Read start images

reference_data = cv.imread(reference_file, -1)
tomatch_data = cv.imread(tomatch_file, -1)

### Template match all

reference_data = ds.DataArray(np.float32(reference_data))
tomatch_data_cropped = ds.DataArray(rot.crop_around_center(np.float32(tomatch_data), template_size, template_size))
# Here a 'template_size' part of the image can be cropped out of the center automatically.
# The center ensures, that the value of rotation calculated for this template is the same as for the full image.
# You could also use the full image as template by using:
# tomatch_data_cropped = tomatch_data, but keep in mind, that the template should be smaller than reference_data

#Plot a rectangle of the template_size into the tomatch_data for visualization of used template
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
	tomatch_data_rotated = rot.rotate_cropped(tomatch_data,best_match[0])
	cv.imwrite(output_dir + 'tomatch_rotated_' + str(np.around(best_match[0],decimals=3)) + 'deg.tif', tomatch_data_rotated)
	tomatch_data_rotated_scaled = scipy.ndimage.zoom(tomatch_data_rotated,best_match[1])
	cv.imwrite(output_dir + 'tomatch_rotated_' + str(np.around(best_match[0],decimals=3)) + 'deg_scale_x' + str(np.around(best_match[1],decimals=3)) + '.tif', tomatch_data_rotated_scaled)

	tomatch_data_rotated_scaled_template = ds.DataArray(rot.crop_around_center(np.float32(tomatch_data_rotated_scaled), template_size, template_size))


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
