"""
This script holds transformation functions for datasets, that are based on interpolation methods for the data points.
"""
__author__ = 'hartelt'

import numpy
import snomtools.data.datasets as datasets
import scipy.interpolate
import snomtools.calcs.units as units


def griddata(dataset, xi, method='linear', fill_value=numpy.nan, rescale=False):
	"""

	:param dataset: The dataset to transform.

	:param xi: The coordinates at which to interpolate data. Given as a tuple of Axes (see datasets.py) with the same
		length as the number of axes of the dataset OR as a numpy array of the same dimensionality as the dataset,
		that is then assumed to have the same physical dimensions as the axes of the dataset.

	:param method:

	:param fill_value:

	:param rescale:

	:return:
	"""
	assert isinstance(dataset, datasets.DataSet), "ERROR: No DataSet instance given."
	raise NotImplementedError("Griddata to be implemented soon...")


if __name__=='__main__':  # Just for testing:
	print("Testing...")
	import snomtools.data.imports.lumerical_mat

	print("Processing input data...")
	infile = "2015-08-03-Sphere4-substrate-532nm-bottomfieldE.mat"
	outfile = infile.replace('.mat', '.hdf5')
	dataset = snomtools.data.imports.lumerical_mat.Efield_3d(infile)
	dataset.swapaxis('l', 'x')
	dataset.swapaxis(1, 2)
	dataset.saveh5(outfile)

	print("Assembling data...")
	x,y = dataset.meshgrid(['x','y'])
	data = dataset.get_datafield(0).get_data()[:, :, 15]

	print("Generating grid...")
	newgrid = numpy.mgrid[-3.:3.:64j, -3.:3.:64j]
	# newx = numpy.arange(-3., 3., .5) * units.ureg('um')
	# newy = numpy.arange(-3., 3., .5) * units.ureg('um')
	# print(newx, newy)
	# newgrid = numpy.meshgrid(newx,newy)

	print("Interpolating...")
	interp = scipy.interpolate.griddata((x.to('um').flatten(),y.to('um').flatten()), data.flatten(), tuple(newgrid),
										method='cubic')

	print("Plotting...")
	import matplotlib.pyplot as plt
	plt.imsave('interpolated.png', interp, cmap='gray')
	print("DONE")
