"""
This script holds transformation functions for datasets, that do a binning of data to reduce data dimensions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import snomtools.data.datasets as ds
from snomtools.data.tools import sliced_shape

# For verbose mode with progress printouts:
if '-v' in sys.argv:
	verbose = True
	import time
else:
	verbose = False


class Binning(object):
	def __init__(self, data=None, binAxisID=None, binFactor=None):

		self.data = data

		if type(binAxisID) is int:
			self.binAxisID = binAxisID
		else:
			self.binAxisID = data.get_axis_index(binAxisID)

		self.binFactor = binFactor


	def bin_axis(self):
		"""
		Gives the new Axis with ticks via np.mean
		:return:
		"""
		oldaxis = self.data.get_axis(self.binAxisID)
		ticks = np.zeros(np.int16(oldaxis.shape[0] / self.binFactor))
		newSubAxis = ds.Axis(data=ticks, unit=oldaxis.get_unit(), label=oldaxis.get_label() + ' binned x'+str(self.binFactor),
						  plotlabel=oldaxis.get_plotlabel())  # Make more elegant
		for i in range(np.int16(oldaxis.shape[0] / self.binFactor)):
			newSubAxis[i] = np.mean(oldaxis.get_data()[self.binFactor * i:self.binFactor * (i + 1)])
		newaxis =self.data.axes
		newaxis[self.binAxisID]=newSubAxis
		return newaxis

	def bin_data(self, h5target=None):
		# TODO: Docstring!

		# Building a new Dataset with shape according to binning
		newshape = list(self.data.shape)
		newshape[self.binAxisID] = np.int16(newshape[self.binAxisID] / self.binFactor)
		newdata = ds.Data_Handler_H5(shape=newshape, unit=self.data.get_datafield(0).get_unit())

		if verbose:
			import time
			print("Start:")
			start_time = time.time()
			print(time.ctime())

		# Calculating the binning chunkwise for performance, therefore slicing the data
		for chunkslice in newdata.iterfastslices():
			# start index is 0 in case of fullslice, which yields None at .start and .stop
			selection_start = chunkslice[self.binAxisID].start or 0
			# stop of chunkslice is matched to actual data in newshape
			selection_along_binaxis = sliced_shape(chunkslice, newshape)[self.binAxisID]

			# binned axis region is a binFactor bigger array along the binAxis
			olddataregion = list(chunkslice)
			olddataregion[self.binAxisID] = slice(selection_start * self.binFactor,
												  (selection_start + selection_along_binaxis)
												  * self.binFactor,
												  None)
			olddataregion = tuple(olddataregion)
			# load olddata from this region
			olddata = self.data.get_datafield(0).data[
				olddataregion].q  # .q necessary, otherwise "olddata.shape = tuple(shapelist)" yields: "AttributeError: can't set attribute"

			# split data in packs that need to be summed up by rearranging the data along an additional axis of shape binFactor in the position of the binAxis and reducing binAxis by a binFactor, so that the amount of arrayelements stays the same
			shapelist = list(olddata.shape)
			shapelist[self.binAxisID] = shapelist[self.binAxisID] // self.binFactor
			shapelist.insert(self.binAxisID, self.binFactor)
			olddata.shape = tuple(shapelist)  # reshape inplace (split binning axis and remaining axis)
			newdata[chunkslice] = np.sum(olddata, axis=self.binAxisID)  # sum along the newly added binning axis

		newdata = ds.DataArray(newdata,
							   label="binned_" + self.data.get_datafield(0).label,
							   plotlabel=self.data.get_datafield(0).plotlabel,
							   h5target=h5target)
		if verbose:
			print("End:")
			print(time.ctime())
			print("{0:.2f} seconds".format(time.time() - start_time))
		return newdata

	def bin(self, h5target = None):
		newaxis = self.bin_axis()
		newda = self.bin_data(h5target=h5target)

		newds = ds.DataSet(self.data.label + " binned", (newda,), newaxis,
												self.data.plotconf, h5target=h5target)
		return newds


if __name__ == '__main__':  # Just for testing:
	print("Testing...")
	path = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs'
	data_dir = path + '\\projected.hdf5'
	# data_dir = path + '\\summed_data.hdf5'
	h5target = path + '\\binned_data.hdf5'
	data = ds.DataSet.from_h5file(data_dir, h5target=h5target)

	# data = ds.DataSet.from_h5file("terra-tr-psi-dld.hdf5")

	binSet = Binning(data=data, binAxisID='energy', binFactor=3)
	newds= binSet.bin(h5target=h5target)

	print("done.")
