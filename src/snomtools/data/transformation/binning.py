"""
This script holds transformation functions for datasets, that do a binning of data to reduce data dimensions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

import snomtools.data.datasets as ds

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

	def reshape_axis(self):
		"""
		Gives the new Axis with ticks via np.mean
		:return:
		"""
		oldaxis = self.data.get_axis(self.binAxisID)
		ticks = np.zeros(np.int8(oldaxis.shape[0] / self.binFactor))
		newaxis = ds.Axis(data=ticks, unit=oldaxis.get_unit(), label=oldaxis.get_label() + ' binned',
						  plotlabel=oldaxis.get_plotlabel())  # Make more elegant
		for i in range(np.int8(oldaxis.shape[0] / self.binFactor)):
			newaxis[i] = np.mean(oldaxis.get_data()[self.binFactor * i:self.binFactor * (i + 1)])
		return newaxis

	def bin_data(self, h5target=None):
		newshape = list(self.data.shape)
		newshape[self.binAxisID] = np.int8(newshape[self.binAxisID] / self.binFactor)
		newdata = ds.Data_Handler_H5(shape=newshape, unit=self.data.get_datafield(0).get_unit())

		if verbose:
			import time
			print("Start:")
			start_time = time.time()
			print(time.ctime())
		for chunkslice in newdata.iterchunkslices():
			olddataregion = list(chunkslice)
			olddataregion[self.binAxisID] = slice(chunkslice[self.binAxisID].start * self.binFactor,
												  (chunkslice[self.binAxisID].start + newdata.chunks[self.binAxisID])
												  * self.binFactor,
												  None)
			olddataregion = tuple(olddataregion)
			olddata = self.data.get_datafield(0).data[olddataregion]

			for i in range(self.binFactor):
				poslist = [np.s_[:] for j in range(self.data.dimensions)]
				poslist[self.binAxisID] = np.s_[i::self.binFactor]
				newdata[chunkslice] += olddata[tuple(poslist)]
		newdata = ds.DataArray(newdata,
							   label="binned_" + self.data.get_datafield(0).label,
							   plotlabel=self.data.get_datafield(0).plotlabel,
							   h5target=h5target)
		if verbose:
			print("End:")
			print(time.ctime())
			print("{0:.2f} seconds".format(time.time() - start_time))
		return newdata


if __name__ == '__main__':  # Just for testing:
	print("Testing...")
	# path = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs'
	# data_dir = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs\\projected.hdf5'
	# data = ds.DataSet.from_h5file(data_dir, h5target=path + '\\uselesscache.hdf5')

	data = ds.DataSet.from_h5file("terra-tr-psi-dld.hdf5")

	binSet = Binning(data=data, binAxisID='channel', binFactor=3)
	newaxis = binSet.reshape_axis()
	newdata = binSet.bin_data()

	print("done.")
