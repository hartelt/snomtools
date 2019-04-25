"""
This script holds transformation functions for datasets, that do a binning of data to reduce data dimensions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

import snomtools.data.datasets as ds
import snomtools.calcs.units as u

#raise NotImplementedError()

# For verbose mode with progress printouts:
if '-v' in sys.argv:
	verbose = True
	import time
else:
	verbose = False


class Binning(object):

	def __init__(self, data = None, binAxisID = None, binFactor = None):

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
		ticks = np.zeros(np.int8(oldaxis.shape[0]/self.binFactor))
		newaxis = ds.Axis(data=ticks, unit=oldaxis.get_unit(),label=oldaxis.get_label() + ' binned', plotlabel=oldaxis.get_plotlabel()) #Make more elegant
		for i in range (np.int8(oldaxis.shape[0]/self.binFactor)):
			newaxis[i] = np.mean(oldaxis.get_data()[self.binFactor*i:self.binFactor*(i+1)])
		return newaxis


	def bin_data(self, h5target=None):
		axList = range(self.data.axes.__len__())
		# ToDo: move binAxis to last position
		axList.remove(self.binAxisID)
		#ToDo: generate new data array of right shape
		newdata = self.data

		for i in range(np.int8(self.data.get_axis(self.binAxisID).shape[0] / self.binFactor)):
			#ToDo: adress correct datafield
			newdata[i] = [[row[axis] for row in matrix] for axis in axList]
			itertools.product

		return newdata


if __name__ == '__main__':  # Just for testing:
	print("Testing...")
	path = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs'
	data_dir = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs\\projected.hdf5'
	data = ds.DataSet.from_h5file(data_dir, h5target=path + '\\uselesscache.hdf5')
	binSet = Binning(data=data, binAxisID= 'energy', binFactor= 3 )
	newaxis = binSet.reshape_axis()
	newdata = binSet.bin_data()

	data.add_axis(newaxis)
	data.add_datafield(newdata)
	print(data.shape)

	print("done.")