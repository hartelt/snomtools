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
        """
        Binns the data of a dataset along the designated axes binAxisID by a factor binFactor.

        :param data: Dataset that should be binned
        :param binAxisID: List of binAxisID's as int or Axes names as str that should be binned
        :param binFactor:  List of binFactors of the corresponding Axis as int
        """
        self.data = data
        self.binAxisID = []
        for axis in binAxisID:
            if type(binAxisID) is int:
                self.binAxisID.append(axis)
            else:
                self.binAxisID.append(data.get_axis_index(axis))

        self.binFactor = list(binFactor)

    # assert shape(self.binFactor)=shape(self.binAxisID)

    def bin_axis(self):
        """
        Gives the new Axis with ticks via np.mean
        :return:
        """
        newaxis = self.data.axes

        for ax in range(len(self.binAxisID)):
            oldaxis = self.data.get_axis(self.binAxisID[ax])
            ticks = np.zeros(np.int16(oldaxis.shape[0] / self.binFactor[ax]))
            newSubAxis = ds.Axis(data=ticks, unit=oldaxis.get_unit(),
                                 label=oldaxis.get_label() + ' binned x' + str(self.binFactor[ax]),
                                 plotlabel=oldaxis.get_plotlabel())  # Make more elegant
            for i in range(np.int16(oldaxis.shape[0] / self.binFactor[ax])):
                newSubAxis[i] = np.mean(oldaxis.get_data()[self.binFactor[ax] * i:self.binFactor[ax] * (i + 1)])
                newaxis[self.binAxisID[ax]] = newSubAxis
        return newaxis

    def bin_data(self, data_id=0, label=None, plotlabel=None, h5target=None):
        # TODO: Docstring!
        data_id = self.data.get_datafield_index(data_id)
        if label is None:
            label = "binned_" + self.data.get_datafield(data_id).label
        if plotlabel is None:
            plotlabel = self.data.get_datafield(data_id).plotlabel

        # Building a new Dataset with shape according to binning
        newshape = list(self.data.shape)
        for ax in range(len(self.binAxisID)):
            newshape[self.binAxisID[ax]] = np.int16(newshape[self.binAxisID[ax]] / self.binFactor[ax])
        newdata = ds.Data_Handler_H5(shape=newshape, unit=self.data.get_datafield(data_id).get_unit())

        if verbose:
            import time
            print("Start:")
            start_time = time.time()
            print(time.ctime())

        # Calculating the binning chunkwise for performance, therefore slicing the data
        for chunkslice in newdata.iterfastslices():
            olddataregion = list(chunkslice)

            for ax in range(len(self.binAxisID)):
                # start index is 0 in case of fullslice, which yields None at .start and .stop
                selection_start = chunkslice[self.binAxisID[ax]].start or 0

                # stop of chunkslice is matched to actual data in newshape:
                selection_along_binaxis = sliced_shape(chunkslice, newshape)[self.binAxisID[ax]]

                # binned axis region has to be a binFactor bigger array along the binAxis
                olddataregion[self.binAxisID[ax]] = slice(selection_start * self.binFactor[ax],
                                                          (selection_start + selection_along_binaxis)
                                                          * self.binFactor[ax],
                                                          None)
            olddataregion = tuple(olddataregion)
            # load olddata from this region into in-memory quantity
            olddata = self.data.get_datafield(data_id).data[olddataregion].q
            # (.q necessary, because we do not want to reshape on the original H5 dataset)

            # split data in packs that need to be summed up by rearranging the data along an additional axis
            # of shape binFactor in the position of the binAxis+1 and reducing binAxis by a binFactor,
            # so that the amount of arrayelements stays the same
            # Example: An axis of len = 50 with binFactor of 2 would be reshaped to (25,2), then summed over the second.
            shapelist = list(olddata.shape)
            binAxisFactor = list(zip(self.binAxisID, self.binFactor))  # zip tuples so we can manipulate them together
            # sort the (binAxisID,binFactor) list declining, so one can add the axis i to the reshape list later and
            # keep track of the new index of the i+1 axis by adding 1 :
            binAxisFactor.sort(reverse=True)

            for ax in range(len(binAxisFactor)):  # binAxisFactor contains pair of [0]=binAxisID [1]=binFactor
                shapelist[binAxisFactor[ax][0]] = shapelist[binAxisFactor[ax][0]] // binAxisFactor[ax][1]
                shapelist.insert(binAxisFactor[ax][0] + 1, binAxisFactor[ax][1])

            olddata_reshaped = olddata.reshape(shapelist)  # reshaped view on data (split remaining axis, binning axis)

            # shift binAxis Index by number of previously added dimensions before it:
            sumaxes = list(range(len(binAxisFactor))) + np.sort(np.array(binAxisFactor)[:, 0]) + 1
            newdata[chunkslice] = np.sum(olddata_reshaped, axis=tuple(sumaxes))  # sum along the binning axis

        newdata = ds.DataArray(newdata,
                               label=label,
                               plotlabel=plotlabel,
                               h5target=h5target)
        if verbose:
            print("End:")
            print(time.ctime())
            print("{0:.2f} seconds".format(time.time() - start_time))
        return newdata

    def bin(self, h5target=None):
        newaxis = self.bin_axis()
        if h5target is not None:
            newda = self.bin_data(h5target=True)
        else:
            newda = self.bin_data(h5target=None)

        newds = ds.DataSet(self.data.label + " binned", (newda,), newaxis,
                           self.data.plotconf, h5target=h5target)
        return newds


if __name__ == '__main__':  # Just for testing:
    print("Testing...")
    test_fakedata = True  # Create and test on a fake dataset that's easier to overview:
    if test_fakedata:
        print("Building fake data...")
        fakearray = np.stack([np.arange(50) for i in range(25)] + [np.arange(50) + 100 for i in range(25)])
        fakedata = ds.DataArray(fakearray, h5target=True, chunks=(5, 5))
        fakeds = ds.DataSet("test", [ds.DataArray(fakedata)],
                            [ds.Axis(np.arange(50), label="y"), ds.Axis(np.arange(50), label="x")],
                            h5target=True)
        fakeds.saveh5("binning_testdata.hdf5")
        print("Test binning on fake data...")
        b = Binning(fakeds, binAxisID=('y', 'x'), binFactor=(2, 8))
        binnedds = b.bin(h5target="binning_outdata.hdf5")
        binnedds.saveh5()

    test_realdata = False  # Testing real data from NFC Session on Ben's PC:
    if test_realdata:
        path = 'E:\\NFC15\\20171207 ZnO+aSiH\\01 DLD PSI -3 to 150 fs step size 400as\\Maximamap\\Driftcorrected\\summed_runs'
        data_dir = path + '\\projected.hdf5'
        # data_dir = path + '\\summed_data.hdf5'
        h5target = path + '\\binned_data.hdf5'
        data = ds.DataSet.from_h5file(data_dir, h5target=h5target)

        # data = ds.DataSet.from_h5file("terra-tr-psi-dld.hdf5")

        binSet = Binning(data=data, binAxisID=('energy', 'delay'), binFactor=(3, 10))
        newds = binSet.bin(h5target=h5target)

    print("done.")
