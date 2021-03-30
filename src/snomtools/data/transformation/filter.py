"""
Filters working on DataSets using scipy.ndimage filters.
"""
import sys
import numpy as np
import scipy.ndimage
import snomtools.data.datasets as ds
import snomtools.calcs.units as u

__author__ = 'Michael Hartelt'

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False


class Filter(object):
    """
    A generic filter class. A filter can be implemented by overwriting __init__ and rawfilter in a derived class.
    """

    def __init__(self, data, axes=None):
        """
        The initializer.

        :param data: The data to filter.
        :type data: ds.DataSet

        :param axes: A list of valid axis identifiers of the axes along which to filter.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), "Invalid input data."
        self.data_original = data
        if axes is not None:
            self.axes_filtered = [self.data_original.get_axis_index(ax) for ax in axes]
            self.axes_kept = [i for i in list(range(self.data_original.dimensions)) if i not in self.axes_filtered]
        else:
            self.axes_filtered = list(range(self.data_original.dimensions))
            self.axes_kept = []
        self.filter_kwargs = {}

    def rawfilter(self, data):
        """
        Handles the numeric filtering in the backend.

        :param data: The raw data to be fed to the scipy.ndimage filter function.
        :type data: array-like

        :return: filtered data
        :rtype np.ndarray
        """
        return None

    def dataarray_filtered(self, data_id, label=None, plotlabel=None,
                           h5target=None,
                           chunks=True,
                           dtype=None,
                           compression='gzip', compression_opts=4):
        """
        Filters the DataArray of the DataSet with the nd filter.
        Full slices along the filtered axes are calculated in memory at once, all other axes are flat-iterated.

        :param data_id: A valid id of the DataArray to rotate.

        :param label: The label for the generated DataArray. Default is `filtered_`+original label
        :type label: str

        :param plotlabel: The plotlabel for the generated DataArray. Default is the original plotlabel.
        :type plotlabel: str

        :param h5target: A hdf5 target to write to, or True for temp file mode.

        :param chunks: (See h5py and data.datasets docs.)

        :param dtype: The data type. Best given as numpy datatype.
            The default will be `numpy.float32`.

        :param compression: (See h5py and data.datasets docs.)

        :param compression_opts: (See h5py and data.datasets docs.)

        :return: The filtered DataArray.
        :rtype: ds.DataArray
        """
        d_original = self.data_original.get_datafield(data_id)
        if label is None:
            label = 'filtered ' + d_original.get_label()
        if plotlabel is None:
            plotlabel = d_original.get_plotlabel()
        if dtype is None:
            dtype = d_original.data.dtype

        if verbose:
            print("Calculating filtered data for {}".format(d_original.get_label()))
            slices_todo = np.prod(
                [d_original.shape[i] for i in range(self.data_original.dimensions) if i in self.axes_kept])
            slices_done = 0
            import time
            print("Start: {}".format(time.ctime()))
            start_time = time.time()

        outda = ds.DataArray.make_empty(d_original.shape, d_original.units, label, plotlabel,
                                        h5target, chunks, dtype, compression, compression_opts)
        for s in outda.data.iterflatslices(dims=self.axes_kept):
            raw_data = d_original.data[s].magnitude
            filtered_data = self.rawfilter(raw_data)
            outda[s] = filtered_data

            if verbose:
                slices_done += 1
                tps = ((time.time() - start_time) / float(slices_done))
                etr = tps * (slices_todo - slices_done)
                print("Slice {0:d} / {1:d}, Time/Slice {3:.2f}s ETR: {2:.1f}s".format(slices_done,
                                                                                      slices_todo,
                                                                                      etr,
                                                                                      tps))
        if verbose:
            print("Done: {}".format(time.ctime()))
        return outda

    def filter_data(self, label=None, h5target=None):
        """
        Filters the full DataSet.
        Returns a DataSet containing the original data with a filtered DataArray added for each DataArray.

        :param label: A label for the new DataSet.
        :type label: str

        :param h5target: A h5 target to write to, or True for temp file mode. See docs for data.datasets.DataSet.

        :return: The filtered DataSet.
        :rtype: ds.DataSet
        """
        if label is None:
            label = 'filtered ' + self.data_original.label
        if h5target:
            dataarrays_filtered = [self.dataarray_filtered(d, h5target=True) for d in self.data_original.dlabels]
        else:
            dataarrays_filtered = [self.dataarray_filtered(d) for d in self.data_original.dlabels]
        self.data_filtered = ds.DataSet(label,
                                        dataarrays_filtered, self.data_original.axes,
                                        h5target=h5target)
        return self.data_filtered

    def data_add_filtered(self, data_id, label=None, plotlabel=None):
        """
        Adds a filtered dataset to the original data that the Filter object is working on.

        :param data_id: A valid identifier of the DataArray to filter.

        :param label: The label for the generated DataArray. Default is `filtered_`+original label
        :type label: str

        :param plotlabel: The plotlabel for the generated DataArray. Default is the original plotlabel.
        :type plotlabel: str

        :return: A reference to the DataSet instance written to.
        """
        assert isinstance(self.data_original, ds.DataSet), "Filter can only write to DataSet, not ROI!"
        if self.data_original.h5target:
            da_h5 = True
        else:
            da_h5 = None
        self.data_original.add_datafield(self.dataarray_filtered(data_id, label, plotlabel, h5target=da_h5))
        return self.data_original

    def filterparam_listify(self, filterparam):
        """
        Makes a list of filter parameters that corresponds to the filtered axes.
        For this, it makes a list if a single element is given, or just checks if the list has the correct length.
        """
        try:
            assert len(filterparam) == len(self.axes_filtered), \
                "Parameter list does not fit to number of filtered axes."
            return filterparam
        except TypeError as e:  # it has no length
            return [filterparam for i in self.axes_filtered]

    def filterparam_indexify(self, filterparam, require_int=False):
        """
        If a filter parameter corresponds to axis values, this converts to pixel values.

        :param filterparam: A filter parameter given as a scalar (pixel-values), quantity (axis-values),
            or a list thereof containing values for each Axis along which to filter.
            If quantities are given, units must be compatible to the respective Axis units,
            and the Axis must be linearly spaced.

        :param require_int: Filter parameters are required as int. Float values are rounded to int.
        :type require_int: bool

        :return: A list containing the filter parameters in pixels for each Axis to filter.
        """
        filterparam_list = self.filterparam_listify(filterparam)
        outlist = []
        for param, axis in zip(filterparam_list, self.axes_filtered):
            axis = self.data_original.get_axis(axis)
            if u.is_quantity(param):
                assert axis.is_linspaced(), "Trying to give filter parameter as Axis value for non-linspaced axis!"
                pxvalue = (param / axis.spacing()).to('1').magnitude
                if require_int:
                    pxvalue = round(pxvalue)
                outlist.append(pxvalue)
            else:
                outlist.append(param)
        return outlist


class GaussFilter(Filter):
    """
    A gaussian filter, using `scipy.ndimage.gaussian_filter`, derived from the generic Filter.
    """

    def __init__(self, data, sigma, axes=None, order=0, mode='reflect', cval=0.0, truncate=4.0):
        """
        The initializeer. See the docs of `scipy.ndimage.gaussian_filter` for details on the filter parameters
        `sigma`, `order`, `mode`, `cval`, and `truncate`.

        :param data: The data to filter.
        :type data: ds.DataSet

        :param sigma: Standard deviation for Gaussian kernel.
            Can be given as a scalar (pixel-values), quantity (axis-values),
            or a list thereof containing values for each Axis along which to filter.
            If quantities are given, units must be compatible to the respective Axis units,
            and the Axis must be linearly spaced.

        :param axes: A list of valid axis identifiers of the axes along which to filter.

        :param order: The order of the derivate of the gaussian(s) to use.
            The default, 0, corresponds to the gaussian itself.
            If a sequence is given, it must contain a value for each filtered axis.
        :type order: int or sequence of ints

        :param mode: The mode parameter, can be `reflect`, `constant`, `nearest`, `mirror`, or `wrap`.
        :type mode: str

        :param cval: The fill value for outside of the edges.
        :type cval: scalar

        :param truncate: Truncate the filter at this many standard deviations. Default is 4.0.
        :type truncate: float
        """
        Filter.__init__(self, data, axes)
        self.sigma = self.filterparam_indexify(sigma)
        self.order = self.filterparam_listify(order)
        self.filter_kwargs['order'] = order
        self.filter_kwargs['mode'] = mode
        self.filter_kwargs['cval'] = cval
        self.filter_kwargs['truncate'] = truncate

    def rawfilter(self, data):
        """
        Handles the numeric filtering in the backend.

        :param data: The raw data to be fed to the scipy.ndimage.gaussian_filter filter function.
        :type data: array-like

        :return: filtered data
        :rtype np.ndarray
        """
        assert len(data.shape) == len(self.axes_filtered), "Wrong data dimensionality."
        return scipy.ndimage.gaussian_filter(data, self.sigma, **self.filter_kwargs)


class MedianFilter(Filter):
    """
    A median filter, using `scipy.ndimage.median_filter`, derived from the generic Filter.
    """

    def __init__(self, data, axes=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0):
        """
        The initializer. See the docs of `scipy.ndimage.median_filter` for details on the filter parameters
        `size`, `footprint`, `mode`, `cval`, and `origin`.

        :param data: The data to filter.
        :type data: ds.DataSet

        :param axes: A list of valid axis identifiers of the axes along which to filter.
        """
        Filter.__init__(self, data, axes)
        if size is not None:
            self.size = self.filterparam_indexify(size, require_int=True)
        else:
            self.size = None
        self.origin = self.filterparam_listify(origin)
        self.filter_kwargs['footprint'] = footprint
        self.filter_kwargs['mode'] = mode
        self.filter_kwargs['cval'] = cval
        self.filter_kwargs['origin'] = origin

    def rawfilter(self, data):
        """
        Handles the numeric filtering in the backend.

        :param data: The raw data to be fed to the `scipy.ndimage.median_filter` filter function.
        :type data: array-like

        :return: filtered data
        :rtype np.ndarray
        """
        assert len(data.shape) == len(self.axes_filtered), "Wrong data dimensionality."
        return scipy.ndimage.median_filter(data, self.size, **self.filter_kwargs)


class MaximumFilter(Filter):
    """
    A maximum filter, using `scipy.ndimage.maximum_filter`, derived from the generic Filter.
    """

    def __init__(self, data, axes=None, size=None, footprint=None, mode='reflect', cval=0.0, origin=0):
        """
        The initializer. See the docs of `scipy.ndimage.maximum_filter` for details on the filter parameters
        `size`, `footprint`, `mode`, `cval`, and `origin`.

        :param data: The data to filter.
        :type data: ds.DataSet

        :param axes: A list of valid axis identifiers of the axes along which to filter.
        """
        Filter.__init__(self, data, axes)
        if size is not None:
            self.size = self.filterparam_indexify(size, require_int=True)
        else:
            self.size = None
        self.origin = self.filterparam_listify(origin)
        self.filter_kwargs['footprint'] = footprint
        self.filter_kwargs['mode'] = mode
        self.filter_kwargs['cval'] = cval
        self.filter_kwargs['origin'] = origin

    def rawfilter(self, data):
        """
        Handles the numeric filtering in the backend.

        :param data: The raw data to be fed to the `scipy.ndimage.maximum_filter` filter function.
        :type data: array-like

        :return: filtered data
        :rtype np.ndarray
        """
        assert len(data.shape) == len(self.axes_filtered), "Wrong data dimensionality."
        return scipy.ndimage.maximum_filter(data, self.size, **self.filter_kwargs)

class LaplaceFilter(Filter):
    """
    A Laplace filter, using `scipy.ndimage.laplace`, derived from the generic Filter.
    """

    def __init__(self, data, axes=None, mode='reflect', cval=0.0):
        """
        The initializeer. See the docs of `scipy.ndimage.laplace` for details on the filter parameters
        `mode` and `cval`.

        :param data: The data to filter.
        :type data: ds.DataSet

        :param axes: A list of valid axis identifiers of the axes along which to filter.

        :param mode: The mode parameter, can be `reflect`, `constant`, `nearest`, `mirror`, or `wrap`.
        :type mode: str

        :param cval: The fill value for outside of the edges.
        :type cval: scalar
        """
        Filter.__init__(self, data, axes)
        self.filter_kwargs['mode'] = mode
        self.filter_kwargs['cval'] = cval

    def rawfilter(self, data):
        """
        Handles the numeric filtering in the backend.

        :param data: The raw data to be fed to the scipy.ndimage.laplace filter function.
        :type data: array-like

        :return: filtered data
        :rtype np.ndarray
        """
        assert len(data.shape) == len(self.axes_filtered), "Wrong data dimensionality."
        return scipy.ndimage.laplace(data, **self.filter_kwargs)



if __name__ == "__main__":
    print("Initializing...")
    testdatah5 = "filtertest_kspace.hdf5"
    data = ds.DataSet.from_h5(testdatah5)
    # gausstest = GaussFilter(data,
    #                         (u.to_ureg(0.05, '1/angstrom'), u.to_ureg(0.05, '1/angstrom')),
    #                         ['k_x', 'k_y'],
    #                         order=[0, 0],
    #                         mode='constant')
    # print("Calculating...")
    # gausstest.data_add_filtered('counts')
    # mediantest = MedianFilter(data,
    #                           ['k_x', 'k_y'],
    #                           (u.to_ureg(0.05, '1/angstrom'), u.to_ureg(0.05, '1/angstrom')),
    #                           mode='constant',
    #                           cval=0)
    # print("Calculating...")
    # mediantest.data_add_filtered('counts')
    # maximumtest = MaximumFilter(data,
    #                              ['k_x', 'k_y'],
    #                              (u.to_ureg(0.05, '1/angstrom'), u.to_ureg(0.05, '1/angstrom')),
    #                              mode='constant',
    #                              cval=0)
    # print("Calculating...")
    # maximumtest.data_add_filtered('counts')
    laplacetest = LaplaceFilter(data,
                                 ['k_x', 'k_y'],
                                 mode='constant',
                                 cval=0)
    print("Calculating...")
    laplacetest.data_add_filtered('counts')
    print("Saving...")
    data.saveh5('filtertest_out.hdf5')
    print("... done.")
