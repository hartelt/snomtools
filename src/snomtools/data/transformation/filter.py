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
    def __init__(self, data, axes=None):
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
        return None

    def dataarray_filtered(self, data_id, label=None, plotlabel=None,
                           h5target=None,
                           chunks=True,
                           dtype=None,
                           compression='gzip', compression_opts=4):
        """
        Rotates a DataArray of the DataSet with the nd filter.
        This is a preliminary version: It takes the full data into a numpy array and rotates it in RAM.

        :param data_id: A valid id of the DataArray to rotate.

        :return: The filtered DataArray.
        :rtype: ds.DataArray
        """
        # TODO: Implement this for arbitrary large data, with chunk-wise optimial I/O!
        d_original = self.data_original.get_datafield(data_id)
        if label is None:
            label = 'filtered ' + d_original.get_label()
        if plotlabel is None:
            plotlabel = d_original.get_plotlabel()
        if dtype is None:
            dtype = d_original.data.dtype

        outda = ds.DataArray.make_empty(d_original.shape, d_original.units, label, plotlabel,
                                        h5target, chunks, dtype, compression, compression_opts)
        for s in outda.data.iterflatslices(dims=self.axes_kept):
            raw_data = d_original.data[s].magnitude
            filtered_data = self.rawfilter(raw_data)
            outda[s] = filtered_data
        return outda

    def filter_data(self, label=None, h5target=None):
        """
        Filters the full DataSet.
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
        """
        filterparam_list = self.filterparam_listify(filterparam)
        outlist = []
        for param, axis in zip(filterparam_list, self.axes_filtered):
            if u.is_quantity(param):
                assert axis.is_linspaced(), "Trying to give filter parameter as Axis value for non-linspaced axis!"
                pxvalue = (param / axis.spacing()).magnitude
                if require_int:
                    pxvalue = round(pxvalue)
                outlist.append(pxvalue)
            else:
                outlist.append(param)
        return outlist


class GaussFilter(Filter):
    def __init__(self, data, sigma, axes=None, order=0, mode='reflect', cval=0.0, truncate=4.0):
        Filter.__init__(self, data, axes)
        self.sigma = self.filterparam_indexify(sigma)
        self.order = self.filterparam_listify(order)
        self.filter_kwargs['order'] = order
        self.filter_kwargs['mode'] = mode
        self.filter_kwargs['cval'] = cval
        self.filter_kwargs['truncate'] = truncate

    def rawfilter(self, data):
        assert len(data.shape) == len(self.axes_filtered), "Wrong data dimensionality."
        return scipy.ndimage.gaussian_filter(data, self.sigma, **self.filter_kwargs)
