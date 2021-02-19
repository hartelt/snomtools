"""
This module contains FFT and frequency filtering classes for DataSets.
"""

from scipy import fftpack
import scipy.signal as signal
import sys
import numpy as np
import snomtools.data.datasets as ds
import snomtools.calcs.units as u
import snomtools.calcs.constants as consts
from snomtools.data.h5tools import probe_chunksize
from snomtools.data.tools import full_slice
import warnings

if '-v' in sys.argv or __name__ == "__main__":
    verbose = True
else:
    verbose = False


class Butterfilter(object):
    """
    This class is a convenience implementation of the butter filter implemented in
    `scipy.signal.butter
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter>`.
    It automatically creates a lowpass, highpass or bandpass filter depending on the parameters given.
    Parameters can be given as Quantities and units are handled accordingly.
    :func:`~Butterfilter.response` can be used to output the characteristic frequency response.
    Data can be filtered using :func:`~Butterfilter.filtered`, which filters data along a specified axis
    along which the values are spaced by the `sampling_delta` given on initialization.

    .. automethod:: __init__
    """

    def __init__(self, sampling_delta, lowcut=None, highcut=None, order=5):
        """
        The initializer of the Butterfilter. All "time and frequency" parameters can and should be given as quantities.
        If raw numerical data is given, it still works, but all values are assumed as dimensionless.

        :param sampling_delta: The spacing of the axis along which to filter the data.
        :type sampling_delta: `pint.Quantity`

        :param lowcut: The low cut, below which the frequency response is faded out.
        :type lowcut: `pint.Quantity`

        :param highcut: The high cut, below which the frequency response is faded out.
        :type highcut: `pint.Quantity`

        :param order: The order of the filter, see scipy docs. If the frequency response looks bad, try increasing this.
        :type order: int
        """
        # Parse Arguments:
        sampling_delta = u.to_ureg(sampling_delta)
        if (lowcut is not None) and (highcut is not None):
            highcut = u.to_ureg(highcut)
            lowcut = u.to_ureg(lowcut)
            assert lowcut.units == highcut.units, "Low and Highcut must be given in same units."
            self.freq_unit = lowcut.units
        elif highcut is not None:
            highcut = u.to_ureg(highcut)
            self.freq_unit = highcut.units
        elif lowcut is not None:
            lowcut = u.to_ureg(lowcut)
            self.freq_unit = lowcut.units
        else:
            raise ValueError("Cannot define filter without high or lowcut.")

        nyq = (1 / (2 * sampling_delta)).to(self.freq_unit)
        self.highcut, self.lowcut = highcut, lowcut
        # Define filters:
        if (lowcut is not None) and (highcut is not None):  # bandpass
            low = lowcut / nyq
            high = highcut / nyq
            # noinspection PyTupleAssignmentBalance,PyTupleAssignmentBalance
            b, a = signal.butter(order, [low, high], btype='band')
        elif lowcut is not None:  # highpass
            low = lowcut / nyq
            # noinspection PyTupleAssignmentBalance
            b, a = signal.butter(order, low, btype='high')
        elif highcut is not None:  # lowpass
            high = highcut / nyq
            # noinspection PyTupleAssignmentBalance
            b, a = signal.butter(order, high, btype='low')
        else:
            raise ValueError("Cannot define filter without high or lowcut.")
        self.b, self.a = b, a
        self.sampling_delta = sampling_delta
        self.order = order

    def response(self, n_freqs=5000):
        """
        The characteristic frequency response of the filter. Returned as a 2-tuple,
        containing frequencies and filter amplitudes.
        The frequency scale given as a Quantity with a unit defined by the low- and highcut upon initialization.
        The amplitudes are given as complex numbers,
        so the absolute value should be used to get the response factor in range 0 to 1.

        :param n_freqs: A number of frequencies to calculate the response for. The outputs will have this length.
        :type n_freqs: int

        :return: The tuple of (frequencies, amplitudes) of the filter response.
        :rtype: tuple[pint.Quantity, numpy.ndarray]
        """
        w, h = signal.freqz(self.b, self.a, worN=n_freqs)
        prefactor = (1 / self.sampling_delta * 0.5 / consts.pi_float).to(self.freq_unit)
        return prefactor * w, h

    def filtered(self, data, axis=-1):
        """
        The frequency filtered data.

        :param data: The data.

        :param axis: The axis of the data along which to apply the frequency filter.
            Of cause the values along this axis should be spaced according to the `sampling_delta` of the filter.
            By default (`-1`), the last axis is filtered.
        :type axis: int

        :return: The filtered data. returned as ndarray with the same shape as the input.
        :rtype: numpy.ndarray
        """
        return signal.filtfilt(self.b, self.a, data, axis=axis)


class FrequencyFilter(object):
    """
    A frequency filter defined on a DataSet, filtering around multiples of a fundamental frequency.
    Typically used for time-resolved data, that is analyzed in multiples of the laser frequencies,
    the so-called omega-components. The filtering is done with butter filters (:class:`Butterfilter`),
    defined for each frequency component.

    Example usage with a DataSet instance containing the time axis `delay`,
    measured with a femtosecond laser with wavelength 800 nm,
    of which the omega_0, omega_1, omega_2 components are extracted:

    .. code-block::

        filterobject = FrequencyFilter(testdata,
                                       (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
                                       'delay',
                                       max_order=2,
                                       widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
                                       butter_orders=[5, 5, 5])
        filtereddata = filterobject.filter_data(h5target='filtered.hdf5')
        filtereddata.saveh5()

    .. automethod:: __init__
    """
    default_widths = u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz')  # Omega components for 800 nm Laser.

    def __init__(self, data, fundamental_frequency, axis=0, max_order=2, widths=None, butter_orders=5):
        """
        The initializer.

        :param data: The data to filter.
        :type data: :class:`~snomtools.data.datasets.DataSet` or :class:`~snomtools.data.datasets.ROI`

        :param fundamental_frequency:
        :type fundamental_frequency: pint.Quantity

        :param axis: The axis along which to apply the frequency filter, given as a valid identifier of the axis.
        :type axis: int or str

        :param max_order: The maximum frequency order to calculate. Default is `2` for up to omega_2 component.
        :type max_order: int

        :param widths: The half-widths of the frequency window around each multiple of the fundamental frequency.
        :type widths: list[pint.Quantity or float]

        :param butter_orders: The order of each butter filter defined for the frequency components.
            Individual orders for each component can be given as a list of ints.
        :type butter_orders: int or list[int]
        """
        assert isinstance(data, (ds.DataSet, ds.ROI))
        self.indata = data
        self.filter_axis_id = data.get_axis_index(axis)
        self.filter_axis = data.get_axis(self.filter_axis_id)

        # Check if axis is ok:
        assert self.filter_axis.is_linspaced(), "Cannot fourier transform an Axis not evenly spaced."
        assert self.filter_axis.size > 1, "Cannot fourier transform a single element."

        self.sampling_delta = self.filter_axis.spacing()

        if fundamental_frequency is u.to_ureg(fundamental_frequency).magnitude:  # Fallback for numerical data.
            fundamental_frequency = u.to_ureg(fundamental_frequency.magnitude, (1 / self.filter_axis.units).units)
        else:
            fundamental_frequency = u.to_ureg(fundamental_frequency)
        assert u.same_dimension((1 / self.filter_axis.units), fundamental_frequency), \
            "Given frequency dimensionality does not match axis."
        self.axis_freq_unit = fundamental_frequency.units
        self.omega = fundamental_frequency

        if widths is None:
            assert max_order <= 3, "Give filter window widths for order >3, defaults are not defined!"
            self.widths = [self.default_widths[i] for i in range(max_order + 1)]
        else:
            self.widths = u.to_ureg(widths, self.axis_freq_unit)

        if isinstance(butter_orders, int):
            # noinspection PyUnusedLocal
            butter_orders = [butter_orders for i in range(max_order + 1)]
        else:
            assert len(butter_orders) == max_order + 1, "Wrong number of Butter orders."

        self.butters = []
        # Initialize lowpass filter for low-frequency component omega_0:
        self.butters.append(Butterfilter(self.sampling_delta, highcut=self.widths[0], order=butter_orders[0]))
        # Initialize bandbass filters for fundamental and its multiples:
        for freq_component in range(1, max_order + 1):
            self.butters.append(Butterfilter(self.sampling_delta,
                                             lowcut=self.omega * freq_component - self.widths[freq_component],
                                             highcut=self.omega * freq_component + self.widths[freq_component],
                                             order=butter_orders[freq_component]))

        self.result = None

    def filteredslice(self, s, component, df=0):
        """
        Filtered slice of the instance input data.

        :param s: A slice addressing a region of the data to return.
            Can be conveniently made with `numpy.s_[]`.
        :type s: slice

        :param component: The frequency component to return.
        :type component: int

        :param df: The DataArray in the given DataSet to filter, can be specified if multiple are present.
            Given as a valid identifier (index or label).
        :type df: int or str

        :return: The filtered data.
        :rtype: numpy.ndarray
        """
        s = full_slice(s, self.indata.dimensions)
        if s[self.filter_axis_id] != np.s_[:]:
            warnings.warn("Frequency filtering a slice that is not full along the filter axis might return bad results")
        df_in = self.indata.get_datafield(df)
        timedata = df_in.data[s]
        filtered_data = self.butters[component].filtered(timedata, axis=self.filter_axis_id)
        return u.to_ureg(filtered_data, df_in.get_unit())

    def filter_direct(self, timedata, component):
        """
        Frequency filter an array given directly as parameter.

        :param timedata: The data to filter. Must have the same number of dimensions as the instance input data.
        :type timedata: numpy.ndarray or pint.Quantity

        :param component: The frequency component to return.
        :type component: int

        :return: The filtered input data, given as ndarray of same shape.
        :rtype: numpy.ndarray
        """
        return self.butters[component].filtered(timedata, axis=self.filter_axis_id)

    # noinspection PyUnusedLocal
    def filter_data(self, components=None, h5target=None, dfs=None, add_to_indata=False):
        """
        Calculate the full frequency components of the instance input data.

        :param components: A list of components to be calculated.
            By default (`None`), all components present are used.
        :type components: None or list[int]

        :param h5target: A HDF5 target to write to, given as a file path or h5py group.
            If this is given, the filtering is performed in a chunk-wise iteration over the data,
            for optimal I/O performance and larger-than-memory support.
            If not given, the FFT will be performed at once in memory.
        :type h5target: str or h5py.Group

        :param dfs: A list of identifiers of DataArrays present in the instance DataSet to filter.
            By default (`None`), all DataArrays present are used.
        :type dfs: None or list[int or str]

        :param add_to_indata: If true, write to the instance input DataSet instead of initializing a new output DataSet.
            In this case, the data of the filtered frequency components are added as new DataArrays
            and a reference to the input data is returned.
        :type add_to_indata: bool

        :return: The DataSet containing the full filtered data.
        :rtype: :class:`~snomtools.data.datasets.DataSet`
        """
        # Handle Parameters:
        if components is None:
            components = list(range(len(self.butters)))
        else:
            assert all([i < len(self.butters) for i in components]), "Frequency filter not available."

        if dfs is None:
            dfs = list(range(len(self.indata.dlabels)))
        else:
            dfs = [self.indata.get_datafield_index(df) for df in dfs]

        new_df_labels = [self.indata.get_datafield(d).label + '_omega{0}'.format(comp)
                         for d in dfs
                         for comp in components]
        new_df_units = [self.indata.get_datafield(d).get_unit()
                        for d in dfs
                        for comp in components]

        # Prepare target dataset:
        if add_to_indata:
            outdata = self.indata
            # Add datafields for filtered data:
            for label, unit in zip(new_df_labels, new_df_units):
                outdata.add_datafield_empty(unit, label)

        else:
            # Prepare DataSet to write to:
            axes = [self.indata.get_axis(label) for label in self.indata.axlabels]

            if h5target:
                chunks = probe_chunksize(self.indata.shape)
                iteration_size = [chunks[i] if i != self.filter_axis_id else self.indata.shape[i]
                                  for i in range(len(self.indata.shape))]
                cache_size_min = np.prod(iteration_size) * len(components) * 8  # 8 Bytes per 64bit number
                use_cache_size = cache_size_min + 64 * 1024 ** 2  # Add 64 MB to be sure.
            else:
                use_cache_size = None

            outdata = ds.DataSet.empty_from_axes("Frequency filtered " + self.indata.label, new_df_labels, axes,
                                                 new_df_units,
                                                 h5target=h5target,
                                                 chunk_cache_mem_size=use_cache_size)

        # For each DataArray, do the actual filtering:
        for i_df in dfs:
            df_in = self.indata.get_datafield(i_df)
            sample_outdf = outdata.get_datafield(
                self.indata.get_datafield(i_df).label + '_omega{0}'.format(components[0]))

            if verbose:
                import time
                print("Frequency filtering: Handling dataset: {0}".format(df_in))
                print("Calculating filtered data of shape: ", df_in.shape)
                if h5target:
                    print("... with chunks of shape: ", sample_outdf.data.ds_data.chunks)
                    # noinspection PyUnboundLocalVariable
                    print("... using cache size {0:d} MB".format(use_cache_size // 1024 ** 2))
                elif add_to_indata:
                    print("... with chunks of shape: ", sample_outdf.data.ds_data.chunks)
                else:
                    print("... in memory")
                start_time = time.time()

            if h5target or add_to_indata:
                # Iterate over chunks and do the Filtering:
                iterdims = [i for i in range(self.indata.dimensions) if i != self.filter_axis_id]
                if verbose:
                    number_of_calcs = np.prod(
                        [sample_outdf.shape[i] // sample_outdf.data.ds_data.chunks[i] for i in iterdims])
                    progress_counter = 0

                for s in sample_outdf.data.iterchunkslices(dims=iterdims):
                    timedata = df_in.data[s]
                    for comp in components:
                        df_out = outdata.get_datafield(self.indata.get_datafield(i_df).label + '_omega{0}'.format(comp))
                        df_out.data[s] = self.filter_direct(timedata, comp)

                    if verbose:
                        # noinspection PyUnboundLocalVariable
                        progress_counter += 1
                        # noinspection PyUnboundLocalVariable
                        tpf = ((time.time() - start_time) / float(progress_counter))
                        # noinspection PyUnboundLocalVariable
                        etr = tpf * (number_of_calcs - progress_counter)
                        print("Filter Chunk {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(progress_counter,
                                                                                                    number_of_calcs,
                                                                                                    etr, tpf))
            else:
                # Do the whole thing at once, user says it should fit into RAM by not providing h5target
                for comp in components:
                    df_out = outdata.get_datafield(self.indata.get_datafield(i_df).label + '_omega{0}'.format(comp))
                    df_out.data = self.filteredslice(np.s_[:], comp, i_df)

        self.result = outdata
        return outdata

    def response_data(self, n_freqs=5000):
        """
        Calculate the frequency response of the filter functions defined and return them as 1D-Dataset,
        containing the frequency axis and a DataArray with complex filter amplitudes for each frequency component.
        The DataArrays will be written in the filter order and labeled `filter response omegaN` for a component N.

        :param n_freqs: Number of frequency steps to calculate for.
        :type n_freqs: int

        :return: The DataSet containing the frequency responses.
        :rtype: :class:`~snomtools.data.datasets.DataSet`
        """
        responses = []
        frequencies = None
        for b in self.butters:
            freqs, response = b.response(n_freqs)
            if frequencies is None:
                frequencies = freqs
            else:
                assert np.allclose(freqs, frequencies), "Butters giving inconsistent frequencies."
            responses.append(response)
        das = [ds.DataArray(responses[i], label="filter response omega{0}".format(i)) for i in range(len(self.butters))]
        data = ds.DataSet("Frequency Filter Response Functions",
                          das,
                          [ds.Axis(frequencies, label='frequency')])
        return data


class FFT(object):
    """
    A fast Fourier-Tranform (FFT), working on a given DataSet.

    Example usage with a DataSet instance containing the time axis `delay`,
    and performing the FFT to generate data with a frequency axis with unit PHz:

    .. code-block::

        fft = FFT(testdata, 'delay', 'PHz')
        fftdata = fft.fft(h5target='FFT.hdf5')
        fftdata.saveh5()

    .. automethod:: __init__
    """

    def __init__(self, data, axis=0, transformed_axis_unit=None):
        """
        The initializer.

        :param data: The data to transform.
        :type data: :class:`~snomtools.data.datasets.DataSet` or :class:`~snomtools.data.datasets.ROI`

        :param axis: The axis along which to apply the FFT, given as a valid identifier of the axis.
        :type axis: int or str

        :param transformed_axis_unit: The unit for the generated frequency axis.
            Must be compatible with the unit of the axis to transform,
            meaning the two units must be inverse of each other.
            If not given, the inverse unit of the Axis to transform is taken,
            e.g. an Axis of unit `fs` is transformed to `1/fs`.
        :type transformed_axis_unit: str or pint.Unit
        """
        assert isinstance(data, (ds.DataSet, ds.ROI))
        self.indata = data
        self.axis_to_transform_id = data.get_axis_index(axis)
        self.axis_to_transform = data.get_axis(self.axis_to_transform_id)

        # Check if axis is ok:
        assert self.axis_to_transform.is_linspaced(), "Cannot fourier transform an Axis not evenly spaced."
        assert self.axis_to_transform.size > 1, "Cannot fourier transform a single element."

        self.sampling_delta = self.axis_to_transform.spacing()
        if transformed_axis_unit is None:
            self.axis_freq_unit = (1 / self.axis_to_transform.units).units
        else:
            assert u.same_dimension(self.axis_to_transform.data, 1 / u.to_ureg(1, transformed_axis_unit))
            self.axis_freq_unit = transformed_axis_unit

        self.result = None

    def transformed_axis(self, unit=None, label=None, plotlabel=None):
        """
        Calculate the transformed axis, e.g. the frequency axis from the time axis.
        To calculate the frequency values, an FFT of a 1-dimensional probe slice is performed
        and the frequency ticks are calculated from the result.

        :param unit: Convert the calculated axis to this unit, deviating from `self.axis_freq_unit`.
        :type unit: None or str

        :param label: A label for the built axis. If none is given, the prefix `FFT_inverse_`
            is put before the label of the original axis.
        :type label: None or str

        :param plotlabel: An (optional) plotlabel for the generated axis.
        :type plotlabel: None or str

        :return: The frequency axis resulting from the FFT.
        :rtype: :class:`~snomtools.data.datasets.Axis`
        """
        # Calculate the frequency ticks:
        probe_slice = tuple([0 if i != self.axis_to_transform_id else np.s_[:] for i in range(self.indata.dimensions)])
        ax_size = fftpack.fftshift(fftpack.fft(self.indata.get_datafield(0).data[probe_slice].magnitude)).size
        fticks = fftpack.fftshift(fftpack.fftfreq(ax_size, self.sampling_delta.magnitude))

        # Build the Axis object with the correct unit and return it:
        if label is None:
            label = "FFT-inverse_" + self.axis_to_transform.get_label()
        ax = ds.Axis(fticks, 1 / self.axis_to_transform.units, label=label, plotlabel=plotlabel)
        if unit is not None:
            ax.set_unit(unit)
        else:
            ax.set_unit(self.axis_freq_unit)
        return ax

    def __getitem__(self, s):
        """
        Get the fourier-transformed data of a specific slice of the first DataArray of the instance input.
        This just links to :func:`FFT.fftslice`.

        :param s: The slice, addressing a selection of the instance input data.
        :type s: slice

        :return: The fourier-transformed data for the selection.
        :rtype: pint.Quantity
        """
        return self.fftslice(s)

    def fftslice(self, s, df=0):
        """
        Get the fourier-transformed data of a specific slice of a DataArray of the instance input.

        :param s: The slice, addressing a selection of the instance input data.
            Can be conveniently made with `numpy.s_[]`.
        :type s: slice

        :param df: An identifier (index or label) of the DataArray to transform.
        :type df: int or str

        :return: The fourier-transformed data for the selection.
        :rtype: pint.Quantity
        """
        s = full_slice(s, self.indata.dimensions)
        if s[self.axis_to_transform_id] != np.s_[:]:
            warnings.warn("FFT of a slice that is not full along the FFT axis might return bad results")
        df_in = self.indata.get_datafield(df)
        timedata = df_in.data[s].magnitude
        freqdata = fftpack.fftshift(fftpack.fft(timedata, axis=self.axis_to_transform_id),
                                    axes=self.axis_to_transform_id)
        return u.to_ureg(freqdata, df_in.get_unit())

    def fft(self, h5target=None, dfs=None):
        """
        Calculate the fourier transform of the instance input data.

        :param h5target: A HDF5 target to write to, given as a file path or h5py group.
            If this is given, the FFT is performed in a chunk-wise iteration over the data,
            for optimal I/O performance and larger-than-memory support.
            If not given, the FFT will be performed at once in memory.
        :type h5target: str or h5py.Group

        :param dfs: A list of identifiers of DataArrays present in the instance DataSet to filter.
            By default (`None`), all DataArrays present are used.
        :type dfs: None or list[int or str]

        :return: The DataSet containing the full fourier-transformed data.
        :rtype: :class:`~snomtools.data.datasets.DataSet`
        """
        if dfs is None:
            dfs = list(range(len(self.indata.dlabels)))
        else:
            dfs = [self.indata.get_datafield_index(df) for df in dfs]

        # Prepare DataSet to write to:
        transformed_axis = self.transformed_axis()
        axes = [self.indata.get_axis(i) if i != self.axis_to_transform_id else transformed_axis
                for i in range(self.indata.dimensions)]
        transformed_shape = tuple([len(ax) for ax in axes])

        if h5target:
            chunks = probe_chunksize(transformed_shape)
            iteration_size = [chunks[i] if i != self.axis_to_transform_id else transformed_shape[i]
                              for i in range(len(transformed_shape))]
            cache_size_min = np.prod(iteration_size) * 16  # 8 Bytes per 64bit number but complex.
            use_cache_size = cache_size_min + 64 * 1024 ** 2  # Add 64 MB to be sure.
        else:
            use_cache_size = None

        df_labels = ["spectral_" + self.indata.get_datafield(i).label for i in dfs]
        outdata = ds.DataSet.empty_from_axes("FFT of " + self.indata.label, df_labels, axes,
                                             [self.indata.get_datafield(i).get_unit() for i in dfs],
                                             h5target=h5target,
                                             chunk_cache_mem_size=use_cache_size,
                                             dtypes=np.complex64)

        # For each DataArray:
        for i_df in dfs:
            df_in = self.indata.get_datafield(i_df)
            df_out = outdata.get_datafield("spectral_" + self.indata.get_datafield(i_df).label)
            if verbose:
                import time
                print("FFT: Handling dataset: {0}".format(df_in))
                print("Calculating FFT data of shape: ", df_out.shape)
                if h5target:
                    print("... with chunks of shape: ", df_out.data.ds_data.chunks)
                    print("... using cache size {0:d} MB".format(use_cache_size // 1024 ** 2))
                else:
                    print("... in memory")
                start_time = time.time()

            if h5target:
                # Iterate over chunks and do the FFT:
                iterdims = [i for i in range(self.indata.dimensions) if i != self.axis_to_transform_id]
                if verbose:
                    number_of_calcs = np.prod([df_out.shape[i] // df_out.data.ds_data.chunks[i] for i in iterdims])
                    progress_counter = 0

                for s in df_out.data.iterchunkslices(dims=iterdims):
                    df_out.data[s] = self.fftslice(s, i_df)

                    if verbose:
                        # noinspection PyUnboundLocalVariable
                        progress_counter += 1
                        # noinspection PyUnboundLocalVariable
                        tpf = ((time.time() - start_time) / float(progress_counter))
                        # noinspection PyUnboundLocalVariable
                        etr = tpf * (number_of_calcs - progress_counter)
                        print("FFT Chunk {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(progress_counter,
                                                                                                 number_of_calcs,
                                                                                                 etr, tpf))
            else:
                # Do the whole thing at once, user says it should fit into RAM by not providing h5target
                df_out = outdata.get_datafield("spectral_" + self.indata.get_datafield(i_df).label)
                df_out.data = self.fftslice(np.s_[:], i_df)

        self.result = outdata
        return outdata


if __name__ == '__main__':
    testdatah5 = "PVL_r10_pol244_25µm_-50-150fs_run1.hdf5"
    testdata = ds.DataSet.in_h5(testdatah5, chunk_cache_mem_size=1000 * 1024 ** 2)
    testroi = ds.ROI(testdata, {'x': [500, 505], 'y': [500, 506]}, by_index=True)

    # Test FFT in memory:
    fftdatah5 = testdatah5.replace(".hdf5", "_roi_FFT.hdf5")
    fft = FFT(testroi, 'delay', 'PHz')
    fftdata = fft.fft()
    fftdata.saveh5(fftdatah5)

    # Test Filtering in memory:
    filtereddatah5 = testdatah5.replace(".hdf5", "_roi_filtered.hdf5")
    filterobject = FrequencyFilter(testroi,
                                   (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
                                   'delay',
                                   max_order=2,
                                   widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
                                   butter_orders=[5, 5, 5])
    filtereddata = filterobject.filter_data()
    filtereddata.saveh5(filtereddatah5)
    responsedata = filterobject.response_data()
    responsedata.saveh5("butter filter response.hdf5")

    # Test FFT on h5:
    fftdatah5 = testdatah5.replace(".hdf5", "_FFT.hdf5")
    fft = FFT(testroi, 'delay', 'PHz')
    fftdata = fft.fft(h5target=fftdatah5)
    fftdata.saveh5()

    # Test Filtering on h5:
    filtereddatah5 = testdatah5.replace(".hdf5", "_filtered.hdf5")
    filterobject = FrequencyFilter(testroi,
                                   (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
                                   'delay',
                                   max_order=2,
                                   widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
                                   butter_orders=[5, 5, 5])
    filtereddata = filterobject.filter_data(h5target=filtereddatah5)
    filtereddata.saveh5()

    # Test Filtering into Set:
    testfile = "frequencytestdata.hdf5"
    testdata_small = testroi.get_DataSet(h5target=testfile)
    testdata_small.saveh5()
    filterobject = FrequencyFilter(testdata_small,
                                   (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
                                   'delay',
                                   max_order=2,
                                   widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
                                   butter_orders=[5, 5, 5])
    filterobject.filter_data(add_to_indata=True)
    testdata_small.saveh5()

    print('done')
