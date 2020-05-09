"""
FFT Script based on the work of Philip.
That is why unused functions e.g. peak_detect are in this file and some notations are strange
"""

from scipy import fftpack
import scipy.signal as signal
from scipy.optimize import curve_fit
import os
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


# --filters
def bandpass(data, lowcut, highcut, srate, order, Nfreqs=5000):
    nyq = 1 / (2 * srate)
    #    nyq = 1.
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    w, h = signal.freqz(b, a, worN=Nfreqs)
    res = signal.filtfilt(b, a, data)
    return [res, w, h]


def lowpass(data, highcut, srate, order, Nfreqs=5000):
    nyq = 1 / (2 * srate)
    #    nyq = 1.
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='low')
    w, h = signal.freqz(b, a, worN=Nfreqs)
    res = signal.filtfilt(b, a, data)
    return [res, w, h]


def highpass(data, lowcut, srate, order, Nfreqs=5000):
    nyq = 1 / (2 * srate)
    #    nyq = 1.
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='high')
    w, h = signal.freqz(b, a, worN=Nfreqs)
    res = signal.filtfilt(b, a, data)
    return [res, w, h]


def doFFT_Filter(timedata, deltaT=0.4, w_c=0.375, d0=0.12, d1=0.075, d2=0.05, d3=0.025):
    '''
    The frequency unit is PHz FFS.

    Wenn der Filter scheisse aussieht an der order drehen!!!

    :param timedata:
    :param deltaT: in fs for default filter values
    :param w_c: Center frequency, default 0.375 (PHz, corresponding to 800nm)
    :param d0:	Lowpass right limit
    :param d1:	First bandpass around w_c +- d1
    :param d2:	Second bandpass around 2*w_c +- d2
    :param d3:	Third bandpass around 3*w_c +- d3
    :return: in fs
    '''
    freqdata = fftpack.fftshift(fftpack.fft(timedata))

    phase = -np.angle(freqdata)
    fticks = fftpack.fftshift(fftpack.fftfreq(freqdata.size, deltaT))

    filtdata3, w3, h3 = highpass(timedata, 3 * w_c - d3, deltaT, 8)  # 1.125
    filtdata2, w2, h2 = bandpass(timedata, 2 * w_c - d2, 2 * w_c + d2, deltaT, 5)  # 0.75
    filtdata1, w1, h1 = bandpass(timedata, 1 * w_c - d1, 1 * w_c + d1, deltaT, 5)  # 800nm =0.375
    filtdata0, w0, h0 = lowpass(timedata, d0, deltaT, 5)

    # Calculate Frequency axis
    prefactor = (1 / deltaT * 0.5 / np.pi)
    return (fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (
        prefactor * w0, prefactor * w1, prefactor * w2, prefactor * w3), (h0, h1, h2, h3), deltaT


# -----------------------------------------------------------------------------------------------------

class Butterfilter(object):
    def __init__(self, sampling_delta, lowcut=None, highcut=None, order=5):
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
            b, a = signal.butter(order, [low, high], btype='band')
        elif lowcut is not None:  # highpass
            low = lowcut / nyq
            b, a = signal.butter(order, low, btype='high')
        elif highcut is not None:  # lowpass
            high = highcut / nyq
            b, a = signal.butter(order, high, btype='low')
        self.b, self.a = b, a
        self.sampling_delta = sampling_delta
        self.order = order

    def response(self, n_freqs=5000):
        w, h = signal.freqz(self.b, self.a, worN=n_freqs)
        prefactor = (1 / self.sampling_delta * 0.5 / consts.pi_float).to(self.freq_unit)
        return prefactor * w, h

    def filtered(self, data, axis=-1):
        return signal.filtfilt(self.b, self.a, data, axis=axis)


class FrequencyFilter(object):
    default_widths = u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz')

    def __init__(self, data, fundamental_frequency, axis=0, max_order=2, widths=None, butter_orders=5):
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
        s = full_slice(s, self.indata.dimensions)
        if s[self.filter_axis_id] != np.s_[:]:
            warnings.warn("Frequency filtering a slice that is not full along the filter axis might return bad results")
        df_in = self.indata.get_datafield(df)
        timedata = df_in.data[s]
        filtered_data = self.butters[component].filtered(timedata, axis=self.filter_axis_id)
        return u.to_ureg(filtered_data, df_in.get_unit())

    def filter_direct(self, timedata, component):
        return self.butters[component].filtered(timedata, axis=self.filter_axis_id)

    def filter_data(self, components=None, h5target=None, dfs=None, add_to_indata=False):
        # Handle Parameters:
        if components is None:
            components = list(range(len(self.butters)))
        else:
            assert all([i < len(self.butters) for i in components]), "Frequency filter not available."

        if dfs is None:
            dfs = list(range(len(self.indata.dlabels)))
        else:
            dfs = [self.indata.get_datafield_index(l) for l in dfs]

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
            axes = [self.indata.get_axis(l) for l in self.indata.axlabels]

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
                    number_of_calcs = np.prod([df_in.shape[i] // df_in.data.ds_data.chunks[i] for i in iterdims])
                    progress_counter = 0

                for s in sample_outdf.data.iterchunkslices(dims=iterdims):
                    timedata = df_in.data[s]
                    for comp in components:
                        df_out = outdata.get_datafield(self.indata.get_datafield(i_df).label + '_omega{0}'.format(comp))
                        df_out.data[s] = self.filter_direct(timedata, comp)

                    if verbose:
                        progress_counter += 1
                        tpf = ((time.time() - start_time) / float(progress_counter))
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
        responses = []
        frequencies = None
        for b in self.butters:
            freqs, response = b.response(n_freqs)
            if frequencies is None:
                frequencies = freqs
            else:
                assert np.allclose(freqs, frequencies), "Butters giving inconsistent frequencies."
            responses.append(response)
        das = [ds.DataArray(responses[i], label="filter curve omega{0}".format(i)) for i in range(len(self.butters))]
        data = ds.DataSet("Frequency Filter Response Functions",
                          das,
                          [ds.Axis(frequencies, label='frequency')])
        return data


class FFT(object):
    def __init__(self, data, axis=0, transformed_axis_unit=None):
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
            self.axis_freq_unit = transformed_axis_unit

        self.result = None

    def transformed_axis(self, unit=None, label=None, plotlabel=None):
        probe_slice = tuple([0 if i != self.axis_to_transform_id else np.s_[:] for i in range(self.indata.dimensions)])
        ax_size = fftpack.fftshift(fftpack.fft(self.indata.get_datafield(0).data[probe_slice].magnitude)).size
        fticks = fftpack.fftshift(fftpack.fftfreq(ax_size, self.sampling_delta.magnitude))

        if label is None:
            label = "FFT-inverse_" + self.axis_to_transform.get_label()
        ax = ds.Axis(fticks, 1 / self.axis_to_transform.units, label=label, plotlabel=plotlabel)
        if unit is not None:
            ax.set_unit(unit)
        else:
            ax.set_unit(self.axis_freq_unit)
        return ax

    def __getitem__(self, s):
        return self.fftslice(s)

    def fftslice(self, s, df=0):
        s = full_slice(s, self.indata.dimensions)
        if s[self.axis_to_transform_id] != np.s_[:]:
            warnings.warn("FFT of a slice that is not full along the FFT axis might return bad results")
        df_in = self.indata.get_datafield(df)
        timedata = df_in.data[s]
        freqdata = fftpack.fftshift(fftpack.fft(timedata, axis=self.axis_to_transform_id),
                                    axes=self.axis_to_transform_id)
        return u.to_ureg(freqdata, df_in.get_unit())

    def fft(self, h5target=None, dfs=None):
        if dfs is None:
            dfs = list(range(len(self.indata.dlabels)))
        else:
            dfs = [self.indata.get_datafield_index(l) for l in dfs]

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
                    number_of_calcs = np.prod([df_in.shape[i] // df_in.data.ds_data.chunks[i] for i in iterdims])
                    progress_counter = 0

                for s in df_out.data.iterchunkslices(dims=iterdims):
                    df_out.data[s] = self.fftslice(s, i_df)

                    if verbose:
                        progress_counter += 1
                        tpf = ((time.time() - start_time) / float(progress_counter))
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

    # # Test FFT in memory:
    # fftdatah5 = testdatah5.replace(".hdf5", "_roi_FFT.hdf5")
    # fft = FFT(testroi, 'delay', 'PHz')
    # fftdata = fft.fft()
    # fftdata.saveh5(fftdatah5)
    #
    # # Test Filtering in memory:
    # filtereddatah5 = testdatah5.replace(".hdf5", "_roi_filtered.hdf5")
    # filterobject = FrequencyFilter(testroi,
    #                                (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
    #                                'delay',
    #                                max_order=2,
    #                                widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
    #                                butter_orders=[5, 5, 5])
    # filtereddata = filterobject.filter_data()
    # filtereddata.saveh5(filtereddatah5)
    # responsedata = filterobject.response_data()
    # responsedata.saveh5("butter filter response.hdf5")
    #
    # # Test Filtering into Set:
    # testfile = "frequencytestdata.hdf5"
    # testdata_small = testroi.get_DataSet(h5target=testfile)
    # testdata_small.saveh5()
    # filterobject = FrequencyFilter(testdata_small,
    #                                (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
    #                                'delay',
    #                                max_order=2,
    #                                widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
    #                                butter_orders=[5, 5, 5])
    # filterobject.filter_data(add_to_indata=True)
    # testdata_small.saveh5()

    # # Test FFT on h5:
    # fftdatah5 = testdatah5.replace(".hdf5", "_FFT.hdf5")
    # fft = FFT(testdata, 'delay', 'PHz')
    # fftdata = fft.fft(h5target=fftdatah5)
    # fftdata.saveh5()

    # Test Filtering on h5:
    filtereddatah5 = testdatah5.replace(".hdf5", "_filtered.hdf5")
    filterobject = FrequencyFilter(testdata,
                                   (consts.c / u.to_ureg(800, 'nm')).to('PHz'),
                                   'delay',
                                   max_order=2,
                                   widths=u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz'),
                                   butter_orders=[5, 5, 5])
    filtereddata = filterobject.filter_data(h5target=filtereddatah5)
    filtereddata.saveh5()

    # import pathlib as pathlib
    #
    # # ----------------------Load Data-------------------------------------
    # path = pathlib.Path(
    #     r"E:\NFC15\20171207 ZnO+aSiH\01 DLD PSI -3 to 150 fs step size 400as\Maximamap\Driftcorrected\summed_runs")
    # file = "ROI_data.hdf5"
    # data_file = path / file
    # data3d = ds.DataSet.from_h5file(os.fspath(data_file), h5target=os.fspath(path / 'chache.hdf5'))
    # for roi in range(0, 10):
    #     roi0_sumpicture = np.sum(data3d.datafields[0][roi][:, 0:45], axis=1)  # TODO: use other stuff than sumpicture
    #     xAxis = data3d.get_axis('delay').data.to('femtoseconds').magnitude
    #     timedata = roi0_sumpicture.magnitude
    #
    #     # ----------------------Apply function to calculate the FFT-------------------------------------
    #     (fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (
    #         h0, h1, h2, h3), deltaDim1 = doFFT_Filter(timedata)
    #
    #     # ----------------------Plotting section starts here-------------------------------------
    #     ###
    #     pltdpi = 100
    #     fontsize_label = 14  # schriftgröße der Labels
    #     freq_lim = 1.25  # limit of frequency spectrum in plots
    #
    #     # ----------------------Phase-------------------------------------
    #     fig = plt.figure(figsize=(8, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' phase ')
    #     plt.xticks(fontsize=fontsize_label)
    #     ax1 = fig.add_subplot(111)
    #     plt.yticks(fontsize=fontsize_label)
    #     ax1.set_xlim(-0.1, freq_lim)
    #     ax1.set_xlabel('Frequenz (PHz)', fontsize=fontsize_label)
    #     ax1.locator_params(nbins=10)
    #     #   ax1.set_yscale('log')
    #     ax1.set_ylabel('norm. spektrale Intensität', fontsize=fontsize_label)
    #     # ax1.set_ylim(ymin=1)
    #     ax1.plot(fticks, abs(freqdata), c='black')
    #     ax1Xs = ax1.get_xticks()[2:]
    #
    #     ax2 = ax1.twiny()
    #     ax2Xs = []
    #
    #     for X in ax1Xs:
    #         ax2Xs.append(299792458.0 / X / 10 ** 6)
    #
    #     for i in range(len(ax2Xs)): ax2Xs[i] = "{:.0f}".format(ax2Xs[i])
    #
    #     ax2.set_xticks(ax1Xs[0:len(ax1Xs) - 1])
    #     ax2.set_xlabel('Wellenlänge (nm)', fontsize=fontsize_label)
    #     ax2.set_xbound(ax1.get_xbound())
    #     ax2.set_xticklabels(ax2Xs, fontsize=fontsize_label)
    #
    #     ax3 = ax1.twinx()
    #     ax3.set_xlim(-0.1, freq_lim)
    #     ax3.set_ylabel('Phase', fontsize=fontsize_label)
    #     plt.yticks(fontsize=fontsize_label)
    #     ax3.plot(fticks, phase, c='orange')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-phase.png'))
    #
    #     # ----------------------Spektrum-------------------------------------
    #     fig = plt.figure(figsize=(8, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' spectrum ')
    #     plt.xticks(fontsize=fontsize_label)
    #     ax1 = fig.add_subplot(111)
    #
    #     ax1.set_xlim(-0.1, freq_lim)
    #     ax1.set_xlabel('Frequenz (PHz)', fontsize=fontsize_label)
    #     ax1.locator_params(nbins=10)
    #     ax1.set_yscale('log')
    #     ax1.set_ylim(bottom=10, top=max(abs(freqdata)))
    #     ax1.set_ylabel('norm. spektrale Intensität', fontsize=fontsize_label)
    #     # ax1.set_ylim(bottom=1)
    #     ax1.plot(fticks, abs(freqdata), c='black')
    #     ax1Xs = ax1.get_xticks()[2:]
    #     plt.yticks(fontsize=fontsize_label)
    #
    #     ax2 = ax1.twiny()
    #     ax2Xs = []
    #
    #     for X in ax1Xs:
    #         ax2Xs.append(299792458.0 / X / 10 ** 6)
    #
    #     for i in range(len(ax2Xs)): ax2Xs[i] = "{:.0f}".format(ax2Xs[i])
    #
    #     ax2.set_xticks(ax1Xs[0:len(ax1Xs) - 1])
    #     ax2.set_xlabel('Wellenlänge (nm)', fontsize=fontsize_label)
    #     ax2.set_xbound(ax1.get_xbound())
    #     ax2.set_xticklabels(ax2Xs, fontsize=fontsize_label)
    #
    #     ax3 = ax1.twinx()
    #     ax3.set_xlim(-0.1, freq_lim)
    #     ax3.set_ylabel('Filter', fontsize=fontsize_label)
    #
    #     ax3.plot(w0, abs(h0), c='blue')
    #     ax3.plot(w1, abs(h1), c='green')
    #     ax3.plot(w2, abs(h2), c='green')
    #     ax3.plot(w3, abs(h3), c='green')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-spec.png'))
    #
    #     # ----------------------w0 Komponente-------------------------------------
    #     plt.figure(figsize=(16, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' w_0 ')
    #     plt.locator_params(axis='x', nbins=20)
    #     plt.xlabel('T (fs)')
    #     plt.ylabel('Intensität (a.u.)')
    #     plt.plot(xAxis, filtdata0, c='blue')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w0.png'))
    #     # ----------------------w1 Komponente-------------------------------------
    #
    #     plt.figure(figsize=(16, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' w_1 ')
    #     plt.locator_params(axis='x', nbins=20)
    #     plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
    #     plt.ylabel('norm. Intensität', fontsize=fontsize_label)
    #     plt.xticks(fontsize=fontsize_label)
    #     plt.yticks(fontsize=fontsize_label)
    #     plt.plot(xAxis, normAC(filtdata1), c='green')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w1.png'))
    #     # ----------------------w2 Komponente-------------------------------------
    #
    #     plt.figure(figsize=(16, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' w_2 ')
    #     plt.locator_params(axis='x', nbins=20)
    #     plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
    #     plt.ylabel('norm. Intensität', fontsize=fontsize_label)
    #     plt.plot(xAxis, normAC(filtdata2), c='red')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w2.png'))
    #     # ----------------------w3 Komponente-------------------------------------
    #
    #     plt.figure(figsize=(16, 4), dpi=pltdpi)
    #     plt.title('Roi' + str(roi) + ' w_3 ')
    #     plt.locator_params(axis='x', nbins=20)
    #     plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
    #     plt.ylabel('norm. Intensität', fontsize=fontsize_label)
    #     plt.plot(xAxis, normAC(filtdata3), c='orange')
    #     plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w3.png'))

    print('done')
