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
from snomtools.data.h5tools import probe_chunksize
import matplotlib.pyplot as plt

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
        nyq = 1 / (2 * sampling_delta)
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
        else:
            raise ValueError("Cannot define filter without high or lowcut.")
        self.b, self.a = b, a
        self.sampling_delta = sampling_delta

    def response(self, Nfreqs=5000):
        w, h = signal.freqz(self.b, self.a, worN=Nfreqs)
        prefactor = (1 / self.sampling_delta * 0.5 / np.pi)
        return w, prefactor * h

    def filtered(self, data, axis=-1):
        return signal.filtfilt(self.b, self.a, data, axis=axis)


class FrequencyFilter(object):
    def __init__(self, data, fundamental_frequency, axis=0, max_order=2, widths=None):
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

        default_widths = u.to_ureg([0.12, 0.075, 0.05, 0.025], 'PHz')  # 800nm Laser omega-components
        if widths is None:
            assert max_order <= 3, "Give filter window widths for order >3, defaults are not defined!"
            self.widths = [default_widths[i] for i in range(max_order + 1)]
        else:
            self.widths = u.to_ureg(widths, self.axis_freq_unit)

        # TODO: Initialize filters.

        self.result = None

    # TODO: Filter data.


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

    def fft(self, h5target=None):
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

        df_labels = ["spectral_" + l for l in self.indata.dlabels]
        outdata = ds.DataSet.empty_from_axes("FFT of " + self.indata.label, df_labels, axes,
                                             [d.get_unit() for d in self.indata.datafields],
                                             h5target=h5target,
                                             chunk_cache_mem_size=use_cache_size,
                                             dtypes=np.complex64)

        # For each DataArray:
        for i_df in range(len(self.indata.datafields)):
            df_in = self.indata.get_datafield(i_df)
            df_out = outdata.get_datafield(i_df)
            if verbose:
                import time
                print("FFT: Handling dataset: {0}".format(df_in))
                print("Calculating FFT data of shape: ", df_in.shape)
                if h5target:
                    print("... with chunks of shape: ", df_in.data.ds_data.chunks)
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
                    timedata = df_in.data[s]
                    freqdata = fftpack.fftshift(fftpack.fft(timedata, axis=self.axis_to_transform_id),
                                                axes=self.axis_to_transform_id)
                    df_out.data[s] = freqdata

                    if verbose:
                        progress_counter += 1
                        tpf = ((time.time() - start_time) / float(progress_counter))
                        etr = tpf * (number_of_calcs - progress_counter)
                        print("Chunk FFT {0:d} / {1:d}, Time/File {3:.2f}s ETR: {2:.1f}s".format(progress_counter,
                                                                                                 number_of_calcs,
                                                                                                 etr, tpf))
                outdata.saveh5()
            else:
                # Do the whole thing at once, user says it should fit into RAM
                outdata.get_datafield(i_df).data = fftpack.fftshift(fftpack.fft(df_in.data,
                                                                                axis=self.axis_to_transform_id),
                                                                    axis=self.axis_to_transform_id)

        self.result = outdata
        return outdata


if __name__ == '__main__':
    testdatah5 = "PVL_r10_pol244_25µm_-50-150fs_run1.hdf5"
    testdata = ds.DataSet.in_h5(testdatah5)

    fftdatah5 = testdatah5.replace(".hdf5", "_FFT.hdf5")
    fft = FFT(testdata, 'delay', 'PHz')
    fftdata = fft.fft(h5target=fftdatah5)
    fftdata.saveh5()

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
    #     print('moep')
