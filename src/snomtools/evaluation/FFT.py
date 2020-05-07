"""
FFT Script based on the work of Philip.
That is why unused functions e.g. peak_detect are in this file and some notations are strange
"""

from scipy import fftpack
import scipy.signal as signal
from scipy.optimize import curve_fit
import os
import numpy as np
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt


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

    filtdata3, w3, h3 = highpass(timedata, 3 * w_c - d3, deltaT, 5)  # 1.125
    filtdata2, w2, h2 = bandpass(timedata, 2 * w_c - d2, 2 * w_c + d2, deltaT, 5)  # 0.75
    filtdata1, w1, h1 = bandpass(timedata, 1 * w_c - d1, 1 * w_c + d1, deltaT, 5)  # 800nm =0.375
    filtdata0, w0, h0 = lowpass(timedata, d0, deltaT, 5)

    # Calculate Frequency axis
    prefactor = (1 / deltaT * 0.5 / np.pi)
    return (fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (
        prefactor * w0, prefactor * w1, prefactor * w2, prefactor * w3), (h0, h1, h2, h3), deltaT


# -----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import pathlib as pathlib

    # ----------------------Load Data-------------------------------------
    path = pathlib.Path(
        r"E:\NFC15\20171207 ZnO+aSiH\01 DLD PSI -3 to 150 fs step size 400as\Maximamap\Driftcorrected\summed_runs")
    file = "ROI_data.hdf5"
    data_file = path / file
    data3d = ds.DataSet.from_h5file(os.fspath(data_file), h5target=os.fspath(path / 'chache.hdf5'))
    for roi in range(0, 10):
        roi0_sumpicture = np.sum(data3d.datafields[0][roi][:, 0:45], axis=1)  # TODO: use other stuff than sumpicture
        xAxis = data3d.get_axis('delay').data.to('femtoseconds').magnitude
        timedata = roi0_sumpicture.magnitude

        # ----------------------Apply function to calculate the FFT-------------------------------------
        (fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (
            h0, h1, h2, h3), deltaDim1 = doFFT_Filter(timedata)

        # ----------------------Plotting section starts here-------------------------------------
        ###
        pltdpi = 100
        fontsize_label = 14  # schriftgröße der Labels
        freq_lim = 1.25  # limit of frequency spectrum in plots

        # ----------------------Phase-------------------------------------
        fig = plt.figure(figsize=(8, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' phase ')
        plt.xticks(fontsize=fontsize_label)
        ax1 = fig.add_subplot(111)
        plt.yticks(fontsize=fontsize_label)
        ax1.set_xlim(-0.1, freq_lim)
        ax1.set_xlabel('Frequenz (PHz)', fontsize=fontsize_label)
        ax1.locator_params(nbins=10)
        #   ax1.set_yscale('log')
        ax1.set_ylabel('norm. spektrale Intensität', fontsize=fontsize_label)
        # ax1.set_ylim(ymin=1)
        ax1.plot(fticks, abs(freqdata), c='black')
        ax1Xs = ax1.get_xticks()[2:]

        ax2 = ax1.twiny()
        ax2Xs = []

        for X in ax1Xs:
            ax2Xs.append(299792458.0 / X / 10 ** 6)

        for i in range(len(ax2Xs)): ax2Xs[i] = "{:.0f}".format(ax2Xs[i])

        ax2.set_xticks(ax1Xs[0:len(ax1Xs) - 1])
        ax2.set_xlabel('Wellenlänge (nm)', fontsize=fontsize_label)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(ax2Xs, fontsize=fontsize_label)

        ax3 = ax1.twinx()
        ax3.set_xlim(-0.1, freq_lim)
        ax3.set_ylabel('Phase', fontsize=fontsize_label)
        plt.yticks(fontsize=fontsize_label)
        ax3.plot(fticks, phase, c='orange')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-phase.png'))

        # ----------------------Spektrum-------------------------------------
        fig = plt.figure(figsize=(8, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' spectrum ')
        plt.xticks(fontsize=fontsize_label)
        ax1 = fig.add_subplot(111)

        ax1.set_xlim(-0.1, freq_lim)
        ax1.set_xlabel('Frequenz (PHz)', fontsize=fontsize_label)
        ax1.locator_params(nbins=10)
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=10, top=max(abs(freqdata)))
        ax1.set_ylabel('norm. spektrale Intensität', fontsize=fontsize_label)
        # ax1.set_ylim(bottom=1)
        ax1.plot(fticks, abs(freqdata), c='black')
        ax1Xs = ax1.get_xticks()[2:]
        plt.yticks(fontsize=fontsize_label)

        ax2 = ax1.twiny()
        ax2Xs = []

        for X in ax1Xs:
            ax2Xs.append(299792458.0 / X / 10 ** 6)

        for i in range(len(ax2Xs)): ax2Xs[i] = "{:.0f}".format(ax2Xs[i])

        ax2.set_xticks(ax1Xs[0:len(ax1Xs) - 1])
        ax2.set_xlabel('Wellenlänge (nm)', fontsize=fontsize_label)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(ax2Xs, fontsize=fontsize_label)

        ax3 = ax1.twinx()
        ax3.set_xlim(-0.1, freq_lim)
        ax3.set_ylabel('Filter', fontsize=fontsize_label)

        ax3.plot(w0, abs(h0), c='blue')
        ax3.plot(w1, abs(h1), c='green')
        ax3.plot(w2, abs(h2), c='green')
        ax3.plot(w3, abs(h3), c='green')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-spec.png'))

        # ----------------------w0 Komponente-------------------------------------
        plt.figure(figsize=(16, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' w_0 ')
        plt.locator_params(axis='x', nbins=20)
        plt.xlabel('T (fs)')
        plt.ylabel('Intensität (a.u.)')
        plt.plot(xAxis, filtdata0, c='blue')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w0.png'))
        # ----------------------w1 Komponente-------------------------------------

        plt.figure(figsize=(16, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' w_1 ')
        plt.locator_params(axis='x', nbins=20)
        plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
        plt.ylabel('norm. Intensität', fontsize=fontsize_label)
        plt.xticks(fontsize=fontsize_label)
        plt.yticks(fontsize=fontsize_label)
        plt.plot(xAxis, normAC(filtdata1), c='green')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w1.png'))
        # ----------------------w2 Komponente-------------------------------------

        plt.figure(figsize=(16, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' w_2 ')
        plt.locator_params(axis='x', nbins=20)
        plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
        plt.ylabel('norm. Intensität', fontsize=fontsize_label)
        plt.plot(xAxis, normAC(filtdata2), c='red')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w2.png'))
        # ----------------------w3 Komponente-------------------------------------

        plt.figure(figsize=(16, 4), dpi=pltdpi)
        plt.title('Roi' + str(roi) + ' w_3 ')
        plt.locator_params(axis='x', nbins=20)
        plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
        plt.ylabel('norm. Intensität', fontsize=fontsize_label)
        plt.plot(xAxis, normAC(filtdata3), c='orange')
        plt.savefig(os.path.join(path, 'out_high/roi' + str(roi) + '-w3.png'))
        print('moep')
