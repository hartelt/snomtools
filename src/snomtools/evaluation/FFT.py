"""
FFT Script based on the work of Philip.
That is why unused functions e.g. peak_detect are in this file and some notations are strange
"""

from scipy import fftpack
import scipy.signal as signal
# from scipy.signal import freqz, butter, lfilter
from scipy.optimize import curve_fit
import os
import pathlib as pathlib
import re
import numpy as np
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt


# --norm functions
def norm(data):
	return (data - data.min()) / (data.max() - data.min())


def normAC(data):
	return data / data[0]


def normAC2(data):
	return data / data[-1]


def normn(data, n):
	return data / data[n]


# --filters
def bandpass(data, lowcut, highcut, srate, order):
	nyq = 1 / (2 * srate)
	#    nyq = 1.
	low = lowcut / nyq
	high = highcut / nyq
	b, a = signal.butter(order, [low, high], btype='band')
	w, h = signal.freqz(b, a, worN=5000)
	res = signal.filtfilt(b, a, data)
	return [res, w, h]


def lowpass(data, highcut, srate, order):
	nyq = 1 / (2 * srate)
	#    nyq = 1.
	high = highcut / nyq
	b, a = signal.butter(order, high, btype='low')  # ,analog = True)
	w, h = signal.freqz(b, a, worN=5000)
	res = signal.filtfilt(b, a, data)
	return [res, w, h]


# --peak detect
def _datacheck_peakdetect(x_axis, y_axis):
	if x_axis is None:
		x_axis = range(len(y_axis))

	if len(y_axis) != len(x_axis):
		raise (ValueError)  # , 'Input vectors y_axis and x_axis must have same length')

	# needs to be a numpy array
	y_axis = np.array(y_axis)
	x_axis = np.array(x_axis)
	return x_axis, y_axis


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0):
	"""
	Converted from/based on a MATLAB script at:
	http://billauer.co.il/peakdet.html

	function for detecting local maximas and minmias in a signal.
	Discovers peaks by searching for values which are surrounded by lower
	or larger values for maximas and minimas respectively

	keyword arguments:
	y_axis -- A list containg the signal over which to find peaks
	x_axis -- (optional) A x-axis whose values correspond to the y_axis list
		and is used in the return to specify the postion of the peaks. If
		omitted an index of the y_axis is used. (default: None)
	lookahead -- (optional) distance to look ahead from a peak candidate to
		determine if it is the actual peak (default: 200)
		'(sample / period) / f' where '4 >= f >= 1.25' might be a good value
	delta -- (optional) this specifies a minimum difference between a peak and
		the following points, before a peak may be considered a peak. Useful
		to hinder the function from picking up false peaks towards to end of
		the signal. To work well delta should be set to delta >= RMSnoise * 5.
		(default: 0)
			delta function causes a 20% decrease in speed, when omitted
			Correctly used it can double the speed of the function

	return -- two lists [max_peaks, min_peaks] containing the positive and
		negative peaks respectively. Each cell of the lists contains a tupple
		of: (position, peak_value)
		to get the average peak value do: np.mean(max_peaks, 0)[1] on the
		results to unpack one of the lists into x, y coordinates do:
		x, y = zip(*tab)
	"""
	max_peaks = []
	min_peaks = []
	dump = []  # Used to pop the first hit which almost always is false

	# check input data
	x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
	# store data length for later use
	length = len(y_axis)

	# perform some checks
	if lookahead < 1:
		raise ValueError  # "Lookahead must be '1' or above in value"
	if not (np.isscalar(delta) and delta >= 0):
		raise ValueError  # "delta must be a positive number"

	# maxima and minima candidates are temporarily stored in
	# mx and mn respectively
	mn, mx = np.Inf, -np.Inf

	# Only detect peak if there is 'lookahead' amount of points after it
	for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
									   y_axis[:-lookahead])):
		if y > mx:
			mx = y
			mxpos = x
		if y < mn:
			mn = y
			mnpos = x

		####look for max####
		if y < mx - delta and mx != np.Inf:
			# Maxima peak candidate found
			# look ahead in signal to ensure that this is a peak and not jitter
			if y_axis[index:index + lookahead].max() < mx:
				max_peaks.append([mxpos, mx])
				dump.append(True)
				# set algorithm to only find minima now
				mx = np.Inf
				mn = np.Inf
				if index + lookahead >= length:
					# end is within lookahead no more peaks can be found
					break
				continue
			# else:  #slows shit down this does
			#    mx = ahead
			#    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

		####look for min####
		if y > mn + delta and mn != -np.Inf:
			# Minima peak candidate found
			# look ahead in signal to ensure that this is a peak and not jitter
			if y_axis[index:index + lookahead].min() > mn:
				min_peaks.append([mnpos, mn])
				dump.append(False)
				# set algorithm to only find maxima now
				mn = -np.Inf
				mx = -np.Inf
				if index + lookahead >= length:
					# end is within lookahead no more peaks can be found
					break
				# else:  #slows shit down this does
				#    mn = ahead
				#    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

	# Remove the false hit on the first value of the y_axis
	try:
		if dump[0]:
			max_peaks.pop(0)
		else:
			min_peaks.pop(0)
		del dump
	except IndexError:
		# no peaks were found, should the function return empty lists?
		pass

	return [max_peaks, min_peaks]


def _peakdetect_parabole_fitter(raw_peaks, x_axis, y_axis, points):
	"""
	Performs the actual parabole fitting for the peakdetect_parabole function.

	keyword arguments:
	raw_peaks -- A list of either the maximium or the minimum peaks, as given
		by the peakdetect_zero_crossing function, with index used as x-axis
	x_axis -- A numpy list of all the x values
	y_axis -- A numpy list of all the y values
	points -- How many points around the peak should be used during curve
		fitting, must be odd.

	return -- A list giving all the peaks and the fitted waveform, format:
		[[x, y, [fitted_x, fitted_y]]]

	"""

	def func(x, k, tau, m):
		return k * ((x - tau) ** 2) + m

	# func = lambda x, k, tau, m: k * ((x - tau) ** 2) + m
	fitted_peaks = []
	for peak in raw_peaks:
		try:
			index = peak[0]
			x_data = x_axis[index - points // 2: index + points // 2 + 1]
			y_data = y_axis[index - points // 2: index + points // 2 + 1]
			# get a first approximation of tau (peak position in time)
			tau = x_axis[index]
			# get a first approximation of peak amplitude
			m = peak[1]

			# build list of approximations
			# k = -m as first approximation?
			p0 = (-m, tau, m)
			popt, pcov = curve_fit(func, x_data, y_data, p0, maxfev=2000)
			# popt, pcov = curve_fit(func, x_data, y_data)
			# retrieve tau and m i.e x and y value of peak
			x, y = popt[1:3]

			# create a high resolution data set for the fitted waveform
			x2 = np.linspace(x_data[0], x_data[-1], points * 10)
			y2 = func(x2, *popt)
		except RuntimeError:
			x = 0
			y = 0
			x2 = 0
			y2 = 0
			pass

		fitted_peaks.append([x, y, [x2, y2]])

	return fitted_peaks


def peakdetect_parabole(y_axis, x_axis, lookahead, delta, points=9):
	"""
	Function for detecting local maximas and minmias in a signal.
	Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
	to the peaks. The amount of points used in the fitting is set by the
	points argument.

	Omitting the x_axis is forbidden as it would make the resulting x_axis
	value silly if it was returned as index 50.234 or similar.

	will find the same amount of peaks as the 'peakdetect_zero_crossing'
	function, but might result in a more precise value of the peak.

	keyword arguments:
	y_axis -- A list containg the signal over which to find peaks
	x_axis -- A x-axis whose values correspond to the y_axis list and is used
		in the return to specify the postion of the peaks.
	points -- (optional) How many points around the peak should be used during
		curve fitting, must be odd (default: 9)

	return -- two lists [max_peaks, min_peaks] containing the positive and
		negative peaks respectively. Each cell of the lists contains a list
		of: (position, peak_value)
		to get the average peak value do: np.mean(max_peaks, 0)[1] on the
		results to unpack one of the lists into x, y coordinates do:
		x, y = zip(*max_peaks)
	"""
	# check input data
	x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
	# make the points argument odd
	points += 1 - points % 2
	# points += 1 - int(points) & 1 slower when int conversion needed

	# get raw peaks
	max_raw, min_raw = peakdetect(y_axis, lookahead=lookahead, delta=delta)

	# define output variable
	max_peaks = []
	min_peaks = []

	max_ = _peakdetect_parabole_fitter(max_raw, x_axis, y_axis, points)
	min_ = _peakdetect_parabole_fitter(min_raw, x_axis, y_axis, points)

	max_peaks = map(lambda x: [x[0], x[1]], max_)
	max_fitted = map(lambda x: x[-1], max_)
	min_peaks = map(lambda x: [x[0], x[1]], min_)
	min_fitted = map(lambda x: x[-1], min_)

	return [max_peaks, min_peaks]


# --custom colors
tab20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
		 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
		 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
		 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
		 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tab20)):
	r, g, b = tab20[i]
	tab20[i] = (r / 255., g / 255., b / 255.)


def loadColorMap(filename):
	colorlist = []
	file = open(filename)
	for line in file.readlines():
		r, g, b = line.split(" ", 2)
		colorlist.append((float(r), float(g), float(b)))
	file.close()
	return colorlist


def doFFT_Filter(timedata, deltaT=0.4, w_c=0.375, d0=0.12, d1=0.075, d2=0.05, d3=(0.025, 0.05)):
	'''

	:param timedata:
	:param deltaT: in fs for default filter values
	:param w_c:
	:param d0:
	:param d1:
	:param d2:
	:param d3:
	:return: in fs
	'''
	freqdata = fftpack.fftshift(fftpack.fft(timedata))

	phase = -np.angle(freqdata)
	fticks = fftpack.fftshift(fftpack.fftfreq(freqdata.size, deltaT))

	filtdata3, w3, h3 = bandpass(timedata, 3 * w_c - d3[0], 3 * w_c + d3[1], deltaT, 5)  # 1.125
	filtdata2, w2, h2 = bandpass(timedata, 2 * w_c - d2, 2 * w_c + d2, deltaT, 5)  # 0.75
	filtdata1, w1, h1 = bandpass(timedata, 1 * w_c - d1, 1 * w_c + d1, deltaT, 5)  # 800nm =0.375
	filtdata0, w0, h0 = lowpass(timedata, d0, deltaT, 5)

	return (fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (h0, h1, h2, h3), deltaT


# -----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	#----------------------Load Data-------------------------------------
	path = pathlib.Path(
		r"E:\NFC15\20171207 ZnO+aSiH\01 DLD PSI -3 to 150 fs step size 400as\Maximamap\Driftcorrected\summed_runs")
	file = "ROI_data.hdf5"
	data_file = path / file
	data3d = ds.DataSet.from_h5file(os.fspath(data_file), h5target=os.fspath(path / 'chache.hdf5'))

	roi0_sumpicture = np.sum(data3d.datafields[0][0], axis=1)  # TODO: use other stuff than sumpicture
	xAxis = data3d.get_axis('delay').data.to('femtoseconds').magnitude
	timedata = roi0_sumpicture.magnitude


	#----------------------Apply function to calculate the FFT-------------------------------------
	(fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (
	h0, h1, h2, h3), deltaDim1 = doFFT_Filter(timedata)



	#----------------------Plotting section starts here-------------------------------------
	###
	pltdpi = 100
	fontsize_label = 14  # schriftgröße der Labels
	freq_lim = 1.25  # limit of frequency spectrum in plots

	# ----------------------Phase-------------------------------------
	fig = plt.figure(figsize=(8, 4), dpi=pltdpi)

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

	# ----------------------Spektrum-------------------------------------
	fig = plt.figure(figsize=(8, 4), dpi=pltdpi)

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

	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w0, abs(h0), c='blue')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w1, abs(h1), c='green')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w2, abs(h2), c='green')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w3, abs(h3), c='green')


	# ----------------------w0 Komponente-------------------------------------
	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_0 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel('T (fs)')
	plt.ylabel('Intensität (a.u.)')
	plt.plot(xAxis, filtdata0, c='blue')

	# ----------------------w1 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_1 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.xticks(fontsize=fontsize_label)
	plt.yticks(fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata1), c='green')
	# ----------------------w2 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_2 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata2), c='red')

	# ----------------------w3 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_3 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata3), c='orange')

	print('moep')

if __name__ == 'not__main__':
	#Analog example to upper one without using the allmighty all in one FFT function but with everything written seperately
	path = pathlib.Path(
		r"E:\NFC15\20171207 ZnO+aSiH\01 DLD PSI -3 to 150 fs step size 400as\Maximamap\Driftcorrected\summed_runs")
	file = "ROI_data.hdf5"
	data_file = path / file

	saveData = False
	# dim1 =  delta in fs
	deltaDim1 = 0.4  # stepsize

	# StartTime of measurement / data in fs
	StartTime = -3  # 22

	# Start and end time to plot in fs
	Tmin = StartTime
	Tmax = 150  # 300

	# Plot dpi
	pltdpi = 100
	fontsize_label = 14  # schriftgröße der Labels

	freq_lim = 1.25  # limit of frequency spectrum in plots
	# -------------------------------------------------------------------------

	data3d = ds.DataSet.from_h5file(os.fspath(data_file), h5target=os.fspath(path / 'chache.hdf5'))

	roi0_sumpicture = np.sum(data3d.datafields[0][0], axis=1)  # TODO: use other stuff than sumpicture
	xAxis = data3d.get_axis('delay').data.to('femtoseconds').magnitude

	plt.plot(data3d.get_axis('delay'), roi0_sumpicture.magnitude)

	timedata = roi0_sumpicture.magnitude

	# ---------------Fouriertrafo-------------------------------------

	freqdata = fftpack.fftshift(fftpack.fft(timedata))

	phase = -np.angle(freqdata)
	fticks = fftpack.fftshift(fftpack.fftfreq(freqdata.size, deltaDim1))

	# --------------------


	# generateXTicks(timedata, deltaDim1, StartTime)  # data3d[0] as represantative of x-Axis

	# ---------------Zeitsignal-------------------------------------
	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title('Time')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Emissionsintensität', fontsize=fontsize_label)
	plt.xlim(Tmin, Tmax)
	plt.xticks(fontsize=fontsize_label)
	plt.yticks(fontsize=fontsize_label)
	plt.plot(xAxis, timedata, color=tab20[0])
	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "timedata.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	w_c = 0.375  # center frequency 800nm =0.375
	filtdata3, w3, h3 = bandpass(timedata, 3 * w_c - 0.025, 3 * w_c + 0.05, deltaDim1, 5)  # 1.125
	filtdata2, w2, h2 = bandpass(timedata, 2 * w_c - 0.05, 2 * w_c + 0.05, deltaDim1, 5)  # 0.75
	filtdata1, w1, h1 = bandpass(timedata, 1 * w_c - 0.075, 1 * w_c + 0.075, deltaDim1, 5)  # 800nm =0.375
	filtdata0, w0, h0 = lowpass(timedata, 0.12, deltaDim1, 5)

	# ----------------------Phase-------------------------------------
	fig = plt.figure(figsize=(8, 4), dpi=pltdpi)

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

	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "phase.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	# ----------------------Spektrum-------------------------------------
	fig = plt.figure(figsize=(8, 4), dpi=pltdpi)

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
	plt.yticks(fontsize=fontsize_label)
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w0, abs(h0), c='blue')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w1, abs(h1), c='green')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w2, abs(h2), c='green')
	ax3.plot((1 / deltaDim1 * 0.5 / np.pi) * w3, abs(h3), c='green')

	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "spectrum.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	# ----------------------w0 Komponente-------------------------------------
	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_0 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel('T (fs)')
	plt.ylabel('Intensität (a.u.)')
	plt.xlim(Tmin, Tmax)
	plt.plot(xAxis, filtdata0, c='blue')
	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + " w0.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	# ----------------------w1 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_1 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.xlim(Tmin, Tmax)
	plt.xticks(fontsize=fontsize_label)
	plt.yticks(fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata1), c='green')
	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "w1.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	# ----------------------w2 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_2 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.xlim(Tmin, Tmax)
	plt.xticks(fontsize=fontsize_label)
	plt.yticks(fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata2), c='red')
	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "w2.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	# ----------------------w3 Komponente-------------------------------------

	plt.figure(figsize=(16, 4), dpi=pltdpi)
	plt.title(' w_3 ')
	plt.locator_params(axis='x', nbins=20)
	plt.xlabel(r'Verzögerungszeit $\tau$ (fs)', fontsize=fontsize_label)
	plt.ylabel('norm. Intensität', fontsize=fontsize_label)
	plt.xlim(Tmin, Tmax)
	plt.xticks(fontsize=fontsize_label)
	plt.yticks(fontsize=fontsize_label)
	plt.plot(xAxis, normAC(filtdata3), c='orange')
	if (saveData == True):
		plt.savefig(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "w3.pdf", dpi=pltdpi,
					bbox_inches='tight', pad_inches=0)

	if (saveData == True):
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "fticks.txt", fticks)
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "spec.txt", abs(freqdata))
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "phase.txt", phase)
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "filterticks.txt",
				(1 / deltaDim1 * 0.5 / np.pi) * w0)
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "filter-w0.txt", abs(h0))
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "filter-w1.txt", abs(h1))
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "filtdata-w0.txt", filtdata0)
		savetxt(workspace + date + '-diploma/' + date + roiList[0][int(dim3Id)][:-30] + "filtdata-w1.txt", filtdata1)
