# --peak detect based on Philip's work (untested)


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

