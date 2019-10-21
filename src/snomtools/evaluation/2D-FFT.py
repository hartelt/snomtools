from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt


def radial_profile(data, center):
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) #determine all radii
	r = r.astype(np.int)	#here, radiusbinning is 1 because of int sized pixels

	tbin = np.bincount(r.ravel(), data.ravel()) #to each radius found in r weight it with it's datapoint
	nr = np.bincount(r.ravel())					#here to amount of occurences of each radius is recorded
	radialprofile = tbin / nr					#norm

	return radialprofile



if __name__ == '__main__':

	# generate example data
	size = 1001
	center = np.int((size - 1) / 2)
	freq = 0.01
	data = np.zeros((size, size))
	for x in range(size):
		for y in range(size):
			r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
			data[y, x] = 0.1 + (10 * np.sin(r * 2 * np.pi * 0.5 * freq) + 10 * np.sin(
				r * 2 * np.pi * 1 * freq) + 10 * np.sin(
				r * 2 * np.pi * 2 * freq) + 10 * np.sin(r * 2 * np.pi * 3 * freq)) * np.exp(-0.01 * r)
	plt.figure()
	plt.imshow(data)

	# 2D FFT
	fft2 = fftpack.fftshift(fftpack.fft2(data))
	plt.figure()
	plt.imshow(abs(fft2))


	#Build axes
	deltaT = 1  # realspace step size
	fticks_x = fftpack.fftshift(fftpack.fftfreq(fft2.shape[0], deltaT))
	fticks_y = fftpack.fftshift(fftpack.fftfreq(fft2.shape[1], deltaT))
	plt.imshow(abs(fft2), extent=(fticks_x.min(), fticks_x.max(), fticks_y.min(), fticks_y.max()))

	# Calculate radial sum
	profile = radial_profile(np.abs(fft2), (center, center))
	profile2 = radial_profile2(np.abs(fft2), (center, center))
	plt.figure()
	plt.plot(fticks_x[center:], np.abs(fft2[center, center:]), label='linescan')
	plt.plot(fticks_x[center:], profile[0:center + 1], label='radial sum')

	print('moep')
