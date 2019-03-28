from scipy import fftpack, ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv



def radial_profile(data, center):
    y,x = np.indices((data.shape)) 				# determine radii of all pixels in relation to the center
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)

    ind = np.argsort(r.flat) 					# get sorted indices
    sr = r.flat[ind] 							# sorted radii
    sim = data.flat[ind] 						# image values sorted by radii
    ri = sr.astype(np.int32) 					# integer part of radii (bin size = 1)

    # determining distance between changes
    deltar = ri[1:] - ri[:-1] 					# assume all radii represented
    rind = np.where(deltar)[0] 					# location of changed radius
    nr = rind[1:] - rind[:-1] 					# number in radius bin

    csim = np.cumsum(sim, dtype=np.float64) 	# cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] 	# sum for image values in radius bins
    radialprofile = tbin/nr 					# the answer
    return radialprofile


if __name__ == '__main__':
    path = 'D:/Auswertungen/20181123 a-SiH on ZnO/opo/'
    image = np.asarray(cv.imread(path+'535.tif',-1))

    fft2 = fftpack.fftshift(fftpack.fft2(image))
    rad=radial_profile(np.log10(abs(fft2)), (fft2.shape[0]/2,fft2.shape[1]/2))
    plt.plot(rad)
    plt.show()