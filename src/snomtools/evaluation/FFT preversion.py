import snomtools.data.datasets as ds
import matplotlib.pyplot  as plt
import snomtools.calcs.units as u
import snomtools.data.transformation.project
import snomtools.data.datasets
import snomtools.evaluation.pes
import snomtools.data.fits
import snomtools.plots.datasets
import snomtools.data.h5tools
import numpy as np
import os
import time
import sys
import matplotlib as matplotlib
import snomtools.evaluation.FFT as stFFT
import snomtools.data.datasets as ds
import cv2 as cv
import h5py
from scipy import optimize
from mpltools import color


# import snomtools.plots.setupmatplotlib as plt


def normAC_r(data):
	meanyield = np.mean(data[-30:])
	return data / meanyield

def exp_func(x,amplitude ,lamb,offset):
	return amplitude * np.exp(lamb * x) + offset

path = os.getcwd()

roi_full_spectrum = False
if roi_full_spectrum:
	# full stack of all energies
	file = "roi_data.hdf5"
	data_dir = os.path.join(path, file)

	data = snomtools.data.datasets.DataSet.from_h5file(data_dir, h5target=data_dir)
	e_0 = 30.3
	xlim = (-0.3, 2.7)

	plot_spectra_TR = False
	if plot_spectra_TR:
		# Spectra Time resolved
		spect_t0 = False
		if spect_t0:
			# Spectra time zero +-0.2fs ROI integrated
			t0 = data.get_axis('delay').get_nearest_index(u.to_ureg(0, unit='fs'))
			plt.figure()
			plt.title('T=0 all hotspots')
			plt.yscale('log')
			plt.plot(data.get_axis('energy').data - e_0,
					 np.nansum(data.datafields[0][:, t0[0]: t0[0] + 1, :], axis=(0, 1)), label='all hotspots')
			plt.ylim(ymin=10)
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/integrated_t0_spectrum.png')

			# For different ROI's
			for i in range(12, 15):
				plt.figure()
				plt.title('T=0 Roi ' + str(i))
				plt.yscale('log')

				plt.plot(data.get_axis('energy').data - e_0,
						 np.nansum(data.datafields[0][i, t0[0]: t0[0] + 1, :], axis=(0)), label='roi ' + str(i))
				plt.legend()
				plt.ylim(ymin=1)
				plt.xlim(xlim[0], xlim[1])
				plt.savefig(path + '/spectra/roi' + str(i) + '_t0_spectrum.png')

			# Comparison of summed with 12,13,14
			plt.figure()
			plt.title('all, normed, T = -0.2 to 0.2 fs')
			plt.yscale('log')
			data_cache = np.nansum(data.datafields[0][:, t0[0]: t0[0] + 1, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100, label='all hotspots')
			for i in range(12, 15):
				data_cache = np.nansum(data.datafields[0][i, t0[0]: t0[0] + 1, :], axis=(0))
				plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
						 label='roi ' + str(i))

			plt.legend()
			plt.ylim(ymin=0.1)
			plt.xlim(xlim[0], xlim[1])
			plt.savefig(path + '/spectra/all_t0_spectrum.png')

		spect_140 = False
		if spect_140:
			# Spectra for times >140fs
			t140 = data.get_axis('delay').get_nearest_index(u.to_ureg(140, unit='fs'))
			plt.figure()
			plt.title('T=140fs all ROIs')
			plt.yscale('log')
			plt.plot(data.get_axis('energy').data - e_0,
					 np.nansum(data.datafields[0][:, t140[0]:, :], axis=(0, 1)), label='all hotspots')
			plt.ylim(ymin=10)
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/integrated_t140_spectrum.png')

			# Comparison sum and single ROI's
			plt.figure()
			plt.title('all, normed T = 140-150 fs')
			data_cache = np.nansum(data.datafields[0][:, t140[0]:, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100, label='all hotspots')
			for i in range(12, 15):
				data_cache = np.nansum(data.datafields[0][i, t140[0]:, :], axis=(0))
				plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
						 label='roi ' + str(i))
			plt.yscale('log')
			plt.ylim(ymin=0.1)
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/all_t140_spectra.png')

		spec_t0_140 = False
		if spec_t0_140:
			# Comparison of 0 and 140fs spectra sum pictures
			plt.figure()
			plt.title('Comparison of 0 and 140+ spectra')
			t0 = data.get_axis('delay').get_nearest_index(u.to_ureg(0, unit='fs'))
			plt.yscale('log')
			data_cache = np.nansum(data.datafields[0][:, t0[0]: t0[0] + 1, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
					 label='T=0 all hotspots ')
			t140 = data.get_axis('delay').get_nearest_index(u.to_ureg(140, unit='fs'))
			data_cache = np.nansum(data.datafields[0][:, t140[0]:, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
					 label='T=140+ all hotspots')
			plt.yscale('log')
			plt.ylim(ymin=0.1)
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/comparison_sum_spectrum.png')

			# Comparison of all 12,13,14 Rois and summed
			plt.figure()
			plt.title('Comparison of 0 and 140+ spectra')
			t0 = data.get_axis('delay').get_nearest_index(u.to_ureg(0, unit='fs'))
			plt.yscale('log')
			data_cache = np.nansum(data.datafields[0][:, t0[0]: t0[0] + 1, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
					 label='T=0 all hotspots ')
			for i in range(12, 15):
				data_cache = np.nansum(data.datafields[0][i, t0[0]: t0[0] + 1, :], axis=(0))
				plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
						 label='T=0 roi ' + str(i))

			t140 = data.get_axis('delay').get_nearest_index(u.to_ureg(140, unit='fs'))
			data_cache = np.nansum(data.datafields[0][:, t140[0]:, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
					 label='T=140+ all hotspots')
			for i in range(12, 15):
				data_cache = np.nansum(data.datafields[0][i, t140[0]:, :], axis=(0))
				plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
						 label='t=140+ roi ' + str(i))
			plt.yscale('log')
			plt.ylim(ymin=0.1)
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/comparison_spectrum.png')

		spec_20fs_bins = False
		if spec_20fs_bins:
			# Time evolution of spectra in 20fs bins
			fig, ax = plt.subplots()
			plt.title('all Hotspots - 20fs bins - normed')
			color.cycle_cmap(10, cmap='viridis', ax=ax)
			plt.yscale('log')
			time_index = data.get_axis('delay').get_nearest_index(u.to_ureg(20, unit='fs'))
			data_cache = np.nansum(data.datafields[0][:, :time_index[0], :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100, label='t = 0-20 fs')
			plt.ylim(ymin=0.1)
			for t in range(20, 150, 20):
				time_index = data.get_axis('delay').get_nearest_index(u.to_ureg(t, unit='fs'))
				data_cache = np.nansum(data.datafields[0][:, time_index[0]:time_index[0] + 50, :], axis=(0, 1))
				plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100,
						 label='t = ' + str(t) + '-' + str(t + 20) + ' fs')
			time_index = data.get_axis('delay').get_nearest_index(u.to_ureg(140, unit='fs'))
			data_cache = np.nansum(data.datafields[0][:, time_index[0]:, :], axis=(0, 1))
			plt.plot(data.get_axis('energy').data - e_0, data_cache / np.max(data_cache) * 100, label='t = 140+ fs')
			plt.xlim(xlim[0], xlim[1])
			plt.legend()
			plt.savefig(path + '/spectra/timeevolution.png')

		print('moep')

roi_e_binned = True
if roi_e_binned:
	roi_ebinned_h5 = path + '/roi_data_ebinned.hdf5'

	# generate new HDF5 with energy axis binned to 3 areas
	generate_new_ebinned = False
	if generate_new_ebinned:
		file = "roi_data.hdf5"
		data_dir = os.path.join(path, file)
		data = snomtools.data.datasets.DataSet.from_h5file(data_dir, h5target=data_dir)

		eslice_limits = [{'energy': [20, 43]}, {'energy': [43, 53]}, {'energy': [53, 67]}]
		stack = []
		for roi in eslice_limits:
			projected_data = snomtools.data.transformation.project.project_2d(ds.ROI(data, roi, by_index=True),
																			  axis1_id='ROI',
																			  axis2_id='delay')
			projected_data.get_datafield(0).data = np.ma.fix_invalid(projected_data.get_datafield(0).data,
																	 fill_value=0.)
			stack.append(projected_data)

		newaxis = ds.Axis([1, 0.5, 0], label='Binned Energy')
		stacked_data = snomtools.data.datasets.stack_DataSets(stack, new_axis=newaxis)
		stacked_data.saveh5(roi_ebinned_h5)

	data_e_binned = snomtools.data.datasets.DataSet.from_h5file(roi_ebinned_h5, h5target=roi_ebinned_h5)

	print('moep')
	fft_h5file = path + '/fft_file.hdf5'
	# generate file with fourier filtered components as datafields
	generate_fft = False
	if generate_fft:
		fft_h5 = h5py.File(fft_h5file, 'w')
		freqgroup = fft_h5.create_group('Frequency Domain')
		filtgroup = fft_h5.create_group('Filter Functions')
		timegroup = fft_h5.create_group('Time Domain')

		# ToDO: make nice Dataarrays labels etc
		time_zero = data_e_binned.get_axis('delay').get_nearest_index(0)[0]+1
		(fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (
			h0, h1, h2, h3), deltaT = stFFT.doFFT_Filter(data_e_binned.datafields[0][0, 0, time_zero:])
		faxis = ds.Axis(fticks, unit='PHz', label='Frequency for spectrum and phase')
		filt_faxis = ds.Axis(w0, unit='PHz', label='Frequency for filters')

		ds_h = ds.DataSet(label='Filter windows',
						  datafields=(ds.DataArray(h0, label='filter h0'),
									  ds.DataArray(h1, label='filter h1'),
									  ds.DataArray(h2, label='filter h2'),
									  ds.DataArray(h3, label='filter h3')),
						  axes=(filt_faxis,),
						  h5target=filtgroup)
		ds_h.saveh5(filtgroup)

		spec_roi_stack = []
		filt_roistack = []

		for roi in data_e_binned.get_axis('ROI'):

			e_spec_list = []
			e_filt_list = []
			for energy in range(data_e_binned.get_axis('Binned Energy').shape[0]):
				(fticks, freqdata, phase), (filtdata0, filtdata1, filtdata2, filtdata3), (w0, w1, w2, w3), (
					h0, h1, h2, h3), deltaT = stFFT.doFFT_Filter(data_e_binned.datafields[0][energy, roi.magnitude, time_zero:])

				ds_spectrum = ds.DataSet(label='Spectrum',
										 datafields=(ds.DataArray(freqdata, label='Spectrum'),
													 ds.DataArray(phase, label='Phase')),
										 axes=(faxis,),
										 h5target=freqgroup)

				ds_filtered = ds.DataSet(label='Filtered w-data',
										 datafields=(ds.DataArray(filtdata0, label='Filtered w0 data'),
													 ds.DataArray(filtdata1, label='Filtered w1 data'),
													 ds.DataArray(filtdata2, label='Filtered w2 data'),
													 ds.DataArray(filtdata3, label='Filtered w3 data')),
										 axes=(data_e_binned.get_axis('delay')[time_zero:],),
										 h5target=timegroup)

				# fft_dim_axis = ds.Axis(range(len(fft_stack)), label='S P w0-3 f0-3')
				e_spec_list.append(ds_spectrum)
				e_filt_list.append(ds_filtered)
			spec_roi_stack.append(ds.stack_DataSets(e_spec_list, new_axis=data_e_binned.get_axis('Binned Energy')))
			filt_roistack.append(ds.stack_DataSets(e_filt_list, new_axis=data_e_binned.get_axis('Binned Energy')))

		fft_ds_spec = ds.stack_DataSets(spec_roi_stack, new_axis=data_e_binned.get_axis('ROI'))
		fft_ds_filt = ds.stack_DataSets(filt_roistack, new_axis=data_e_binned.get_axis('ROI'))

		fft_ds_spec.saveh5(freqgroup)
		fft_ds_filt.saveh5(timegroup)
		fft_h5.close()

	# ToDo:plot new filter windows
	fft_h5_open = h5py.File(fft_h5file, 'r')
	fft_data = ds.DataSet.from_h5file(fft_h5_open['/Frequency Domain'], h5target=fft_h5_open['/Frequency Domain'])
	time_data = ds.DataSet.from_h5file(fft_h5_open['/Time Domain'], h5target=fft_h5_open['/Time Domain'])
	filter_data = ds.DataSet.from_h5file(fft_h5_open['/Filter Functions'], h5target=fft_h5_open['/Filter Functions'])
	print('moep')

	phasefront_analysis = True
	if phasefront_analysis:
		roi_distances = True
		if roi_distances:
			roilimits = np.array([[131, 274, 137, 282],
								  [137, 274, 144, 283],
								  [146, 274, 150, 283],
								  [153, 277, 159, 284],
								  [158, 283, 164, 290],
								  [164, 281, 171, 292],
								  [171, 281, 178, 294],
								  [178, 284, 185, 293],
								  [186, 282, 192, 291],
								  [193, 282, 198, 291],
								  [199, 283, 206, 293],
								  [207, 285, 211, 292],
								  [211, 292, 216, 300],
								  [218, 293, 223, 301],
								  [225, 295, 230, 301],
								  [231, 293, 237, 300],
								  [238, 291, 243, 298],
								  [246, 291, 257, 301],
								  [267, 287, 274, 293],
								  [266, 297, 273, 304],
								  [239, 304, 247, 310],
								  [252, 310, 259, 316],
								  [262, 311, 266, 317],
								  [165, 272, 172, 279],
								  [174, 272, 180, 279],
								  [182, 275, 187, 281]])
			roi_centerpositions = []
			for roi in range(roilimits.shape[0]):
				roi_centerpositions.append((roilimits[roi, 0] + (roilimits[roi, 2] - roilimits[roi, 0]) / 2,
											roilimits[roi, 1] + (roilimits[roi, 3] - roilimits[roi, 1]) / 2))
			roi_centerpositions = np.asarray(roi_centerpositions)
			roi_distance_next = []
			for i in range(16):
				roi_distance_next.append(np.linalg.norm(roi_centerpositions[i] - roi_centerpositions[i + 1]))

		# -----------Plot roi values with some norm to their position---------
		cmap = matplotlib.cm.get_cmap('viridis')
		plt.ioff()
		# -----------w0---------
		w0_analysis = True
		if w0_analysis:
			roi_max = []
			ebinned_w0data = np.nansum(time_data.datafields[0], axis=(1))
			exp_substract = True
			if exp_substract:
				decay_list =  []
				for i, roi in enumerate(roi_centerpositions):
					res, garbage = optimize.curve_fit(exp_func,time_data.get_axis('delay').data.magnitude/1000,ebinned_w0data[i],p0=(2500,-0.01,100))
					decay_list.append(res)
					plot_w0_expfit = False
					if plot_w0_expfit:
						plt.figure()
						plt.plot(time_data.get_axis('delay').data.magnitude/1000, ebinned_w0data[i])
						plt.plot(time_data.get_axis('delay').data.magnitude/1000, exp_func(time_data.get_axis('delay').data.magnitude/1000,res[0],res[1],res[2]))
						plt.title('ROI ' + str(i) +'  lambda = ' + str(res[1]) )
						plt.savefig(path+'/FFT/w0/Roi'+ str(i)+'.png')
						plt.close()

				w0_substractet = np.zeros(ebinned_w0data.shape)
				for i, roi in enumerate(roi_centerpositions):
					for t,time in enumerate(time_data.get_axis('delay').data.magnitude/1000):
						w0_substractet[i,t]=ebinned_w0data[i,t]-exp_func(time,decay_list[i][0],decay_list[i][1],decay_list[i][2])
					plot_w0_substract =False
					if plot_w0_substract:
						plt.figure()
						plt.plot(time_data.get_axis('delay'),w0_substractet[i])
						plt.savefig(path+'/FFT/w0/substracted/Roi'+str(i)+'.png')
						plt.close()
				scatterplot_w0_substracted = False
				if scatterplot_w0_substracted:
					roi_max=[]
					roi_min=[]
					for i, roi in enumerate(roi_centerpositions):
						roi_max.append(np.max(w0_substractet[i, 30:]))
						roi_min.append(np.min(w0_substractet[i,30:]))
					for time in range(30,time_data.get_axis('delay').shape[0]):
						fig = plt.figure(figsize=(5, 16), dpi=100)
						plt.style.use('dark_background')
						plt.title('Time = ' + str(time_data.get_axis('delay')[time]))
						print(str(time))
						for i,roi in enumerate(roi_centerpositions):
							plotvalue_normed = (w0_substractet[i,time]-roi_min[i])/(roi_max[i]-roi_min[i])*256
							plt.scatter(roi[1], roi[0], s=300 , color=cmap(np.int(plotvalue_normed)))
						plt.ylim(290, 130)
						plt.savefig(path+'/roi analysis/w0/'+str(time)+'.png')
						plt.close()


			scatter_w0_rawdata = False
			if scatter_w0_rawdata:
				for i, roi in enumerate(roi_centerpositions):
					roi_max.append(np.max(ebinned_w0data[i, 11:100]))
				for time in range(time_data.get_axis('delay').shape[0]):
					plt.figure()
					plt.style.use('dark_background')
					plt.title('Time = ' + str(time_data.get_axis('delay')[time]))
					print(str(time))
					for i,roi in enumerate(roi_centerpositions):
						plotvalue_normed = ebinned_w0data[i,time]/roi_max[i]*256
						plt.scatter(roi[1], roi[0], s=300 , color=cmap(np.int(plotvalue_normed)))
					plt.ylim(290, 130)
					plt.savefig(path+'/roi analysis/w0/'+str(time)+'.png')
					plt.close()

		#-----------w1---------
		w1_analysis = True
		if w1_analysis:
			ebinned_w1data = np.nansum(time_data.datafields[1], axis=(1))
			roi_max =[]
			for i, roi in enumerate(roi_centerpositions):
				roi_max.append(np.max(ebinned_w1data[i, 130:150]))
			for time in range(130,time_data.get_axis('delay').shape[0]):
				fig = plt.figure(figsize=(5, 16), dpi=100)
				plt.style.use('dark_background')
				plt.title('Time = ' + str(time_data.get_axis('delay')[time]))
				print(str(time))

				for i,roi in enumerate(roi_centerpositions):
					plotvalue_normed = ebinned_w1data[i,time]/roi_max[i]*128 + 128
					plt.scatter(roi[1], roi[0], s=300 , color=cmap(np.int(plotvalue_normed)))
				plt.ylim(290, 130)
				plt.savefig(path+'/roi analysis/w1/'+str(time)+'.png')
				plt.close()

		#-----------w2---------
		w2_analysis = True
		if w2_analysis:
			ebinned_w2data = np.nansum(time_data.datafields[2], axis=(1))
			roi_max =[]
			for i, roi in enumerate(roi_centerpositions):
				roi_max.append(np.max(ebinned_w2data[i, 130:150]))
			for time in range(130,time_data.get_axis('delay').shape[0]):
				fig = plt.figure(figsize=(5, 16), dpi=100)
				plt.style.use('dark_background')
				plt.title('Time = ' + str(time_data.get_axis('delay')[time]))
				print(str(time))

				for i,roi in enumerate(roi_centerpositions):
					plotvalue_normed = ebinned_w2data[i,time]/roi_max[i]*128 + 128
					plt.scatter(roi[1], roi[0], s=300 , color=cmap(np.int(plotvalue_normed)))
				plt.ylim(290, 130)
				plt.savefig(path+'/roi analysis/w2/'+str(time)+'.png')
				plt.close()
		#-----------w3---------
		w3_analysis = True
		if w3_analysis:
			ebinned_w3data = np.nansum(time_data.datafields[3], axis=(1))
			roi_max =[]
			for i, roi in enumerate(roi_centerpositions):
				roi_max.append(np.max(ebinned_w3data[i, 130:150]))
			for time in range(130,time_data.get_axis('delay').shape[0]):
				fig = plt.figure(figsize=(5, 16), dpi=100)
				plt.style.use('dark_background')
				plt.title('Time = ' + str(time_data.get_axis('delay')[time]))
				print(str(time))

				for i,roi in enumerate(roi_centerpositions):
					plotvalue_normed = ebinned_w3data[i,time]/roi_max[i]*128 + 128
					plt.scatter(roi[1], roi[0], s=300 , color=cmap(np.int(plotvalue_normed)))
				plt.ylim(290, 130)
				plt.savefig(path+'/roi analysis/w3/'+str(time)+'.png')
				plt.close()


	end_w1 = False
	if end_w1:
		start = 120
		w = 1
		data12 = np.nansum(time_data.datafields[w][12], axis=0)[start:] / np.max(
			np.nansum(time_data.datafields[w][12], axis=0)[start:])
		for roi in range(0, 23):
			fig, ax = plt.subplots()
			data_roi_cache = np.nansum(time_data.datafields[w][roi], axis=0)[start:] / np.max(
				np.nansum(time_data.datafields[w][roi], axis=0)[start:])
			ax.plot(time_data.get_axis('delay')[start:].data, data_roi_cache, label=str(roi))
			ax.plot(time_data.get_axis('delay')[start:].data, data12, label=str(12))
			ax.legend()
			fig.savefig(path + '/w1/' + str(roi) + 'png')
	plot_fft_spec = True
	if plot_fft_spec:
		data12 = np.nansum(fft_data.datafields[0][12], axis=0)
		for roi in range(0, 23):
			fig, ax = plt.subplots()
			data_roi_cache = np.nansum(fft_data.datafields[0][roi], axis=0)
			ax.plot(fft_data.get_axis('Frequency for spectrum and phase').data, abs(data_roi_cache), label=str(roi))
			ax.plot(fft_data.get_axis('Frequency for spectrum and phase').data, abs(data12), label=str(12))
			ax.legend()
			fig.savefig(path + '/spec/' + str(roi) + 'png')

			overview_analysis = False
			if overview_analysis:
				path = os.getcwd()
			file = "roi_overview_data.hdf5"
			data_dir = os.path.join(path, file)

			overview_data = snomtools.data.datasets.DataSet.from_h5file(data_dir, h5target=data_dir)
			sumpath = path + '/sum/'
			for i in range(383):
				imdata = np.nansum(overview_data.datafields[0][i], axis=(0))
			plt.imshow(imdata)
			# plt.imsave(sumpath + str(i) + '.tif', imdata)

			roi_binned_overview = False  # does not work properly yet
			if roi_binned_overview:
				path = os.getcwd()
			file = "roi_overview_data.hdf5"
			data_dir = os.path.join(path, file)

			overview_data = snomtools.data.datasets.DataSet.from_h5file(data_dir, h5target=data_dir)

			roilimits = np.array([[131, 274, 137, 282],
								  [137, 274, 144, 283],
								  [146, 274, 150, 283],
								  [153, 277, 159, 284],
								  [158, 283, 164, 290],
								  [164, 281, 171, 292],
								  [171, 281, 178, 294],
								  [178, 284, 185, 293],
								  [186, 282, 192, 291],
								  [193, 282, 198, 291],
								  [199, 283, 206, 293],
								  [207, 285, 211, 292],
								  [211, 292, 216, 300],
								  [218, 293, 223, 301],
								  [225, 295, 230, 301],
								  [231, 293, 237, 300],
								  [238, 291, 243, 298],
								  [246, 291, 257, 301],
								  [267, 287, 274, 293],
								  [266, 297, 273, 304],
								  [239, 304, 247, 310],
								  [252, 310, 259, 316],
								  [262, 311, 266, 317],
								  [165, 272, 172, 279],
								  [174, 272, 180, 279],
								  [182, 275, 187, 281]])
			roilimits[:, 0] = roilimits[:, 0] - 120
			roilimits[:, 2] = roilimits[:, 2] - 120
			roilimits[:, 1] = roilimits[:, 1] - 260
			roilimits[:, 3] = roilimits[:, 3] - 260

			ovbin_data = np.nansum(overview_data.datafields[0], axis=1)
			# norm the data
			ovbin_data = normAC_r(ovbin_data)
			norm_w = []
			for i in range(time_data.shape[0]):
				norm_w.append(np.mean(time_data.datafields[0][i, 1, :-30].data))

			for delay in range(ovbin_data.shape[0]):
				for roi in range(roilimits.shape[0]):
					# ovbin_data[delay, roilimits[roi, 0]:roilimits[roi, 2], roilimits[roi, 1]:roilimits[roi, 3]] = np.ones((-roilimits[roi, 0] + roilimits[roi, 2], -roilimits[roi, 1] + roilimits[roi, 3])) * 0][roi, 1, delay].magnitude  # / norm_w[roi]
					print('moep')
			newds = ds.DataSet(label='manipulatedData', datafields=(ovbin_data,),
							   axes=(overview_data.get_axis('delay'), overview_data.get_axis('y'),
									 overview_data.get_axis('x')))
			newds.saveh5(path + '/manipulated_w1.hdf5')

			print('moep')
