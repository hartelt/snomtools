__author__ = 'hartelt'
"""
This scripts imports matlab files generated by the lumerical data export scripts. The methods defined here will read
those files and return the data as a DataSet instances. There might be different methods for different simulation
types.
"""
import snomtools.data.datasets
import h5py
import os
import numpy


def Efield_3d(filepath, first_coord='l', second_coord='x', third_coord='y', first_unit='m', second_unit='m',
			  third_unit='m'):
	"""
	Reads a matlab file where the electric field of a frequency domain monitor stored in a grid coordinate system of
	three coordinates. The components Ex, Ey, Ez and the Intensity E^2 are included.
	CAUTION! The right order of the axes cannot be checked. This means the coordinates have to be given in the same
	order as the corresponding values are stored in the electric field arrays!
	:param filepath: String: The (absolute or relative) path of input file.
	:param first_coord: String: The label of the first coordinate, by default 'x'.
	:param second_coord: String: The label of the second coordinate, by default 'y'.
	:param third_coord: String: The label of the third coordinate, by default 'l' (for lambda).
	:return: The DataSet instance.
	"""
	# Initialize the input file object:
	filepath = os.path.abspath(filepath)
	infile = h5py.File(filepath, 'r')

	# Assemble the axes datasets (thereby checking if they exist by the specified names):
	coord_list = [first_coord, second_coord, third_coord]
	axes_sets = []
	for coord_str in coord_list:
		axes_sets.append(infile[coord_str])

	# Same for the E-field datasets:
	field_list = ['E2', 'Ex', 'Ey', 'Ez']
	field_sets = []
	for coord_str in field_list:
		field_sets.append(infile[coord_str])

	# Initialize the dataset. Data consistency (dimension and shape) will be checked by the init of DataSet:
	unit_list = [first_unit, second_unit, third_unit]
	field_label_list = ['Intensity / arb. unit', 'E_x / arb. unit', 'E_y / arb. unit', 'E_z / arb. unit']
	coord_label_list = []
	for s in coord_list:
		if s == 'l':
			coord_label_list.append('lambda')
		elif s == 'f':
			coord_label_list.append('frequency')
		else:
			coord_label_list.append(s)
	dataarrays = []
	field_data_list = []
	field_data_list.append(numpy.array(field_sets[0]))
	for i in range(1,4):
		field_data_list.append(numpy.array(field_sets[i]).view(numpy.complex))
	for i in range(len(field_sets)):
		dataarrays.append(snomtools.data.datasets.DataArray(field_data_list[i], label=field_list[i],
															plotlabel=field_label_list[i]))
	axes = []
	for i in range(len(axes_sets)):
		axes.append(snomtools.data.datasets.Axis(axes_sets[i], unit=unit_list[i], label=coord_list[i],
												 plotlabel=coord_label_list[i]+" / "+unit_list[i]))
	return snomtools.data.datasets.DataSet(os.path.basename(filepath),dataarrays,axes)

if True: # Just for testing
	infile = "2015-08-03-Sphere4-substrate-532nm-bottomfieldE.mat"
	outfile = infile.replace('.mat','.hdf5')
	dataset = Efield_3d(infile)
	dataset.saveh5(outfile)