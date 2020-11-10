"""
This file provides a script that transforms a DataSet measured in k-space from pixels to inverse angstrom.
The transformation is applied in both directions (x-, y-axis) individually.
The DataSet is projected onto (energy,pixel) plane (energy distribution map) and plotted with a free electron parabola.
For adjusting the parabola to the photoemission horizon the plots are shown and will be saved with used parameters.
The methods work on a nD DataSet imported with snomtools.
"""

import snomtools.data.datasets as ds
import snomtools.plots.datasets
import snomtools.evaluation.kscalecalibration
import snomtools.calcs.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Load experimental data, copy to new target and project dispersion data:
# Define run you want to scale
file = "HDF5 file to scale"  # example : "example_data_set.hdf5"
# If you don't want to create new file with same data but only scaled 'x', 'y' axis, which only doubles amount of data.
# Ignore 'full_data' and use 'file' instead
full_data = ds.DataSet.from_h5(file)

# Parameters for fitting the Parabola to your data
x_scalefactor = None  # example : u.to_ureg(0.00270, 'angstrom**-1 per pixel')
y_scalefactor = None  # example : u.to_ureg(0.00275, 'angstrom**-1 per pixel')
e_offset = None  # example : u.to_ureg(29.9, 'eV')
x_zero = None  # example : u.to_ureg(320, 'pixel')
y_zero = None  # example : u.to_ureg(321, 'pixel')
kfov = None  # example : None

# Set axes labels
x_axisid = "x-axis"  # example : 'x'
y_axisid = "y-axis"  # example : 'y'
e_axisid = "energy-axis"  # example : 'energy'

# Projects dataset on energy, pixel plane for both pixel axes individually
# Set d_axisid = False for static data, add x_window="Amount" e.g. 6 for "summing" pixels for better signal to noise
# y_axisid describes respective k plane, x_axisid is used for statistic binning
x_dispersion_data = snomtools.evaluation.kscalecalibration.load_dispersion_data(full_data, x_axisid, y_axisid, e_axisid,
                                                                                d_axisid=False)
y_dispersion_data = snomtools.evaluation.kscalecalibration.load_dispersion_data(full_data, y_axisid, x_axisid, e_axisid,
                                                                                d_axisid=False)

# Trigger for saving Imgae, with figname as name of saved file and scale parameter in label
save = False
figname = 'Figure Name'

# Show k-space scaling by plotting parabola along data:
# Along k_x axis:
(x_pixels, x_parab_data), x_scalefactor, x_zeropoint = snomtools.evaluation.kscalecalibration.show_kscale(
    x_dispersion_data, x_zero, x_scalefactor, e_offset, kfov, x_axisid)
# Along k_y axis:
(y_pixels, y_parab_data), y_scalefactor, y_zeropoint = snomtools.evaluation.kscalecalibration.show_kscale(
    y_dispersion_data, y_zero, y_scalefactor, e_offset, kfov, y_axisid)

# Plot dispersion and fitted parabola for both directions
plt.figure(figsize=(6.4, 11.4))
ax = plt.subplot(211)
ay = plt.subplot(212)
snomtools.plots.datasets.project_2d(x_dispersion_data, ax, e_axisid, x_axisid, norm=colors.LogNorm())
snomtools.plots.datasets.project_2d(y_dispersion_data, ay, e_axisid, y_axisid, norm=colors.LogNorm())
ax.plot(x_pixels, x_parab_data, 'r-', label="fit parabola")
ay.plot(y_pixels, y_parab_data, 'r-', label="fit parabola")
# project_2d flips the y axis as it assumes standard matrix orientation, so flip it back.
ax.invert_yaxis()
ay.invert_yaxis()
ax.set_xlabel("$k_{x}$ [pixel]", fontsize=14)
ay.set_xlabel("$k_{y}$ [pixel]", fontsize=14)
ax.set_ylabel("$E_{Interm.}$ [eV]", fontsize=14)
ay.set_ylabel("$E_{Interm.}$ [eV]", fontsize=14)
ax.set_title("Scalefactor=" + str(x_scalefactor.magnitude) + " & Zero=" + str(x_zeropoint.magnitude), fontsize=14)
ay.set_title("Scalefactor=" + str(y_scalefactor.magnitude) + " & Zero=" + str(y_zeropoint.magnitude), fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ay.tick_params(axis='both', labelsize=12)
if save:
    plt.savefig(figname)
plt.show()
print("x-Axis Param. :", x_scalefactor, x_zeropoint)
print("y-Axis Param. :", y_scalefactor, y_zeropoint)

# Scale k-space axes according to above scaling factors and save the scaled DataSet:
# Set to save = True, if fit is good to save and rescale your data
if save:
    # Applies k-scale calibration in k_x and k_y direction
    full_data = ds.DataSet.from_h5(file, file.replace('.hdf5', '_Kscaled.hdf5'))
    snomtools.evaluation.kscalecalibration.kscale_axes(full_data, y_scalefactor, x_scalefactor, y_zero, x_zero,
                                                       y_axisid, x_axisid)
    # Save scaled DataSet with target file name
    full_data.saveh5()
