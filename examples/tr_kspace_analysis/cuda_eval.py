"""
This script is supposed to run with cuda on high performance graphic cards, e.g. elwe2
This file provides a script to simulate lifetimes for a time resolved DataSet measured in k-space.
The script works on a defined roi, usually one energy slice, which is defined in the file that calls the script.
The script runs e.g. a lorentzian fit to calculate start parameters.
The methods work on a nD DataSet imported with snomtools.
"""

import sys
import os

wd = os.getcwd()

# Recommended folder structure : /home/username/repos/evaluation/year/month/datafolder
home = os.path.expanduser("~")
# On elwe, can't install packages, so after we cloned them to a folder (repos in our home), we use them from source.
# This means we have to add their folders to PATH, to allow python to find them:
sys.path.insert(0, os.path.join(home, "repos/snomtools/src/"))
sys.path.insert(0, os.path.join(home, "repos/pint/"))
# Specify path to data:
wd = "path to datafolder" # example : "/home/username/repos/evaluation/2020/01 January/20200128_Au788_Zheng_blue"

import snomtools.data.datasets as ds
import snomtools.calcs.units as u
import snomtools.evaluation.cuda.obe_copol
from snomtools.evaluation.pumpprobe import delay_apply_timescale

# Used to define the energy slice when submitting the job
for arg in sys.argv:
    if "-roi" in arg:
        roistr = arg.partition('-roi')[-1]
        roi = int(roistr)
        print("ROI: {0:d}".format(roi))
if '-v' in sys.argv:
    verbose = True
else:
    verbose = False


# Define time zero, if it's deviating from 0 in DataSet to shift delay axis so time_zero equals 0fs
time_zero = "time zero value" # example : u.to_ureg(3, 'um').to('fs', 'light')
# Define laser FWHM as evaluated with a gaussfit in energys close to E_Fermi and "off state"
laser_fwhm = "laser FWHM value" # example : u.to_ureg(25, 'um').to('fs', 'light')

infile = os.path.abspath(os.path.join(wd,"Kscaled HDF5")) # example : "1. Durchlauf_e4_binned_int_kscaled.hdf5"
indata = ds.DataSet.from_h5(infile)
delay_apply_timescale(indata)
delay_axis = indata.get_axis('delay')
delay_axis -= time_zero

# Roi that will be simulated, as defined in the "cuda_eval_job" that calls this script on elwe2
# For smaller Roi (e.g. only part of one energy slice add : x binned x10':[43,43], 'y binned x10':[43,43] in Roi
roi_to_fit = ds.ROI(indata, {"energy-axis": [roi, roi],
                             # example : 'energy binned x4'
                             }, by_index=True)

# Define start parameters for fitting algorithm
obeobject = snomtools.evaluation.cuda.obe_copol.OBEfit_Copol(roi_to_fit,
                                                             laser_lambda=u.to_ureg(400, 'nm'),
                                                             laser_AC_FWHM=laser_fwhm,
                                                             time_zero=u.to_ureg(0, 'fs'),
                                                             max_time_zero_offset=u.to_ureg(40, 'fs'),
                                                             max_lifetime=u.to_ureg(50,'fs'),
                                                             fit_mode='gauss')
                                                             
obeobject.optimize_gpu_blocksize()
# Saving simulated lifetime data for the energy slice
oberesults = obeobject.obefit()
oberesults.saveh5(infile.replace('.hdf5', '') + "_lifetimes_eslice{0}.hdf5".format(roi))
# Saving fitparameter in separate DataSet for quality control
obefits = obeobject.resultACdata()
obefits.saveh5(infile.replace('.hdf5', '') + "_lifetimes_eslice{0}_fitdata.hdf5".format(roi))

print("done")
