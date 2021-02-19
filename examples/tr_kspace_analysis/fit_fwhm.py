# coding=utf-8

"""
This file provides a script that approximates the FWHM by fitting a certain curve to the time resolved data.
The script provides the possibility of fitting a multidimensional Gaussian, Lorentzian or Sech2 curve.
The fit parameters for the FWHM and for recreating the fitted curve are exported and saved.
The methods work on time resolved nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import snomtools.data.fits
import sys

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False

# Define run to evaluate
fith5 = os.path.abspath("TR Data HDF5")  # example : "example_data_set_tr.hdf5"

# Read in data
fitdata = ds.DataSet.from_h5(fith5)

# Chose which curve you want to fit to data
fit_type = "fit mode" # example: 'gaussian'

# Fit Gaussian:
if fit_type == 'gaussian':
    fit = snomtools.data.fits.Gauss_Fit_nD(fitdata)
    fit_param = snomtools.data.fits.Gauss_Fit_nD.export_parameters(fit)
    fit_param.saveh5(fith5.replace('.hdf5', '_Gaussfit.hdf5'))

# Fit Lorentzian:
elif fit_type == 'lorentzian':
    fit = snomtools.data.fits.Lorentz_Fit_nD(fitdata)
    fit_param = snomtools.data.fits.Lorentz_Fit_nD.export_parameters(fit)
    fit_param.saveh5(fith5.replace('.hdf5', '_Lorentzfit.hdf5'))

# Fit Sech2:
elif fit_type == 'sech2':
    fit = snomtools.data.fits.Sech2_Fit_nD(fitdata)
    fit_param = snomtools.data.fits.Sech2_Fit_nD.export_parameters(fit)
    fit_param.saveh5(fith5.replace('.hdf5', '_Sech2fit.hdf5'))

print("Fit done.")