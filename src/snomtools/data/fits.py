"""
This file provides miscellaneous fitting scripts for data.
For furter info about data structures, see:
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.data.datasets as ds
import snomtools.calcs.units as u
import snomtools.data.tools
import numpy as np
from scipy.optimize import curve_fit
import sys

__author__ = 'hartelt'

# For verbose mode with progress printouts:
if '-v' in sys.argv:
    verbose = True
    import time
else:
    verbose = False
print_interval = 50


def fit_xy_linear(xdata, ydata):
    """
    A simple linear fit to data given as x and y values. Fits y = m*x + c and returns c tuple of (m, c), where m and
    c are quantities according to the physical dimensions of data. Numerical data is assumed as dimensionless.

    :param xdata: DataArray or Quantity or numpy array: The x values.

    :param ydata: DataArray or Quantity or numpy array: The y values.

    :return:tuple: (m, c)
    """
    if isinstance(xdata, ds.DataArray):
        xdata = xdata.get_data()
    else:
        xdata = u.to_ureg(xdata)
    if isinstance(ydata, ds.DataArray):
        ydata = ydata.get_data()
    else:
        ydata = u.to_ureg(ydata)
    xdata = snomtools.data.tools.assure_1D(xdata)
    ydata = snomtools.data.tools.assure_1D(ydata)

    m, c = np.polyfit(xdata.magnitude, ydata.magnitude, deg=1, full=False)

    one_xunit = u.to_ureg(str(xdata.units))
    one_yunit = u.to_ureg(str(ydata.units))
    m = u.to_ureg(m, one_yunit / one_xunit)
    c = u.to_ureg(c, one_yunit)
    return m, c


class Bell_Fit(object):
    """
    Base class for fits of a bell function to given data. A Bell function in this context is defined as a symmetric
    function that has parameters x_0 (peak position), width, A (Amplitude) and C (Constant background).
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
        if data:
            self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
            xunit = self.data.get_axis(0).get_unit()
            yunit = self.data.get_datafield(0).get_unit()

            if normalize:
                take_data = 0
            else:
                take_data = 1

            self.x_0_unit = xunit
            self.width_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit

            self.coeffs, pcov = self.fit_function(self.data.get_axis(0).get_data(),
                                                  self.data.get_datafield(take_data).get_data(),
                                                  guess)
            # Take absolute value for width, because negative widths don't make sense and curve is identical:
            self.coeffs[1] = abs(self.coeffs[1])
            try:
                self.accuracy = np.sqrt(np.diag(pcov))
            except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                self.accuracy = np.full(4, np.inf)

            if not keepdata:
                self.data = None

    func = None

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @x_0.setter
    def x_0(self, newvalue):
        newvalue = u.to_ureg(newvalue, self.x_0_unit)
        self.coeffs[0] = newvalue.magnitude

    @property
    def width(self):
        return u.to_ureg(self.coeffs[1], self.width_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        raise NotImplementedError("FWHM not implemented for baseclass Bell_Fit.")

    @classmethod
    def from_coeffs(cls, coeffs):
        new_instance = cls()
        new_instance.coeffs = coeffs
        return new_instance

    @classmethod
    def from_xy(cls, xdata, ydata, guess=None):
        new_instance = cls()
        xdata = u.to_ureg(xdata)
        xunit = str(xdata.units)
        ydata = u.to_ureg(ydata)
        yunit = str(ydata.units)
        new_instance.coeffs, pcov = cls.fit_function(xdata, ydata, guess)
        try:
            new_instance.accuracy = np.sqrt(np.diag(pcov))
        except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
            new_instance.accuracy = np.full(4, np.inf)
        new_instance.x_0_unit = xunit
        new_instance.sigma_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    def eval_function(self, x):
        """
        The fit function corresponding to the fit values of the Gauss_Fit instance.

        :param x: The value for which to evaluate. (Quantity or numerical in correct unit).

        :return: The value of the function at the value x. Returned as Quantity in whichever unit the fit
            data was given
        """
        x = u.to_ureg(x, self.x_0_unit)
        return self.func(x, self.x_0, self.width, self.A, self.C)

    @staticmethod
    def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :param label: string: label for the produced DataSet

        :return: 1D-DataSet with projected Y Data and X Axis.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        data_projected = data_full.project_nd(xaxis_index)
        data_projected = ds.DataArray(data_projected, label='projected data')
        # Normalize by scaling to 1:
        data_projected_norm = data_projected / data_projected.max()
        data_projected_norm.set_label("projected data normalized")
        # Initialize the DataSet containing only the projected powerlaw data;
        return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

    @staticmethod
    def extract_data_raw(data, data_id=0, axis_id=0):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :return: xdata, ydata: tuple of quantities with the projected data.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        return xaxis.get_data(), data_full.project_nd(xaxis_index)

    @classmethod
    def fit_function(cls, xdata, ydata, guess=None):
        """
        This function fits a gauss function to data. Uses numpy.optimize.curve_fit under the hood.

        :param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

        :param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

        :param guess: optional: A tuple of start parameters (x_0, sigma, A, C) as defined in gaussian method.

        :return: The coefficients and uncertainties of the fitted gaussian (x_0, sigma, A, C), as defined in gaussian
            method.
        """
        xdata = u.to_ureg(xdata)
        ydata = u.to_ureg(ydata)
        if guess is None:
            guess = (np.mean(xdata), (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
        # to assure the guess is represented in the correct units:
        xunit = xdata.units
        yunit = ydata.units
        unitslist = [xunit, xunit, yunit, yunit]
        guesslist = []
        for guesselement, guessunit in zip(guess, unitslist):
            guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
        guess = tuple(guesslist)
        return curve_fit(cls.func, xdata.magnitude, ydata.magnitude, guess)


class Bell_Fit_nD(object):
    """
    Base class for a moredimensional Fit of given data, in the sense of fitting a Bell function along a specified axis for
    all points on all other axes. This can be used to generate "peak maps" of given data, for example an
    autocorrelation width map of a time-resolved measurement.
    A Bell function in this context is defined as a symmetric
    function that has parameters x_0 (peak position), width, A (Amplitude) and C (Constant background).
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True):
        global print_counter, start_time
        if data:
            data_id = data.get_datafield_index(data_id)
            axis_id = data.get_axis_index(axis_id)
            xaxis = data.get_axis(axis_id)
            ydata = data.get_datafield(data_id)
            xunit = data.get_axis(axis_id).get_unit()
            yunit = data.get_datafield(data_id).get_unit()

            map_shape_list = list(data.shape)
            del map_shape_list[axis_id]
            map_shape = tuple(map_shape_list)
            coeff_shape = tuple([4] + map_shape_list)
            self.coeffs = np.empty(coeff_shape)
            self.accuracy = np.empty(coeff_shape)

            # to assure the guess is represented in the correct units:
            if guess is not None:
                unitslist = [xunit, xunit, yunit, yunit]
                guesslist = []
                for guesselement, guessunit in zip(guess, unitslist):
                    guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
                guess = tuple(guesslist)

            if verbose:
                print("Fitting {0} Bell functions".format(np.prod(map_shape)))
                start_time = time.time()
                print_counter = 0

            xvals = xaxis.get_data_raw()
            for index in np.ndindex(map_shape):
                indexlist = list(index)
                indexlist.insert(axis_id, np.s_[:])
                slicetup = tuple(indexlist)
                yvals = ydata.get_data()[slicetup].magnitude

                if guess is None:
                    pointguess = (np.mean(xvals), (np.max(xvals) - np.min(xvals)) / 4, np.max(yvals), np.min(yvals))
                    coeffs, pcov = curve_fit(self.func, xvals, yvals, pointguess)
                else:
                    coeffs, pcov = curve_fit(self.func, xvals, yvals, guess)
                # Take absolute value for width, because negative widths don't make sense and curve is identical:
                coeffs[1] = abs(coeffs[1])
                self.coeffs[(np.s_[:],) + index] = coeffs
                try:
                    self.accuracy[(np.s_[:],) + index] = np.sqrt(np.diag(pcov))
                except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                    self.accuracy[(np.s_[:],) + index] = np.inf

                if verbose:
                    print_counter += 1
                    if print_counter % print_interval == 0:
                        tpf = ((time.time() - start_time) / float(print_counter))
                        etr = tpf * (np.prod(map_shape) - print_counter)
                        print("Fit {0:d} / {1:d}, Time/Fit {3:.4f}s ETR: {2:.1f}s".format(print_counter,
                                                                                          np.prod(map_shape),
                                                                                          etr, tpf))

            self.x_0_unit = xunit
            self.width_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            if keepdata:
                self.data = data

    func = None

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @property
    def width(self):
        return u.to_ureg(self.coeffs[1], self.width)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        raise NotImplementedError("FWHM not implemented for baseclass Bell_Fit.")

    def eval_function(self, x, sel):
        """
        The Bell function corresponding to the fit values of the Bell_Fit instance at a given position of the
        data.

        :param x: The value for which to evaluate. (Quantity or numerical in correct unit).

        :param sel: An index tuple addressing the selected point.
        :type sel: tuple of int

        :return: The value of the function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return self.func(x, self.x_0[sel], self.width[sel], self.A[sel], self.C[sel])


def gaussian(x, x_0, sigma, A, C):
    """
    A Gauss function of the form gaussian(x) = A * exp(-(x-x_0)**2 / 2 sigma**2) + C
    All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
    assumed.

    :param x: The variable x.

    :param x_0: (Same unit as x.) The center of the gaussian.

    :param sigma: (Same unit as x.) The width (standard deviation) of the gaussian. Relates to FWHM by:
        FWHM = 2 sqrt(2 ln 2) sigma

    :param A: (Same unit as C.) The amplitude of the gaussian bell relative to background.

    :param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

    :return: (Same unit as A and C.) The result of the gaussian function.
    """
    return A * np.exp(-(x - x_0) ** 2 / (2 * sigma ** 2)) + C


def gaussian_2D(xydata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    #TODO: This function was moved here from Ben's code and still needs to be integrated in fitting routines.

    :param xydata_tuple
    :param amplitude:
    :param xo:  x center of gaussian
    :param yo: y center of gaussian
    :param sigma_x: gauss size
    :param sigma_y: gauss size
    :param theta: rotation of the 2D gauss 'potato'
    :param offset: offset
    :return:
    This code is based on https://stackoverflow.com/a/21566831/8654672
    Working example:

    # Create x and y indices
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)

    #create data
    data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

    # plot twoD_Gaussian data generated above
    plt.figure()
    plt.imshow(data.reshape(201, 201))
    plt.colorbar()
    '''
    (x, y) = xydata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo) ** 2)))
    return g.ravel()


class Gauss_Fit(object):
    """
    A Gauss Fit of given data with benefits.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
        if data:
            self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
            xunit = self.data.get_axis(0).get_unit()
            yunit = self.data.get_datafield(0).get_unit()

            if normalize:
                take_data = 0
            else:
                take_data = 1

            self.coeffs, pcov = self.fit_gaussian(self.data.get_axis(0).get_data(),
                                                  self.data.get_datafield(take_data).get_data(),
                                                  guess)
            try:
                self.accuracy = np.sqrt(np.diag(pcov))
            except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                self.accuracy = np.full(4, np.inf)
            self.x_0_unit = xunit
            self.sigma_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            if not keepdata:
                self.data = None

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @x_0.setter
    def x_0(self, newvalue):
        newvalue = u.to_ureg(newvalue, self.x_0_unit)
        self.coeffs[0] = newvalue.magnitude

    @property
    def sigma(self):
        return u.to_ureg(self.coeffs[1], self.sigma_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def sigma_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.sigma_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma_accuracy

    @classmethod
    def from_coeffs(cls, coeffs):
        x_0, sigma, A, C = coeffs
        x_0 = u.to_ureg(x_0)
        sigma = u.to_ureg(sigma)
        A = u.to_ureg(A)
        C = u.to_ureg(C)
        xunit = x_0.units
        yunit = A.units

        new_instance = cls()
        new_instance.coeffs = coeffs
        new_instance.x_0_unit = xunit
        new_instance.sigma_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    @classmethod
    def from_xy(cls, xdata, ydata, guess=None):
        new_instance = cls()
        xdata = u.to_ureg(xdata)
        xunit = str(xdata.units)
        ydata = u.to_ureg(ydata)
        yunit = str(ydata.units)
        new_instance.coeffs, pcov = cls.fit_gaussian(xdata, ydata, guess)
        try:
            new_instance.accuracy = np.sqrt(np.diag(pcov))
        except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
            new_instance.accuracy = np.full(4, np.inf)
        new_instance.x_0_unit = xunit
        new_instance.sigma_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    def gaussian(self, x):
        """
        The Gaussian function corresponding to the fit values of the Gauss_Fit instance.

        :param x: The value for which to evaluate the gaussian. (Quantity or numerical in correct unit).

        :return: The value of the gaussian function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return gaussian(x, self.x_0, self.sigma, self.A, self.C)

    @staticmethod
    def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :param label: string: label for the produced DataSet

        :return: 1D-DataSet with projected Intensity Data and Power Axis.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        data_projected = data_full.project_nd(xaxis_index)
        data_projected = ds.DataArray(data_projected, label='projected data')
        # Normalize by scaling to 1:
        data_projected_norm = data_projected / data_projected.max()
        data_projected_norm.set_label("projected data normalized")
        # Initialize the DataSet containing only the projected powerlaw data;
        return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

    @staticmethod
    def extract_data_raw(data, data_id=0, axis_id=0):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :return: xdata, ydata: tuple of quantities with the projected data.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        return xaxis.get_data(), data_full.project_nd(xaxis_index)

    @staticmethod
    def fit_gaussian(xdata, ydata, guess=None):
        """
        This function fits a gauss function to data. Uses numpy.optimize.curve_fit under the hood.

        :param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

        :param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

        :param guess: optional: A tuple of start parameters (x_0, sigma, A, C) as defined in gaussian method.

        :return: The coefficients and uncertainties of the fitted gaussian (x_0, sigma, A, C), as defined in gaussian
            method.
        """
        xdata = u.to_ureg(xdata)
        ydata = u.to_ureg(ydata)
        if guess is None:
            guess = (xdata[np.argmax(ydata)], (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
        # to assure the guess is represented in the correct units:
        xunit = xdata.units
        yunit = ydata.units
        unitslist = [xunit, xunit, yunit, yunit]
        guesslist = []
        for guesselement, guessunit in zip(guess, unitslist):
            guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
        guess = tuple(guesslist)
        return curve_fit(gaussian, xdata.magnitude, ydata.magnitude, guess)

    @staticmethod
    def sigma_from_FWHM(FWHM):
        return FWHM / (2 * np.sqrt(2 * np.log(2)))

    @staticmethod
    def FWHM_from_sigma(sigma):
        return sigma * (2 * np.sqrt(2 * np.log(2)))


class Gauss_Fit_nD(object):
    """
    A moredimensional Gauss Fit of given data, in the sense of fitting a Gauss function along a specified axis for
    all points on all other axes. This can be used to generate "peak maps" of given data, for example an
    autocorrelation width map of a time-resolved measurement.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True):
        global print_counter, start_time
        if data:
            data_id = data.get_datafield_index(data_id)
            axis_id = data.get_axis_index(axis_id)
            xaxis = data.get_axis(axis_id)
            ydata = data.get_datafield(data_id)
            xunit = data.get_axis(axis_id).get_unit()
            yunit = data.get_datafield(data_id).get_unit()

            map_shape_list = list(data.shape)
            del map_shape_list[axis_id]
            map_shape = tuple(map_shape_list)
            coeff_shape = tuple([4] + map_shape_list)
            self.coeffs = np.empty(coeff_shape)
            self.accuracy = np.empty(coeff_shape)

            # to assure the guess is represented in the correct units:
            if guess is not None:
                unitslist = [xunit, xunit, yunit, yunit]
                guesslist = []
                for guesselement, guessunit in zip(guess, unitslist):
                    guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
                guess = tuple(guesslist)

            if verbose:
                print("Fitting {0} gaussian functions".format(np.prod(map_shape)))
                start_time = time.time()
                print_counter = 0

            xvals = xaxis.get_data_raw()
            for index in np.ndindex(map_shape):
                indexlist = list(index)
                indexlist.insert(axis_id, np.s_[:])
                slicetup = tuple(indexlist)
                yvals = ydata.get_data()[slicetup].magnitude

                try:
                    if guess is None:
                        pointguess = (np.mean(xvals), (np.max(xvals) - np.min(xvals)) / 4, np.max(yvals), np.min(yvals))
                        coeffs, pcov = curve_fit(gaussian, xvals, yvals, pointguess)
                    else:
                        coeffs, pcov = curve_fit(gaussian, xvals, yvals, guess)
                except RuntimeError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)
                except ValueError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)
                # Take absolute value for sigma, because negative widths don't make sense and curve is identical:
                coeffs[1] = abs(coeffs[1])
                self.coeffs[(np.s_[:],) + index] = coeffs
                try:
                    self.accuracy[(np.s_[:],) + index] = np.sqrt(np.diag(pcov))
                except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                    self.accuracy[(np.s_[:],) + index] = np.inf

                if verbose:
                    print_counter += 1
                    if print_counter % print_interval == 0:
                        tpf = ((time.time() - start_time) / float(print_counter))
                        etr = tpf * (np.prod(map_shape) - print_counter)
                        print("tiff {0:d} / {1:d}, Time/Fit {3:.4f}s ETR: {2:.1f}s".format(print_counter,
                                                                                           np.prod(map_shape),
                                                                                           etr, tpf))

            self.x_0_unit = xunit
            self.sigma_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            self.axes = [ax for i, ax in enumerate(data.axes) if i != axis_id]
            if keepdata:
                self.data = data

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @property
    def sigma(self):
        return u.to_ureg(self.coeffs[1], self.sigma_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def sigma_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.sigma_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma_accuracy

    def export_parameters(self, h5target=None):
        if h5target:
            da_h5target = True
        else:
            da_h5target = None
        da1 = ds.DataArray(self.FWHM, label="fwhm", plotlabel="Fit Width (FWHM) / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da2 = ds.DataArray(self.A, label="amplitude", plotlabel="Fit Amplitude / Counts",
                           h5target=da_h5target)
        da3 = ds.DataArray(self.C, label="background", plotlabel="Fit Background / Counts",
                           h5target=da_h5target)
        da4 = ds.DataArray(self.x_0, label="center", plotlabel="Fit Center / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da5 = ds.DataArray(self.sigma, label="sigma", plotlabel="Fit Sigma) / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da1_acc = ds.DataArray(self.FWHM_accuracy, label="fwhm_accuracy",
                               plotlabel="Fit Width (FWHM) Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da2_acc = ds.DataArray(self.A_accuracy, label="amplitude_accuracy",
                               plotlabel="Fit Amplitude Fit Accuracy / Counts", h5target=da_h5target)
        da3_acc = ds.DataArray(self.C_accuracy, label="background_accuracy",
                               plotlabel="Fit Background Fit Accuracy / Counts", h5target=da_h5target)
        da4_acc = ds.DataArray(self.x_0_accuracy, label="center_accuracy",
                               plotlabel="Fit Center Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da5_acc = ds.DataArray(self.sigma_accuracy, label="sigma_accuracy",
                               plotlabel="Fit Sigma Fit Accuracy) / \\si{\\femto\\second}", h5target=da_h5target)
        return ds.DataSet("Gauss Fit", [da1, da2, da3, da4, da5, da1_acc, da2_acc, da3_acc, da4_acc, da5_acc],
                          self.axes, h5target=h5target)

    def gaussian(self, x, sel):
        """
        The Gauss function corresponding to the fit values of the Gauss_Fit instance at a given position of the
        data.

        :param x: The value for which to evaluate the gaussian. (Quantity or numerical in correct unit).

        :param sel: An index tuple addressing the selected point.
        :type sel: tuple of int

        :return: The value of the gaussian function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return gaussian(x, self.x_0[sel], self.sigma[sel], self.A[sel], self.C[sel])

    @staticmethod
    def sigma_from_FWHM(FWHM):
        return FWHM / (2 * np.sqrt(2 * np.log(2)))

    @staticmethod
    def FWHM_from_sigma(sigma):
        return sigma * (2 * np.sqrt(2 * np.log(2)))


def lorentzian(x, x_0, gamma, A, C):
    """
    A Lorentz function of the form lorentzian(x) = A * ( gamma**2 / ( (x - x_0)**2 + gamma**2 ) ) + C
    All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
    assumed.

    :param x: The variable x.

    :param x_0: (Same unit as x.) The center (peak position) of the distribution.

    :param gamma: (Same unit as x.) The scale parameter. Relates to FWHM by:
        FWHM = 2 * gamma

    :param A: (Same unit as C.) The amplitude of the peak relative to background.

    :param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

    :return: (Same unit as A and C.) The result of the gaussian function.
    """
    return A * (gamma ** 2 / ((x - x_0) ** 2 + gamma ** 2)) + C


class Lorentz_Fit(object):
    """
    A Lorentz Fit of given data with benefits.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
        if data:
            self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
            xunit = self.data.get_axis(0).get_unit()
            yunit = self.data.get_datafield(0).get_unit()

            if normalize:
                take_data = 0
            else:
                take_data = 1

            self.coeffs, pcov = self.fit_lorentzian(self.data.get_axis(0).get_data(),
                                                    self.data.get_datafield(take_data).get_data(),
                                                    guess)
            # Take absolute value for gamma, because negative widths don't make sense and curve is identical:
            self.coeffs[1] = abs(self.coeffs[1])
            try:
                self.accuracy = np.sqrt(np.diag(pcov))
            except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                self.accuracy = np.full(4, np.inf)

            self.x_0_unit = xunit
            self.gamma_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            if not keepdata:
                self.data = None

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @x_0.setter
    def x_0(self, newvalue):
        newvalue = u.to_ureg(newvalue, self.x_0_unit)
        self.coeffs[0] = newvalue.magnitude

    @property
    def gamma(self):
        return u.to_ureg(self.coeffs[1], self.gamma_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * self.gamma

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def gamma_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.gamma_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * self.gamma_accuracy

    @classmethod
    def from_coeffs(cls, coeffs):
        x_0, gamma, A, C = coeffs
        x_0 = u.to_ureg(x_0)
        gamma = u.to_ureg(gamma)
        A = u.to_ureg(A)
        C = u.to_ureg(C)
        xunit = x_0.units
        yunit = A.units

        new_instance = cls()
        new_instance.coeffs = coeffs
        new_instance.x_0_unit = xunit
        new_instance.gamma_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    @classmethod
    def from_xy(cls, xdata, ydata, guess=None):
        new_instance = cls()
        xdata = u.to_ureg(xdata)
        xunit = str(xdata.units)
        ydata = u.to_ureg(ydata)
        yunit = str(ydata.units)
        new_instance.coeffs, pcov = cls.fit_lorentzian(xdata, ydata, guess)
        try:
            new_instance.accuracy = np.sqrt(np.diag(pcov))
        except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
            new_instance.accuracy = np.full(4, np.inf)
        new_instance.x_0_unit = xunit
        new_instance.gamma_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    def lorentzian(self, x):
        """
        The Lorentz function corresponding to the fit values of the Lorentz_Fit instance.

        :param x: The value for which to evaluate the lorentzian. (Quantity or numerical in correct unit).

        :return: The value of the lorentzian function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return lorentzian(x, self.x_0, self.gamma, self.A, self.C)

    @staticmethod
    def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :param label: string: label for the produced DataSet

        :return: 1D-DataSet with projected Intensity Data and Power Axis.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        data_projected = data_full.project_nd(xaxis_index)
        data_projected = ds.DataArray(data_projected, label='projected data')
        # Normalize by scaling to 1:
        data_projected_norm = data_projected / data_projected.max()
        data_projected_norm.set_label("projected data normalized")
        # Initialize the DataSet containing only the projected powerlaw data;
        return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

    @staticmethod
    def extract_data_raw(data, data_id=0, axis_id=0):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :return: xdata, ydata: tuple of quantities with the projected data.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        return xaxis.get_data(), data_full.project_nd(xaxis_index)

    @staticmethod
    def fit_lorentzian(xdata, ydata, guess=None):
        """
        This function fits a lorentz function to data. Uses numpy.optimize.curve_fit under the hood.

        :param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

        :param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

        :param guess: optional: A tuple of start parameters (x_0, gamma, A, C) as defined in lorentzian method.

        :return: The coefficients and uncertainties of the fitted gaussian (x_0, gamma, A, C), as defined in lorentzian
            method.
        """
        xdata = u.to_ureg(xdata)
        ydata = u.to_ureg(ydata)
        if guess is None:
            guess = (xdata[np.argmax(ydata)], (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
        # to assure the guess is represented in the correct units:
        xunit = xdata.units
        yunit = ydata.units
        unitslist = [xunit, xunit, yunit, yunit]
        guesslist = []
        for guesselement, guessunit in zip(guess, unitslist):
            guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
        guess = tuple(guesslist)
        return curve_fit(lorentzian, xdata.magnitude, ydata.magnitude, guess)

    @staticmethod
    def gamma_from_FWHM(FWHM):
        return FWHM / 2.

    @staticmethod
    def FWHM_from_gamma(gamma):
        return gamma * 2.


class Lorentz_Fit_nD(object):
    """
    A moredimensional Lorentz Fit of given data, in the sense of fitting a Lorentz function along a specified axis for
    all points on all other axes. This can be used to generate "peak maps" of given data, for example an
    autocorrelation width map of a time-resolved measurement.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True):
        global print_counter, start_time
        if data:
            data_id = data.get_datafield_index(data_id)
            axis_id = data.get_axis_index(axis_id)
            xaxis = data.get_axis(axis_id)
            ydata = data.get_datafield(data_id)
            xunit = data.get_axis(axis_id).get_unit()
            yunit = data.get_datafield(data_id).get_unit()

            map_shape_list = list(data.shape)
            del map_shape_list[axis_id]
            map_shape = tuple(map_shape_list)
            coeff_shape = tuple([4] + map_shape_list)
            self.coeffs = np.empty(coeff_shape)
            self.accuracy = np.empty(coeff_shape)

            # to assure the guess is represented in the correct units:
            if guess is not None:
                unitslist = [xunit, xunit, yunit, yunit]
                guesslist = []
                for guesselement, guessunit in zip(guess, unitslist):
                    guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
                guess = tuple(guesslist)

            if verbose:
                print("Fitting {0} lorentzian functions".format(np.prod(map_shape)))
                start_time = time.time()
                print_counter = 0

            xvals = xaxis.get_data_raw()
            for index in np.ndindex(map_shape):
                indexlist = list(index)
                indexlist.insert(axis_id, np.s_[:])
                slicetup = tuple(indexlist)
                yvals = ydata.get_data()[slicetup].magnitude

                try:
                    if guess is None:
                        pointguess = (np.mean(xvals), (np.max(xvals) - np.min(xvals)) / 4, np.max(yvals), np.min(yvals))
                        coeffs, pcov = curve_fit(lorentzian, xvals, yvals, pointguess)
                    else:
                        coeffs, pcov = curve_fit(lorentzian, xvals, yvals, guess)
                except RuntimeError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)
                except ValueError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)

                # Take absolute value for gamma, because negative widths don't make sense and curve is identical:
                coeffs[1] = abs(coeffs[1])
                self.coeffs[(np.s_[:],) + index] = coeffs
                try:
                    self.accuracy[(np.s_[:],) + index] = np.sqrt(np.diag(pcov))
                except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                    self.accuracy[(np.s_[:],) + index] = np.inf

                if verbose:
                    print_counter += 1
                    if print_counter % print_interval == 0:
                        tpf = ((time.time() - start_time) / float(print_counter))
                        etr = tpf * (np.prod(map_shape) - print_counter)
                        print("tiff {0:d} / {1:d}, Time/Fit {3:.4f}s ETR: {2:.1f}s".format(print_counter,
                                                                                           np.prod(map_shape),
                                                                                           etr, tpf))

            self.x_0_unit = xunit
            self.gamma_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            self.axes = [ax for i, ax in enumerate(data.axes) if i != axis_id]
            if keepdata:
                self.data = data

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @property
    def gamma(self):
        return u.to_ureg(self.coeffs[1], self.gamma_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * self.gamma

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def gamma_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.gamma_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * self.gamma_accuracy

    def export_parameters(self, h5target=None):
        if h5target:
            da_h5target = True
        else:
            da_h5target = None
        da1 = ds.DataArray(self.FWHM, label="fwhm", plotlabel="Fit Width (FWHM) / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da2 = ds.DataArray(self.A, label="amplitude", plotlabel="Fit Amplitude / Counts",
                           h5target=da_h5target)
        da3 = ds.DataArray(self.C, label="background", plotlabel="Fit Background / Counts",
                           h5target=da_h5target)
        da4 = ds.DataArray(self.x_0, label="center", plotlabel="Fit Center / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da5 = ds.DataArray(self.gamma, label="gamma", plotlabel="Fit Gamma / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da1_acc = ds.DataArray(self.FWHM_accuracy, label="fwhm_accuracy",
                               plotlabel="Fit Width (FWHM) Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da2_acc = ds.DataArray(self.A_accuracy, label="amplitude_accuracy",
                               plotlabel="Fit Amplitude Fit Accuracy / Counts", h5target=da_h5target)
        da3_acc = ds.DataArray(self.C_accuracy, label="background_accuracy",
                               plotlabel="Fit Background Fit Accuracy / Counts", h5target=da_h5target)
        da4_acc = ds.DataArray(self.x_0_accuracy, label="center_accuracy",
                               plotlabel="Fit Center Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da5_acc = ds.DataArray(self.gamma_accuracy, label="gamma_accuracy",
                               plotlabel="Fit Gamma Fit Accuracy) / \\si{\\femto\\second}", h5target=da_h5target)
        return ds.DataSet("Lorentz Fit", [da1, da2, da3, da4, da5, da1_acc, da2_acc, da3_acc, da4_acc, da5_acc],
                          self.axes, h5target=h5target)

    def lorentzian(self, x, sel):
        """
        The Lorentz function corresponding to the fit values of the Lorentz_Fit instance at a given position of the
        data.

        :param x: The value for which to evaluate the lorentzian. (Quantity or numerical in correct unit).

        :param sel: An index tuple addressing the selected point.
        :type sel: tuple of int

        :return: The value of the lorentzian function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return lorentzian(x, self.x_0[sel], self.gamma[sel], self.A[sel], self.C[sel])

    @staticmethod
    def gamma_from_FWHM(FWHM):
        return FWHM / 2.

    @staticmethod
    def FWHM_from_gamma(gamma):
        return gamma * 2.


def sech2(x, x_0, tau, A, C):
    """
    A Sech squared function of the form sech2(x) = A * 2 / (np.exp((x - x_0) / tau) + np.exp(-(x - x_0) / tau)) ** 2 + C
    All parameters can be given as quantities, if so, unit checks are done automatically. If not, correct units are
    assumed.

    :param x: The variable x.

    :param x_0: (Same unit as x.) The center of the sech^2.

    :param tau: (Same unit as x.) The width of the sech^2. Relates to FWHM by:
        FWHM = 2 log(1+sqrt(2)) * tau

    :param A: (Same unit as C.) The amplitude of the sech^2 function relative to background.

    :param C: (Same unit as A.) The constant offset (background) of the curve relative to zero.

    :return: (Same unit as A and C.) The result of the sech^2 function.
    """
    return A * 4 / (np.exp((x - x_0) / tau) + np.exp(-(x - x_0) / tau)) ** 2 + C


class Sech2_Fit(object):
    """
    A Sech squared Fit of given data with benefits.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True, normalize=False):
        if data:
            self.data = self.extract_data(data, data_id=data_id, axis_id=axis_id)
            xunit = self.data.get_axis(0).get_unit()
            yunit = self.data.get_datafield(0).get_unit()

            if normalize:
                take_data = 0
            else:
                take_data = 1

            self.coeffs, pcov = self.fit_sech2(self.data.get_axis(0).get_data(),
                                               self.data.get_datafield(take_data).get_data(),
                                               guess)
            try:
                self.accuracy = np.sqrt(np.diag(pcov))
            except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                self.accuracy = np.full(4, np.inf)
            self.x_0_unit = xunit
            self.tau_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            if not keepdata:
                self.data = None

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @x_0.setter
    def x_0(self, newvalue):
        newvalue = u.to_ureg(newvalue, self.x_0_unit)
        self.coeffs[0] = newvalue.magnitude

    @property
    def tau(self):
        return u.to_ureg(self.coeffs[1], self.tau_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * np.log(1 + np.sqrt(2)) * self.tau

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def tau_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.tau_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * np.log(1 + np.sqrt(2)) * self.tau_accuracy

    @classmethod
    def from_coeffs(cls, coeffs):
        x_0, tau, A, C = coeffs
        x_0 = u.to_ureg(x_0)
        tau = u.to_ureg(tau)
        A = u.to_ureg(A)
        C = u.to_ureg(C)
        xunit = x_0.units
        yunit = A.units

        new_instance = cls()
        new_instance.coeffs = coeffs
        new_instance.x_0_unit = xunit
        new_instance.tau_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    @classmethod
    def from_xy(cls, xdata, ydata, guess=None):
        new_instance = cls()
        xdata = u.to_ureg(xdata)
        xunit = str(xdata.units)
        ydata = u.to_ureg(ydata)
        yunit = str(ydata.units)
        new_instance.coeffs, pcov = cls.fit_sech2(xdata, ydata, guess)
        try:
            new_instance.accuracy = np.sqrt(np.diag(pcov))
        except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
            new_instance.accuracy = np.full(4, np.inf)
        new_instance.x_0_unit = xunit
        new_instance.tau_unit = xunit
        new_instance.A_unit = yunit
        new_instance.C_unit = yunit
        return new_instance

    def sech2(self, x):
        """
        The Hyperbolic secant squared function corresponding to the fit values of the Sech2_Fit instance.

        :param x: The value for which to evaluate the sech2. (Quantity or numerical in correct unit).

        :return: The value of the sech2 function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return sech2(x, self.x_0, self.tau, self.A, self.C)

    @staticmethod
    def extract_data(data, data_id=0, axis_id=0, label="fitdata"):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :param label: string: label for the produced DataSet

        :return: 1D-DataSet with projected Intensity Data and Power Axis.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        data_projected = data_full.project_nd(xaxis_index)
        data_projected = ds.DataArray(data_projected, label='projected data')
        # Normalize by scaling to 1:
        data_projected_norm = data_projected / data_projected.max()
        data_projected_norm.set_label("projected data normalized")
        # Initialize the DataSet containing only the projected powerlaw data;
        return ds.DataSet(label, [data_projected_norm, data_projected], [xaxis])

    @staticmethod
    def extract_data_raw(data, data_id=0, axis_id=0):
        """
        Extracts the datapoints to be fitted out of a dataset. Therefore, it takes the chosen axis of the input data,
        and projects the chosen datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the data.

        :param data_id: Identifier of the DataField to use. By default, the first DataField is used.

        :param axis_id: Identifier of the axis to use. By default, the first Axis is used.

        :return: xdata, ydata: tuple of quantities with the projected data.
        """
        assert isinstance(data, ds.DataSet) or isinstance(data, ds.ROI), \
            "ERROR: No dataset or ROI instance given to fit data extraction."
        xaxis = data.get_axis(axis_id)
        data_full = data.get_datafield(data_id)
        xaxis_index = data.get_axis_index(axis_id)
        return xaxis.get_data(), data_full.project_nd(xaxis_index)

    @staticmethod
    def fit_sech2(xdata, ydata, guess=None):
        """
        This function fits a sech^2 function to data. Uses numpy.optimize.curve_fit under the hood.

        :param xdata: A quantity or array. If no quantity, dimensionless data are assumed.

        :param ydata: Quantity or array: The corresponding y values. If no quantity, dimensionless data are assumed.

        :param guess: optional: A tuple of start parameters (x_0, tau, A, C) as defined in sech squared method.

        :return: The coefficients and uncertainties of the fitted gaussian (x_0, tau, A, C), as defined in sech2
            method.
        """
        xdata = u.to_ureg(xdata)
        ydata = u.to_ureg(ydata)
        if guess is None:
            guess = (xdata[np.argmax(ydata)], (np.max(xdata) - np.min(xdata)) / 4, np.max(ydata), np.min(ydata))
        # to assure the guess is represented in the correct units:
        xunit = xdata.units
        yunit = ydata.units
        unitslist = [xunit, xunit, yunit, yunit]
        guesslist = []
        for guesselement, guessunit in zip(guess, unitslist):
            guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
        guess = tuple(guesslist)
        return curve_fit(sech2, xdata.magnitude, ydata.magnitude, guess)

    @staticmethod
    def tau_from_FWHM(FWHM):
        return FWHM / (2 * np.log(1 + np.sqrt(2)))

    @staticmethod
    def FWHM_from_tau(tau):
        return 2 * np.log(1 + np.sqrt(2)) * tau


class Sech2_Fit_nD(object):
    """
    A moredimensional Hyperbolic Secant Squared Fit of given data, in the sense of fitting a Sech2 function along a specified axis for
    all points on all other axes. This can be used to generate "peak maps" of given data, for example an
    autocorrelation width map of a time-resolved measurement.
    """

    def __init__(self, data=None, guess=None, data_id=0, axis_id=0, keepdata=True):
        global print_counter, start_time
        if data:
            data_id = data.get_datafield_index(data_id)
            axis_id = data.get_axis_index(axis_id)
            xaxis = data.get_axis(axis_id)
            ydata = data.get_datafield(data_id)
            xunit = data.get_axis(axis_id).get_unit()
            yunit = data.get_datafield(data_id).get_unit()

            map_shape_list = list(data.shape)
            del map_shape_list[axis_id]
            map_shape = tuple(map_shape_list)
            coeff_shape = tuple([4] + map_shape_list)
            self.coeffs = np.empty(coeff_shape)
            self.accuracy = np.empty(coeff_shape)

            # to assure the guess is represented in the correct units:
            if guess is not None:
                unitslist = [xunit, xunit, yunit, yunit]
                guesslist = []
                for guesselement, guessunit in zip(guess, unitslist):
                    guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
                guess = tuple(guesslist)

            if verbose:
                print("Fitting {0} sech2 functions".format(np.prod(map_shape)))
                start_time = time.time()
                print_counter = 0

            xvals = xaxis.get_data_raw()
            for index in np.ndindex(map_shape):
                indexlist = list(index)
                indexlist.insert(axis_id, np.s_[:])
                slicetup = tuple(indexlist)
                yvals = ydata.get_data()[slicetup].magnitude

                try:
                    if guess is None:
                        pointguess = (np.mean(xvals), (np.max(xvals) - np.min(xvals)) / 4, np.max(yvals), np.min(yvals))
                        coeffs, pcov = curve_fit(gaussian, xvals, yvals, pointguess)
                    else:
                        coeffs, pcov = curve_fit(sech2, xvals, yvals, guess)
                except RuntimeError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)
                except ValueError as e:
                    coeffs = np.zeros(4)
                    pcov = np.full((4, 4), np.inf)
                # Take absolute value for tau, because negative widths don't make sense and curve is identical:
                coeffs[1] = abs(coeffs[1])
                self.coeffs[(np.s_[:],) + index] = coeffs
                try:
                    self.accuracy[(np.s_[:],) + index] = np.sqrt(np.diag(pcov))
                except ValueError as e:  # Fit failed. pcov = inf, diag(inf) throws exception:
                    self.accuracy[(np.s_[:],) + index] = np.inf

                if verbose:
                    print_counter += 1
                    if print_counter % print_interval == 0:
                        tpf = ((time.time() - start_time) / float(print_counter))
                        etr = tpf * (np.prod(map_shape) - print_counter)
                        print("tiff {0:d} / {1:d}, Time/Fit {3:.4f}s ETR: {2:.1f}s".format(print_counter,
                                                                                           np.prod(map_shape),
                                                                                           etr, tpf))

            self.x_0_unit = xunit
            self.tau_unit = xunit
            self.A_unit = yunit
            self.C_unit = yunit
            self.axes = [ax for i, ax in enumerate(data.axes) if i != axis_id]
            if keepdata:
                self.data = data

    @property
    def x_0(self):
        return u.to_ureg(self.coeffs[0], self.x_0_unit)

    @property
    def tau(self):
        return u.to_ureg(self.coeffs[1], self.tau_unit)

    @property
    def A(self):
        return u.to_ureg(self.coeffs[2], self.A_unit)

    @property
    def C(self):
        return u.to_ureg(self.coeffs[3], self.C_unit)

    @property
    def FWHM(self):
        return 2 * np.log(1 + np.sqrt(2)) * self.tau

    @property
    def x_0_accuracy(self):
        return u.to_ureg(self.accuracy[0], self.x_0_unit)

    @property
    def tau_accuracy(self):
        return u.to_ureg(self.accuracy[1], self.tau_unit)

    @property
    def A_accuracy(self):
        return u.to_ureg(self.accuracy[2], self.A_unit)

    @property
    def C_accuracy(self):
        return u.to_ureg(self.accuracy[3], self.C_unit)

    @property
    def FWHM_accuracy(self):
        return 2 * np.log(1 + np.sqrt(2)) * self.tau_accuracy

    def export_parameters(self, h5target=None):
        if h5target:
            da_h5target = True
        else:
            da_h5target = None
        da1 = ds.DataArray(self.FWHM, label="fwhm", plotlabel="Fit Width (FWHM) / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da2 = ds.DataArray(self.A, label="amplitude", plotlabel="Fit Amplitude / Counts",
                           h5target=da_h5target)
        da3 = ds.DataArray(self.C, label="background", plotlabel="Fit Background / Counts",
                           h5target=da_h5target)
        da4 = ds.DataArray(self.x_0, label="center", plotlabel="Fit Center / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da5 = ds.DataArray(self.tau, label="tau", plotlabel="Fit Tau / \\si{\\femto\\second}",
                           h5target=da_h5target)
        da1_acc = ds.DataArray(self.FWHM_accuracy, label="fwhm_accuracy",
                               plotlabel="Fit Width (FWHM) Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da2_acc = ds.DataArray(self.A_accuracy, label="amplitude_accuracy",
                               plotlabel="Fit Amplitude Fit Accuracy / Counts", h5target=da_h5target)
        da3_acc = ds.DataArray(self.C_accuracy, label="background_accuracy",
                               plotlabel="Fit Background Fit Accuracy / Counts", h5target=da_h5target)
        da4_acc = ds.DataArray(self.x_0_accuracy, label="center_accuracy",
                               plotlabel="Fit Center Fit Accuracy / \\si{\\femto\\second}", h5target=da_h5target)
        da5_acc = ds.DataArray(self.tau_accuracy, label="tau_accuracy",
                               plotlabel="Fit Tau Fit Accuracy) / \\si{\\femto\\second}", h5target=da_h5target)
        return ds.DataSet("Sech2 Fit", [da1, da2, da3, da4, da5, da1_acc, da2_acc, da3_acc, da4_acc, da5_acc],
                          self.axes, h5target=h5target)

    def sech2(self, x, sel):
        """
        The Hyperbolic secant squared function corresponding to the fit values of the Sech2_Fit instance at a given position of the
        data.

        :param x: The value for which to evaluate the sech2. (Quantity or numerical in correct unit).

        :param sel: An index tuple addressing the selected point.
        :type sel: tuple of int

        :return: The value of the sech2 function at the value x. Returned as Quantity in whichever unit the fit
            data was given.
        """
        x = u.to_ureg(x, self.x_0_unit)
        return sech2(x, self.x_0[sel], self.tau[sel], self.A[sel], self.C[sel])

    @staticmethod
    def tau_from_FWHM(FWHM):
        return FWHM / (2 * np.log(1 + np.sqrt(2)))

    @staticmethod
    def FWHM_from_tau(tau):
        return 2 * np.log(1 + np.sqrt(2)) * tau
