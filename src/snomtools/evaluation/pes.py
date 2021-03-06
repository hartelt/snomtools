"""
This file provides data evaluation scripts for photoemission spectroscopy (PES) data. This includes anything that is
not experiment-specific, but can be applied for all photoemission spectra.
For further info about data structures, see:
data.imports.tiff.py
data.datasets.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import snomtools.calcs.units as u
import numpy as np
import snomtools.data.datasets
from scipy.optimize import curve_fit
import scipy.special
import snomtools.calcs.constants as const

__author__ = 'hartelt'

k_B = const.k_B  # The Boltzmann constant
Temp = u.to_ureg(300, "K")  # The Temperature, for now hardcoded as room temperature.
kBT_in_eV = (k_B.to("eV/K") * Temp)


def fermi_edge(E, E_f, dE, c, d):
    """
    The typical shape of a fermi edge for constant DOS. Suitable as a fit function, therefore it takes only floats,
    no quantities.

    :param E: The x-Axis of the data consists of energies in eV

    :param E_f: The Fermi energy in eV.

    :param dE: Energy Resolution. The broadening of the Fermi edge on top of the thermal broadening,
        which is introduced by all experimental errors, in eV.

    :param c: The height of the Fermi edge, in whichever units the data is given, e.g. "counts".

    :param d: Offset of the lower level of the fermi edge, e.g. "dark counts".

    :return: The value of the Fermi distribution at the energy E.
    """
    E = u.to_ureg(E, 'eV').magnitude
    E_f = u.to_ureg(E_f, 'eV').magnitude
    dE = u.to_ureg(dE, 'eV').magnitude
    c = u.to_ureg(c)
    d = u.to_ureg(d)
    return 0.5 * (1 +
                  scipy.special.erf((E_f - E) / (np.sqrt(((1.7 * kBT_in_eV.magnitude) ** 2) + dE ** 2) * np.sqrt(2)))
                  ) * c + d


class FermiEdge:
    """
    A fermi edge in a spectrum...
    """

    def __init__(self, data=None, guess=None, keepdata=True, normalize=False):
        """
        Extract the fermi edge of a given data set or ROI.

        :param data: Choose the dataset containing the spectral data that should be extracted.

        :param guess: Optional parameter, set tuple of the start parameters (E_f, dE, c, d)
            defined in the fermi_edge method.

        :param keepdata: Keep the the data that was fitted to. If `false`, keep just the extracted parameters.

        :param normalize: Normalize the spectrum before fitting to (0,1).
            This discards the information about the absolute value of the spectrum,
            but makes guessing the start parameters very simple.
        """
        if data:
            self.data = self.extract_data(data)
            energyunit = self.data.get_axis(0).get_unit()
            countsunit = self.data.get_datafield(0).get_unit()
            if normalize:
                take_data = 0
            else:
                take_data = 1

            self.coeffs, self.accuracy = self.fit_fermi_edge(self.data.get_axis(0).get_data(),
                                                             self.data.get_datafield(take_data).get_data(),
                                                             guess)
            self.E_f_unit = energyunit
            self.dE_unit = energyunit
            self.c_unit = countsunit
            self.d_unit = countsunit
            if not keepdata:
                self.data = None

    def __getattr__(self, item):
        """
        This method provides dynamical naming in instances. It is called any time an attribute of the instance is
        not found with the normal naming mechanism. Raises an AttributeError if the name cannot be resolved.

        :param item: The name to get the corresponding attribute.

        :return: The attribute corresponding to the given name.
        """
        raise AttributeError("Attribute \'{0}\' of Fermi_Edge instance cannot be resolved.".format(item))

    @property
    def E_f(self):
        return u.to_ureg(self.coeffs[0], self.E_f_unit)

    @E_f.setter
    def E_f(self, newvalue):
        newvalue = u.to_ureg(newvalue, self.E_f_unit)
        self.coeffs[0] = newvalue.magnitude

    @property
    def dE(self):
        return u.to_ureg(self.coeffs[1], self.dE_unit)

    @property
    def c(self):
        return u.to_ureg(self.coeffs[2], self.c_unit)

    @property
    def d(self):
        return u.to_ureg(self.coeffs[3], self.d_unit)

    @classmethod
    def from_coeffs(cls, coeffs, energyunit='eV', countsunit=None):
        # Parse input and handle units:
        E_f, d_E, c, d = coeffs
        E_f = u.to_ureg(E_f, energyunit)
        d_E = u.to_ureg(d_E, energyunit)
        c = u.to_ureg(c, countsunit)
        d = u.to_ureg(d, countsunit)

        # Prepare variables:
        energyunit = str(E_f.units)
        countsunit = str(c.units)
        coeffs = (E_f.magnitude, d_E.magnitude, c, d)

        # Make instance:
        f = cls()
        f.coeffs = coeffs
        f.E_f_unit = energyunit
        f.dE_unit = energyunit
        f.c_unit = countsunit
        f.d_unit = countsunit
        return f

    @classmethod
    def from_xy(cls, energies, intensities, guess):
        f = cls()
        f.coeffs, f.accuracy = cls.fit_fermi_edge(energies, intensities, guess)
        return f

    def fermi_edge(self, E):
        """
        The shape of a fermi edge for the known fit parameters of the Fermi_Edge instance.
        This is the equivalent of the standalone `fermi_fit` function outside the class,
        only it is aware of the units of the class parameters.

        :param E: Electron Energy (Quantity or numerical in eV).

        :return: The value of the Fermi distribution at the energy E. Returned as Quantity in whichever unit the fit
            data was given.
        """
        E = u.to_ureg(E, self.E_f_unit)
        E_f = self.E_f
        dE = self.dE
        c = self.c
        d = self.d
        return 0.5 * (1 + scipy.special.erf(
            ((E_f - E) / (np.sqrt(((1.7 * kBT_in_eV) ** 2) + dE ** 2) * np.sqrt(2))).magnitude)
                      ) * c + d

    @staticmethod
    def extract_data_raw(data, data_id=0, axis_id=None):
        """
        Extracts the energies and intensities out of a dataset. Therefore, it takes the energy axis of the input data,
        and projects the datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the spectral data.

        :param data_id: Identifier of the DataField to use.

        :param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
            to a Power in its physical dimension is taken.

        :return: energies, intensities: tuple of quantities with the projected data.
        """
        assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
            "ERROR: No dataset or ROI instance given to Powerlaw data extraction."
        if axis_id is None:
            energy_axis = data.get_axis_by_dimension("eV")
        else:
            energy_axis = data.get_axis(axis_id)
        count_data = data.get_datafield(data_id)
        energy_axis_index = data.get_axis_index(energy_axis.get_label())
        return energy_axis.get_data(), count_data.project_nd(energy_axis_index)

    @staticmethod
    def extract_data(data, data_id=0, axis_id=None, label="fermiedge"):
        """
        Extracts the energies and intensities out of a dataset. Therefore, it takes the energy axis of the input data,
        and projects the datafield onto that axis by summing over all the other axes.

        :param data: Dataset containing the spectral data.

        :param data_id: Identifier of the DataField to use.

        :param axis_id: optional, Identifier of the power axis to use. If not given, the first axis that corresponds
            to a Power in its physical dimension is taken.

        :param label: string: label for the produced DataSet

        :return: 1D-DataSet with projected Intensity Data and Power Axis.
        """
        assert isinstance(data, snomtools.data.datasets.DataSet) or isinstance(data, snomtools.data.datasets.ROI), \
            "ERROR: No dataset or ROI instance given to Powerlaw data extraction."
        if axis_id is None:
            energy_axis = data.get_axis_by_dimension("eV")
        else:
            energy_axis = data.get_axis(axis_id)
        count_data = data.get_datafield(data_id)
        energy_axis_index = data.get_axis_index(energy_axis.get_label())
        count_data_projected = count_data.project_nd(energy_axis_index, ignorenan=True)
        count_data_projected = snomtools.data.datasets.DataArray(count_data_projected, label='intensity')
        # Normalize by scaling to 1:
        count_data_projected_norm = count_data_projected / count_data_projected.max()
        count_data_projected_norm.set_label("intensity_normalized")
        # Initialize the DataSet containing only the projected powerlaw data;
        return snomtools.data.datasets.DataSet(label, [count_data_projected_norm, count_data_projected], [energy_axis])

    @staticmethod
    def fit_fermi_edge(energies, intensities, guess=None):
        """
        This function fits a fermi edge to data. Uses numpy.optimize.curve_fit under the hood.

        :param energies: A quantity or array of energies. If no quantity, electronvolts are assumed.

        :param intensities: Quantity or array: The corresponding intensity values to powers.

        :param guess: optional: A tuple of start parameters (E_f, dE, c, d) as defined in fermi_edge method.

        :return: The coefficients and uncertainties of the fitted fermi edge E_f, dE, c, d, as defined in fermi_edge
            method.
        """
        if u.is_quantity(energies):
            assert u.same_dimension(energies, "eV")
            energies = u.to_ureg(energies)
        else:
            energies = u.to_ureg(energies, 'eV')
        intensities = u.to_ureg(intensities)
        if guess is None:
            guess = (29.6, 0.1, 1.0, 0.01)  # Just typical values
        else:  # to assure the guess is represented in the correct units:
            energyunit = energies.units
            countsunit = intensities.units
            unitslist = [energyunit, energyunit, countsunit, countsunit]
            guesslist = []
            for guesselement, guessunit in zip(guess, unitslist):
                guesslist.append(u.to_ureg(guesselement, guessunit).magnitude)
            guess = tuple(guesslist)
        return curve_fit(fermi_edge, energies.magnitude, intensities.magnitude, guess)

# def fermi_fit(data, energy_axis=None, range=None, guess=None):
# 	"""
# 	Fit a Fermi Distribution to the given data.
#
# 	:param data:
#
# 	:param energy_axis:
#
# 	:param range:
#
# 	:param guess:
#
# 	:return:
# 	"""
# 	raise NotImplementedError()
# fermi_fit is redundant I will remove this when i am ready with all the other tasks
