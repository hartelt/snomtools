"""
This script provides tabulated literature values for material properties.
It reads the data files in literature as extracted from the papers, and converts them to numpy arrays.
The data array will be in the following form:
wavelength, dielectric constant, refraction index

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import os
import scipy.interpolate
import snomtools.calcs.prefixes as pref
import snomtools.calcs.conversions as conv
import snomtools.calcs.units as u

__author__ = 'hartelt'


# TODO: Compatibility with pint

class Literature_Material(object):
    def __init__(self, wavelengths, epsilons=None, ns=None, spline_order=3):
        self.literature_wavelengths = u.to_ureg(wavelengths, 'nanometer').real
        if (epsilons is None) and (ns is None):
            raise ValueError("Literature Material cannot be initializes without values.")
        if epsilons is not None:
            self.literature_epsilons = u.to_ureg(epsilons, 'dimensionless')
            self.eps_r_spline = scipy.interpolate.splrep(self.literature_wavelengths.magnitude,
                                                         self.literature_epsilons.real.magnitude,
                                                         k=spline_order)
            self.eps_i_spline = scipy.interpolate.splrep(self.literature_wavelengths.magnitude,
                                                         self.literature_epsilons.imag.magnitude,
                                                         k=spline_order)
        else:
            self.literature_epsilons = None
        if ns is not None:
            self.literature_ns = u.to_ureg(ns, 'dimensionless')
            self.n_r_spline = scipy.interpolate.splrep(self.literature_wavelengths.magnitude,
                                                       self.literature_ns.real.magnitude,
                                                       k=spline_order)
            self.n_i_spline = scipy.interpolate.splrep(self.literature_wavelengths.magnitude,
                                                       self.literature_ns.imag.magnitude,
                                                       k=spline_order)
        else:
            self.literature_ns = None

    def epsilon(self, omega):
        """
        This method approximates the complex dielectric function of the material
        at a given frequency,
        using spline interpolation of the literature data.
        If the frequency is outside of literature data range, a ValueError is thrown.
        If only refractive index data is available, the dielectric function is calculated from it,
        assuming non-magnetic behaviour.

        :param omega: the frequency at which the dielectric function shall be calculated (float or numpy array) in rad/s

        :return: the complex dielectric function (dimensionless)
        """
        omega = u.to_ureg(omega, 'rad/s')
        wl = conv.omega2lambda(omega).to('nanometer')
        if self.literature_epsilons is not None:
            return u.to_ureg(scipy.interpolate.splev(wl, self.eps_r_spline, ext=2) +
                             1j * scipy.interpolate.splev(wl, self.eps_i_spline, ext=2))
        else:
            return u.to_ureg(conv.n2epsilon(self.n(omega)))

    def n(self, omega):
        """
        This method approximates the complex refraction index of the material
        at a given frequency,
        using spline interpolation of the literature data.
        If the frequency is outside of literature data range, a ValueError is thrown.
        If only dielectric function data is available, the refractive index is calculated from it,
        assuming non-magnetic behaviour.

        :param omega: the frequency at which the refraction index shall be calculated (float or numpy array) in rad/s

        :return: the complex refraction index (dimensionless)
        """
        omega = u.to_ureg(omega, 'rad/s')
        wl = conv.omega2lambda(omega).to('nanometer')
        if self.literature_ns is not None:
            return u.to_ureg(scipy.interpolate.splev(wl, self.n_r_spline, ext=2) +
                             1j * scipy.interpolate.splev(wl, self.n_i_spline, ext=2))
        else:
            return u.to_ureg(conv.epsilon2n(self.epsilon(omega)))


ownpath = os.path.dirname(os.path.realpath(__file__))

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/johnson-christy.dat")))
# datafile is formatted: wavelength in micrometers, n, k
wl = raw[:, 0] * pref.micro
n = raw[:, 1] + (raw[:, 2] * 1j)
eps = conv.n2epsilon(n)
Au_Johnson_Christy_data_raw = numpy.column_stack((wl, eps, n))
Au_Johnson_Christy_data = [u.to_ureg(wl, 'meter'), eps, n]
Au_Johnson_Christy = Literature_Material(u.to_ureg(wl, 'meter'), eps, n)
"""
Au by Johnson and Christy

.. code-block:: bibtex

	@Article{johnsonchristy1972,
	  Title                    = {Optical constants of the noble metals},
	  Author                   = {Johnson, Peter B and Christy, R. W.},
	  Journal                  = {Physical Review B},
	  Year                     = {1972},
	  Number                   = {12},
	  Pages                    = {4370},
	  Volume                   = {6},
	  Doi                      = {http://dx.doi.org/10.1103/PhysRevB.6.4370},
	  File                     = {:Paper/Johnson-Christy-Dielectric constants/PhysRevB.6.4370.pdf:PDF},
	  Publisher                = {APS},
	  Timestamp                = {2015.03.26}
	}

"""

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/Olmon_PRB2012_EV.dat")))
wl = raw[:, 1]
eps = raw[:, 2] + 1j * raw[:, 3]
n = raw[:, 4] + 1j * raw[:, 5]
Au_Olmon_evaporated_data_raw = numpy.column_stack((wl, eps, n))
Au_Olmon_evaporated_data = [u.to_ureg(wl, 'meter'), eps, n]
Au_Olmon_evaporated = Literature_Material(u.to_ureg(wl, 'meter'), eps, n)

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/Olmon_PRB2012_SC.dat")))
wl = raw[:, 1]
eps = raw[:, 2] + 1j * raw[:, 3]
n = raw[:, 4] + 1j * raw[:, 5]
Au_Olmon_singlecristalline_data_raw = numpy.column_stack((wl, eps, n))
Au_Olmon_singlecristalline_data = [u.to_ureg(wl, 'meter'), eps, n]
Au_Olmon_singlecristalline = Literature_Material(u.to_ureg(wl, 'meter'), eps, n)

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/Olmon_PRB2012_TS.dat")))
wl = raw[:, 1]
eps = raw[:, 2] + 1j * raw[:, 3]
n = raw[:, 4] + 1j * raw[:, 5]
Au_Olmon_templatestripped_data_raw = numpy.column_stack((wl, eps, n))
Au_Olmon_templatestripped_data = [u.to_ureg(wl, 'meter'), eps, n]
Au_Olmon_templatestripped = Literature_Material(u.to_ureg(wl, 'meter'), eps, n)
"""
Au by Olmon et al.

.. code-block:: bibtex

	@Article{Olmon2012,
	  Title                    = {Optical dielectric function of gold},
	  Author                   = {Olmon, Robert L. and Slovick, Brian and Johnson, Timothy W. and Shelton, David and Oh, Sang-Hyun and Boreman, Glenn D. and Raschke, Markus B.},
	  Journal                  = {Phys. Rev. B},
	  Year                     = {2012},

	  Month                    = {Dec},
	  Number                   = {23},
	  Volume                   = {86},

	  Doi                      = {10.1103/physrevb.86.235147},
	  File                     = {:Paper/Olmon-PRB2012-Optical dielectric function of gold/PhysRevB.86.235147.pdf:PDF},
	  ISSN                     = {1550-235X},
	  Keywords                 = {dielectric constants},
	  Publisher                = {American Physical Society (APS)},
	  Timestamp                = {2015.03.26},
	  Url                      = {http://dx.doi.org/10.1103/PhysRevB.86.235147}
	}

There are three different datasets:
- evaporated (EV)
- single cristalline (SC)
- template stripped (TS)
Datafiles are formatted: Photon energy/eV	Wavelength/m	ep1	ep2	n	k
"""

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/ITO/ITO-Koenig.dat")))
# Datafile format: wl/um	n		k
wl = raw[:, 0] * pref.micro
n = raw[:, 1] + (raw[:, 2] * 1j)
eps = conv.n2epsilon(n)
ITO_Koenig_data_raw = numpy.column_stack((wl, eps, n))
ITO_Koenig_data = [u.to_ureg(wl, 'meter'), eps, n]
ITO_Koenig = Literature_Material(u.to_ureg(wl, 'meter'), eps, n)
"""
ITO by Koenig et al.
http://refractiveindex.info
Numerical data provided by Tobias Koenig
Koenig et al.:
Electrically Tunable Plasmonic Behavior of Nanocube-Polymer Nanomaterials Induced by a Redox-Active Electrochromic Polymer
ACS Nano, American Chemical Society (ACS), 2014, 8, 6182-6192
"""

# for testing:
if __name__ == "__main__":
    mat = Au_Johnson_Christy
    from matplotlib import pyplot as plt

    xvals = u.to_ureg(numpy.arange(400, 800, .05), 'nanometer')
    plt.plot(xvals.magnitude, mat.n(conv.lambda2omega(xvals)).real,
             '-', label='approx')

    plt.plot(Au_Johnson_Christy_data_raw[:, 0] * 1e9, Au_Johnson_Christy_data_raw[:, 2].real, '.', label='Olmon')

    plt.legend()
    plt.show()
