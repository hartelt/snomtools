"""
This script provides tabulated literature values for material properties.
It reads the data files in literature as extracted from the papers, and converts them to numpy arrays.
The data array will be in the following form:
wavelength, dielectric constant, refraction index

"""
__author__ = 'hartelt'

# TODO: Compatibility with pint

import numpy
import os
import snomtools.calcs.prefixes as pref
import snomtools.calcs.conversions as conv

ownpath = os.path.dirname(os.path.realpath(__file__))

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/johnson-christy.dat")))
# datafile is formatted: wavelength in micrometers, n, k
wl = raw[:, 0] * pref.micro
n = raw[:, 1] + (raw[:, 2] * 1j)
eps = conv.n2epsilon(n)
Au_Johnson_Christy = numpy.column_stack((wl, eps, n))
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
Au_Olmon_evaporated = numpy.column_stack((wl, eps, n))

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/Olmon_PRB2012_SC.dat")))
wl = raw[:, 1]
eps = raw[:, 2] + 1j * raw[:, 3]
n = raw[:, 4] + 1j * raw[:, 5]
Au_Olmon_singlecristalline = numpy.column_stack((wl, eps, n))

raw = numpy.loadtxt(os.path.abspath(os.path.join(ownpath, "literature/Au/Olmon_PRB2012_TS.dat")))
wl = raw[:, 1]
eps = raw[:, 2] + 1j * raw[:, 3]
n = raw[:, 4] + 1j * raw[:, 5]
Au_Olmon_templatestripped = numpy.column_stack((wl, eps, n))
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
ITO_Koenig = numpy.column_stack((wl, eps, n))
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
	print(Au_Olmon_singlecristalline)
