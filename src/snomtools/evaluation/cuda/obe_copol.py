# -*- coding:utf-8 -*-

"""
This file contains evaluation methods that use an Optical Bloch Equations (OBE) model,
assuming co-polarized laser pulses incident on a three-level-system.
It is typically used to evaluate time-resolved 2-Photon-Photoemission (2PPE) data.
"""

import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.cumath
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from scipy import fftpack
import scipy.signal as signal
from scipy.signal import freqz, butter, lfilter
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from timeit import default_timer as timer
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline
import re
import h5py
import sys
import os
from snomtools.evaluation.cuda import load_cuda_source
import snomtools.data.datasets as ds
import snomtools.calcs.constants

# For running snomtools on elwe, including cuda evaluations:
# Import snomtools from source by pulling the repo to ~/repos/ and then using
# home = os.path.expanduser("~")
# sys.path.insert(0, os.path.join(home, "repos/snomtools/src/"))
# sys.path.insert(0, os.path.join(home, "repos/pint/"))

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False

# Initialize Nvidia Graphics Card for pycuda:
if verbose:
    print('_____INITIATING PYCUDA DEVICE_____')
dev = pycuda.autoinit.device
gpuOBE_blocksize = int(dev.max_block_dim_x / 4)
if verbose:
    print('Device: ' + str(drv.Device.name(dev)))
    print('Compute capability: ' + str(drv.Device.compute_capability(dev)[0]) + '.' + str(
        drv.Device.compute_capability(dev)[1]))
    print('Total memory: ' + str(drv.Device.total_memory(dev) / 1024 / 1024) + ' MB')
    print('Device clock rate: ' + str(dev.clock_rate / 1000) + ' MHz')
    print('Max grid size x: ' + str(dev.max_grid_dim_x))
    print('Max grid size y: ' + str(dev.max_grid_dim_y))
    print('Max block size x: ' + str(dev.max_block_dim_x))
    print('Max block size y: ' + str(dev.max_block_dim_y))
    print('Max block size z: ' + str(dev.max_block_dim_z))
    print('Multiprocessor count: ' + str(dev.multiprocessor_count))
    print('shared_memory_per_block: ' + str(dev.shared_memory_per_block))
    print('max_shared_memory_per_block: ' + str(dev.max_shared_memory_per_block))
    print('registers_per_block: ' + str(dev.registers_per_block))
    print('max_registers_per_block: ' + str(dev.max_registers_per_block))
    print('max_threads_per_multiprocessor: ' + str(dev.max_threads_per_multiprocessor))
    print('max_threads_per_block: ' + str(dev.max_threads_per_block))
    print('Setting max buffer size to half that value (for double precision?): ' + str(gpuOBE_blocksize))
    print('__________')

# Load and compile Cuda Source:
CUDA_SOURCEFILE = "obe_source_module_copol.cu"
if verbose:
    print('Compiling cuda source module...')
mod = SourceModule(load_cuda_source(CUDA_SOURCEFILE))

# Constants:
c = snomtools.calcs.constants.c_float  # TODO: Change to proper quantity usage.

# Simulation overhead - Do more timesteps than delaysteps to avoid that the
# system reached equilibrium. If you have a dropping signal at the border
# of your simulation increase this value
gpuOBE_simOverhead = 1.3  # 1.2

# Tau for FWHM relax - This small tau is used to fit the FWHM of the
# corresponding laser pulse. Don't use 0, otherwise the numerics will crash
gpuOBE_simFWHMTau = 0.05


def lowpass(data, highcut, srate, order):
    nyq = 1 / (2 * srate)
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='low')  # ,analog = True)
    w, h = freqz(b, a, worN=5000)
    res = signal.filtfilt(b, a, data)
    return [res, w, h]


def norm(data):
    return (data - data.min()) / (data.max() - data.min())


def getw0CoPol(timedata, stepsize, normparameter=False):
    filtdata0, w0, h0 = lowpass(timedata, 0.1, stepsize, 5)
    if (normparameter == False):
        return filtdata0
    elif (normparameter == True):
        return norm(filtdata0)


def CreateCoPolDelay(stepsize, MaxDelay):
    # use global vars
    global gpuOBE_buffersize
    global gpuOBE_gridsize

    # depending on stepsize, choose buffersize large enough to fit the whole curve in at once
    # slicing the curve and calculating multiple parts consecutively on gpu requires a large
    # simOverhead factor; otherwise curve will not fit together after concatenate

    # number of points needed to fit MaxDelay into buffer
    numPoints = MaxDelay / stepsize

    # determine buffer size
    # if its smaller, you can do the calculation on the cpu...
    gpuOBE_buffersize = 64
    if (numPoints > 64):
        gpuOBE_buffersize = 128
    if (numPoints > 128):
        gpuOBE_buffersize = 256
    if (numPoints > 256):
        gpuOBE_buffersize = 512
    if (numPoints > 512):
        gpuOBE_buffersize = 1024
    if (numPoints > 1024):
        gpuOBE_buffersize = 2048
    if (numPoints > 2048):
        gpuOBE_buffersize = 4096
    if (numPoints > 4096):
        gpuOBE_buffersize = 8192
    if (numPoints > 8192):
        gpuOBE_buffersize = 16384
    if (numPoints > 16384):
        gpuOBE_buffersize = 32768
    if (numPoints > 32768):
        gpuOBE_buffersize = 65536
    if (numPoints > 65536):
        gpuOBE_buffersize = 131072
    if (numPoints > 131072):
        gpuOBE_buffersize = 131072 * 2
    if (numPoints > 131072 * 2):
        gpuOBE_buffersize = 131072 * 4
    if (numPoints > 131072 * 4):
        gpuOBE_buffersize = 131072 * 8
    if (numPoints > 131072 * 8):
        gpuOBE_buffersize = 131072 * 16
    if (numPoints > 131072 * 16):
        print('Fuck you')
        return

    # if buffer size > blocksize, we need a grid; otherwise parts of the curve are missing
    if (gpuOBE_buffersize > gpuOBE_blocksize):
        gpuOBE_gridsize = int(gpuOBE_buffersize / gpuOBE_blocksize)
    else:
        gpuOBE_gridsize = 1

    # print(gpuOBE_buffersize,gpuOBE_gridsize)

    # create delay list
    Delays = []
    # create pm values
    Delays = np.arange(0.0, round(gpuOBE_buffersize * stepsize, 2), stepsize)

    return np.asarray(Delays)


def coreTauACCoPol(x, tau, L, FWHM):
    global gpuOBE_buffersize
    IAC = gpuOBE_ACBlauCoPolTest(x, tau, L, FWHM, gpuOBE_buffersize)
    return IAC


def TauACCoPol(ExpDelays, L, FWHM, tau, Amp, Offset, Center, normparameter=False, Phase=False):
    global gpuOBE_stepsize
    maxDelay = np.amax(np.array(ExpDelays))
    simDelays = CreateCoPolDelay(gpuOBE_stepsize, maxDelay)

    IAC = coreTauACCoPol(simDelays, tau, L, FWHM)

    if (normparameter == False):
        xi = simDelays + Center
        if (Phase == False):
            yi = getw0CoPol(IAC, gpuOBE_stepsize, normparameter=False)
        elif (Phase == True):
            yi = IAC
        x = ExpDelays
        s = InterpolatedUnivariateSpline(xi, yi, k=1)
        intpAC = s(x)

    elif (normparameter == True):
        IAC = norm(IAC)
        xi = simDelays + Center
        if (Phase == False):
            yi = np.array(getw0CoPol(IAC, gpuOBE_stepsize, normparameter=True))
        elif (Phase == True):
            yi = IAC
        x = ExpDelays
        s = InterpolatedUnivariateSpline(xi, yi, k=1)
        intpAC = s(x)
    return Amp * intpAC + Offset


def fitTauACblauCoPol(ExpDelays, tau, Amp, Offset, Center):
    # print '%.2f'%tau,
    global gpuOBE_laserBlau
    global gpuOBE_LaserBlauFWHM
    global gpuOBE_normparameter
    global gpuOBE_Phaseresolution
    return TauACCoPol(ExpDelays, gpuOBE_laserBlau, gpuOBE_LaserBlauFWHM, tau, Amp, Offset, Center,
                      normparameter=gpuOBE_normparameter, Phase=gpuOBE_Phaseresolution)


def gpuOBE_ACBlauCoPolTest(Delaylist, tau, laserWavelength, laserFWHM, buffersize):
    f = c * 1e9 / laserWavelength
    w = 2 * np.pi * f * 1e-15

    skal = 2 * np.log(np.sqrt(2) + 1)
    # sech²(t/dt): tau_pulse = dt * 2*ln(sqrt(2)+1) = 1.763;    Dtau_IntensityAC / Dtau_pulse = 1.5427 - removed
    FWHM = laserFWHM / skal / 1.5427  # sech
    # Time constants
    T1 = 1.0
    T2 = 2. * tau
    T3 = np.inf

    G1 = 1. / T1
    G2 = 1. / T2
    G3 = 1. / T3

    IAC = np.zeros(gpuOBE_buffersize, dtype=np.float64)
    t_min = gpuOBE_simOverhead * max(abs(Delaylist))

    simOBErb = mod.get_function("simOBEcudaCoPolTest")

    grid_x = gpuOBE_gridsize
    grid_y = 1

    block_x = gpuOBE_blocksize
    block_y = 1
    block_z = 1
    # assert buffersize <= 1024, "Maximum number of threads per block exceeded"

    delays = np.array(Delaylist, dtype=np.float64)

    simOBErb(drv.Out(IAC), drv.In(delays), np.float64(w), np.float64(FWHM),
             np.float64(G1 + G2), np.float64(G1 + G3), np.float64(G2 + G3), np.float64(t_min),
             grid=(grid_x, grid_y), block=(block_x, block_y, block_z))

    return IAC


minimal_example_test = False
if __name__ == '__main__':
    minimal_example_test = True

# Example to test functionality
if minimal_example_test:
    if verbose:
        print("Testing minimal example...")
    gpuOBE_stepsize = 1.0
    Delays = CreateCoPolDelay(gpuOBE_stepsize, 200.0)
    starttime = timer()
    # Delays = np.arange(-200,200,0.1)
    # print ('length',len(Delays))

    L = 400.0
    FWHM = 34.7
    tau = 5.0
    Amp = 1.0
    Offset = 0.0
    Center = 0.0
    IAC = TauACCoPol(Delays, L, FWHM, tau, Amp, Offset, Center, normparameter=False, Phase=False)
    # LaserAC(ExpDelays,L,FWHM,Amp,Offset,Center):
    # IAC2 = LaserAC(Delays, L, FWHM, Amp, Offset, Center)
    endtime = timer()
    print("... done in {0:.3f} seconds".format(endtime - starttime))
    # figure()
    # plot((-300,300),(5.0/3.0,5.0/3.0))
    # plot((-300,300),(2.0,2.0))
    # plot(Delays,normAC(IAC))

evaluation_test = False
if evaluation_test:
    # Script for evaluation on local PC
    wd = "/home/hartelt/repos/evaluation/2018/08 August/BFoerster crosspol"

    HfO2folder = os.path.join(wd, "20180829 BFoerster Au HfO2")
    HfO2datah5 = os.path.join(HfO2folder, "04 - TRER - sum Runs 5-8 - xy_integrated.hdf5")
    HfO2FWHMh5 = os.path.join(HfO2folder, "HfO2 sp AC Fit Lorentz.hdf5")
    HfO2outfolder = os.path.join(HfO2folder, "cudaresults")
    if not os.path.exists(HfO2outfolder):
        os.makedirs(HfO2outfolder)

    HfO2data = ds.DataSet.from_h5(HfO2datah5)
    HfO2FWHM = ds.DataSet.from_h5(HfO2FWHMh5)

    print(HfO2data)
    print(HfO2FWHM.get_datafield(0).min())
    HfO2_AC_FWHM = HfO2FWHM.get_datafield(0).min().magnitude

    print("Start: ", datetime.datetime.now().isoformat())

    fitparams = []

    for i, energy in enumerate(HfO2data.get_axis('energy').data):
        print("HfO2 ", i, energy)
        ExpDelays = HfO2data.get_axis('delay').data.magnitude
        ExpData = HfO2data.get_datafield(0).data[:, i].magnitude
        LorentzFWHM = HfO2FWHM.get_datafield(0).data[i]
        # ExpData = ExpData / np.max(ExpData)

        gpuOBE_stepsize = 1.0
        gpuOBE_laserBlau = 400.
        gpuOBE_LaserBlauFWHM = HfO2_AC_FWHM - 4.

        global gpuOBE_normparameter
        gpuOBE_normparameter = True
        global gpuOBE_Phaseresolution
        gpuOBE_Phaseresolution = False

        guess_lifetime = max(1., LorentzFWHM.magnitude - gpuOBE_LaserBlauFWHM)
        guess_lifetime = min(guess_lifetime, 200.)
        guess_Offset = np.min(ExpData)
        guess_Amp = np.max(ExpData) - guess_Offset
        p0 = (guess_lifetime, guess_Amp, guess_Offset, -4.0)
        # fitTauACblau(ExpDelays,FWHM,Amp,Offset,Center)

        try:
            # popt, pcov = curve_fit(fitTauACblau, ExpDelays, ExpData, p0,
            # 					   bounds=([0, 0., 0., -20.], [200., np.inf, np.inf, 20.]))
            popt, pcon = curve_fit(fitTauACblauCoPol, ExpDelays, ExpData, p0,
                                   bounds=([0.0, 0.0, 0.0, -20.], [200., np.inf, np.inf, 20.]))
        except RuntimeError as e:  # Fit failed
            popt = np.full((4,), np.nan)
            pcon = np.full((4, 4), np.nan)
            print("Fit of HfO2 sp AC for {0:.2f} eV failed.".format(energy))

        fitparams.append(popt)

    print("HFO2 Finished: ", datetime.datetime.now().isoformat())

    HfO2lifeTimes = ds.DataArray(np.array([popt[0] for popt in fitparams]), unit='fs',
                                 label="lifetimes", plotlabel="Intermediate State Lifetime / \\si{\\femto\\second}")
    HfO2amplitude = ds.DataArray(np.array([popt[1] for popt in fitparams]), unit='count',
                                 label="amplitude", plotlabel="AC Amplitude / arb. unit.")
    HfO2offset = ds.DataArray(np.array([popt[2] for popt in fitparams]), unit='count',
                              label="offset", plotlabel="AC Background / arb. unit.")
    HfO2center = ds.DataArray(np.array([popt[3] for popt in fitparams]), unit='fs',
                              label="center", plotlabel="AC Center / \\si{\\femto\\second}")
    HfO2result = ds.DataSet('HfO2 lifetime data', [HfO2lifeTimes, HfO2amplitude, HfO2offset, HfO2center],
                            [HfO2data.get_axis('energy')])
    HfO2result.saveh5(os.path.join(HfO2outfolder, 'HfO2 sp Lifetimes ownpulselength-4.hdf5'))

    print("HfO2 data stored: ", datetime.datetime.now().isoformat())
