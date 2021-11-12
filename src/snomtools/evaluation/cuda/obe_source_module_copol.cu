#include <stdio.h>
#define CUDART_PI_F 3.141592654f

__device__ void stepOBE (double* dxdy, const double* rho, const double laser, const double w, const double w2, const double g12, const double g13, const double g23, const double g22)
// This is some kind of differential step of the time evolution of the density matrix, as govered by the OBEs.
{
    //12R
    dxdy[0] = -laser*rho[3] - w*rho[1] - g12*rho[0];
    //12I
    dxdy[1] = laser*(-rho[7] + rho[6] + rho[2]) + w*rho[0] - g12*rho[1];
    //13R
    dxdy[2] = laser*(rho[5] - rho[1]) - w2*rho[3] - g13*rho[2];
    //13I
    dxdy[3] = laser*(-rho[4] + rho[0]) + w2*rho[2] -g13*rho[3];
    //23R
    dxdy[4] = laser*rho[3] - w*rho[5] - g23*rho[4];
    //23I
    dxdy[5] = laser*(-rho[8] + rho[7] - rho[2]) + w*rho[4] - g23*rho[5];

    //dxdy[6] = -2*laser*rho[1] - g11*rho[6];
    //11R
    dxdy[6] = -2*laser*rho[1] + g22*rho[7];
    //22R
    dxdy[7] = 2*laser*(rho[1] - rho[5]) - g22*rho[7];
    //33R
    dxdy[8] = 2*laser*rho[5];
}

__global__ void simOBEcudaCoPolTest (double* AC, const double* Delaylist, const double w, const double FWHM, const double G1, const double G2, const double G3, const double t_min)
// RUNGE-KUTTA Verfahren happens here to solve OBE for full density matrix.
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // Unique index

    // This is increadibly bad because it does not correspond to the correct physical definitions, but it encodes the lifetime assumptions:
    // G1 corresponds to T2* for the polarizations with 1, rho12 and rho13 (usually something short like 1fs)
    // G2 corresponds to T1 of the intermediate state, the fit parameter.
    // G3 corresponds to T1 of the final state, usually infinity.
    // T2* of rho23 is implicitly set as infinity.
    // T1 of rho11 is implicitly set as infinity.
    //double g11 = 2*G1;
    //double g12 = G1;
    double g12 = G1 + G2;
    double g13 = G1 + G3;
    double g23 = G2 + G3;
    double g22 = 2*G2;

    const double w2 = 2*w;

    // Number of time steps to simulate, -tmin to +tmin divided by the time step of 0.1fs
    int n = round(2*t_min/0.1);
    // Simulation time step, basically the 0.1fs from above +- float precision...
    double delta_t = 2*t_min/n;

    double delay = Delaylist[idx];
    double laser = 0;
    double time = 0;

    // rho = {12R,12I,13R,13I,23R,23I,11R,22R,33R}
    double rho[9] = {0,0,0,0,0,0, 1,0,0};

    double k1[9] = {0,0,0,0,0,0, 0,0,0};
    double k2[9] = {0,0,0,0,0,0, 0,0,0};
    double k3[9] = {0,0,0,0,0,0, 0,0,0};
    double k4[9] = {0,0,0,0,0,0, 0,0,0};
    double tmp[9] = {0,0,0,0,0,0, 0,0,0};

    double laser1 = 0, laser2 = 0;

    double h = delta_t;
    double hh = delta_t/2;


    for(int i=0; i<n; i++)
    {

        time = i*delta_t;
        // No idea what the prefactors mean, but the cosine brackets are the electric field of the two pulses:
        laser = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time-t_min)) / cosh(-(time-t_min)/(FWHM)) + cos(w*(time-t_min+delay)) / cosh(-(time-t_min+delay)/(FWHM)) );
        // ...and half a time step later:
        laser1 = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time+hh-t_min)) / cosh(-(time+hh-t_min)/(FWHM)) + cos(w*(time+hh-t_min+delay)) / cosh(-(time+hh-t_min+delay)/(FWHM)) );
        // ...and a full time step later:
        laser2 = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time+h-t_min)) / cosh(-(time+h-t_min)/(FWHM)) + cos(w*(time+h-t_min+delay)) / cosh(-(time+h-t_min+delay)/(FWHM)) );

        // These are the incremental terms for the RK4 Method (Runge-Kutta) for the differential time evolution.
        // see: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        // or in german: https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
        stepOBE(k1, rho, laser, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k1[i];
        stepOBE(k2, tmp, laser1, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k2[i];
        stepOBE(k3, tmp, laser1, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + h * k3[i];
        stepOBE(k4, tmp, laser2, w, w2, g12, g13, g23, g22);

        // Add the incrementals to the density matrix:
        for(int j=0; j<9; j++)
        {
            rho[j] = rho[j] + h/6 * (k1[j] + k4[j] + 2*(k2[j] + k3[j]));
        }
    }

    AC[idx] = rho[8];
}