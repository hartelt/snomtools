#include <stdio.h>
#define CUDART_PI_F 3.141592654f

__device__ void stepOBE (double* dxdy, const double* rho, const double laser, const double w, const double w2, const double g12, const double g13, const double g23, const double g22)
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

__global__ void simOBEcuda (double* AC, const double* Delaylist, const double w, const double FWHM, const double G1, const double G2, const double G3, const double t_min)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // Unique index

    //double g11 = 2*G1;
    //double g12 = G1;
    double g12 = G1 + G2;
    double g13 = G1 + G3;
    double g23 = G2 + G3;
    double g22 = 2*G2;

    const double w2 = 2*w;

    int n = round(2*t_min/0.1);

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
        laser = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time-t_min)) / cosh(-(time-t_min)/(FWHM)) + cos(w*(time-t_min+delay)) / cosh(-(time-t_min+delay)/(FWHM)) );
        laser1 = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time+hh-t_min)) / cosh(-(time+hh-t_min)/(FWHM)) + cos(w*(time+hh-t_min+delay)) / cosh(-(time+hh-t_min+delay)/(FWHM)) );
        laser2 = 0.5 * 0.0012 / CUDART_PI_F * 2 * ( cos(w*(time+h-t_min)) / cosh(-(time+h-t_min)/(FWHM)) + cos(w*(time+h-t_min+delay)) / cosh(-(time+h-t_min+delay)/(FWHM)) );

        stepOBE(k1, rho, laser, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k1[i];
        stepOBE(k2, tmp, laser1, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k2[i];
        stepOBE(k3, tmp, laser1, w, w2, g12, g13, g23, g22);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + h * k3[i];
        stepOBE(k4, tmp, laser2, w, w2, g12, g13, g23, g22);

        for(int j=0; j<9; j++)
        {
            rho[j] = rho[j] + h/6 * (k1[j] + k4[j] + 2*(k2[j] + k3[j]));
        }
    }

    AC[idx] = rho[8];
}