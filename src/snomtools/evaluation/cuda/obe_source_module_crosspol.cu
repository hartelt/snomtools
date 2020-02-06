#include <stdio.h>
#define CUDART_PI_F 3.141592654f

__device__ void stepOBECrossPol (double* dxdy, const double* rho, const double laser1, const double w1, const double laser2, const double w2, const double g12, const double g13, const double g23, const double g22, const double alpha, const double beta)
{


    const double pulse1=laser1*cos(alpha)-laser2*sin(alpha);
    const double pulse2=(laser1*cos(beta)-laser2*sin(beta));

    //12R
    dxdy[0] = -pulse2*rho[3] - w1*rho[1] - g12*rho[0];
    //12I
    dxdy[1] = pulse1*(-rho[7] + rho[6]) + pulse2*(rho[2]) + w1*rho[0] - g12*rho[1];
    //13R
    dxdy[2] = pulse1*(rho[5]) - pulse2*(rho[1]) - (w1+w2)*rho[3] - g13*rho[2];
    //13I
    dxdy[3] = pulse1*(-rho[4]) + pulse2*(rho[0]) + (w1+w2)*rho[2] -g13*rho[3];
    //23R
    dxdy[4] = pulse1*rho[3] - (w2)*rho[5] - g23*rho[4];
    //23I
    dxdy[5] = pulse2*(-rho[8] + rho[7]) - pulse1*(rho[2]) + (w2)*rho[4] - g23*rho[5];

    //dxdy[6] = -2*pulse1*rho[1] - g11*rho[6];
    //11R
    dxdy[6] = -2*pulse1*rho[1] + g22*rho[7];
    //22R
    dxdy[7] = 2*pulse1*(rho[1]) - 2*pulse2*(rho[5]) - g22*rho[7];
    //33R
    dxdy[8] = 2*pulse2*rho[5];
}


__global__ void simOBEcudaCrossPoltest (double* AC, const double* delays, const double* alphas, const double* betas, const double w, const double FWHM, const double g12, const double g13, const double g23, const double t_min)
{

    //int blockId = blockIdx.x + blockIdx.y * gridDim.x;
     //int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;

    //int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z);
    //threadId += (threadIdx.z * (blockDim.x * blockDim.y));
    //threadId += (threadIdx.y * blockDim.x);
    //threadId += threadIdx.x;


    double delay = delays[threadId];
    double alpha = alphas[threadId]*CUDART_PI_F/180;
    double beta = betas[threadId]*CUDART_PI_F/180;

    int n = round(2*t_min/0.1);
    double delta_t = 2*t_min/n;
    double laser1 = 0, laser1a = 0, laser1b = 0;
    double laser2 = 0, laser2a = 0, laser2b = 0;
    double time = 0;

    double rho[9] = {0,0,0,0,0,0, 1,0,0};
    double k1[9] = {0,0,0,0,0,0, 0,0,0};
    double k2[9] = {0,0,0,0,0,0, 0,0,0};
    double k3[9] = {0,0,0,0,0,0, 0,0,0};
    double k4[9] = {0,0,0,0,0,0, 0,0,0};
    double tmp[9] = {0,0,0,0,0,0, 0,0,0};


    double h = delta_t;
    double hh = delta_t/2.0;


    for(int i=0; i<n; i++)
    {
        time = i*delta_t;


        laser1 = 0.5 * 0.00153 / CUDART_PI_F * 2 *  ( cos(w*(time-t_min)) / cosh(-(time-t_min)/(FWHM)) );
        laser1a = 0.5 * 0.00153 / CUDART_PI_F * 2 * ( cos(w*(time+hh-t_min)) / cosh(-(time+hh-t_min)/(FWHM)) );
        laser1b = 0.5 * 0.00153 / CUDART_PI_F * 2 * ( cos(w*(time+h-t_min)) / cosh(-(time+h-t_min)/(FWHM)) );

        laser2 = 0.5 * 0.00153 / CUDART_PI_F * 2 * ( cos(w*(time-t_min+delay)) / cosh(-(time-t_min+delay)/(FWHM)) );
        laser2a = 0.5 * 0.00153 / CUDART_PI_F * 2 * ( cos(w*(time+hh-t_min+delay)) / cosh(-(time+hh-t_min+delay)/(FWHM)) );
        laser2b = 0.5 * 0.00153 / CUDART_PI_F * 2 * ( cos(w*(time+h-t_min+delay)) / cosh(-(time+h-t_min+delay)/(FWHM)) );

        stepOBECrossPol(k1, rho, laser1, w, laser2, w, g12, g13, g23, g12+g23-g13, alpha, beta);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k1[i];
        stepOBECrossPol(k2, tmp, laser1a, w, laser2a, w, g12, g13, g23, g12+g23-g13, alpha, beta);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + hh * k2[i];
        stepOBECrossPol(k3, tmp, laser1a, w, laser2a, w, g12, g13, g23, g12+g23-g13, alpha, beta);
        for(int i = 0; i<9; i++) tmp[i] = rho[i] + h * k3[i];
        stepOBECrossPol(k4, tmp, laser1b, w, laser2b, w, g12, g13, g23, g12+g23-g13, alpha, beta);

        for(int j=0; j<9; j++)
        {
            rho[j] = rho[j] + h/6 * (k1[j] + k4[j] + 2*(k2[j] + k3[j]));
        }



    }
      AC[threadId] = rho[8];
}
