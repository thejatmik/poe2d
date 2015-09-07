
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <fstream>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void kersigmaxyz(int *DDIMX, int *DDIMY, int *DDIMZ, float *memory_dvx_dx, float *memory_dvy_dy, float *memory_dvz_dz, float *a_x_half, float *a_y, float *a_z, float *b_x_half, float *b_y, float *b_z, float *K_x_half, float *K_y, float *K_z, float *DELTAT_lambdaplus2mu, float *DELTAT_lambda, float *sigmaxx, float *sigmayy, float *sigmazz, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vx, float *vy, float *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int right = offset + 1;
	int ybottom = offset - DDIMX[0];
	int zbottom = offset - DDIMX[0] * DDIMY[0];


	if ((index_z >= 2) && (index_z <= DDIMZ[0])) {
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float value_dvx_dx = (vx[right] - vx[offset])*ONE_OVER_DELTAX[0];
				float value_dvy_dy = (vy[offset] - vy[ybottom])*ONE_OVER_DELTAY[0];
				float value_dvz_dz = (vz[offset] - vz[zbottom])*ONE_OVER_DELTAZ[0];

				memory_dvx_dx[offset] = b_x_half[index_x] * memory_dvx_dx[offset] + a_x_half[index_x] * value_dvx_dx;
				memory_dvy_dy[offset] = b_y[index_y] * memory_dvy_dy[offset] + a_y[index_y] * value_dvy_dy;
				memory_dvz_dz[offset] = b_z[index_z] * memory_dvz_dz[offset] + a_z[index_z] * value_dvz_dz;

				value_dvx_dx = value_dvx_dx / K_x_half[index_x] + memory_dvx_dx[offset];
				value_dvy_dy = value_dvy_dy / K_y[index_y] + memory_dvy_dy[offset];
				value_dvz_dz = value_dvz_dz / K_z[index_z] + memory_dvz_dz[offset];

				sigmaxx[offset] = DELTAT_lambdaplus2mu[0] * value_dvx_dx + DELTAT_lambda[0] * (value_dvy_dy + value_dvz_dz) + sigmaxx[offset];
				sigmayy[offset] = DELTAT_lambda[0] * (value_dvx_dx + value_dvz_dz) + DELTAT_lambdaplus2mu[0] * value_dvy_dy + sigmayy[offset];
				sigmazz[offset] = DELTAT_lambda[0] * (value_dvx_dx + value_dvy_dy) + DELTAT_lambdaplus2mu[0] * value_dvz_dz + sigmazz[offset];
			}
		}
	}

}

__global__ void kersigmaxy(int *DDIMX, int *DDIMY, int *DDIMZ, float *memory_dvy_dx, float *memory_dvx_dy, float *a_x, float *a_y_half, float *b_x, float *b_y_half, float *K_x, float *K_y_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *vx, float *vy, float *DELTAT_mu, float *sigmaxy) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int left = offset - 1;
	int ytop = offset + DDIMX[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0])) {
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float value_dvy_dx = (vy[offset] - vy[left])*ONE_OVER_DELTAX[0];
				float value_dvx_dy = (vx[ytop] - vx[offset])*ONE_OVER_DELTAY[0];

				memory_dvy_dx[offset] = b_x[index_x] * memory_dvy_dx[offset] + a_x[index_x] * value_dvy_dx;
				memory_dvx_dy[offset] = b_y_half[index_y] * memory_dvx_dy[offset] + a_y_half[index_y] * value_dvx_dy;

				value_dvy_dx = value_dvy_dx / K_x[index_x] + memory_dvy_dx[offset];
				value_dvx_dy = value_dvx_dy / K_y_half[index_y] + memory_dvx_dy[offset];

				sigmaxy[offset] = DELTAT_mu[0] * (value_dvy_dx + value_dvx_dy) + sigmaxy[offset];
			}
		}
	}

}

__global__ void kersigmaxzyz(int *DDIMX, int *DDIMY, int *DDIMZ, float *memory_dvz_dx, float *memory_dvx_dz, float *memory_dvz_dy, float *memory_dvy_dz, float *a_x, float *a_z, float *a_y_half, float *a_z_half, float *b_x, float *b_y_half, float *b_z_half, float *K_x, float *K_y_half, float *K_z_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vx, float *vy, float *vz, float *DELTAT_mu, float *sigmaxz, float *sigmayz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int left = offset - 1;
	int ztop = offset + DDIMX[0] * DDIMY[0];
	int ytop = offset + DDIMX[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0])) {
		//sigmaxz
		if ((index_y >= 1) && (index_y <= DDIMY[0])) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float value_dvz_dx = (vz[offset] - vz[left]) * ONE_OVER_DELTAX[0];
				float value_dvx_dz = (vx[ztop] - vx[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dx[offset] = b_x[index_x] * memory_dvz_dx[offset] + a_x[index_x] * value_dvz_dx;
				memory_dvx_dz[offset] = b_z_half[index_z] * memory_dvx_dz[offset] + a_z_half[index_z] * value_dvx_dz;

				value_dvz_dx = value_dvz_dx / K_x[index_x] + memory_dvz_dx[offset];
				value_dvx_dz = value_dvx_dz / K_z_half[index_z] + memory_dvx_dz[offset];

				sigmaxz[offset] = DELTAT_mu[0] * (value_dvz_dx + value_dvx_dz) + sigmaxz[offset];
			}
		}

		//sigmayz
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 1) && (index_z <= DDIMX[0])) {
				float value_dvz_dy = (vz[ytop] - vz[offset]) * ONE_OVER_DELTAY[0];
				float value_dvy_dz = (vy[ztop] - vy[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dy[offset] = b_y_half[index_y] * memory_dvz_dy[offset] + a_y_half[index_y] * value_dvz_dy;
				memory_dvy_dz[offset] = b_z_half[index_z] * memory_dvy_dz[offset] + a_z_half[index_z] * value_dvy_dz;

				value_dvz_dy = value_dvz_dy / K_y_half[index_y] + memory_dvz_dy[offset];
				value_dvy_dz = value_dvy_dz / K_z_half[index_z] + memory_dvy_dz[offset];

				sigmayz[offset] = DELTAT_mu[0] * (value_dvz_dy + value_dvy_dz) + sigmayz[offset];
			}
		}
	}

}

__global__ void kervxvy(int *DDIMX, int *DDIMY, int *DDIMZ,float *sigmaxx, float *sigmaxy, float *sigmaxz, float *sigmayy, float *sigmayz, float *memory_dsigmaxx_dx, float *memory_dsigmaxy_dy, float *memory_dsigmaxz_dz, float *memory_dsigmaxy_dx, float *memory_dsigmayy_dy, float *memory_dsigmayz_dz, float *a_x, float *a_y, float *a_z, float *a_x_half, float *a_y_half, float *b_x, float *b_y, float *b_z, float *b_x_half, float *b_y_half, float *K_x, float *K_y, float *K_z, float *K_x_half, float *K_y_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *DELTAT_over_rho, float *vx, float *vy) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int left = offset - 1;
	int ybottom = offset - DDIMX[0];
	int zbottom = offset - DDIMX[0] * DDIMY[0];
	int right = offset + 1;
	int ytop = offset + DDIMX[0];


	if ((index_z >= 2) && (index_z <= DDIMZ[0])) {
		//vx
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float value_dsigmaxx_dx = (sigmaxx[offset] - sigmaxx[left]) * ONE_OVER_DELTAX[0];
				float value_dsigmaxy_dy = (sigmaxy[offset] - sigmaxy[ybottom]) * ONE_OVER_DELTAY[0];
				float value_dsigmaxz_dz = (sigmaxz[offset] - sigmaxz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxx_dx[offset] = b_x[index_x] * memory_dsigmaxx_dx[offset] + a_x[index_x] * value_dsigmaxx_dx;
				memory_dsigmaxy_dy[offset] = b_y[index_y] * memory_dsigmaxy_dy[offset] + a_y[index_y] * value_dsigmaxy_dy;
				memory_dsigmaxz_dz[offset] = b_z[index_z] * memory_dsigmaxz_dz[offset] + a_z[index_z] * value_dsigmaxz_dz;

				value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[index_x] + memory_dsigmaxx_dx[offset];
				value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[index_y] + memory_dsigmaxy_dy[offset];
				value_dsigmaxz_dz = value_dsigmaxz_dz / K_z[index_z] + memory_dsigmaxz_dz[offset];

				vx[offset] = DELTAT_over_rho[0] * (value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + vx[offset];
			}
		}

		//vy
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float value_dsigmaxy_dx = (sigmaxy[right] - sigmaxy[offset]) * ONE_OVER_DELTAX[0];
				float value_dsigmayy_dy = (sigmayy[ytop] - sigmayy[offset]) * ONE_OVER_DELTAY[0];
				float value_dsigmayz_dz = (sigmayz[offset] - sigmayz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxy_dx[offset] = b_x_half[index_x] * memory_dsigmaxy_dx[offset] + a_x_half[index_x] * value_dsigmaxy_dx;
				memory_dsigmayy_dy[offset] = b_y_half[index_y] * memory_dsigmayy_dy[offset] + a_y_half[index_y] * value_dsigmayy_dy;
				memory_dsigmayz_dz[offset] = b_z[index_z] * memory_dsigmayz_dz[offset] + a_z[index_z] * value_dsigmayz_dz;

				value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[index_x] + memory_dsigmaxy_dx[offset];
				value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[index_y] + memory_dsigmayy_dy[offset];
				value_dsigmayz_dz = value_dsigmayz_dz / K_z[index_z] + memory_dsigmayz_dz[offset];

				vy[offset] = DELTAT_over_rho[0] * (value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + vy[offset];
			}
		}
	}

}

__global__ void kervz(int *DDIMX, int *DDIMY, int *DDIMZ, float *sigmaxz, float *sigmayz, float *sigmazz, float *memory_dsigmaxz_dx, float *memory_dsigmayz_dy, float *memory_dsigmazz_dz, float *b_x_half, float *b_y, float *b_z_half, float *a_x_half, float *a_y, float *a_z_half, float *K_x_half, float *K_y, float *K_z_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vz, float *DELTAT_over_rho) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;
	int right = offset + 1;
	int ybottom = offset - DDIMX[0];
	int ztop = offset + DDIMX[0] * DDIMY[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0] - 1)) {
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float value_dsigmaxz_dx = (sigmaxz[right] - sigmaxz[offset]) * ONE_OVER_DELTAX[0];
				float value_dsigmayz_dy = (sigmayz[offset] - sigmayz[ybottom]) * ONE_OVER_DELTAY[0];
				float value_dsigmazz_dz = (sigmazz[ztop] - sigmazz[offset]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxz_dx[offset] = b_x_half[index_x] * memory_dsigmaxz_dx[offset] + a_x_half[index_x] * value_dsigmaxz_dx;
				memory_dsigmayz_dy[offset] = b_y[index_y] * memory_dsigmayz_dy[offset] + a_y[index_y] * value_dsigmayz_dy;
				memory_dsigmazz_dz[offset] = b_z_half[index_z] * memory_dsigmazz_dz[offset] + a_z_half[index_z] * value_dsigmazz_dz;

				value_dsigmaxz_dx = value_dsigmaxz_dx / K_x_half[index_x] + memory_dsigmaxz_dx[offset];
				value_dsigmayz_dy = value_dsigmayz_dy / K_y[index_y] + memory_dsigmayz_dy[offset];
				value_dsigmazz_dz = value_dsigmazz_dz / K_z_half[index_z] + memory_dsigmazz_dz[offset];

				vz[offset] = DELTAT_over_rho[0] * (value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + vz[offset];
			}
		}
	}

}

__global__ void keraddSource(int *iit, int *ISOURCE, int *JSOURCE, int *KSOURCE, float *ANGLE_FORCE, float *DEGREES_TO_RADIANS, float *DELTAT, float *factor, float *t0, float *ff0, float *DPI, float *vx, float *vy, float *rho) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;

	float a = DPI[0] * DPI[0] * ff0[0] * ff0[0];
	float t = float(iit[0] - 1)*DELTAT[0];

	//Gaussian
	//float source_term = factor * expf(-a*powf((t - t0), 2));

	//first derivative of a Gaussian
	float source_term = -factor[0] * 2.0*a*(t - t0[0])*expf(-a*powf((t - t0[0]), 2));

	//Ricker source time function(second derivative of a Gaussian)
	//float source_term = factor*(1.0 - 2.0*a*powf((t - t0), 2))*expf(-a*powf(t - t0, 2));

	float force_x = sinf(ANGLE_FORCE[0] * DEGREES_TO_RADIANS[0])*source_term;
	float force_y = cosf(ANGLE_FORCE[0] * DEGREES_TO_RADIANS[0])*source_term;

	if (index_z == KSOURCE[0]) {
		if (index_y == JSOURCE[0]) {
			if (index_x == ISOURCE[0]) {
				vx[offset] = vx[offset] + force_x*DELTAT[0] / rho[0];
				vy[offset] = vy[offset] + force_y*DELTAT[0] / rho[0];
			}
		}
	}
}

__global__ void kerDirichletBoundary(int *DDIMX, int *DDIMY, int *DDIMZ, float *vx, float *vy, float *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = blkId * aaaa + bbbb + cccc;

	if ((index_z == 1) || (index_z == DDIMZ[0])) {
		if ((index_y == 1) || (index_y == DDIMY[0])) {
			if ((index_x == 1) || (index_z == DDIMX[0])) {
				vy[offset] = 0.0;
				vx[offset] = 0.0;
				vz[offset] = 0.0;
			}
		}
	}
}

int main()
{
	int NIMX, NIMY, NIMZ;
	NIMX = 200;
	NIMY = 200;
	NIMZ = 200;

	int NSTEP = 2500;

	int DIMX, DIMY, DIMZ;
	DIMX = NIMX + 1; DIMY = NIMY + 1; DIMZ = NIMZ + 1;

	int DELTAX, DELTAY, DELTAZ;
	DELTAX = 1.5; DELTAY = DELTAX; DELTAZ = DELTAX;
	float ONE_OVER_DELTAXX, ONE_OVER_DELTAYY, ONE_OVER_DELTAZZ;
	ONE_OVER_DELTAXX = 1 / float(DELTAX);
	ONE_OVER_DELTAZZ = ONE_OVER_DELTAXX; ONE_OVER_DELTAYY = ONE_OVER_DELTAXX;

	float *ONE_OVER_DELTAX, *ONE_OVER_DELTAY, *ONE_OVER_DELTAZ;
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAX, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAY, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAZ, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAX, &ONE_OVER_DELTAXX, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAY, &ONE_OVER_DELTAYY, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAZ, &ONE_OVER_DELTAZZ, sizeof(float), cudaMemcpyHostToDevice));

	float tempcp, tempcs, temprho, mu, lambda, lambdaplustwomu;
	tempcp = 3300.0;
	tempcs = tempcp / 1.732;
	temprho = 3000.0;
	float *rho;
	HANDLE_ERROR(cudaMalloc((void**)&rho, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(rho, &temprho, sizeof(float), cudaMemcpyHostToDevice));
	mu = temprho*tempcs*tempcs;
	lambda = temprho*(tempcp*tempcp - 2 * tempcs*tempcs);
	lambdaplustwomu = temprho*tempcp*tempcp;

	float DELTATT = 1e-5;
	float *DELTAT;
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DELTAT, &DELTATT, sizeof(float), cudaMemcpyHostToDevice));

	float f0, tt0, factorr;
	f0 = 7;
	tt0 = 1.2 / f0;
	float *ff0, *t0;
	HANDLE_ERROR(cudaMalloc((void**)&t0, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(t0, &tt0, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&ff0, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ff0, &f0, sizeof(float), cudaMemcpyHostToDevice));
	factorr = 1e+7;
	float *factor;
	HANDLE_ERROR(cudaMalloc((void**)&factor, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(factor, &factorr, sizeof(float), cudaMemcpyHostToDevice));

	int NPOINTS_PML = 10;

	int ISOURCEE, KSOURCEE, JSOURCEE;
	ISOURCEE = (NIMX) / 2;
	JSOURCEE = (NIMY) / 2;
	KSOURCEE = (NIMZ) / 2;
	int *ISOURCE, *KSOURCE, *JSOURCE;
	HANDLE_ERROR(cudaMalloc((void**)&ISOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(ISOURCE, &ISOURCEE, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&JSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(JSOURCE, &JSOURCEE, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&KSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(KSOURCE, &KSOURCEE, sizeof(int), cudaMemcpyHostToDevice));

	float ANGLE_FORCEE = 90;
	float *ANGLE_FORCE;
	HANDLE_ERROR(cudaMalloc((void**)&ANGLE_FORCE, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ANGLE_FORCE, &ANGLE_FORCEE, sizeof(float), cudaMemcpyHostToDevice));
	int IT_OUTPUT = 200;

	float PI = 3.141592653589793238462643;
	float *DPI;
	HANDLE_ERROR(cudaMalloc((void**)&DPI, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DPI, &PI, sizeof(float), cudaMemcpyHostToDevice));
	float DEGREES_TO_RADIANSS = PI / 180;
	float *DEGREES_TO_RADIANS;
	HANDLE_ERROR(cudaMalloc((void**)&DEGREES_TO_RADIANS, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DEGREES_TO_RADIANS, &DEGREES_TO_RADIANSS, sizeof(float), cudaMemcpyHostToDevice));

	float NPOWER = 2;
	float K_MAX_PML = 1;
	float ALPHA_MAX_PML = 2 * PI*(f0 / 2);

	float *tempvx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempvx[ijk] = 0;
			}
		}
	}
	float *vx;
	HANDLE_ERROR(cudaMalloc((void**)&vx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(vx, tempvx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempvx);

	float *tempvy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempvy[ijk] = 0;
			}
		}
	}
	float *vy;
	HANDLE_ERROR(cudaMalloc((void**)&vy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(vy, tempvy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempvy);

	float *tempvz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempvz[ijk] = 0;
			}
		}
	}
	float *vz;
	HANDLE_ERROR(cudaMalloc((void**)&vz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(vz, tempvz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempvz);

	float *tempsigmaxx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmaxx[ijk] = 0;
			}
		}
	}
	float *sigmaxx;
	HANDLE_ERROR(cudaMalloc((void**)&sigmaxx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmaxx, tempsigmaxx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmaxx);

	float *tempsigmaxy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmaxy[ijk] = 0;
			}
		}
	}
	float *sigmaxy;
	HANDLE_ERROR(cudaMalloc((void**)&sigmaxy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmaxy, tempsigmaxy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmaxy);

	float *tempsigmayy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmayy[ijk] = 0;
			}
		}
	}
	float *sigmayy;
	HANDLE_ERROR(cudaMalloc((void**)&sigmayy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmayy, tempsigmayy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmayy);

	float *tempsigmazz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmazz[ijk] = 0;
			}
		}
	}
	float *sigmazz;
	HANDLE_ERROR(cudaMalloc((void**)&sigmazz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmazz, tempsigmazz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmazz);

	float *tempsigmaxz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmaxz[ijk] = 0;
			}
		}
	}
	float *sigmaxz;
	HANDLE_ERROR(cudaMalloc((void**)&sigmaxz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmaxz, tempsigmaxz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmaxz);

	float *tempsigmayz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempsigmayz[ijk] = 0;
			}
		}
	}
	float *sigmayz;
	HANDLE_ERROR(cudaMalloc((void**)&sigmayz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(sigmayz, tempsigmayz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempsigmayz);

	float *tempmemory_dvx_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvx_dx[ijk] = 0;
			}
		}
	}
	float *memory_dvx_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvx_dx, tempmemory_dvx_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvx_dx);

	float *tempmemory_dvx_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvx_dy[ijk] = 0;
			}
		}
	}
	float *memory_dvx_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvx_dy, tempmemory_dvx_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvx_dy);

	float *tempmemory_dvx_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvx_dz[ijk] = 0;
			}
		}
	}
	float *memory_dvx_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvx_dz, tempmemory_dvx_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvx_dz);

	float *tempmemory_dvy_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvy_dx[ijk] = 0;
			}
		}
	}
	float *memory_dvy_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvy_dx, tempmemory_dvy_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvy_dx);

	float *tempmemory_dvy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvy_dy[ijk] = 0;
			}
		}
	}
	float *memory_dvy_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvy_dy, tempmemory_dvy_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvy_dy);

	float *tempmemory_dvy_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvy_dz[ijk] = 0;
			}
		}
	}
	float *memory_dvy_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvy_dz, tempmemory_dvy_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvy_dz);

	float *tempmemory_dvz_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvz_dx[ijk] = 0;
			}
		}
	}
	float *memory_dvz_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvz_dx, tempmemory_dvz_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvz_dx);

	float *tempmemory_dvz_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvz_dy[ijk] = 0;
			}
		}
	}
	float *memory_dvz_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvz_dy, tempmemory_dvz_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvz_dy);

	float *tempmemory_dvz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dvz_dz[ijk] = 0;
			}
		}
	}
	float *memory_dvz_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dvz_dz, tempmemory_dvz_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dvz_dz);

	float *tempmemory_dsigmaxx_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmaxx_dx[ijk] = 0;
			}
		}
	}
	float *memory_dsigmaxx_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxx_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmaxx_dx, tempmemory_dsigmaxx_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmaxx_dx);

	float *tempmemory_dsigmayy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmayy_dy[ijk] = 0;
			}
		}
	}
	float *memory_dsigmayy_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayy_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmayy_dy, tempmemory_dsigmayy_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmayy_dy);

	float *tempmemory_dsigmazz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmazz_dz[ijk] = 0;
			}
		}
	}
	float *memory_dsigmazz_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmazz_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmazz_dz, tempmemory_dsigmazz_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmazz_dz);

	float *tempmemory_dsigmaxy_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmaxy_dx[ijk] = 0;
			}
		}
	}
	float *memory_dsigmaxy_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxy_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmaxy_dx, tempmemory_dsigmaxy_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmaxy_dx);

	float *tempmemory_dsigmaxy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmaxy_dy[ijk] = 0;
			}
		}
	}
	float *memory_dsigmaxy_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxy_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmaxy_dy, tempmemory_dsigmaxy_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmaxy_dy);

	float *tempmemory_dsigmaxz_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmaxz_dx[ijk] = 0;
			}
		}
	}
	float *memory_dsigmaxz_dx;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxz_dx, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmaxz_dx, tempmemory_dsigmaxz_dx, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmaxz_dx);

	float *tempmemory_dsigmaxz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmaxz_dz[ijk] = 0;
			}
		}
	}
	float *memory_dsigmaxz_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxz_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmaxz_dz, tempmemory_dsigmaxz_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmaxz_dz);

	float *tempmemory_dsigmayz_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmayz_dy[ijk] = 0;
			}
		}
	}
	float *memory_dsigmayz_dy;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayz_dy, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmayz_dy, tempmemory_dsigmayz_dy, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmayz_dy);

	float *tempmemory_dsigmayz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*DIMX + k*DIMX*DIMY;
				tempmemory_dsigmayz_dz[ijk] = 0;
			}
		}
	}
	float *memory_dsigmayz_dz;
	HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayz_dz, DIMX*DIMY*DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(memory_dsigmayz_dz, tempmemory_dsigmayz_dz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice));
	free(tempmemory_dsigmayz_dz);

	float *d_x, *K_x, *alpha_x, *a_x, *b_x, *d_x_half, *K_x_half, *alpha_x_half, *a_x_half, *b_x_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_x, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_x, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_x, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_x, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_x, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_x_half, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_x_half, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_x_half, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_x_half, DIMX*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_x_half, DIMX*sizeof(float)));

	float *d_y, *K_y, *alpha_y, *a_y, *b_y, *d_y_half, *K_y_half, *alpha_y_half, *a_y_half, *b_y_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_y, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_y, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_y, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_y, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_y, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_y_half, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_y_half, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_y_half, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_y_half, DIMY*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_y_half, DIMY*sizeof(float)));

	float *d_z, *K_z, *alpha_z, *a_z, *b_z, *d_z_half, *K_z_half, *alpha_z_half, *a_z_half, *b_z_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_z, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_z, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_z, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_z, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_z, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_z_half, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_z_half, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_z_half, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_z_half, DIMZ*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_z_half, DIMZ*sizeof(float)));

	float thickness_PML_x, thickness_PML_y, thickness_PML_z;
	float xoriginleft, xoriginright, yoriginbottom, yorigintop, zoriginbottom, zorigintop;
	float Rcoef, d0_x, d0_y, d0_z, xval, yval, zval, abscissa_in_PML, abscissa_normalized;
	float a, t, force_x, force_y, source_term;
	float epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_xz, epsilon_yz;
	float total_energy_kinetic, total_energy_potential;
	float *total_energy = (float*)malloc(sizeof(float)*NSTEP);

	float tDELTAT_lambda = DELTATT*lambda;
	float tDELTAT_mu = DELTATT*mu;
	float tDELTAT_lambdaplus2mu = DELTATT*lambdaplustwomu;
	float tDELTAT_over_rho = DELTATT / temprho;
	float Courant_number;

	float *DELTAT_lambda, *DELTAT_mu, *DELTAT_lambdaplus2mu, *DELTAT_over_rho;
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT_lambda, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT_mu, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT_lambdaplus2mu, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT_over_rho, sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(DELTAT_lambda, &tDELTAT_lambda, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DELTAT_mu, &tDELTAT_mu, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DELTAT_lambdaplus2mu, &tDELTAT_lambdaplus2mu, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DELTAT_over_rho, &tDELTAT_over_rho, sizeof(float), cudaMemcpyHostToDevice));

	thickness_PML_x = NPOINTS_PML * DELTAX;
	thickness_PML_y = NPOINTS_PML * DELTAY;
	thickness_PML_z = NPOINTS_PML * DELTAZ;
	Rcoef = 0.001;

	d0_x = -(NPOWER + 1) * tempcp * logf(Rcoef) / (2.0 * thickness_PML_x);
	d0_y = -(NPOWER + 1) * tempcp * logf(Rcoef) / (2.0 * thickness_PML_y);
	d0_z = -(NPOWER + 1) * tempcp * logf(Rcoef) / (2.0 * thickness_PML_z);

	//------------------PML X
	float *tempd_x = (float*)malloc(sizeof(float)*DIMX);
	float *tempd_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *tempa_x = (float*)malloc(sizeof(float)*DIMX);
	float *tempa_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *tempb_x = (float*)malloc(sizeof(float)*DIMX);
	float *tempb_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *tempK_x = (float*)malloc(sizeof(float)*DIMX);
	float *tempK_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *tempalpha_x = (float*)malloc(sizeof(float)*DIMX);
	float *tempalpha_x_half = (float*)malloc(sizeof(float)*DIMX);

	for (int i = 1; i < DIMX; i++) {
		tempd_x[i] = 0.0;
		tempd_x_half[i] = 0.0;
		tempK_x[i] = 1.0;
		tempK_x_half[i] = 1.0;
		tempalpha_x[i] = 0.0;
		tempalpha_x_half[i] = 0.0;
		tempa_x[i] = 0.0;
		tempa_x_half[i] = 0.0;
		tempb_x[i] = 0.0;
		tempb_x_half[i] = 0.0;
	}

	xoriginleft = thickness_PML_x;
	xoriginright = (NIMX - 1)*DELTAX - thickness_PML_x;
	for (int i = 1; i <= NIMX; i++) {
		xval = DELTAX*float(i - 1);
		abscissa_in_PML = xoriginleft - xval;//PML XMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval - xoriginright;//PML XMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval + DELTAX / 2.0 - xoriginright;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}
		if (tempalpha_x[i] < 0.0) { tempalpha_x[i] = 0.0; }
		if (tempalpha_x_half[i] < 0.0) { tempalpha_x_half[i] = 0.0; }
		tempb_x[i] = expf(-(tempd_x[i] / tempK_x[i] + tempalpha_x[i])*DELTATT);
		tempb_x_half[i] = expf(-(tempd_x_half[i] / tempK_x_half[i] + tempalpha_x_half[i])*DELTATT);

		if (fabs(tempd_x[i]) > 1e-6) { tempa_x[i] = tempd_x[i] * (tempb_x[i] - 1.0) / (tempK_x[i] * (tempd_x[i] + tempK_x[i] * tempalpha_x[i])); }
		if (fabs(tempd_x_half[i]) > 1e-6) { tempa_x_half[i] = tempd_x_half[i] * (tempb_x_half[i] - 1.0) / (tempK_x_half[i] * (tempd_x_half[i] + tempK_x_half[i] * tempalpha_x_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_x, tempd_x, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_x_half, tempd_x_half, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_x, tempa_x, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_x_half, tempa_x_half, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_x, tempalpha_x, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_x_half, tempalpha_x_half, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_x, tempb_x, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_x_half, tempb_x_half, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_x, tempK_x, sizeof(float)*DIMX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_x_half, tempK_x_half, sizeof(float)*DIMX, cudaMemcpyHostToDevice));

	free(tempd_x); free(tempd_x_half); free(tempa_x); free(tempa_x_half); free(tempalpha_x); free(tempalpha_x_half); free(tempb_x); free(tempb_x_half); free(tempK_x); free(tempK_x_half);

	//-----------------PML Y
	float *tempd_y = (float*)malloc(sizeof(float)*DIMY);
	float *tempd_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *tempa_y = (float*)malloc(sizeof(float)*DIMY);
	float *tempa_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *tempb_y = (float*)malloc(sizeof(float)*DIMY);
	float *tempb_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *tempK_y = (float*)malloc(sizeof(float)*DIMY);
	float *tempK_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *tempalpha_y = (float*)malloc(sizeof(float)*DIMY);
	float *tempalpha_y_half = (float*)malloc(sizeof(float)*DIMY);

	for (int i = 1; i < DIMY; i++) {
		tempd_y[i] = 0.0;
		tempd_y_half[i] = 0.0;
		tempK_y[i] = 1.0;
		tempK_y_half[i] = 1.0;
		tempalpha_y[i] = 0.0;
		tempalpha_y_half[i] = 0.0;
		tempa_y[i] = 0.0;
		tempa_y_half[i] = 0.0;
		tempb_y[i] = 0.0;
		tempb_y_half[i] = 0.0;
	}

	yoriginbottom = thickness_PML_y;
	yorigintop = (NIMY - 1)*DELTAY - thickness_PML_y;
	for (int i = 1; i <= NIMY; i++) {
		yval = DELTAY*float(i - 1);
		abscissa_in_PML = yoriginbottom - yval;//PML YMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval - yorigintop;//PML YMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval + DELTAY / 2.0 - yorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		tempb_y[i] = expf(-(tempd_y[i] / tempK_y[i] + tempalpha_y[i])*DELTATT);
		tempb_y_half[i] = expf(-(tempd_y_half[i] / tempK_y_half[i] + tempalpha_y_half[i])*DELTATT);

		if (fabs(tempd_y[i]) > 1e-6) { tempa_y[i] = tempd_y[i] * (tempb_y[i] - 1.0) / (tempK_y[i] * (tempd_y[i] + tempK_y[i] * tempalpha_y[i])); }
		if (fabs(tempd_y_half[i]) > 1e-6) { tempa_y_half[i] = tempd_y_half[i] * (tempb_y_half[i] - 1.0) / (tempK_y_half[i] * (tempd_y_half[i] + tempK_y_half[i] * tempalpha_y_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_y, tempd_y, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_y_half, tempd_y_half, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_y, tempa_y, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_y_half, tempa_y_half, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_y, tempalpha_y, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_y_half, tempalpha_y_half, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_y, tempb_y, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_y_half, tempb_y_half, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_y, tempK_y, sizeof(float)*DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_y_half, tempK_y_half, sizeof(float)*DIMY, cudaMemcpyHostToDevice));

	free(tempd_y); free(tempd_y_half); free(tempa_y); free(tempa_y_half); free(tempalpha_y); free(tempalpha_y_half); free(tempb_y); free(tempb_y_half); free(tempK_y); free(tempK_y_half);

	//-----------------PML Z
	float *tempd_z = (float*)malloc(sizeof(float)*DIMZ);
	float *tempd_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *tempa_z = (float*)malloc(sizeof(float)*DIMZ);
	float *tempa_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *tempb_z = (float*)malloc(sizeof(float)*DIMZ);
	float *tempb_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *tempK_z = (float*)malloc(sizeof(float)*DIMZ);
	float *tempK_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *tempalpha_z = (float*)malloc(sizeof(float)*DIMZ);
	float *tempalpha_z_half = (float*)malloc(sizeof(float)*DIMZ);

	for (int i = 1; i < DIMZ; i++) {
		tempd_z[i] = 0.0;
		tempd_z_half[i] = 0.0;
		tempK_z[i] = 1.0;
		tempK_z_half[i] = 1.0;
		tempalpha_z[i] = 0.0;
		tempalpha_z_half[i] = 0.0;
		tempa_z[i] = 0.0;
		tempa_z_half[i] = 0.0;
		tempb_z[i] = 0.0;
		tempb_z_half[i] = 0.0;
	}

	zoriginbottom = thickness_PML_z;
	zorigintop = (NIMZ - 1)*DELTAZ - thickness_PML_z;
	for (int i = 1; i <= NIMZ; i++) {
		zval = DELTAZ*float(i - 1);
		abscissa_in_PML = zoriginbottom - zval;//PML ZMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = zval - zorigintop;//PML ZMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_z;
			tempd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = zval + DELTAZ / 2.0 - zorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		tempb_z[i] = expf(-(tempd_z[i] / tempK_z[i] + tempalpha_z[i])*DELTATT);
		tempb_z_half[i] = expf(-(tempd_z_half[i] / tempK_z_half[i] + tempalpha_z_half[i])*DELTATT);

		if (fabs(tempd_z[i]) > 1e-6) { tempa_z[i] = tempd_z[i] * (tempb_z[i] - 1.0) / (tempK_z[i] * (tempd_z[i] + tempK_z[i] * tempalpha_z[i])); }
		if (fabs(tempd_z_half[i]) > 1e-6) { tempa_z_half[i] = tempd_z_half[i] * (tempb_z_half[i] - 1.0) / (tempK_z_half[i] * (tempd_z_half[i] + tempK_z_half[i] * tempalpha_z_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_z, tempd_z, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_z_half, tempd_z_half, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_z, tempa_z, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_z_half, tempa_z_half, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_z, tempalpha_z, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_z_half, tempalpha_z_half, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_z, tempb_y, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_z_half, tempb_z_half, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_z, tempK_z, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_z_half, tempK_z_half, sizeof(float)*DIMZ, cudaMemcpyHostToDevice));

	free(tempd_z); free(tempd_z_half); free(tempa_z); free(tempa_z_half); free(tempalpha_z); free(tempalpha_z_half); free(tempb_z); free(tempb_z_half); free(tempK_z); free(tempK_z_half);

	int *DDIMX, *DDIMY, *DDIMZ;
	HANDLE_ERROR(cudaMalloc((void**)&DDIMX, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMY, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMZ, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(DDIMX, &NIMX, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMY, &NIMY, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMZ, &NIMZ, sizeof(int), cudaMemcpyHostToDevice));

	dim3 threads;
	threads.x = 100;
	threads.y = 100;
	threads.z = 100;

	dim3 blocks;
	blocks.x = NIMX / threads.x;
	blocks.y = NIMY / threads.y;
	blocks.z = NIMZ / threads.z;

	int *iit;
	HANDLE_ERROR(cudaMalloc((void**)&iit, sizeof(int)));

	for (int it = 1; it <= NSTEP; it++) {
		kersigmaxyz << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ, memory_dvx_dx, memory_dvy_dy, memory_dvz_dz, a_x_half, a_y, a_z, b_x_half, b_y, b_z, K_x_half, K_y, K_z, DELTAT_lambdaplus2mu, DELTAT_lambda, sigmaxx, sigmayy, sigmazz, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vx, vy, vz);

		kersigmaxy << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ, memory_dvy_dx, memory_dvx_dy, a_x, a_y_half, b_x, b_y_half, K_x, K_y_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, vx, vy, DELTAT_mu, sigmaxy);

		kersigmaxzyz << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ,memory_dvz_dx, memory_dvx_dz, memory_dvz_dy, memory_dvy_dz, a_x, a_z, a_y_half, a_z_half, b_x, b_y_half, b_z_half, K_x, K_y_half, K_z_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vx, vy, vz, DELTAT_mu, sigmaxz, sigmayz);

		kervxvy << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ, sigmaxx, sigmaxy, sigmaxz, sigmayy, sigmayz, memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxz_dz, memory_dsigmaxy_dx, memory_dsigmayy_dy, memory_dsigmayz_dz, a_x, a_y, a_z, a_x_half, a_y_half, b_x, b_y, b_z, b_x_half, b_y_half, K_x, K_y, K_z, K_x_half, K_y_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, DELTAT_over_rho, vx, vy);

		kervz << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ, sigmaxz, sigmayz, sigmazz, memory_dsigmaxz_dx, memory_dsigmayz_dy, memory_dsigmazz_dz, b_x_half, b_y, b_z_half, a_x_half, a_y, a_z_half, K_x_half, K_y, K_z_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vz, DELTAT_over_rho);

		HANDLE_ERROR(cudaMemcpy(iit, &it, sizeof(int), cudaMemcpyHostToDevice));
		keraddSource << <blocks, threads >> >(iit, ISOURCE, JSOURCE, KSOURCE, ANGLE_FORCE, DEGREES_TO_RADIANS, DELTAT, factor, t0, ff0, DPI, vx, vy, rho);

		kerDirichletBoundary << <blocks, threads >> >(DDIMX, DDIMY, DDIMZ, vx, vy, vz);

		if (fmod(it, 200) == 0){
			float *tempvz = (float*)malloc(sizeof(float)*DIMX*DIMY*DIMZ);
			float *sxvz = (float*)malloc(sizeof(float)*NIMY*NIMZ);
			float *syvz = (float*)malloc(sizeof(float)*NIMX*NIMZ);
			float *szvz = (float*)malloc(sizeof(float)*NIMX*NIMY);
			HANDLE_ERROR(cudaMemcpy(tempvz, vz, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyDeviceToHost));

			//slicing for snapshot
			for (int k = 1; k <= NIMZ; k++) {
				for (int j = 1; j <=NIMY; j++) {
					int jk = (j-1) + (k-1)*NIMY;
					int ijk = ISOURCEE + j*NIMX + k*NIMX*NIMY;
					sxvz[jk] = tempvz[ijk];
				}

				for (int i = 1; i <= NIMX - 1; i++) {
					int ik = (i-1) + (k-1)*NIMX;
					int ijk = i + JSOURCEE*NIMX + k*NIMX*NIMY;
					syvz[ik] = tempvz[ijk];
				}
			}
			for (int j = 1; j <= NIMY; j++) {
				for (int i = 1; i <= NIMX; i++) {
					int ij = (i-1) + (j-1)*NIMX;
					int ijk = i + j*NIMX + KSOURCEE * NIMX*NIMY;
					szvz[ij] = tempvz[ijk];
				}
			}
			//save to file
			char nmfile1[50]; char nmfile2[50]; char nmfile3[50];
			sprintf_s(nmfile1, "sxvz%05i.sxvz", it);
			sprintf_s(nmfile2, "syvz%05i.syvz", it);
			sprintf_s(nmfile3, "szvz%05i.szvz", it);
			errno_t err;
			FILE *file1, *file2, *file3;
			err = fopen_s(&file1, nmfile1, "wb");
			if (err == 0) {
				printf("Capturing sxvz %05i \n", it);
				for (int k = 0; k < NIMZ; k++) {
					for (int j = 0; j < NIMY; j++) {
						int jk = j + k*NIMY;
						float f1 = sxvz[jk];
						fwrite(&f1, sizeof(float), 1, file1);
					}
				}
				fclose(file1);
			}
			err = fopen_s(&file2, nmfile2, "wb");
			if (err == 0) {
				printf("Capturing syvz %05i \n", it);
				for (int k = 0; k < NIMZ; k++) {
					for (int i = 0; i < NIMX; i++) {
						int ik = i + k*NIMX;
						float f2 = syvz[ik];
						fwrite(&f2, sizeof(float), 1, file2);
					}
				}
				fclose(file2);
			}
			err = fopen_s(&file3, nmfile3, "wb");
			if (err == 0) {
				printf("Capturing szvz %05i \n", it);
				for (int j = 0; j < NIMY; j++) {
					for (int i = 0; i < NIMX; i++) {
						int ij = i + j*NIMX;
						float f3 = szvz[ij];
						fwrite(&f3, sizeof(float), 1, file3);
					}
				}
				fclose(file3);
			}
			_fcloseall();
			//save to file END
			free(tempvz);
			free(sxvz); free(syvz); free(szvz);
		}
	}

	HANDLE_ERROR(cudaFree(sigmaxx)); HANDLE_ERROR(cudaFree(sigmayy)); HANDLE_ERROR(cudaFree(sigmazz));
	HANDLE_ERROR(cudaFree(sigmaxy)); HANDLE_ERROR(cudaFree(sigmaxz)); HANDLE_ERROR(cudaFree(sigmayz));
	HANDLE_ERROR(cudaFree(memory_dsigmaxx_dx)); HANDLE_ERROR(cudaFree(memory_dsigmaxy_dx)); HANDLE_ERROR(cudaFree(memory_dsigmaxy_dy));
	HANDLE_ERROR(cudaFree(memory_dsigmaxz_dx)); HANDLE_ERROR(cudaFree(memory_dsigmaxz_dz)); HANDLE_ERROR(cudaFree(memory_dsigmayy_dy));
	HANDLE_ERROR(cudaFree(memory_dsigmayz_dy)); HANDLE_ERROR(cudaFree(memory_dsigmayz_dz)); HANDLE_ERROR(cudaFree(memory_dsigmazz_dz));
	HANDLE_ERROR(cudaFree(memory_dvx_dx)); HANDLE_ERROR(cudaFree(memory_dvx_dy)); HANDLE_ERROR(cudaFree(memory_dvx_dz));
	HANDLE_ERROR(cudaFree(memory_dvy_dx)); HANDLE_ERROR(cudaFree(memory_dvy_dy)); HANDLE_ERROR(cudaFree(memory_dvy_dz));
	HANDLE_ERROR(cudaFree(memory_dvz_dx)); HANDLE_ERROR(cudaFree(memory_dvz_dy)); HANDLE_ERROR(cudaFree(memory_dvz_dz));
	return 0;
}
