// 2D poroelastic finite-difference code in velocity and stress formulation
// with Convolution-PML (C-PML) absorbing conditions
// with and without viscous dissipation
//
// Roland Martin, University of Pau, France, October 2009.
// based on the elastic code of Komatitsch and Martin, 2007.
//Modified for CUDA GPGPU by Sudarmaji and Leo Cahya D
//
// The fourth-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used:
//
//            ^ y
//            |
//            |
//
//            +-------------------+
//            |                   |
//            |                   |
//            |                   |
//            |                   |
//            |        v_y        |
//   sigma_xy +---------+         |
//            |         |         |
//            |         |         |
//            |         |         |
//            |         |         |
//            |         |         |
//            +---------+---------+  ---> x
//           v_x    sigma_xx
//                  sigma_yy
//
#include "cuda.h"
#include "conio.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>
#include "book.h"
//#include "cpu_anim.h"
#include <fstream>
#include <string>
#define PI 3.1415926535897932f
#define nilai_maks 1.0e30
#define batas_kestabilan 1.0e25
#define d_pml 10.0
#define faktor 1.0e10
    
using namespace std;

	__global__ void stress1_kernel(float *i_gpu, int *NNX, int *NNZ, int *is, int *js,float *ff0, float *DDELTAX, float *DDELTAZ, float *DDELTAT, float *vx, float *vz, float *memory_dx_sigmaxx, float *memory_dz_sigmaxx, float *b_x_half_x,float *a_x_half_x,float *b_z,float *a_z,float *gamma11,float *gamma22,float *K_x_half_x,float *K_z,float *vxf,float *vzf, float *xi_1, float *xi_2, float *memory_dx_sigma2vxf, float *memory_dz_sigma2vzf, float *sigma2, float *alpha, float *rbM) 
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;		
		//  memory of sigmaxx

		int right= offset+1; int right2=offset+2;
		int left= offset-1; 
		int bottom=offset-(NNX[0]+1); int bottom2=bottom-(NNX[0]+1);
		int top=offset+(NNX[0]+1);
		if ((x >= 1) && (x < NNX[0]))   
		{
			if ((y >= 2) && (y < NNZ[0]-1))   
			{
        float value_dx_sigmaxx =(27.0*vx[right]-27.0*vx[offset]-vx[right2]+vx[left])/DDELTAX[0]/24.0;
        float value_dz_sigmaxx =(27.0*vz[offset]-27.0*vz[bottom]-vz[top]+vz[bottom2])/DDELTAZ[0]/24.0;
        
        memory_dx_sigmaxx[offset] = b_x_half_x[x] * memory_dx_sigmaxx[offset] + a_x_half_x[x] * value_dx_sigmaxx;
        memory_dz_sigmaxx[offset] = b_z[y] * memory_dz_sigmaxx[offset] + a_z[y] * value_dz_sigmaxx;
        
        
        gamma11[offset] = gamma11[offset]+DDELTAT[0]*(value_dx_sigmaxx / K_x_half_x[x] + memory_dx_sigmaxx[offset]);
        
        gamma22[offset] = gamma22[offset]+DDELTAT[0]*(value_dz_sigmaxx / K_z[y] + memory_dz_sigmaxx[offset]);
        
		// sigma2
        float value_dx_sigma2vxf=(27.0*vxf[right]-27.0* vxf[offset]-vxf[right2]+vxf[left]) / DDELTAX[0]/24.0;
        float value_dz_sigma2vzf=(27.0*vzf[offset]-27.0*vzf[bottom]-vzf[top]+vzf[bottom2]) / DDELTAZ[0]/24.0;
        
        memory_dx_sigma2vxf[offset] = b_x_half_x[x] * memory_dx_sigma2vxf[offset] + a_x_half_x[x] * value_dx_sigma2vxf;
        memory_dz_sigma2vzf[offset] = b_z[y] * memory_dz_sigma2vzf[offset] + a_z[y] * value_dz_sigma2vzf;
        
        xi_1[offset] = xi_1[offset] -(value_dx_sigma2vxf/ K_x_half_x[x] + memory_dx_sigma2vxf[offset])*DDELTAT[0];
        
        xi_2[offset] = xi_2[offset]-(value_dz_sigma2vzf/K_z[y]+memory_dz_sigma2vzf[offset])*DDELTAT[0];
        
        sigma2[offset]=-alpha[offset]*rbM[offset]*(gamma11[offset]+gamma22[offset])+rbM[offset]*(xi_1[offset]+xi_2[offset]);
			}
		}
		
		if ((x == is[0]) && (y == js[0]))   
		{
		////add the source (point source located at a given grid point)
		float a = PI*PI*ff0[0]*ff0[0];
		float t = i_gpu[0]*DDELTAT[0];
		float t0=1.0/ff0[0];
		//float source_term =faktor * exp(-a*pow((t-t0),2))/(-2.0*a); //zhu
		float source_term =  faktor * 2.0*a*(t-t0)*exp(-a*pow((t-t0),2));
		////float source_term =  faktor * (t-1.2)*exp(-a*pow(t-1.2/ff0[0],2));
		//float source_term =  faktor * (2.0*a*pow(t-1.2/ff0[0],2))*exp(-a*pow(t-1.2/ff0[0],2));
		//float source_term = faktor * exp(-a*pow((t-t0),2));
		float w=0.0;
		float f1=10.0; float f2=11.0; float f3=39.0; float f4=40.0; float tt0=0.15;
		w = (PI*f4*f4) * (sin(PI*f4*t)/(PI*f4*(t-tt0)))*(sin(PI*f4*(t-tt0))/(PI*f4*(t-tt0)))/(f4-f3);
		w = w- (PI*f3*f3) * (sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))*(sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))/(f4-f3);
		w = w- (PI*f2*f2) * (sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))*(sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))/(f2-f1);
		w = w+ (PI*f1*f1) * (sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))*(sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))/(f2-f1);
		if (t<0.4)
		{sigma2[offset]=sigma2[offset]+source_term;
		}
		i_gpu[0] = i_gpu[0] + 1.0;
		}
	}
//save1_kernel<<<blocks,threads>>>(a0_pos,a1_pos,panj2,panj,samprek,i_steps,seismo,seismo1,seismo2,seismo3,seismo4,seismo0,NNX, NNZ, is, js, vz);
	__global__ void save1_kernel(int a0_pos,int a1_pos,int panj2,int panj,int samprek,int i,float *seismo,float *deismo,float *seismo1,float *deismo1, float *seismo2,float *seismo3,float *seismo4,float *seismo0,int *NNX, int *NNZ, int *is, int *js,float *vx, float *vz)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;	
	
		int length2=(NNZ[0]+1)/panj;
		if (x==is[0])         //if(x==((NNX[0]+1)/2)-1)
			//if(x==3500)
		{

			if (i==0)
			{
			    if(((y+1)%length2==0))
		        {
				seismo[((y)/length2)+((1)-1)*(panj)]=vz[offset];
				deismo[((y)/length2)+((1)-1)*(panj)]=vx[offset];
			    }
			}
			if((i%samprek==0)&&(i>0))
			{
			    if(((y+1)%length2==0))
				{
				seismo[((y)/length2)+((i/samprek)-1)*(panj)]=vz[offset];	
				deismo[((y)/length2)+((i/samprek)-1)*(panj)]=vx[offset];
			    }
			}
		  
		}
		int length=(NNX[0]+1)/panj2;
		if(y==10)
		{

			if (i==0)
			{
			    if(((x+1)%length==0))
		        {
				seismo1[((x)/length)+((1)-1)*(panj2)]=vz[offset];
				deismo1[((x)/length)+((1)-1)*(panj2)]=vx[offset];
			    }
			}
			if((i%samprek==0)&&(i>0))
			{
			    if(((x+1)%length==0))
				{
				seismo1[((x)/length)+((i/samprek)-1)*(panj2)]=vz[offset];
				deismo1[((x)/length)+((i/samprek)-1)*(panj2)]=vx[offset];
			    }
			}
		  
		}
	//	if((y==1500/8000*(NNZ[0]+1))&&(x==(NNX[0]+1)/2))
	//	{
	//	seismo0[i-1]=vz[offset];
	//	}
	//	
	//	if((y==2900/8000*(NNZ[0]+1))&&(x==(NNX[0]+1)/2))
	//	{
	//	seismo1[i-1]=vz[offset];
	//	}
	//	

    //
	//	if((y==7000/8000*(NNZ[0]+1))&&(x==(NNX[0]+1)/2))
	//	{
	//	seismo4[i-1]=vz[offset];
	//	}
		if((y==50)&&(x==(NNX[0]+1)/2-1))
		{
		seismo0[i-1]=vz[offset];
		}
		
		
		if((y==(a0_pos))&&(x==(NNX[0]+1)/2-1))
		{
		seismo2[i-1]=vz[offset];
		}
    
		if((y==(a1_pos))&&(x==(NNX[0]+1)/2-1))
		{
		seismo3[i-1]=vz[offset];
		}

		if((y==js[0])&&(x==(NNX[0]+1)/2-1))
		{
		seismo4[i-1]=vz[offset];
		}
	}	
	__global__ void stress2_kernel(int *NNX, int *NNZ, float *DDELTAX, float *DDELTAZ, float *DDELTAT, float *vx, float *vz, float *rmu,float *memory_dx_sigmaxz,float *memory_dz_sigmaxz,float *b_x,float *a_x,float *b_z_half_z,float *a_z_half_z,float *sigmaxz,float *K_x,float *K_z)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;
		int right= offset+1; 
		int left= offset-1;  int left2=offset-2;
		int bottom=offset-(NNX[0]+1); 
		int top=offset+(NNX[0]+1); int top2=top+(NNX[0]+1);
		
		if ((x >= 2) && (x < NNX[0]))   
		{
		if ((y >= 1) && (y < NNZ[0]-1))   
		{
		// interpolate material parameters at the right location in the staggered grid cell
		float c33_half_z = 2.0/(1.0/rmu[offset]+1.0/rmu[top]);
		c33_half_z = rmu[top];
		
		float value_dx_sigmaxz = (27.0*vz[offset] - 27.0*vz[left]-vz[right]+vz[left2]) / DDELTAX[0]/24.0;
		float value_dz_sigmaxz = (27.0*vx[top] - 27.0*vx[offset]-vx[top2]+vx[bottom]) / DDELTAZ[0]/24.0;
		
		memory_dx_sigmaxz[offset] = b_x[x] * memory_dx_sigmaxz[offset] + a_x[x] * value_dx_sigmaxz;
		memory_dz_sigmaxz[offset] = b_z_half_z[y] * memory_dz_sigmaxz[offset] + a_z_half_z[y] * value_dz_sigmaxz;
		
		sigmaxz[offset] = sigmaxz[offset] + c33_half_z/1.0 * (value_dx_sigmaxz / K_x[x] + memory_dx_sigmaxz[offset] + value_dz_sigmaxz / K_z[y] + memory_dz_sigmaxz[offset]) * DDELTAT[0];
		}
		}
	}

		__global__ void stress3_kernel(int samprek,int i,float *seismo,float *seismo1,float *seismo2,float *seismo3,float *seismo4,float *seismo0,float *i_gpu, int *NNX, int *NNZ, int *is, int *js,float *ff0, float *DDELTAX, float *DDELTAZ, float *DDELTAT, float *sigmaxx,float *sigmazz,float *rlambdac,float *rmu,float *gamma11,float *gamma22,float *alpha,float *sigma2)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;
		
		if ((x >= 1) && (x < NNX[0]-1))   
		{
		if ((y >= 2) && (y < NNZ[0]))   
		{
			sigmaxx[offset]=(rlambdac[offset]+2.0*rmu[offset])*gamma11[offset]+rlambdac[offset]*gamma22[offset] -alpha[offset]*sigma2[offset];
			sigmazz[offset]=rlambdac[offset]*gamma11[offset]+(rlambdac[offset]+2.0*rmu[offset])*gamma22[offset] -alpha[offset]*sigma2[offset];
		}
		}
		
		if ((x == is[0]) && (y == js[0]))   
		{
		//add the source (point source located at a given grid point)
		//float a = PI*PI*ff0[0]*ff0[0];
		//float t = i_gpu[0]*DDELTAT[0];
		//float t0=1.0/ff0[0];
		//float source_term =  faktor * (1.0 - 2.0*a*pow(t-1.2/ff0[0],2))*exp(-a*pow(t-1.2/ff0[0],2));
		//float source_term =  faktor * (t-1.2)*exp(-a*pow(t-1.2/ff0[0],2));
		//float source_term =  faktor * (2.0*a*pow(t-1.2/ff0[0],2))*exp(-a*pow(t-1.2/ff0[0],2));
		// Gaussian
        //float source_term = faktor * exp(-a*pow((t-t0),2))/(-2.0*a);
        //float source_term = faktor * exp(-a*pow((t-t0),2));
        //first derivative of a Gaussian
        // source_term =  factor * 2.d0*a*(t-t0)*exp(-a*(t-t0)**2)
          // source_term =  factor *(t-t0)*exp(-a*(t-t0)**2)
        
		//float source_term =-faktor * exp(-a*pow((t-t0),2));

		//source_term = factor * exp(-a*(t-t0)^2); matlab
		//float w=0.0;
		//float f1=0.1; float f2=1.0; float f3=99.0; float f4=100.0; float tt0=0.1;
		//w = (PI*f4*f4) * (sin(PI*f4*t)/(PI*f4*(t-tt0)))*(sin(PI*f4*(t-tt0))/(PI*f4*(t-tt0)))/(f4-f3);
		//w = w- (PI*f3*f3) * (sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))*(sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))/(f4-f3);
		//w = w- (PI*f2*f2) * (sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))*(sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))/(f2-f1);
		//w = w+ (PI*f1*f1) * (sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))*(sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))/(f2-f1);
		//sigmaxx[offset]=sigmaxx[offset]+source_term;
		//sigmazz[offset]=sigmazz[offset]+source_term;
		//sigma2[offset]=sigma2[offset]+source_term;
		//i_gpu[0] = i_gpu[0] + 1.0;
		}
		
	}
	
		__global__ void vx_kernel(int *NNX, int *NNZ, float *DDELTAX, float *DDELTAZ, float *DDELTAT, float *vx, float *rho,float *rsm,float *rhof,float *etaokappa,float *vxf,float *sigmaxx,float *sigma2,float *sigmaxz,float *memory_dx_vx1,float *memory_dx_vx2,float *memory_dz_vx,float *b_x,float *a_x,float *a_z,float *b_z,float *K_x,float *K_z)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;
		int right= offset+1; 
		int left= offset-1;  int left2=offset-2;
		int bottom=offset-(NNX[0]+1); int bottom2=bottom-(NNX[0]+1);
		int top=offset+(NNX[0]+1); 	
		
		if ((x >= 2) && (x < NNX[0]))   
		{
		if ((y >= 2) && (y < NNZ[0]))   
		{	
		float co=(rho[offset]*rsm[offset]-rhof[offset]*rhof[offset])/DDELTAT[0];
		float c1=co+rho[offset]*etaokappa[offset]*0.50;
		float c2=co-rho[offset]*etaokappa[offset]*0.50;
		float vtemp=vxf[offset];
		float value_dx_vx1 = (27.0*sigmaxx[offset] - 27.0*sigmaxx[left]-sigmaxx[right]+sigmaxx[left2]) / DDELTAX[0]/24.0;
		float value_dx_vx2 = (27.0*sigma2[offset] - 27.0*sigma2[left]-sigma2[right]+sigma2[left2]) / DDELTAX[0]/24.0;
		float value_dz_vx = (27.0*sigmaxz[offset] - 27.0*sigmaxz[bottom]-sigmaxz[top]+sigmaxz[bottom2]) / DDELTAZ[0]/24.0;
		
		
		
		memory_dx_vx1[offset] = b_x[x] * memory_dx_vx1[offset] + a_x[x] * value_dx_vx1;
		memory_dx_vx2[offset] = b_x[x] * memory_dx_vx2[offset] + a_x[x] * value_dx_vx2;
		memory_dz_vx[offset] = b_z[y] * memory_dz_vx[offset] + a_z[y] * value_dz_vx;
		
		vxf[offset] = (c2*vxf[offset] +(-rhof[offset]*(value_dx_vx1/ K_x[x] + memory_dx_vx1[offset]+ value_dz_vx / K_z[y] + memory_dz_vx[offset])-rho[offset]*(value_dx_vx2/ K_x[x] + memory_dx_vx2[offset]))) /c1;
		
		vtemp=(vtemp+vxf[offset])*0.50;
		
		vx[offset] = vx[offset]+(rsm[offset]*(value_dx_vx1/ K_x[x] + memory_dx_vx1[offset]+value_dz_vx / K_z[y] + memory_dz_vx[offset])+rhof[offset]*(value_dx_vx2/ K_x[x] + memory_dx_vx2[offset])+rhof[offset]*etaokappa[offset]*vtemp)/co;
		}
		}
	}

		__global__ void vz_kernel(float *i_gpu, int *NNX, int *NNZ, int *is, int *js,float *ff0, float *DDELTAX, float *DDELTAZ, float *DDELTAT,float *vz, float *rho,float *rsm,float *rhof,float *etaokappa,float *vzf,float *sigmaxz,float *sigmazz,float *sigma2, float *memory_dz_vz1,float *memory_dz_vz2,float *memory_dx_vz,float *b_x_half_x,float *a_x_half_x,float *a_z_half_z,float *b_z_half_z,float *K_x_half_x,float *K_z_half_z)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;
		int right= offset+1; int right2= offset+2; 
		int left= offset-1; 
		int bottom=offset-(NNX[0]+1);// int bottom2=bottom-NX;
		int top=offset+(NNX[0]+1); 		int top2=top+(NNX[0]+1);	
		
		if ((x >= 1) && (x < NNX[0]-1))   
		{
		if ((y >= 1) && (y < NNZ[0]-1))   
		{	
		float rho_half_x_half_z = rho[top];
		float rsm_half_x_half_z = rsm[top];
		float rhof_half_x_half_z = rhof[top];
		float etaokappa_half_x_half_z = etaokappa[top];
		 
		
		
		float co=(rho_half_x_half_z*rsm_half_x_half_z-rhof_half_x_half_z*rhof_half_x_half_z)/DDELTAT[0];
		float c1=co+rho_half_x_half_z*etaokappa_half_x_half_z*0.50;
		float c2=co-rho_half_x_half_z*etaokappa_half_x_half_z*0.50;
		float vtemp=vzf[offset];
		
		float value_dx_vz = (27.0*sigmaxz[right] - 27.0*sigmaxz[offset]-sigmaxz[right2]+sigmaxz[left]) / DDELTAX[0]/24.0;
		float value_dz_vz1 = (27.0*sigmazz[top]- 27.0*sigmazz[offset]-sigmazz[top2]+sigmazz[bottom]) / DDELTAZ[0]/24.0;
		float value_dz_vz2 = (27.0*sigma2[top] - 27.0*sigma2[offset]-sigma2[top2]+sigma2[bottom]) / DDELTAZ[0]/24.0;
		
		memory_dx_vz[offset]  = b_x_half_x[x] * memory_dx_vz[offset] + a_x_half_x[x] * value_dx_vz;
		memory_dz_vz1[offset] = b_z_half_z[y] * memory_dz_vz1[offset] + a_z_half_z[y] * value_dz_vz1;
		memory_dz_vz2[offset] = b_z_half_z[y] * memory_dz_vz2[offset] + a_z_half_z[y] * value_dz_vz2;
		
		vzf[offset] = (c2*vzf[offset] +(-rhof_half_x_half_z*(value_dx_vz / K_x_half_x[x] + memory_dx_vz[offset]+value_dz_vz1 / K_z_half_z[y] + memory_dz_vz1[offset])-rho_half_x_half_z*(value_dz_vz2 / K_z_half_z[y] + memory_dz_vz2[offset]))) /c1;
		vtemp=(vtemp+vzf[offset])*0.50;
		
		vz[offset] = vz[offset] +(rsm_half_x_half_z*(value_dx_vz / K_x_half_x[x] + memory_dx_vz[offset]+ value_dz_vz1 / K_z_half_z[y] + memory_dz_vz1[offset])+ rhof_half_x_half_z*(value_dz_vz2 / K_z_half_z[y] + memory_dz_vz2[offset])+ rhof_half_x_half_z*etaokappa_half_x_half_z*vtemp)/co;
		}
		}
		
		//if ((x == is[0]) && (y == js[0]))   
		//{
		////add the source (point source located at a given grid point)
		//float a = PI*PI*ff0[0]*ff0[0];
		//float t = i_gpu[0]*DDELTAT[0];
		//float t0=1.0/ff0[0];
		//float source_term =  faktor * (1.0 - 2.0*a*pow(t-1.2/ff0[0],2))*exp(-a*pow(t-1.2/ff0[0],2));
		////float source_term =  faktor * (t-1.2)*exp(-a*pow(t-1.2/ff0[0],2));
		////float source_term =  faktor * (2.0*a*pow(t-1.2/ff0[0],2))*exp(-a*pow(t-1.2/ff0[0],2));
		//// Gaussian
        ////float source_term = faktor * exp(-a*pow((t-t0),2))/(-2.0*a);
        //
        ////first derivative of a Gaussian
        //// source_term =  factor * 2.d0*a*(t-t0)*exp(-a*(t-t0)**2)
        //  // source_term =  factor *(t-t0)*exp(-a*(t-t0)**2)
        //
		////float source_term =faktor * 2.0*a*(t-t0)*exp(-a*pow((t-t0),2));
		//float w=0.0;
		////float f1=ff0[0]; float f2=ff0[0]+4.0; float f3=ff0[0]+96.0; float f4=ff0[0]+106.0; float tt0=0.1;
		//float f1=5.0; float f2=10.0; float f3=90.0; float f4=100.0; float tt0=0.1;
		//w = (PI*f4*f4) * (sin(PI*f4*t)/(PI*f4*(t-tt0)))*(sin(PI*f4*(t-tt0))/(PI*f4*(t-tt0)))/(f4-f3);
		//w = w- (PI*f3*f3) * (sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))*(sin(PI*f3*(t-tt0))/(PI*f3*(t-tt0)))/(f4-f3);
		//w = w- (PI*f2*f2) * (sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))*(sin(PI*f2*(t-tt0))/(PI*f2*(t-tt0)))/(f2-f1);
		//w = w+ (PI*f1*f1) * (sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))*(sin(PI*f1*(t-tt0))/(PI*f1*(t-tt0)))/(f2-f1);
		//vz[offset]=source_term;
		//i_gpu[0] = i_gpu[0] + 1.0;
		//}
		
		
	}
	__global__ void zero_kernel(float *memory_dx_sigmaxx,float *memory_dz_sigmaxx,float *gamma11,float *gamma22,float *memory_dx_sigma2vxf,float *memory_dz_sigma2vzf,float *xi_1,float *xi_2,float *sigma2,float *memory_dx_sigmaxz,float *memory_dz_sigmaxz,float *sigmaxz,float *sigmaxx,float *sigmazz,float *memory_dx_vx1 ,float *memory_dx_vx2 ,float *memory_dz_vx  	,float *vxf,float *vx,float *vz ,float *vzf ,float *i_gpu,float *memory_dx_vz ,float *memory_dz_vz1,float *memory_dz_vz2) 
	{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

	memory_dx_sigmaxx[offset] = 0.0;
	memory_dz_sigmaxx[offset] = 0.0;
	gamma11[offset] = 0.0;
	gamma22[offset] = 0.0;
	memory_dx_sigma2vxf[offset] =0.0;
	memory_dz_sigma2vzf[offset] =0.0;
	xi_1[offset] = 0.0;
	xi_2[offset] = 0.0;
	sigma2[offset]=0.0;
	
    memory_dx_sigmaxz[offset] = 0.0;	
    memory_dz_sigmaxz[offset] = 0.0;	
    sigmaxz[offset] = 0.0;
	
	sigmaxx[offset]=0.0;
	sigmazz[offset]=0.0;
	
	memory_dx_vx1[offset] =0.0;
	memory_dx_vx2[offset] =0.0;
	memory_dz_vx[offset] = 0.0;	
	vxf[offset] = 0.0;
	vx[offset] = 0.0;
	
	vz[offset] =0.0;
	vzf[offset] =0.0;
	i_gpu[0]=0.0;
	memory_dx_vz[offset] =0.0;
	memory_dz_vz1[offset]=0.0;
	memory_dz_vz2[offset]=0.0;
	}
	
	float max_element (float a[], int lng)
	{
	float mx=a[0];
	for(int i=1;i<lng;i++)
		{
			if(mx<a[i])
			{
				mx = a[i];
			}
		}
	return mx;
	}

	float min_element (float a[], int lng)
	{
	float mx=a[0];
	for(int i=1;i<lng;i++)
		{
			if(mx>a[i])
			{
				mx = a[i];
			}
		}
	return mx;
	}
	
// ********************* Template untuk alokasi memory 2D
template <typename T>
T **AllocateDynamicArray( int nRows, int nCols)
{
      T **dynamicArray;

      dynamicArray = new T*[nRows];
      for( int i = 0 ; i < nRows ; i++ )
      dynamicArray[i] = new T [nCols];

      return dynamicArray;
}

template <typename T>
void FreeDynamicArray(T** dArray)
{
      delete [] *dArray;
      delete [] dArray;
}
	
int  main( void ) 
{
		std::ifstream in_file1("input1.txt");
	    if(!in_file1.is_open())
	    {
	    	printf("File not opened...");
	    	return 1;
	    }

		std::ifstream in_file2("input2.txt");
	    if(!in_file2.is_open())
	    {
	    	printf("File not opened...");
	    	return 1;
	    }

  int NX = 13599;
  int NZ = 2439;
  //NX=NX-1;
  //NZ=NZ-1;
    //int NX = 799;
    //int NZ = 799;
  std::string line;
//size of a grid cell
  float DELTAX = 1.25;
  float DELTAZ = DELTAX;
  int panj=122; //vsp
  int panj2=680; //hor
  cout<<panj<<endl;
  int NSTEP;
  in_file1>>NSTEP;
  int samp =  5000;
  int samprek = 1;
//time step in seconds
  float DELTAT =95e-06;
  //cout<<"masukkan deltat = "; cin>>DELTAT;
// parameters for the source

  float factor =1.e02;
  // thickness of the PML layer in grid points
  int NPOINTS_PML = 10;

//heterogeneous model and height of the interface
  //logical, parameter :: HETEROGENEOUS_MODEL = .true.
  bool HETEROGENEOUS_MODEL  = true;
  float INTERFACE_HEIGHT =(NZ*DELTAZ)/2.0+NPOINTS_PML*DELTAZ;
  //int JINTERFACE=(INTERFACE_HEIGHT/DELTAZ)+1;
  //float co,c1,c2,vtemp;

  // model mem allocation
  float *cp=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cs=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *Kb=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *Ks=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *Kf=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *Ksat=   (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *eta=  (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *kappa= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *phi=  (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *a=    (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crhos= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crhof= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  
  float *medium= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));  
  //
  float *crho = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
 // float *crhof = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crsm = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crmu = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crlambdac = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *crbM = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *calpha = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cetaokappa = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));  
  float *crlambdao = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));    
//reading model
		//std::ifstream in_file0("modporo.ctxt");
		//if(!in_file0.is_open())
		//{
		//	printf("File not opened...");
		//	return 1;
		//}
// float *crhof= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
		
int jmlmed=4;
	


//zhu shale datar
//float kfm   []=   {2.25e09 ,2.25e09  ,1.0e09  ,2.0e07     };
//float etam  []=   {0.001   ,1.0e-3   ,4.0e-3  ,1.0e-4     };
//float kappam[]=   {1.e-18  ,1.e-9    ,1.e-9   ,1.0e-12    }; 
//float phim  []=   {0.001   ,0.3      ,0.3     ,0.25       };
//float am    []=   {2.5     ,2.2      ,2.2     ,2.2        };
//float rhofm []=   {1000.0  ,1000.0   ,700.0   ,150.0      };

float kfm   []=   {2.4e09 ,2.4e09  ,0.7e09  ,0.02e09     };// Bulk Moduli of Fluid (GPa) from Batzle and Wang, 1992 and Quintal, B., (2012)
float etam  []=   {0.003   ,0.003   ,0.03  ,1.0e-5     };//viscosity (Pa.s) from Batzle and Wang (1992) and Quintal, B., (2012)
float kappam[]=   {1.5e-17  ,500.0e-15    ,500.0e-15   ,500.0e-15	  }; //permeability (1D=0.986923e-12 m2) from dutta(1983) for shale kappa=0.087x phi^6
float am    []=   {2.0     ,2.0      ,2.0     ,2.0        };// Tortousity from Wenzlau, F., 2009, Disertation Uni Berli
//Kondisi poroelastik
//float phim  []=   {0.01   ,0.25      ,0.25     ,0.01        };// kondisi poroelastik
//float rhofm []=   {1000.0  ,1000.0   ,700.0   ,160.0      };//kondisi poroelastik

//Kondisi elastik
float phim  []=   {1.e-6   ,1.e-6    ,1.e-6   ,1.e-6      };//kondisi elastik
float rhofm []=   {10.0  ,10.0   ,10.0   ,10.0      }; ///  kondisi elastik


float i_frek[]=   {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,110.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0};



cout<<"NSTEP="<<NSTEP<<endl;




//bates model : parameter ini menunjukkan batas grid lapisan agar program tidak terlalu panjang

//int minz   []={1900,1913,1925,1938,1950,1963,1975,1988,2000,2013};
//int maxz   []={1913,1925,1938,1950,1963,1975,1988,2000,2013,2025};
//float *medium= (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));  



int tb_lapis=10;
int h_rsvr=3333;
int tbl_rsvr=340;

int hom;
cout<<"\nhomogen? (1=ya) =";
	hom=1;
	cout<<endl;if (hom==1)	
{ 
	cout<<"test meh load";
		//line="model_zhu_10000.bin";
		line="mar_poro_2440x13600.bin";
        char *inname1 = const_cast<char*>(line.c_str());	
		FILE* file = fopen(inname1,"rb");
		for(int i=0; i<(NX+1)*(NZ+1); i++){
		float f;
		fread(&f,sizeof(float),1,file);
		medium[i]=f;
		}
		fclose(file);
		
 		line="mar_cp_2440x13600.bin";
        char *inname2 = const_cast<char*>(line.c_str());	
		FILE* file2 = fopen(inname2,"rb");
		for(int i=0; i<(NX+1)*(NZ+1); i++){
		float f2;
		fread(&f2,sizeof(float),1,file2);
		cp[i]=f2;
		}
		fclose(file2);

 		line="mar_cs_2440x13600.bin";
        char *inname3 = const_cast<char*>(line.c_str());	
		FILE* file3 = fopen(inname3,"rb");
		for(int i=0; i<(NX+1)*(NZ+1); i++){
		float f3;
		fread(&f3,sizeof(float),1,file3);
		cs[i]=f3;
		}
		fclose(file3);

 		line="mar_rho_2440x13600.bin";
        char *inname4 = const_cast<char*>(line.c_str());	
		FILE* file4 = fopen(inname4,"rb");
		for(int i=0; i<(NX+1)*(NZ+1); i++){
		float f4;
		fread(&f4,sizeof(float),1,file4);
		crhos[i]=f4;
		}
		fclose(file4);
		
 		for(int j=0; j<(NX+1)*(NZ+1); j++)
		{	
			for(int i=0; i<jmlmed; i++)
	        {
				if(medium[j]==i)
				{
				 
				 Kf[j]=kfm   [i];
				 eta[j]=etam  [i];
				 kappa[j]=kappam[i];
				 phi[j]=phim  [i];
				 a[j]=am    [i];
				 crhof[j]=rhofm [i];
				}
			}
		}
		cout<<"test wis load";
      }


	 else 
     {


     }

  
cout<<"MASUKIN PARAMETER SELESAI"<<endl;
// poroelastic model mud by sudarmaji
 //float maxcptop=max_element(cp,((NX+1)*(NZ+1))); float maxcpbottom=maxcptop;
 //float mincptop=min_element(cp,((NX+1)*(NZ+1))); float mincpbottom=mincptop;
 

  
  
  // hitung variabel medium untuk perhitungan utama
  	for(int j=0; j < NZ+1; j++)
    for(int i=0; i < NX+1; i++)
	{
	int ij=(NX+1)*j + i;	
	//crho[ij]= phi[ij]*crhof[ij]+(1.0-phi[ij])*crhos[ij]; //untuk rho dry 
	crho[ij]=crhos[ij]; // jika rho input saturated
	crmu[ij]= cs[ij]*cs[ij]*crho[ij];
	
	Ksat[ij]=crhos[ij]*(cp[ij]*cp[ij]-(4.0/3.0)*(cs[ij]*cs[ij]));
	
	float vgede=(3.0*Ksat[ij]-2.0*crmu[ij])/(2.0*(3.0*Ksat[ij]+crmu[ij]));
	
	Kb[ij]=((2.0*vgede/(1.0-2.0*vgede))+(2.0/3.0))*crmu[ij];
	Ks[ij]=Kb[ij]/(1.0-phi[ij]);
	
	crsm[ij] = a[ij]*crhof[ij]/phi[ij];
	cetaokappa[ij]=eta[ij]/kappa[ij];
    calpha[ij]=1.0-Kb[ij]/Ks[ij];
	crbM[ij] = 1.0/(phi[ij]/Kf[ij]+(calpha[ij]-phi[ij])/Ks[ij]);	
	if((eta[ij]==0.0)||(kappa[ij]==0.0))
	{
	cetaokappa[ij]=0.0;
	}
	//float C= calpha[ij]*crbM[ij];
    //crmu[ij]= cs[ij]*cs[ij]*crho[ij];
    crlambdao[ij] = Kb[ij]-(2.0/3.0)*crmu[ij];
	crlambdac[ij] = crlambdao[ij]+calpha[ij]*calpha[ij]*crbM[ij];

	//Ksat[ij]=(((Kb[ij])/(Ks[ij]-Kb[ij]) +(Kf[ij])/(phi[ij]*(Ks[ij]-Kf[ij]))   )*Ks[ij])/(1+((Kb[ij])/(Ks[ij]-Kb[ij]) +(Kf[ij])/(phi[ij]*(Ks[ij]-Kf[ij]))));
	//cs[ij]=sqrtf(crmu[ij]/crho[ij]);
	//cp[ij]=sqrtf((Ksat[ij]/crho[ij]) +( 1.33333333333*cs[ij]*cs[ij]));
	}
	
//cek kesetabilan	
 float maxcptop=max_element(cp,((NX+1)*(NZ+1))); float maxcpbottom=maxcptop;
 float mincptop=min_element(cp,((NX+1)*(NZ+1))); float mincpbottom=mincptop;
 
 
 
  //parameter check
	int  ij=1;
  
  cout<<"Kb= "<<Kb[ij]<<endl;
  cout<<"Kf= "<<Kf[ij]<<endl;
  cout<<"Ks= "<<Ks[ij]<<endl;
  cout<<"phi= "<<phi[ij]<<endl;
  cout<<"mu= "<<crmu[ij]<<endl;
  cout<<"rhof= "<<crhof[ij]<<endl;
  cout<<"rhos= "<<crhos[ij]<<endl;
  cout<<"rho= "<<crho[ij]<<endl;
  cout<<"alpha= "<<calpha[ij]<<endl;
  cout<<"rbM= "<<crbM[ij]<<endl;
  
  cout<<"rsm= "<<crsm[ij]<<endl;
  cout<<"lambdao= "<<crlambdao[ij]<<endl;
  cout<<"rlambdac= "<<crlambdac[ij]<<endl;
  cout<<"Ksat= "<<Ksat[ij]<<endl;
  cout<<"cp= "<<cp[ij]<<endl;
  cout<<"cs= "<<cs[ij]<<endl;
  cout<<"cp/cs= "<<(cp[ij]/cs[ij])<<endl;
  cout<<"etaokappa= "<<cetaokappa[ij]<<endl;
  

  
  //end

   // total number of time steps


//source
  int ISOURCE;
  int JSOURCE;
  //ISOURCE = (NX+1)/2;
  //JSOURCE = (NZ+1)/2;

		in_file1>>ISOURCE;
		in_file1>>JSOURCE;
		int a0_pos;
		int a1_pos;
	    in_file1>>a0_pos;
		in_file1>>a1_pos;
		
		
  //cout<<"/nmasukkan ISOURCE = "; cin>>ISOURCE;
  //cout<<"/nmasukkan JSOURCE = "; cin>>JSOURCE;
  int IDEB =  (NX / 2) + 1;
  int JDEB = ( NZ / 2) + 1;
  float xsource = DELTAX * ISOURCE;
  float zsource = DELTAZ * JSOURCE;
//angle of source force clockwise with respect to vertical (Z) axis
  float ANGLE_FORCE = 0.e0;

//receivers
 // int NREC = 20
 // float precision, parameter :: zdeb = NPOINTS_PML*DELTAZ+10.D0   // first receiver z in meters
 // float precision, parameter :: zfin = NZ*DELTAZ-NPOINTS_PML*DELTAZ-10.d0   // final receiver z in meters
 // float precision, parameter :: xdeb =NPOINTS_PML*DELTAX+10.D0  // first receiver x in meters
 // float precision, parameter :: xfin =NX*DELTAX-NPOINTS_PML*DELTAX-10.d0   // final receiver x in meters

// display information on the screen from time to time
  int IT_DISPLAY = 200;

// value of PI
  //float PI = 3.141592653589793238462643e0;

// conversion from degrees to radians
  float DEGREES_TO_RADIANS = PI / 180.e0;

// zero
  float ZERO = 0.e0;

// large value for maximum
  float HUGEVAL = 1.e+30;

// velocity threshold above which we consider that the code became unstable
  float STABILITY_THRESHOLD = 1.e+25;

//main arrays
  float *cvx = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cvz = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *csigmaxx = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *csigma2 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *calp_sigma2 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *csigmazz = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *csigmaxz = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cvnorm = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  
  float *cvxf = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cvzf = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));

 	

  //float *cp = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  //float *cs = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  //float *cps = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  
//to interpolate material parameters at the right location in the staggered grid cell
  //float rho_half_x_half_z,rhof_half_x_half_z,rsm_half_x_half_z,etaokappa_half_x_half_z;

// for evolution of total energy in the medium
//  float epsilon_xx,epsilon_zz,epsilon_xz;
//  float precision, dimension(NSTEP) :: total_energy_kinetic,total_energy_potential
  //float c33_half_z;


  		float f0;
	
		//cout<<"\nfrekuensi =";
		in_file2>>f0;
		cout<<f0<<endl;
		float *ff0;
		HANDLE_ERROR( cudaMalloc( (void**)&ff0, sizeof(float)));
		
        float t0 = 1.0/f0;
	    HANDLE_ERROR( cudaMemcpy( ff0, &f0,sizeof(float),cudaMemcpyHostToDevice ) );	
  
  
// power to compute d0 profile
  float NPOWER = 2.e0;

// float precision, parameter :: K_MAX_PML = 7.d0   ! from Gedney page 8.11
  float K_MAX_PML = 1.e0;   // from Gedney page 8.11
// float precision, parameter :: ALPHA_MAX_PML = 0.05d0   ! from Gedney page 8.22
  float ALPHA_MAX_PML = 2.e0*PI*(f0/2.e0);   // from Festa and Vilotte

// 2D arrays for the memory variables
  //float precision, dimension(0:NX+1,0:NZ+1) :: gamma11,gamma22
  //float precision, dimension(0:NX+1,0:NZ+1) :: gamma12_1
  //float precision, dimension(0:NX+1,0:NZ+1) :: xi_1,xi_2

  float *cgamma11 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cgamma22 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cgamma12_1 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cxi_1 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
  float *cxi_2 = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));

  
  //float precision, dimension(0:NX+1,0:NZ+1) :: &
     //memory_dx_vx1,memory_dx_vx2,memory_dz_vx,memory_dx_vz,memory_dz_vz1,memory_dz_vz2,&
     //memory_dx_sigmaxx,memory_dx_sigmazz,memory_dx_sigmaxz,&
    // memory_dx_sigma2vx,memory_dx_sigma2vxf,memory_dz_sigma2vz,memory_dz_sigma2vzf, &
    // memory_dz_sigmaxx,memory_dz_sigmazz,memory_dz_sigmaxz

  float *cmemory_dx_vx1 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dx_vx2 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_vx 		= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dx_vz 		= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_vz1 		= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_vz2 		= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dx_sigmaxx 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dx_sigmazz 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));  
  float *cmemory_dx_sigmaxz 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));  
  float *cmemory_dx_sigma2vx = (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dx_sigma2vxf= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_sigma2vz = (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_sigma2vzf= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));
  float *cmemory_dz_sigmaxx 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));  
  float *cmemory_dz_sigmazz 	= (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));  
  float *cmemory_dz_sigmaxz  = (float*)malloc( sizeof(float)*(((NX+1)*(NZ+1))));  
	 
// 1D arrays for the damping profiles
	// float precision, dimension(NX) :: d_x,K_x,alpha_x,a_x,b_x,d_x_half_x,K_x_half_x,alpha_x_half_x,a_x_half_x,b_x_half_x
	// float precision, dimension(NZ) :: d_z,K_z,alpha_z,a_z,b_z,d_z_half_z,K_z_half_z,alpha_z_half_z,a_z_half_z,b_z_half_z

  float *cd_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *cK_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *calpha_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *ca_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *cb_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *cd_x_half_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *cK_x_half_x = (float*)malloc( sizeof(float)*((NX+1)));
  float *calpha_x_half_x = (float*)malloc( sizeof(float)*((NX+1)));  
  float *ca_x_half_x = (float*)malloc( sizeof(float)*((NX+1)));  
  float *cb_x_half_x = (float*)malloc( sizeof(float)*((NX+1)));    
  
  float *cd_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *cK_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *calpha_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *ca_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *cb_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *cd_z_half_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *cK_z_half_z = (float*)malloc( sizeof(float)*((NZ+1)));
  float *calpha_z_half_z = (float*)malloc( sizeof(float)*((NZ+1)));  
  float *ca_z_half_z = (float*)malloc( sizeof(float)*((NZ+1)));  
  float *cb_z_half_z = (float*)malloc( sizeof(float)*((NZ+1)));  
  
  float thickness_PML_x,thickness_PML_z,xoriginleft,xoriginright,zoriginbottom,zorigintop;
  float Rcoef,d0_x,d0_z,xval,zval,abscissa_in_PML;//,abscissa_normalized;
  //float value_dx_vx1,value_dx_vx2,value_dx_vz,value_dx_sigmaxx,value_dx_sigmaxz;
  //float value_dz_vz1,value_dz_vz2,value_dz_vx,value_dz_sigmaxx,value_dz_sigmaxz;
  //float value_dx_sigma2vxf,value_dz_sigma2vzf;

// for the source
  //float a,t,source_term;

// for receivers
 //float precision xspacerec,zspacerec,distval,dist
 //integer, dimension(NREC) :: ix_rec,iz_rec
 //float precision, dimension(NREC) :: xrec,zrec

// for seismograms
  //float precision, dimension(NSTEP,NREC) :: sisvx,sisvz,sisvxf,sisvzf,sisp

  //int i,j,it,irec;

  float Courant_number_bottom,Courant_number_top;//,velocnorm_all;//,max_amplitude;
  float Dispersion_number_bottom,Dispersion_number_top;
  
// float  Courant_number_l1,Courant_number_l2;
  //float  Dispersion_number_l1,Dispersion_number_l2;
  
  
//---
//--- program starts here
//--- 
 // cps_bottom=(-b1_b-sqrt(delta_b))/(2.e0*a1_b);
 // cps_bottom=sqrt(cps_bottom);
 //
 // cps_top=(-b1_t-sqrt(delta_t))/(2.e0*a1_t);
 // cps_top=sqrt(cps_top);
 //
 // cps_l1=(-b1_l1-sqrt(delta_l1))/(2.e0*a1_l1);
 // cps_l1=sqrt(cps_l1);
 //
 // cps_l2=(-b1_l2-sqrt(delta_l2))/(2.e0*a1_l2);
 // cps_l2=sqrt(cps_l2);

 //cout<<"cp_bottom		= "<<cp_bottom<<"\n";
 //cout<<"cps_bottom		= "<<cps_bottom<<"\n";
 //cout<<"cs_bottom		= "<<cs_bottom<<"\n";
 //cout<<"cp_top			= "<<cp_top<<"\n";
 //cout<<"cps_top		= "<<cps_top<<"\n";
 //cout<<"cs_top			= "<<cs_top<<"\n";
 //
 //cout<<"rho_bottom			="<<rho_bottom<<"\n";
 //cout<<"rsm_bottom			="<<rsm_bottom<<"\n";
 //cout<<"rho_top			="<<rho_top<<"\n";
 //cout<<"rsm_top			="<<rsm_top<<"\n";
 //cout<<"rmu_bottom			="<<rmu_bottom<<"\n";
 //cout<<"rlambdac_bottom	="<<rlambdac_bottom<<"\n";
 //cout<<"alpha_bottom		="<<alpha_bottom<<"\n";
 //cout<<"etaokappa_bottom	="<<etaokappa_bottom<<"\n";
 //cout<<"rlambdac_top		="<<rlambdac_top<<"\n";
 //cout<<"rlambdao_top		="<<rlambdao_top<<"\n";
 //cout<<"alpha_top			="<<alpha_top<<"\n";
 //cout<<"rbM_top			="<<rbM_top<<"\n";
 //cout<<"etaokappa_top		="<<etaokappa_top<<"\n";  

 //cout<<"\nDELTAT CPML		="<<DELTAT<<"\n"; 
 //cout<<"==============\n"; 
 //cout<<"2D poroelastic finite-difference code in velocity and stress formulation with C-PML\n"; 
 //cout<<"=============\n"; 
// display size of the model
 //cout<<"\n";
 //cout<<"NX ="<<NX<<"\n";
 //cout<<"NZ ="<<NZ<<"\n";    
 //cout<<"\n";
 //  cout<<"size of the model along X = "<<(NX - 1.0) * DELTAX<<"\n";  
 //  cout<<"size of the model along Z = "<<(NZ - 1.0) * DELTAZ<<"\n";  
 //cout<<"\n";
 //  cout<<"Total number of grid points = "<<NX*NZ<<"\n";  
 //cout<<"\n";

//--- define profile of absorption in PML region

// thickness of the PML layer in meters
  thickness_PML_x = (NPOINTS_PML) * DELTAX;
  thickness_PML_z = (NPOINTS_PML) * DELTAZ;

// reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
  Rcoef = 0.001e0;  
  
    if(NPOWER < 1)
    {
        cout << "\nnpower harus lebih besar dari 1!\n";
        return 0;
    }

         if(HETEROGENEOUS_MODEL)
         {d0_x = - (NPOWER + 1.0) * maxcptop * log(Rcoef) / (2.0 * thickness_PML_x);
		 d0_z = - (NPOWER + 1.0) * maxcptop * log(Rcoef) / (2.0 * thickness_PML_z);
		 cout<<"hiya!/n";}
		  //if(HETEROGENEOUS_MODEL) then
    //d0_x = - (NPOWER + 1) * max(cp_bottom,cp_top) * log(Rcoef) / (2.d0 * thickness_PML_x)
    //d0_z = - (NPOWER + 1) * max(cp_bottom,cp_top) * log(Rcoef) / (2.d0 * thickness_PML_z)
         else
         d0_x = - (NPOWER + 1.0) * maxcpbottom * log(Rcoef) / (2.0 * thickness_PML_x);
         d0_z = - (NPOWER + 1.0) * maxcpbottom * log(Rcoef) / (2.0 * thickness_PML_z);
		
		cout<<"\nd0_x = "<<d0_x<<"\nd0_z = "<<d0_z;

    for(int i=0; i < NX+1; i++)
    {
        cd_x[i] = 0.0;
        cK_x[i] = 1.0;
        cd_x_half_x[i] = 0.0;
        cK_x_half_x[i] = 1.0;
        ca_x[i] = 0.0;
        ca_x_half_x[i] = 0.0;
        cb_x[i] = 0.0;
        cb_x_half_x[i] = 0.0;
        calpha_x[i] = 0.0;
        calpha_x_half_x[i] = 0.0;
    }
    for(int i=0; i < NZ+1; i++)
    {
        cd_z[i] = 0.0;
        cd_z_half_z[i] = 0.0;
        cK_z[i] = 1.0;
        cK_z_half_z[i] = 1.0;
        calpha_z[i] = 0.0;
        calpha_z_half_z[i] = 0.0;
        ca_z[i] = 0.0;
        ca_z_half_z[i] = 0.0;
		cb_z[i] = 0.0;
        cb_z_half_z[i] = 0.0;
		}
	    
	// ================= hitung penyerap pada arah x
    // titik asal lapisan PML
    xoriginleft = thickness_PML_x+(3*DELTAX);
    xoriginright = (NX-2)*DELTAX - thickness_PML_x;
	
	for(int i=0; i < NX+1; i++)
    { //PMLX
		xval = DELTAX * (float)(i);
		//float xval2= DELTAX * (float)(i+1);
		    //PML kiri
			abscissa_in_PML = xoriginleft - xval;
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_x;
                cd_x[i] = d0_x * pow(abscissanorm, NPOWER);
                cK_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
            }
            //hitung profile penyerap di posisi separoh grid
            abscissa_in_PML = xoriginleft - (xval + DELTAX/2.0);
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_x;
                cd_x_half_x[i] = d0_x * pow(abscissanorm, NPOWER);
                cK_x_half_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_x_half_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
            }
			//PML kiri selesai
			
			//PML kanan
            abscissa_in_PML = xval - xoriginright;
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_x;
                cd_x[i] = d0_x * pow(abscissanorm, NPOWER);
                cK_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
            }
            //hitung profile penyerap di posisi separoh grid
            abscissa_in_PML = (xval + DELTAX/2.0) - xoriginright;
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_x;
                cd_x_half_x[i] = d0_x * pow(abscissanorm, NPOWER);
                cK_x_half_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_x_half_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
            }
			
			if(calpha_x[i] < 0.0) calpha_x[i] = 0.0;
				if(calpha_x_half_x[i] < 0.0) calpha_x_half_x[i] = 0.0;
				cb_x[i] = exp(- (cd_x[i] / cK_x[i] + calpha_x[i]) * DELTAT);
				cb_x_half_x[i] = exp(- (cd_x_half_x[i] / cK_x_half_x[i] + calpha_x_half_x[i]) * DELTAT);
			// cegah pembagian dengan nol diluar PML
			if(fabs(cd_x[i]) > 1.e-6) ca_x[i] = cd_x[i] * (cb_x[i] - 1.0) / (cK_x[i] * (cd_x[i] + cK_x[i] * calpha_x[i]));
			if(fabs(cd_x_half_x[i]) > 1.e-6)
            ca_x_half_x[i] = cd_x_half_x[i] * (cb_x_half_x[i] - 1.0) / (cK_x_half_x[i] * (cd_x_half_x[i] + cK_x_half_x[i] * calpha_x_half_x[i]));
			//PML kanan selesai
	} //PMLX

	// ================= hitung penyerap pada arah Z
    // titik asal lapisan PML
    zoriginbottom = (NZ-2)*DELTAZ - thickness_PML_z;  
    zorigintop = thickness_PML_z;    

	for(int i=0; i < NZ+1; i++)
    { //PMLZ
			zval = DELTAZ * (float)i;	

		//PML ATAS JENG JENG
           //define damping profile at the grid points
             abscissa_in_PML = zorigintop- zval;
             if(abscissa_in_PML >= ZERO)
             { float  abscissanorm = abscissa_in_PML / thickness_PML_z;
               cd_z[i] = d0_z * pow(abscissanorm, NPOWER);
         //this taken from Gedney page 8.2
               cK_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
               calpha_z[i] = ALPHA_MAX_PML * (1.0 - abscissanorm) + 0.10 * ALPHA_MAX_PML;
             }
         
         //define damping profile at half the grid points
             abscissa_in_PML = zorigintop - (zval + DELTAZ/2.0);
             if(abscissa_in_PML >= ZERO)
              {float  abscissanorm = abscissa_in_PML / thickness_PML_z;
               cd_z_half_z[i] = d0_z * pow(abscissanorm, NPOWER);
         //this taken from Gedney page 8.2
               cK_z_half_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
               calpha_z_half_z[i]= ALPHA_MAX_PML * (1.0 - abscissanorm) + 0.10 * ALPHA_MAX_PML;
              }

			
			//PML BAWAH
			abscissa_in_PML = zval - zoriginbottom;
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_z;
                cd_z[i] = d0_z * pow(abscissanorm, NPOWER);
                cK_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_z[i] = ALPHA_MAX_PML  * (1.0 - abscissanorm);
            }
            //hitung profile penyerap di posisi separoh grid
            abscissa_in_PML = (zval + DELTAZ/2.0) - zoriginbottom;
            if(abscissa_in_PML >= 0.0)
            {
                float abscissanorm = abscissa_in_PML / thickness_PML_z;
                cd_z_half_z[i] = d0_z * pow(abscissanorm, NPOWER);
                cK_z_half_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
                calpha_z_half_z[i] = ALPHA_MAX_PML  * (1.0 - abscissanorm);
            }
			//PML BAWAH
			
			if(calpha_z[i] < 0.0) calpha_z[i] = 0.0;
			if(calpha_z_half_z[i] < 0.0) calpha_z_half_z[i] = 0.0;
			cb_z[i] = exp(- (cd_z[i] / cK_z[i] + calpha_z[i]) * DELTAT);

			cb_z_half_z[i] = exp(- (cd_z_half_z[i] / cK_z_half_z[i] + calpha_z_half_z[i]) * DELTAT);
			// cegah pembagian dengan nol diluar PML
			if(fabs(cd_z[i]) > 1.e-6) ca_z[i] = cd_z[i] * (cb_z[i] - 1.0) / (cK_z[i] * (cd_z[i] + cK_z[i] * calpha_z[i]));
			if(fabs(cd_z_half_z[i]) > 1.e-6)
            ca_z_half_z[i] = cd_z_half_z[i] * (cb_z_half_z[i] - 1.0) / (cK_z_half_z[i] * (cd_z_half_z[i] + cK_z_half_z[i] * calpha_z_half_z[i]));
	} //PMLZ
	//PMLDEBUG
	 //  ofstream haha("pro.txt"); // output, normal file
     //  for (int ii=0; ii<12; ii++) 
     //  {
     //      //for (int ii=0; ii<NX; ii++) 
     //     // {
     //      //int ij=NX*jj + ii;               
     //      haha<<cK_x_half_x[ii]<<"\n";   
     //      //}
     //      //outseya<<"\n";
     //  }
	 //  cout<<"\n";
	 //  for (int jj=0; jj<NX; jj++) {
	 //  cout<<cK_x_half_x[jj]<<" ";
	 //  }
	 //  ofstream outseyaa("profilporoZ.txt"); // output, normal file
     //  for (int jj=0; jj<NZ; jj++) 
     //  {
     //      //for (int ii=0; ii<NX; ii++) 
     //     // {
     //      //int ij=NX*jj + ii;               
     //      outseyaa<<cK_z[jj]<<" ";   
     //      //}
     //      //outseya<<"\n";
     //  }
	//PMLDEBUGEND
	
	//Parameter end
	
//PARAMETER DEBUG

 
	//PARAMETER DEBUG END

	
	// check the Courant stability condition for the explicit time scheme
	// R. Courant et K. O. Friedrichs et H. Lewy (1928)
	Courant_number_bottom = maxcptop * DELTAT / DELTAX;
	Dispersion_number_bottom=maxcpbottom/(2.50*f0*DELTAX);
//	cout<<"\nCourant number at the bottom is : "<<Courant_number_bottom<<"\n";
//	cout<<"Dispersion number at the bottom is : "<<Dispersion_number_bottom<<"\n";

	if(Courant_number_bottom > 1.0/sqrt(2.0))
	{cout<<"'time step is too large, simulation will be unstable'"; return 0; }	
	
	if(HETEROGENEOUS_MODEL)
		Courant_number_top = maxcptop * DELTAT / DELTAX;
		Dispersion_number_top=800.0/(2.50*f0*DELTAX);

//	cout<<"Courant number at the top is : "<<Courant_number_top<<"\n";
//	cout<<"Dispersion number at the top is : "<<Dispersion_number_top<<"\n";		
	if(Courant_number_top > 6.0/7.0/sqrt(2.0))
	{cout<<"'time step is too large, simulation will be unstable'"; return 0; }
	
	// initialize arrays
    for(int i=0; i < (NX+1)*(NZ+1); i++)	
    {	
	cvx[i] = ZERO;
	cvz[i] = ZERO;
	csigmaxx[i] = ZERO;
	csigmazz[i] = ZERO;
	csigmaxz[i] = ZERO;
	csigma2[i] = ZERO;
	calp_sigma2[i] = ZERO;
	cgamma11[i]=ZERO;
	cgamma22[i]=ZERO;
	cgamma12_1[i]=ZERO;
	cgamma12_1[i]=ZERO;
	cxi_1[i]=ZERO;
	cxi_2[i]=ZERO;
	cvxf[i] = ZERO;
	cvzf[i] = ZERO;
	
    cmemory_dx_vx1[i]=ZERO;
    cmemory_dx_vx2[i]=ZERO;
    cmemory_dz_vx[i]=ZERO;
    cmemory_dx_vz[i]=ZERO;
    cmemory_dz_vz1[i]=ZERO;
    cmemory_dz_vz2[i]=ZERO;
    cmemory_dx_sigmaxx[i]=ZERO;
    cmemory_dx_sigmazz[i]=ZERO;
    cmemory_dx_sigmaxz[i]=ZERO;
    cmemory_dx_sigma2vx[i]=ZERO;
    cmemory_dx_sigma2vxf[i]=ZERO;
    cmemory_dz_sigmaxx[i]=ZERO;
    cmemory_dz_sigmazz[i]=ZERO;
    cmemory_dz_sigmaxz[i]=ZERO;
    cmemory_dz_sigma2vz[i]=ZERO;
    cmemory_dz_sigma2vzf[i]=ZERO;
	}

	float *sseismo=(float*)malloc(sizeof(float)*(panj)*NSTEP/samprek);
	float *seismo;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo, (panj)*NSTEP/samprek* sizeof(float)));
	
	float *sdeismo=(float*)malloc(sizeof(float)*(panj)*NSTEP/samprek);
	float *deismo;
	HANDLE_ERROR( cudaMalloc( (void**)&deismo, (panj)*NSTEP/samprek* sizeof(float)));

	float *sseismo1=(float*)malloc(sizeof(float)*(panj2)*NSTEP/samprek);
	float *seismo1;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo1, (panj2)*NSTEP/samprek* sizeof(float)));
	
	float *sdeismo1=(float*)malloc(sizeof(float)*(panj2)*NSTEP/samprek);
	float *deismo1;
	HANDLE_ERROR( cudaMalloc( (void**)&deismo1, (panj2)*NSTEP/samprek* sizeof(float)));
	
	float *sseismo2=(float*)malloc(sizeof(float)*(1)*NSTEP);
	float *seismo2;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo2, (1)*NSTEP*sizeof(float)));
	
	float *sseismo3=(float*)malloc(sizeof(float)*(1)*NSTEP);
	float *seismo3;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo3, (1)*NSTEP*sizeof(float)));

	float *sseismo4=(float*)malloc(sizeof(float)*(1)*NSTEP);
	float *seismo4;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo4, (1)*NSTEP*sizeof(float)));

	float *sseismo0=(float*)malloc(sizeof(float)*(1)*NSTEP);
	float *seismo0;
	HANDLE_ERROR( cudaMalloc( (void**)&seismo0, (1)*NSTEP*sizeof(float)));	

	
	for (int ij=0; ij<(panj)*NSTEP/samprek; ij++)
	{		
		sseismo[ij]=0.0;
		sdeismo[ij]=0.0;
	}
	for (int ij=0; ij<(panj2)*NSTEP/samprek; ij++)
	{	
	sseismo1[ij]=0.0; 
	sdeismo1[ij]=0.0; 
	}
	for (int ij=0; ij<NSTEP; ij++)
	{	
	sseismo2[ij]=0.0; sseismo3[ij]=0.0; sseismo4[ij]=0.0; sseismo0[ij]=0.0; 
	}
	HANDLE_ERROR( cudaMemcpy( seismo, sseismo,sizeof(float)*((panj)*NSTEP/samprek),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( deismo, sdeismo,sizeof(float)*((panj)*NSTEP/samprek),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( seismo1, sseismo1,sizeof(float)*((panj2)*NSTEP/samprek),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( deismo1, sseismo1,sizeof(float)*((panj2)*NSTEP/samprek),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( seismo2, sseismo2,sizeof(float)*((1)*NSTEP),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( seismo3, sseismo3,sizeof(float)*((1)*NSTEP),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( seismo4, sseismo4,sizeof(float)*((1)*NSTEP),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( seismo0, sseismo0,sizeof(float)*((1)*NSTEP),cudaMemcpyHostToDevice ) );	
	
	//CPU2GPU
	int *NNX;
	int *NNZ;
	int *is;
	int *js;
	float *DDELTAX;
	float *DDELTAZ;

	float *DDELTAT;

	HANDLE_ERROR( cudaMalloc( (void**)&NNX, sizeof(int)));
	HANDLE_ERROR( cudaMalloc( (void**)&NNZ, sizeof(int)));
    HANDLE_ERROR( cudaMalloc( (void**)&is, sizeof(int)));
	HANDLE_ERROR( cudaMalloc( (void**)&js, sizeof(int)));
    HANDLE_ERROR( cudaMalloc( (void**)&DDELTAX, sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&DDELTAZ, sizeof(float)));

	HANDLE_ERROR( cudaMalloc( (void**)&DDELTAT, sizeof(float)));

	HANDLE_ERROR( cudaMemcpy( NNX, &NX,sizeof(int),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( NNZ, &NZ,sizeof(int),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( is, &ISOURCE,sizeof(int),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( js, &JSOURCE,sizeof(int),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( DDELTAX, &DELTAX,sizeof(float),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( DDELTAZ, &DELTAX,sizeof(float),cudaMemcpyHostToDevice ) );		
		
	HANDLE_ERROR( cudaMemcpy( DDELTAT, &DELTAT,sizeof(float),cudaMemcpyHostToDevice ) );
	
	float *sigmaxz	; 
	float *sigmaxx	; 
	float *sigmazz	; 
	float *vz		; 
	float *vx		; 
	float *sigma2	; 
	float *alp_sigma2;
	float *gamma11	; 
	float *gamma22	; 
	float *gamma12_1; 
	//float *gamma12_1; 
	float *vxf		; 
	float *vzf		; 
		float *xi_1		; 
	float *xi_2		; 
	
	HANDLE_ERROR( cudaMalloc( (void**)&sigmaxz	 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&sigmaxx	 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&sigmazz	 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&vz		 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&vx		 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&sigma2	 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&alp_sigma2, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&gamma11	 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&gamma22	 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&gamma12_1 ,  (NX+1)*(NZ+1) * sizeof(float)));
    //HANDLE_ERROR( cudaMalloc( (void**)&gamma12_1,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&vxf		 , (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&vzf		 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&xi_1		 , (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&xi_2		 , (NX+1)*(NZ+1) * sizeof(float)));
	
	HANDLE_ERROR( cudaMemcpy( sigmaxz	, csigmaxz	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sigmaxx	, csigmaxx	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( sigmazz	, csigmazz	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( vz		, cvz		,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );                          
	HANDLE_ERROR( cudaMemcpy( vx		, cvx		,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( sigma2	, csigma2	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( alp_sigma2, calp_sigma2,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( gamma11	, cgamma11	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( gamma22	, cgamma22	,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( gamma12_1,  cgamma12_1,sizeof(float)*  (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );	
	//HANDLE_ERROR( cudaMemcpy( gamma12_1,  cgamma12_1,sizeof(float)*  (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( vxf		, cvxf		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( vzf		, cvzf		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( xi_1		, cxi_1		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( xi_2		, cxi_2		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );	
	float *memory_dx_vx1		;	
	float *memory_dx_vx2		;	
	float *memory_dz_vx 		;	
	float *memory_dx_vz 		;	
	float *memory_dz_vz1		;	
	float *memory_dz_vz2		;	
	float *memory_dx_sigmaxx ;
	float *memory_dx_sigmazz ;
	float *memory_dx_sigmaxz  ;
	float *memory_dx_sigma2vx ;
	float *memory_dx_sigma2vxf;
	float *memory_dz_sigmaxx	;	
	float *memory_dz_sigmazz	;	
	float *memory_dz_sigmaxz	;	
	float *memory_dz_sigma2vz;
	float *memory_dz_sigma2vzf;
	
	HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_vx1, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_vx2, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_vx , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_vz ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_vz1,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_vz2, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_sigmaxx 	, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_sigmazz 	, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_sigmaxz  	,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_sigma2vx 	,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dx_sigma2vxf	,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_sigmaxx	, (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_sigmazz	, (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_sigmaxz	,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_sigma2vz	, (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&memory_dz_sigma2vzf	, (NX+1)*(NZ+1) * sizeof(float)));

	HANDLE_ERROR( cudaMemcpy( memory_dx_vx1			,cmemory_dx_vx1			,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_vx2			,cmemory_dx_vx2			,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dz_vx 			,cmemory_dz_vx 			,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_vz 			,cmemory_dx_vz 			,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );                          
	HANDLE_ERROR( cudaMemcpy( memory_dz_vz1			,cmemory_dz_vz1			,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( memory_dz_vz2			,cmemory_dz_vz2			,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_sigmaxx 	,cmemory_dx_sigmaxx 	,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_sigmazz 	,cmemory_dx_sigmazz 	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_sigmaxz  	,cmemory_dx_sigmaxz  	,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dx_sigma2vx 	,cmemory_dx_sigma2vx 	,sizeof(float)*  (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( memory_dx_sigma2vxf	,cmemory_dx_sigma2vxf	,sizeof(float)*  (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( memory_dz_sigmaxx		,cmemory_dz_sigmaxx		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dz_sigmazz		,cmemory_dz_sigmazz		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( memory_dz_sigmaxz		,cmemory_dz_sigmaxz		,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( memory_dz_sigma2vz	,cmemory_dz_sigma2vz	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );		
	HANDLE_ERROR( cudaMemcpy( memory_dz_sigma2vzf	,cmemory_dz_sigma2vzf	,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );	
	
	float *rho		;  	
	float *rhof	  	;	
	float *rsm		;  	
	float *rmu		;  	
	float *rlambdac ; 	
	float *rbM		;  	
	float *alpha	;	 	
	float *etaokappa;  	
	float *rlambdao ;   	
	
    HANDLE_ERROR( cudaMalloc( (void**)&rho		 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&rhof	 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&rsm		 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&rmu		 ,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&rlambdac , (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&rbM		 , (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&alpha	,  (NX+1)*(NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&etaokappa, (NX+1)*(NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&rlambdao , (NX+1)*(NZ+1) * sizeof(float)));	
	
	HANDLE_ERROR( cudaMemcpy( rho		  	,crho		  ,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( rhof	  		,crhof	   ,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( rsm		  	,crsm		  ,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( rmu		  	,crmu		  ,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );                          
	HANDLE_ERROR( cudaMemcpy( rlambdac  	,crlambdac  ,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( rbM		  	,crbM		  ,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( alpha		 	,calpha		,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( etaokappa  	,cetaokappa ,sizeof(float)*(NX+1)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( rlambdao    	,crlambdao  ,sizeof(float)* (NX+1)*(NZ+1),cudaMemcpyHostToDevice ) );
	
//	HANDLE_ERROR( cudaMemcpy( crho		, rho		,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crhof	    , rhof	  	,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crsm		, rsm		,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crmu		, rmu		,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crlambdac , rlambdac ,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crbM		, rbM		,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( calpha	, alpha		,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( cetaokappa, etaokappa,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//	HANDLE_ERROR( cudaMemcpy( crlambdao , rlambdao ,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
//


				//std::ofstream outvz(nmfile); // output, normal file
				FILE* file11 = fopen ("mediumrho.ctxt", "wb");
				for (int jj=0; jj<NZ+1; jj++) 
				{
					for (int ii=0; ii<NX+1; ii++) 
					{
					int ij=(NX+1)*jj + ii;           
					//outvz<<cvz[ij]<<" ";
					float f2 = cp[ij];
					fwrite(&f2, sizeof(float), 1, file11);						
					}
				//outvz<<"\n";
				}
				fclose(file11);


//ofstream outseya11("mediumrho.ctxt"); // output, normal file
//for (int jj=0; jj<NZ+1; jj++) 
//{
//    for (int ii=0; ii<NX+1; ii++) 
//    {
//    int ij=(NX+1)*jj + ii;               
//    outseya11<<cp[ij]<<" ";   
//    }
//   outseya11<<"\n";
//}
//
//if ((f0>4.0)&&(f0<6.0)) 
//{
//ofstream outseya11("mediumcp.ctxt"); // output, normal file
//for (int jj=0; jj<NZ+1; jj++) 
//{
//    for (int ii=0; ii<NX+1; ii++) 
//    {
//    int ij=(NX+1)*jj + ii;               
//    outseya11<<cp[ij]<<" ";   
//    }
//    outseya11<<"\n";
//}
//}

//
//ofstream outseya111("mediumcs.ctxt"); // output, normal file
//for (int jj=0; jj<NZ+1; jj++) 
//{
//    for (int ii=0; ii<NX+1; ii++) 
//    {
//    int ij=(NX+1)*jj + ii;               
//    outseya111<<cs[ij]<<" ";   
//    }
//    outseya111<<"\n";
//}

	
	float *d_x			; 
	float *K_x			; 
	float *d_x_half_x	; 
	float *K_x_half_x	; 
	float *a_x			; 
	float *a_x_half_x	; 
	float *b_x			; 
	float *b_x_half_x	; 
	float *alpha_x		; 
	float *alpha_x_half_x; 
	float *d_z       	 ;
	float *d_z_half_z	 ;
	float *K_z			 ;
	float *K_z_half_z	 ;
	float *alpha_z		 ;
	float *alpha_z_half_z;
	float *a_z			 ;
	float *a_z_half_z	 ;
	float *b_z			 ;
	float *b_z_half_z	 ;
	
    HANDLE_ERROR( cudaMalloc( (void**)&d_x				 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&K_x				 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&d_x_half_x		 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&K_x_half_x		 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&a_x				 , (NX+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&a_x_half_x		 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&b_x				 , (NX+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&b_x_half_x		 , (NX+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&alpha_x			 , (NX+1) * sizeof(float)));		
	HANDLE_ERROR( cudaMalloc( (void**)&alpha_x_half_x    , (NX+1) * sizeof(float)));	
	
	HANDLE_ERROR( cudaMemcpy( d_x				 ,cd_x				 ,sizeof(float)*(NX+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( K_x				 ,cK_x				 ,sizeof(float)*(NX+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( d_x_half_x		 ,cd_x_half_x		 ,sizeof(float)*(NX+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( K_x_half_x		 ,cK_x_half_x		 ,sizeof(float)*(NX+1),cudaMemcpyHostToDevice ) );                          
	HANDLE_ERROR( cudaMemcpy( a_x				 ,ca_x				 ,sizeof(float)*(NX+1),cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( a_x_half_x		 ,ca_x_half_x		 ,sizeof(float)*(NX+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( b_x				 ,cb_x				 ,sizeof(float)*(NX+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( b_x_half_x		 ,cb_x_half_x		 ,sizeof(float)*(NX+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( alpha_x			 ,calpha_x			 ,sizeof(float)*(NX+1),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( alpha_x_half_x     ,calpha_x_half_x    ,sizeof(float)*(NX+1),cudaMemcpyHostToDevice ) );	
	
    HANDLE_ERROR( cudaMalloc( (void**)&d_z       		, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&d_z_half_z		, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&K_z				, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&K_z_half_z		, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&alpha_z			, (NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&alpha_z_half_z	, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&a_z				, (NZ+1) * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&a_z_half_z		, (NZ+1) * sizeof(float)));
	HANDLE_ERROR( cudaMalloc( (void**)&b_z				, (NZ+1) * sizeof(float)));		
	HANDLE_ERROR( cudaMalloc( (void**)&b_z_half_z		, (NZ+1) * sizeof(float)));	
	
	HANDLE_ERROR( cudaMemcpy( d_z       			 ,cd_z       			 ,sizeof(float)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( d_z_half_z			 ,cd_z_half_z			 ,sizeof(float)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( K_z				     ,cK_z				 	 ,sizeof(float)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( K_z_half_z			 ,cK_z_half_z			 ,sizeof(float)*(NZ+1),cudaMemcpyHostToDevice ) );                          
	HANDLE_ERROR( cudaMemcpy( alpha_z			 	 ,calpha_z				 ,sizeof(float)*(NZ+1),cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( alpha_z_half_z		 ,calpha_z_half_z		 ,sizeof(float)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( a_z				 	 ,ca_z				 	 ,sizeof(float)*(NZ+1),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( a_z_half_z			 ,ca_z_half_z			 ,sizeof(float)*(NZ+1) ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( b_z				 	 ,cb_z					 ,sizeof(float)*(NZ+1),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( b_z_half_z		     ,cb_z_half_z		     ,sizeof(float)*(NZ+1),cudaMemcpyHostToDevice ) );	
	

	                     
	//main calculation
		int xthread=10;
		int ythread=10;
		dim3    blocks((NX+1)/xthread,(NZ+1)/ythread);
		dim3    threads(xthread,ythread);
		cudaEvent_t start, stop;
		float time;
		float totalwaktu;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		totalwaktu=0.0;
		printf("\nSimulasi numerik di mulai : \n");
		float *i_gpu;
		float tempgpu=0.0;
		
        HANDLE_ERROR( cudaMalloc( (void**)&i_gpu, sizeof(float)));		
		HANDLE_ERROR( cudaMemcpy( i_gpu, &tempgpu,sizeof(float),cudaMemcpyHostToDevice ) );	

		
		for(int ishot=1; ishot < 2; ishot++)
		{

        
	    int i_steps=0;
		for(int it=0; it < NSTEP/samp; it++)
		{
		//totalwaktu=0.0;
		//printf("\nShot %i \n",sht);
		cudaEventRecord(start, 0);
		for(int ix=0; ix < samp; ix++)
		{
			i_steps++;	
			stress1_kernel<<<blocks,threads>>>(i_gpu, NNX, NNZ, is, js,ff0, DDELTAX, DDELTAZ, DDELTAT, vx, vz, memory_dx_sigmaxx, memory_dz_sigmaxx, b_x_half_x,a_x_half_x,b_z,a_z,gamma11,gamma22,K_x_half_x,K_z,vxf,vzf,xi_1,xi_2, memory_dx_sigma2vxf, memory_dz_sigma2vzf, sigma2, alpha, rbM);
			save1_kernel<<<blocks,threads>>>(a0_pos,a1_pos,panj2,panj,samprek,i_steps,seismo,deismo,seismo1,deismo1,seismo2,seismo3,seismo4,seismo0,NNX, NNZ, is, js,vx, vz);
			//int a0_pos,int a1_pos,int panj2,int panj,int samprek,int i,int *NNX, int *NNZ, int *is, int *js,float *vz
			stress2_kernel<<<blocks,threads>>>(NNX,NNZ, DDELTAX, DDELTAZ, DDELTAT, vx, vz, rmu,memory_dx_sigmaxz,memory_dz_sigmaxz,b_x,a_x,b_z_half_z,a_z_half_z,sigmaxz,K_x,K_z);
		    stress3_kernel<<<blocks,threads>>>(samprek,i_steps,seismo,seismo1,seismo2,seismo3,seismo4,seismo0,i_gpu, NNX, NNZ, is, js,ff0, DDELTAX, DDELTAZ, DDELTAT,sigmaxx,sigmazz,rlambdac,rmu,gamma11,gamma22,alpha,sigma2);
			vx_kernel<<<blocks,threads>>>(NNX, NNZ, DDELTAX, DDELTAZ, DDELTAT, vx, rho,rsm,rhof,etaokappa,vxf,sigmaxx,sigma2,sigmaxz,memory_dx_vx1,memory_dx_vx2,memory_dz_vx,b_x,a_x,a_z,b_z,K_x,K_z);
			vz_kernel<<<blocks,threads>>>(i_gpu, NNX, NNZ, is, js,ff0, DDELTAX, DDELTAZ, DDELTAT,vz, rho,rsm,rhof,etaokappa,vzf,sigmaxz,sigmazz,sigma2, memory_dz_vz1,memory_dz_vz2,memory_dx_vz,b_x_half_x,a_x_half_x,a_z_half_z,b_z_half_z,K_x_half_x,K_z_half_z);
			
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);	
		cudaEventElapsedTime(&time, start, stop);
		
		
		HANDLE_ERROR( cudaMemcpy( cvz, vz,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( cvx, vx,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( cvzf, vzf,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( cvxf, vxf,sizeof(float)*((NX+1)*(NZ+1)),cudaMemcpyDeviceToHost ) );
		
		totalwaktu=totalwaktu+(time/1000.0);
		printf( "\nFrekuensi=%f hz;   Step =  %5d ; Waktu perhitungan = %5.2f s \n",f0, i_steps, totalwaktu);  
		float maxvz=min_element(cvz,((NX+1)*(NZ+1)));
		cout<<"\nmax velocity : "<<maxvz; 
		cout<<"\nwaktu simulasi : "<<i_steps*DELTAT;
						//outvx.close();
				if ((ishot%11==0) || (ishot==1)) 
				{
				char nmfile[50];
				sprintf(nmfile, "vz%05i.vzo",i_steps  );
				//std::ofstream outvz(nmfile); // output, normal file
				FILE* file1 = fopen (nmfile, "wb");
				for (int jj=0; jj<NZ+1; jj++) 
				{
					for (int ii=0; ii<NX+1; ii++) 
					{
					int ij=(NX+1)*jj + ii;           
					//outvz<<cvz[ij]<<" ";
					float f2 = cvz[ij];
					fwrite(&f2, sizeof(float), 1, file1);						
					}
				//outvz<<"\n";
				}
				fclose(file1);
				
            //
            //
			//	////fclose(file2);
			//	char nmfile2[50];
			//	sprintf(nmfile2, "vxf%05i.vxf",i_steps  );
			//	//std::ofstream outvz2(nmfile2); // output, normal file
			//	FILE* file2 = fopen (nmfile2, "wb");
			//	for (int jj=0; jj<NZ+1; jj++) 
			//	{
			//		for (int ii=0; ii<NX+1; ii++) 
			//		{
			//		int ij=(NX+1)*jj + ii;           
			//		//outvz2<<cvx[ij]<<" ";
			//		float f2 = cvxf[ij];
			//		fwrite(&f2, sizeof(float), 1, file2);						
			//		}
			//	//outvz2<<"\n";
			//	}
			//	fclose(file2);
			//	
			//	char nmfile3[50];
			//	sprintf(nmfile3, "vzf%05i.vzf",i_steps );
			//	//std::ofstream outvz(nmfile); // output, normal file
			//	FILE* file3 = fopen (nmfile3, "wb");
			//	for (int jj=0; jj<NZ+1; jj++) 
			//	{
			//		for (int ii=0; ii<NX+1; ii++) 
			//		{
			//		int ij=(NX+1)*jj + ii;           
			//		//outvz<<cvz[ij]<<" ";
			//		float f2 = cvzf[ij];
			//		fwrite(&f2, sizeof(float), 1, file3);						
			//		}
			//	//outvz<<"\n";
			//	}
			//	fclose(file3);
			//	
            //
            //
			//	////fclose(file2);
				char nmfile4[50];
				sprintf(nmfile4, "vx%05i.vxo",i_steps  );
				//std::ofstream outvz2(nmfile2); // output, normal file
				FILE* file4 = fopen (nmfile4, "wb");
				for (int jj=0; jj<NZ+1; jj++) 
				{
					for (int ii=0; ii<NX+1; ii++) 
					{
					int ij=(NX+1)*jj + ii;           
					//outvz2<<cvx[ij]<<" ";
					float f2 = cvx[ij];
					fwrite(&f2, sizeof(float), 1, file4);						
					}
				//outvz2<<"\n";
				}
				fclose(file4);	
				
				}
				
		}
				//HANDLE_ERROR( cudaMemcpy( &ISOURCE, is,sizeof(float),cudaMemcpyDeviceToHost ) );
				//ISOURCE=ISOURCE+20;
				printf("posisi shot : X= %i Y=%i \n",ISOURCE,JSOURCE);
				//HANDLE_ERROR( cudaMemcpy( is, &ISOURCE,sizeof(float),cudaMemcpyHostToDevice ) );
				zero_kernel<<<blocks,threads>>>( memory_dx_sigmaxx,memory_dz_sigmaxx,gamma11,gamma22,memory_dx_sigma2vxf,memory_dz_sigma2vzf,xi_1,xi_2,sigma2,memory_dx_sigmaxz,memory_dz_sigmaxz,sigmaxz,sigmaxx,sigmazz,memory_dx_vx1 ,memory_dx_vx2 ,memory_dz_vx  	,vxf,vx,vz ,vzf ,i_gpu,memory_dx_vz ,memory_dz_vz1,memory_dz_vz2);
				HANDLE_ERROR( cudaMemcpy( sseismo, seismo,sizeof(float)*((panj)*NSTEP/samprek),cudaMemcpyDeviceToHost ) );
				HANDLE_ERROR( cudaMemcpy( sdeismo, deismo,sizeof(float)*((panj)*NSTEP/samprek),cudaMemcpyDeviceToHost ) );
				char nmfile3[100]; char nmfile6[100]; char nmfile6d[100]; char nmfile7[100]; char nmfile7d[100];
				printf("\nmenyimpan file... \n");

				sprintf(nmfile6, "recvspvz%08i.vsp",int(f0));
				sprintf(nmfile6d, "recvspvx%08i.vsp",int(f0));
				sprintf(nmfile7, "rechorvz%08i.hor",int(f0));	
				sprintf(nmfile7d, "rechorvx%08i.hor",int(f0));


				if(f0 <1.0)
				{
				sprintf(nmfile6, "recvspvz%08i.vsp",int(f0));
				sprintf(nmfile6d, "recvspvx%08i.vsp",int(f0));
				sprintf(nmfile7, "rechorvz%08i.hor",int(f0));	
				sprintf(nmfile7d, "rechorvx%08i.hor",int(f0));		
						
				}	
				
				sprintf(nmfile3, "seismohor%4.4f_%ix%ipos%iiter%08i.sxt",f0,NX+1,NZ+1,ISOURCE,i_steps );
				//std::ofstream outvz3(nmfile3); // output, normal file
				//FILE* file3 = fopen (nmfile3, "wb");
				//for (int jj=0; jj<NSTEP/samprek; jj++) 
				//{
				//	for (int ii=0; ii<NZ+1; ii++) 
				//	{
				//	int ij=(NZ+1)*jj + ii;           
				//	//outvz3<<sseismo[ij]<<" ";
				//	float f3 = sseismo[ij];
				//	fwrite(&f3, sizeof(float), 1, file3);						
				//	}
				////outvz3<<"\n";
				//}
				//fclose(file3);
				ofstream outvz3(nmfile6); // output, normal file
				for (int jj=0; jj<NSTEP/samprek; jj++) 
				{
					for (int ii=0; ii<panj; ii++) 
					{
					int ij=(panj)*jj + ii;      
					outvz3<<sseismo[ij]<<" ";  
					}
				outvz3<<"\n";
				}		
				
				ofstream outvz3d(nmfile6d); // output, normal file
				for (int jj=0; jj<NSTEP/samprek; jj++) 
				{
					for (int ii=0; ii<panj; ii++) 
					{
					int ij=(panj)*jj + ii;      
					outvz3d<<sdeismo[ij]<<" ";  
					}
				outvz3d<<"\n";
				}
				
				
				
				HANDLE_ERROR( cudaMemcpy( sseismo1, seismo1,sizeof(float)*((panj2)*NSTEP/samprek),cudaMemcpyDeviceToHost ) );
				HANDLE_ERROR( cudaMemcpy( sdeismo1, deismo1,sizeof(float)*((panj2)*NSTEP/samprek),cudaMemcpyDeviceToHost ) );
				//char nmfile4[50];
				//sprintf(nmfile4, "seismo1%05i_%ix%iiter%08i.sxt",ishot,NX+1,NZ+1,i_steps );
				////std::ofstream outvz3(nmfile3); // output, normal file
				//FILE* file4 = fopen (nmfile4, "wb");
				//for (int jj=0; jj<NSTEP; jj++) 
				//{
				//	float f4 = sseismo1[ij];
				//	fwrite(&f4, sizeof(float), 1, file4);						
				////outvz3<<"\n";
				//}
				//fclose(file4);
                //
				HANDLE_ERROR( cudaMemcpy( sseismo2, seismo2,sizeof(float)*(NSTEP),cudaMemcpyDeviceToHost ) );
				//char nmfile5[50];
				//sprintf(nmfile5, "seismo2%05i_%ix%iiter%08i.sxt",ishot,NX+1,NZ+1,i_steps );
				////std::ofstream outvz3(nmfile3); // output, normal file
				//FILE* file5 = fopen (nmfile5, "wb");
				//for (int jj=0; jj<NSTEP; jj++) 
				//{
				//	float f5 = sseismo2[ij];
				//	fwrite(&f5, sizeof(float), 1, file5);						
				////outvz3<<"\n";
				//}
				//fclose(file5);
                //
				HANDLE_ERROR( cudaMemcpy( sseismo3, seismo3,sizeof(float)*(NSTEP),cudaMemcpyDeviceToHost ) );
				//char nmfile6[50];
				//sprintf(nmfile6, "seismo3%05i_%ix%iiter%08i.sxt",ishot,NX+1,NZ+1,i_steps );
				////std::ofstream outvz3(nmfile3); // output, normal file
				//FILE* file6 = fopen (nmfile6, "wb");
				//for (int jj=0; jj<NSTEP; jj++) 
				//{
				//	float f6 = sseismo3[ij];
				//	fwrite(&f6, sizeof(float), 1, file6);						
				////outvz3<<"\n";
				//}
				//fclose(file6);	
				HANDLE_ERROR( cudaMemcpy( sseismo4, seismo4,sizeof(float)*(NSTEP),cudaMemcpyDeviceToHost ) );
				HANDLE_ERROR( cudaMemcpy( sseismo0, seismo0,sizeof(float)*(NSTEP),cudaMemcpyDeviceToHost ) );
				//HANDLE_ERROR( cudaMemcpy( stzz, tzz,sizeof(float)*(NSTEP),cudaMemcpyDeviceToHost ) );
				sprintf(nmfile3, "seismovsp%4.4f_%ix%ipos%iiter%08i.sxt",f0,NX+1,NZ+1,ISOURCE,i_steps );
				//std::ofstream outvz3(nmfile3); // output, normal file
				//FILE* file3 = fopen (nmfile3, "wb");
				//for (int jj=0; jj<NSTEP/samprek; jj++) 
				//{
				//	for (int ii=0; ii<NZ+1; ii++) 
				//	{
				//	int ij=(NZ+1)*jj + ii;           
				//	//outvz3<<sseismo[ij]<<" ";
				//	float f3 = sseismo[ij];
				//	fwrite(&f3, sizeof(float), 1, file3);						
				//	}
				////outvz3<<"\n";
				//}
				//fclose(file3);
				ofstream outseya1(nmfile7); // output, normal file
				for (int jj=0; jj<NSTEP/samprek; jj++) 
				{
					for (int ii=0; ii<panj2; ii++) 
					{
					int ij=(panj2)*jj + ii;      
					outseya1<<sseismo1[ij]<<" ";  
					}
				outseya1<<"\n";
				}	
				
				ofstream outseya1d(nmfile7d); // output, normal file
				for (int jj=0; jj<NSTEP/samprek; jj++) 
				{
					for (int ii=0; ii<panj2; ii++) 
					{
					int ij=(panj2)*jj + ii;      
					outseya1d<<sdeismo1[ij]<<" ";  
					}
				outseya1d<<"\n";
				}	
				
			 char nmfile4[100]; char nmfile5[100]; 
				
				sprintf(nmfile3, "recsrc%08i.src",int(f0));	
				sprintf(nmfile4, "recats%08i.ats",int(f0));		
				sprintf(nmfile5, "recbwh%08i.bwh",int(f0));		

				if(f0 <1.0)
				{
				sprintf(nmfile3, "recsrc%08f.src",(f0));	
				sprintf(nmfile4, "recats%08f.ats",(f0));		
				sprintf(nmfile5, "recbwh%08f.bwh",(f0));						
				}	
	

				ofstream outseya2(nmfile4); // output, normal file
				for (int ij=0; ij<NSTEP; ij++) 
				{            
					outseya2<<sseismo2[ij]<<" ";   
				}

				ofstream outseya3(nmfile5); // output, normal file
				for (int ij=0; ij<NSTEP; ij++) 
				{            
					outseya3<<sseismo3[ij]<<" ";   
				}		

	      		ofstream outseya4(nmfile3); // output, normal file
	      		for (int ij=0; ij<NSTEP; ij++) 
	      		{            
	      			outseya4<<sseismo4[ij]<<" ";   
	      		}		
	//			sprintf(nmfile3, "seismo%fhz_50.sxt",f0 );
	//			ofstream outseya0(nmfile3); // output, normal file
	//			for (int ij=0; ij<NSTEP; ij++) 
	//			{            
	//				outseya0<<sseismo0[ij]<<" ";   
	//			}		
				//ofstream outseya5("tzz.ctxt"); // output, normal file
				//for (int ij=0; ij<NSTEP; ij++) 
				//{            
				//	outseya5<<stzz[ij]<<" ";   
				//}	
	}
	
	// END MAIN CALCULATION

	
}
