#include "conio.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <string>
#define PI 3.1415926535897932f
#define nilai_maks 1.0e30
#define batas_kestabilan 1.0e25
#define d_pml 10.0
#define faktor 1.0e10

using namespace std;

float max_element(float a[], int lng)
{
	float mx = a[0];
	for (int i = 1; i<lng; i++)
	{
		if (mx<a[i])
		{
			mx = a[i];
		}
	}
	return mx;
}

float min_element(float a[], int lng)
{
	float mx = a[0];
	for (int i = 1; i<lng; i++)
	{
		if (mx>a[i])
		{
			mx = a[i];
		}
	}
	return mx;
}

int _tmain(int argc, _TCHAR* argv[])
{
	std::ifstream in_file1("input1.txt");
	if (!in_file1.is_open())
	{
		printf("File not opened...");
		return 1;
	}

	std::ifstream in_file2("input2.txt");
	if (!in_file2.is_open())
	{
		printf("File not opened...");
		return 1;
	}

	int NX = 13599;
	int NZ = 2439;

	std::string line;
	//size of a grid cell
	float DELTAX = 1.25;
	float DELTAZ = DELTAX;
	int panj = 122; //vsp
	int panj2 = 680; //hor
	cout << panj << endl;
	int NSTEP;
	in_file1 >> NSTEP;
	int samp = 5000;
	int samprek = 1;
	//time step in seconds
	float DELTAT = 95e-06;

	float factor = 1.e02;
	// thickness of the PML layer in grid points
	int NPOINTS_PML = 10;

	bool HETEROGENEOUS_MODEL = true;
	float INTERFACE_HEIGHT = (NZ*DELTAZ) / 2.0 + NPOINTS_PML*DELTAZ;

	float *cp = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cs = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *Kb = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *Ks = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *Kf = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *Ksat = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *eta = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *kappa = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *phi = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *a = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crhos = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crhof = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *medium = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crho = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	// float *crhof = (float*)malloc( sizeof(float)*((NX+1)*(NZ+1)));
	float *crsm = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crmu = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crlambdac = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crbM = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *calpha = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cetaokappa = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *crlambdao = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));

	float kfm[] = { 2.4e09, 2.4e09, 0.7e09, 0.02e09 };// Bulk Moduli of Fluid (GPa) from Batzle and Wang, 1992 and Quintal, B., (2012)
	float etam[] = { 0.003, 0.003, 0.03, 1.0e-5 };//viscosity (Pa.s) from Batzle and Wang (1992) and Quintal, B., (2012)
	float kappam[] = { 1.5e-17, 500.0e-15, 500.0e-15, 500.0e-15 }; //permeability (1D=0.986923e-12 m2) from dutta(1983) for shale kappa=0.087x phi^6
	float am[] = { 2.0, 2.0, 2.0, 2.0 };// Tortousity from Wenzlau, F., 2009, Disertation Uni Berli
	//Kondisi poroelastik
	//float phim  []=   {0.01   ,0.25      ,0.25     ,0.01        };// kondisi poroelastik
	//float rhofm []=   {1000.0  ,1000.0   ,700.0   ,160.0      };//kondisi poroelastik

	//Kondisi elastik
	float phim[] = { 1.e-6, 1.e-6, 1.e-6, 1.e-6 };//kondisi elastik
	float rhofm[] = { 10.0, 10.0, 10.0, 10.0 }; ///  kondisi elastik

	int jmlmed = 4;

	cout << "test meh load";
	//line="model_zhu_10000.bin";
	line = "mar_poro_2440x13600.bin";
	char *inname1 = const_cast<char*>(line.c_str());
	FILE* file = fopen(inname1, "rb");
	for (int i = 0; i < (NX + 1)*(NZ + 1); i++){
		float f;
		fread(&f, sizeof(float), 1, file);
		medium[i] = f;
	}
	fclose(file);

	line = "mar_cp_2440x13600.bin";
	char *inname2 = const_cast<char*>(line.c_str());
	FILE* file2 = fopen(inname2, "rb");
	for (int i = 0; i < (NX + 1)*(NZ + 1); i++){
		float f2;
		fread(&f2, sizeof(float), 1, file2);
		cp[i] = f2;
	}
	fclose(file2);

	line = "mar_cs_2440x13600.bin";
	char *inname3 = const_cast<char*>(line.c_str());
	FILE* file3 = fopen(inname3, "rb");
	for (int i = 0; i < (NX + 1)*(NZ + 1); i++){
		float f3;
		fread(&f3, sizeof(float), 1, file3);
		cs[i] = f3;
	}
	fclose(file3);

	line = "mar_rho_2440x13600.bin";
	char *inname4 = const_cast<char*>(line.c_str());
	FILE* file4 = fopen(inname4, "rb");
	for (int i = 0; i < (NX + 1)*(NZ + 1); i++){
		float f4;
		fread(&f4, sizeof(float), 1, file4);
		crhos[i] = f4;
	}
	fclose(file4);

	for (int j = 0; j < (NX + 1)*(NZ + 1); j++)
	{
		for (int i = 0; i < jmlmed; i++)
		{
			if (medium[j] == i)
			{

				Kf[j] = kfm[i];
				eta[j] = etam[i];
				kappa[j] = kappam[i];
				phi[j] = phim[i];
				a[j] = am[i];
				crhof[j] = rhofm[i];
			}
		}
	}
	cout << "test wis load";

	for (int j = 0; j < NZ + 1; j++) {
		for (int i = 0; i < NX + 1; i++) {
			int ij = (NX + 1)*j + i;
			//crho[ij]= phi[ij]*crhof[ij]+(1.0-phi[ij])*crhos[ij]; //untuk rho dry 
			crho[ij] = crhos[ij]; // jika rho input saturated
			crmu[ij] = cs[ij] * cs[ij] * crho[ij];

			Ksat[ij] = crhos[ij] * (cp[ij] * cp[ij] - (4.0 / 3.0)*(cs[ij] * cs[ij]));

			float vgede = (3.0*Ksat[ij] - 2.0*crmu[ij]) / (2.0*(3.0*Ksat[ij] + crmu[ij]));

			Kb[ij] = ((2.0*vgede / (1.0 - 2.0*vgede)) + (2.0 / 3.0))*crmu[ij];
			Ks[ij] = Kb[ij] / (1.0 - phi[ij]);

			crsm[ij] = a[ij] * crhof[ij] / phi[ij];
			cetaokappa[ij] = eta[ij] / kappa[ij];
			calpha[ij] = 1.0 - Kb[ij] / Ks[ij];
			crbM[ij] = 1.0 / (phi[ij] / Kf[ij] + (calpha[ij] - phi[ij]) / Ks[ij]);
			if ((eta[ij] == 0.0) || (kappa[ij] == 0.0))
			{
				cetaokappa[ij] = 0.0;
			}
			//float C= calpha[ij]*crbM[ij];
			//crmu[ij]= cs[ij]*cs[ij]*crho[ij];
			crlambdao[ij] = Kb[ij] - (2.0 / 3.0)*crmu[ij];
			crlambdac[ij] = crlambdao[ij] + calpha[ij] * calpha[ij] * crbM[ij];

			//Ksat[ij]=(((Kb[ij])/(Ks[ij]-Kb[ij]) +(Kf[ij])/(phi[ij]*(Ks[ij]-Kf[ij]))   )*Ks[ij])/(1+((Kb[ij])/(Ks[ij]-Kb[ij]) +(Kf[ij])/(phi[ij]*(Ks[ij]-Kf[ij]))));
			//cs[ij]=sqrtf(crmu[ij]/crho[ij]);
			//cp[ij]=sqrtf((Ksat[ij]/crho[ij]) +( 1.33333333333*cs[ij]*cs[ij]));
		}
	}

	float maxcptop = max_element(cp, ((NX + 1)*(NZ + 1))); float maxcpbottom = maxcptop;
	float mincptop = min_element(cp, ((NX + 1)*(NZ + 1))); float mincpbottom = mincptop;

	int  ij = 1;

	cout << "Kb= " << Kb[ij] << endl;
	cout << "Kf= " << Kf[ij] << endl;
	cout << "Ks= " << Ks[ij] << endl;
	cout << "phi= " << phi[ij] << endl;
	cout << "mu= " << crmu[ij] << endl;
	cout << "rhof= " << crhof[ij] << endl;
	cout << "rhos= " << crhos[ij] << endl;
	cout << "rho= " << crho[ij] << endl;
	cout << "alpha= " << calpha[ij] << endl;
	cout << "rbM= " << crbM[ij] << endl;

	cout << "rsm= " << crsm[ij] << endl;
	cout << "lambdao= " << crlambdao[ij] << endl;
	cout << "rlambdac= " << crlambdac[ij] << endl;
	cout << "Ksat= " << Ksat[ij] << endl;
	cout << "cp= " << cp[ij] << endl;
	cout << "cs= " << cs[ij] << endl;
	cout << "cp/cs= " << (cp[ij] / cs[ij]) << endl;
	cout << "etaokappa= " << cetaokappa[ij] << endl;

	int ISOURCE;
	int JSOURCE;
	//ISOURCE = (NX+1)/2;
	//JSOURCE = (NZ+1)/2;

	in_file1 >> ISOURCE;
	in_file1 >> JSOURCE;
	int a0_pos;
	int a1_pos;
	in_file1 >> a0_pos;
	in_file1 >> a1_pos;


	//cout<<"/nmasukkan ISOURCE = "; cin>>ISOURCE;
	//cout<<"/nmasukkan JSOURCE = "; cin>>JSOURCE;
	int IDEB = (NX / 2) + 1;
	int JDEB = (NZ / 2) + 1;
	float xsource = DELTAX * ISOURCE;
	float zsource = DELTAZ * JSOURCE;
	//angle of source force clockwise with respect to vertical (Z) axis
	float ANGLE_FORCE = 0.e0;

	// display information on the screen from time to time
	int IT_DISPLAY = 200;

	// conversion from degrees to radians
	float DEGREES_TO_RADIANS = PI / 180.e0;

	// zero
	float ZERO = 0.e0;

	// large value for maximum
	float HUGEVAL = 1.e+30;

	// velocity threshold above which we consider that the code became unstable
	float STABILITY_THRESHOLD = 1.e+25;

	//main arrays
	float *cvx = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cvz = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *csigmaxx = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *csigma2 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *calp_sigma2 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *csigmazz = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *csigmaxz = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cvnorm = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));

	float *cvxf = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cvzf = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));

	float f0;

	//cout<<"\nfrekuensi =";
	in_file2 >> f0;
	cout << f0 << endl;
	float *ff0;
	float t0 = 1.0 / f0;

	float NPOWER = 2.e0;
	float K_MAX_PML = 1.e0; // from Gedney page 8.11
	float ALPHA_MAX_PML = 2.e0*PI*(f0 / 2.e0); // from Festa and Vilotte

	float *cgamma11 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cgamma22 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cgamma12_1 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cxi_1 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));
	float *cxi_2 = (float*)malloc(sizeof(float)*((NX + 1)*(NZ + 1)));

	float *cmemory_dx_vx1 = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_vx2 = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_vx = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_vz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_vz1 = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_vz2 = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_sigmaxx = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_sigmazz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_sigmaxz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_sigma2vx = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dx_sigma2vxf = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_sigma2vz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_sigma2vzf = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_sigmaxx = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_sigmazz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));
	float *cmemory_dz_sigmaxz = (float*)malloc(sizeof(float)*(((NX + 1)*(NZ + 1))));

	float *cd_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *cK_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *calpha_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *ca_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *cb_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *cd_x_half_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *cK_x_half_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *calpha_x_half_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *ca_x_half_x = (float*)malloc(sizeof(float)*((NX + 1)));
	float *cb_x_half_x = (float*)malloc(sizeof(float)*((NX + 1)));

	float *cd_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *cK_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *calpha_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *ca_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *cb_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *cd_z_half_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *cK_z_half_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *calpha_z_half_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *ca_z_half_z = (float*)malloc(sizeof(float)*((NZ + 1)));
	float *cb_z_half_z = (float*)malloc(sizeof(float)*((NZ + 1)));

	float thickness_PML_x, thickness_PML_z, xoriginleft, xoriginright, zoriginbottom, zorigintop;
	float Rcoef, d0_x, d0_z, xval, zval, abscissa_in_PML;//,abscissa_normalized;

	float Courant_number_bottom, Courant_number_top;//,velocnorm_all;//,max_amplitude;
	float Dispersion_number_bottom, Dispersion_number_top;

	//--- define profile of absorption in PML region

	// thickness of the PML layer in meters
	thickness_PML_x = (NPOINTS_PML)* DELTAX;
	thickness_PML_z = (NPOINTS_PML)* DELTAZ;

	// reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
	Rcoef = 0.001e0;

	if (NPOWER < 1)
	{
		cout << "\nnpower harus lebih besar dari 1!\n";
		return 0;
	}

	if (HETEROGENEOUS_MODEL)
	{
		d0_x = -(NPOWER + 1.0) * maxcptop * log(Rcoef) / (2.0 * thickness_PML_x);
		d0_z = -(NPOWER + 1.0) * maxcptop * log(Rcoef) / (2.0 * thickness_PML_z);
		cout << "hiya!/n";
	}
	//if(HETEROGENEOUS_MODEL) then
	//d0_x = - (NPOWER + 1) * max(cp_bottom,cp_top) * log(Rcoef) / (2.d0 * thickness_PML_x)
	//d0_z = - (NPOWER + 1) * max(cp_bottom,cp_top) * log(Rcoef) / (2.d0 * thickness_PML_z)
	else
		d0_x = -(NPOWER + 1.0) * maxcpbottom * log(Rcoef) / (2.0 * thickness_PML_x);
	d0_z = -(NPOWER + 1.0) * maxcpbottom * log(Rcoef) / (2.0 * thickness_PML_z);

	cout << "\nd0_x = " << d0_x << "\nd0_z = " << d0_z;

	for (int i = 0; i < NX + 1; i++)
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
	for (int i = 0; i < NZ + 1; i++)
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
	xoriginleft = thickness_PML_x + (3 * DELTAX);
	xoriginright = (NX - 2)*DELTAX - thickness_PML_x;

	for (int i = 0; i < NX + 1; i++)
	{ //PMLX
		xval = DELTAX * (float)(i);
		//float xval2= DELTAX * (float)(i+1);
		//PML kiri
		abscissa_in_PML = xoriginleft - xval;
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_x;
			cd_x[i] = d0_x * pow(abscissanorm, NPOWER);
			cK_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
		}
		//hitung profile penyerap di posisi separoh grid
		abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0);
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_x;
			cd_x_half_x[i] = d0_x * pow(abscissanorm, NPOWER);
			cK_x_half_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_x_half_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
		}
		//PML kiri selesai

		//PML kanan
		abscissa_in_PML = xval - xoriginright;
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_x;
			cd_x[i] = d0_x * pow(abscissanorm, NPOWER);
			cK_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
		}
		//hitung profile penyerap di posisi separoh grid
		abscissa_in_PML = (xval + DELTAX / 2.0) - xoriginright;
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_x;
			cd_x_half_x[i] = d0_x * pow(abscissanorm, NPOWER);
			cK_x_half_x[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_x_half_x[i] = ALPHA_MAX_PML * (1.0 - abscissanorm);
		}

		if (calpha_x[i] < 0.0) calpha_x[i] = 0.0;
		if (calpha_x_half_x[i] < 0.0) calpha_x_half_x[i] = 0.0;
		cb_x[i] = exp(-(cd_x[i] / cK_x[i] + calpha_x[i]) * DELTAT);
		cb_x_half_x[i] = exp(-(cd_x_half_x[i] / cK_x_half_x[i] + calpha_x_half_x[i]) * DELTAT);
		// cegah pembagian dengan nol diluar PML
		if (fabs(cd_x[i]) > 1.e-6) ca_x[i] = cd_x[i] * (cb_x[i] - 1.0) / (cK_x[i] * (cd_x[i] + cK_x[i] * calpha_x[i]));
		if (fabs(cd_x_half_x[i]) > 1.e-6)
			ca_x_half_x[i] = cd_x_half_x[i] * (cb_x_half_x[i] - 1.0) / (cK_x_half_x[i] * (cd_x_half_x[i] + cK_x_half_x[i] * calpha_x_half_x[i]));
		//PML kanan selesai
	} //PMLX

	// ================= hitung penyerap pada arah Z
	// titik asal lapisan PML
	zoriginbottom = (NZ - 2)*DELTAZ - thickness_PML_z;
	zorigintop = thickness_PML_z;

	for (int i = 0; i < NZ + 1; i++)
	{ //PMLZ
		zval = DELTAZ * (float)i;

		//PML ATAS JENG JENG
		//define damping profile at the grid points
		abscissa_in_PML = zorigintop - zval;
		if (abscissa_in_PML >= ZERO)
		{
			float  abscissanorm = abscissa_in_PML / thickness_PML_z;
			cd_z[i] = d0_z * pow(abscissanorm, NPOWER);
			//this taken from Gedney page 8.2
			cK_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_z[i] = ALPHA_MAX_PML * (1.0 - abscissanorm) + 0.10 * ALPHA_MAX_PML;
		}

		//define damping profile at half the grid points
		abscissa_in_PML = zorigintop - (zval + DELTAZ / 2.0);
		if (abscissa_in_PML >= ZERO)
		{
			float  abscissanorm = abscissa_in_PML / thickness_PML_z;
			cd_z_half_z[i] = d0_z * pow(abscissanorm, NPOWER);
			//this taken from Gedney page 8.2
			cK_z_half_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_z_half_z[i] = ALPHA_MAX_PML * (1.0 - abscissanorm) + 0.10 * ALPHA_MAX_PML;
		}


		//PML BAWAH
		abscissa_in_PML = zval - zoriginbottom;
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_z;
			cd_z[i] = d0_z * pow(abscissanorm, NPOWER);
			cK_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_z[i] = ALPHA_MAX_PML  * (1.0 - abscissanorm);
		}
		//hitung profile penyerap di posisi separoh grid
		abscissa_in_PML = (zval + DELTAZ / 2.0) - zoriginbottom;
		if (abscissa_in_PML >= 0.0)
		{
			float abscissanorm = abscissa_in_PML / thickness_PML_z;
			cd_z_half_z[i] = d0_z * pow(abscissanorm, NPOWER);
			cK_z_half_z[i] = 1.0 + (K_MAX_PML - 1.0) * pow(abscissanorm, NPOWER);
			calpha_z_half_z[i] = ALPHA_MAX_PML  * (1.0 - abscissanorm);
		}
		//PML BAWAH

		if (calpha_z[i] < 0.0) calpha_z[i] = 0.0;
		if (calpha_z_half_z[i] < 0.0) calpha_z_half_z[i] = 0.0;
		cb_z[i] = exp(-(cd_z[i] / cK_z[i] + calpha_z[i]) * DELTAT);

		cb_z_half_z[i] = exp(-(cd_z_half_z[i] / cK_z_half_z[i] + calpha_z_half_z[i]) * DELTAT);
		// cegah pembagian dengan nol diluar PML
		if (fabs(cd_z[i]) > 1.e-6) ca_z[i] = cd_z[i] * (cb_z[i] - 1.0) / (cK_z[i] * (cd_z[i] + cK_z[i] * calpha_z[i]));
		if (fabs(cd_z_half_z[i]) > 1.e-6)
			ca_z_half_z[i] = cd_z_half_z[i] * (cb_z_half_z[i] - 1.0) / (cK_z_half_z[i] * (cd_z_half_z[i] + cK_z_half_z[i] * calpha_z_half_z[i]));
	} //PMLZ

	Courant_number_bottom = maxcptop * DELTAT / DELTAX;
	Dispersion_number_bottom = maxcpbottom / (2.50*f0*DELTAX);

	if (Courant_number_bottom > 1.0 / sqrt(2.0))
	{
		cout << "'time step is too large, simulation will be unstable'"; return 0;
	}

	if (HETEROGENEOUS_MODEL) {
		Courant_number_top = maxcptop * DELTAT / DELTAX;
		Dispersion_number_top = 800.0 / (2.50*f0*DELTAX);
	}

	if (Courant_number_top > 6.0 / 7.0 / sqrt(2.0))
	{
		cout << "'time step is too large, simulation will be unstable'"; return 0;
	}

	for (int i = 0; i < (NX + 1)*(NZ + 1); i++)
	{
		cvx[i] = ZERO;
		cvz[i] = ZERO;
		csigmaxx[i] = ZERO;
		csigmazz[i] = ZERO;
		csigmaxz[i] = ZERO;
		csigma2[i] = ZERO;
		calp_sigma2[i] = ZERO;
		cgamma11[i] = ZERO;
		cgamma22[i] = ZERO;
		cgamma12_1[i] = ZERO;
		cgamma12_1[i] = ZERO;
		cxi_1[i] = ZERO;
		cxi_2[i] = ZERO;
		cvxf[i] = ZERO;
		cvzf[i] = ZERO;

		cmemory_dx_vx1[i] = ZERO;
		cmemory_dx_vx2[i] = ZERO;
		cmemory_dz_vx[i] = ZERO;
		cmemory_dx_vz[i] = ZERO;
		cmemory_dz_vz1[i] = ZERO;
		cmemory_dz_vz2[i] = ZERO;
		cmemory_dx_sigmaxx[i] = ZERO;
		cmemory_dx_sigmazz[i] = ZERO;
		cmemory_dx_sigmaxz[i] = ZERO;
		cmemory_dx_sigma2vx[i] = ZERO;
		cmemory_dx_sigma2vxf[i] = ZERO;
		cmemory_dz_sigmaxx[i] = ZERO;
		cmemory_dz_sigmazz[i] = ZERO;
		cmemory_dz_sigmaxz[i] = ZERO;
		cmemory_dz_sigma2vz[i] = ZERO;
		cmemory_dz_sigma2vzf[i] = ZERO;
	}

	float *sseismo = (float*)malloc(sizeof(float)*(panj)*NSTEP / samprek);
	float *sdeismo = (float*)malloc(sizeof(float)*(panj)*NSTEP / samprek);
	float *sseismo1 = (float*)malloc(sizeof(float)*(panj2)*NSTEP / samprek);
	float *sdeismo1 = (float*)malloc(sizeof(float)*(panj2)*NSTEP / samprek);
	float *sseismo2 = (float*)malloc(sizeof(float)*(1)*NSTEP);
	float *sseismo3 = (float*)malloc(sizeof(float)*(1)*NSTEP);
	float *sseismo4 = (float*)malloc(sizeof(float)*(1)*NSTEP);
	float *sseismo0 = (float*)malloc(sizeof(float)*(1)*NSTEP);
	for (int ij = 0; ij<(panj)*NSTEP / samprek; ij++)
	{
		sseismo[ij] = 0.0;
		sdeismo[ij] = 0.0;
	}
	for (int ij = 0; ij<(panj2)*NSTEP / samprek; ij++)
	{
		sseismo1[ij] = 0.0;
		sdeismo1[ij] = 0.0;
	}
	for (int ij = 0; ij<NSTEP; ij++)
	{
		sseismo2[ij] = 0.0; 
		sseismo3[ij] = 0.0; 
		sseismo4[ij] = 0.0; 
		sseismo0[ij] = 0.0;
	}

	int i_steps = 0;
	for (int it = 0; it < NSTEP / samp; it++) {
		for (int ix = 0; ix < samp; ix++) {
			i_steps++;
			//stress1
			for (int y = 2; y < NZ - 1; y++) {
				for (int x = 1; x < NX; x++) {
					int offset = x + y*NX;
					int right = offset + 1; int right2 = offset + 2;
					int left = offset - 1;
					int bottom = offset - (NX + 1); 
					int bottom2 = bottom - (NX + 1);
					int top = offset + (NX + 1);

					float value_dx_sigmaxx = (27.0*cvx[right] - 27.0*cvx[offset] - cvx[right2] + cvx[left]) / DELTAX / 24.0;
					float value_dz_sigmaxx = (27.0*cvz[offset] - 27.0*cvz[bottom] - cvz[top] + cvz[bottom2]) / DELTAZ / 24.0;

					cmemory_dx_sigmaxx[offset] = cb_x_half_x[x] * cmemory_dx_sigmaxx[offset] + ca_x_half_x[x] * value_dx_sigmaxx;
					cmemory_dz_sigmaxx[offset] = cb_z[y] * cmemory_dz_sigmaxx[offset] + ca_z[y] * value_dz_sigmaxx;

					cgamma11[offset] = cgamma11[offset] + DELTAT * (value_dx_sigmaxx / cK_x_half_x[x] + cmemory_dx_sigmaxx[offset]);
					cgamma22[offset] = cgamma22[offset] + DELTAT * (value_dz_sigmaxx / cK_z[y] + cmemory_dz_sigmaxx[offset]);

					float value_dx_sigma2vxf = (27.0*cvxf[right] - 27.0* cvxf[offset] - cvxf[right2] + cvxf[left]) / DELTAX / 24.0;
					float value_dz_sigma2vzf = (27.0*cvzf[offset] - 27.0*cvzf[bottom] - cvzf[top] + cvzf[bottom2]) / DELTAZ / 24.0;

					cxi_1[offset] = cxi_1[offset] - (value_dx_sigma2vxf / cK_x_half_x[x] + cmemory_dx_sigma2vxf[offset])*DELTAT;

					cxi_2[offset] = cxi_2[offset] - (value_dz_sigma2vzf / cK_z[y] + cmemory_dz_sigma2vzf[offset])*DELTAT;

					csigma2[offset] = -calpha[offset] * crbM[offset] * (cgamma11[offset] + cgamma22[offset]) + crbM[offset] * (cxi_1[offset] + cxi_2[offset]);

					if ((x == ISOURCE) && (y == JSOURCE)) {
						float a = PI*PI*f0 * f0;
						float t = i_steps * DELTAT;
						float t0 = 1.0 / f0;

						float source_term = faktor * 2.0*a*(t - t0)*exp(-a*pow((t - t0), 2));

						csigma2[offset] = csigma2[offset] + source_term;
					}
				}
			}

			//save file
			for (int y = 0; y < NZ; y++) {
				for (int x = 0; x < NX; x++) {
					int offset = x + y*NX;
					int length2 = (NZ + 1) / panj;
					if (x == ISOURCE)         //if(x==((NNX[0]+1)/2)-1)
						//if(x==3500)
					{

						if (i_steps == 0)
						{
							if (((y + 1) % length2 == 0))
							{
								sseismo[((y) / length2) + ((1) - 1)*(panj)] = cvz[offset];
								sdeismo[((y) / length2) + ((1) - 1)*(panj)] = cvx[offset];
							}
						}
						if ((i_steps%samprek == 0) && (i_steps>0))
						{
							if (((y + 1) % length2 == 0))
							{
								sseismo[((y) / length2) + ((i_steps / samprek) - 1)*(panj)] = cvz[offset];
								sdeismo[((y) / length2) + ((i_steps / samprek) - 1)*(panj)] = cvx[offset];
							}
						}

					}
				}
			}

			for (int y = 0; y < NZ; y++) {
				for (int x = 0; x < NX; x++) {
					int offset = x + y*NX;
					int length = (NX + 1) / panj2;
					if (y == 10)
					{

						if (i_steps == 0)
						{
							if (((x + 1) % length == 0))
							{
								sseismo1[((x) / length) + ((1) - 1)*(panj2)] = cvz[offset];
								sdeismo1[((x) / length) + ((1) - 1)*(panj2)] = cvx[offset];
							}
						}
						if ((i_steps%samprek == 0) && (i_steps>0))
						{
							if (((x + 1) % length == 0))
							{
								sseismo1[((x) / length) + ((i_steps / samprek) - 1)*(panj2)] = cvz[offset];
								sdeismo1[((x) / length) + ((i_steps / samprek) - 1)*(panj2)] = cvx[offset];
							}
						}

					}
				}
			}

			for (int y = 0; y < NZ; y++) {
				for (int x = 0; x < NX; x++) {
					int offset = x + y*NX;
					if ((y == 50) && (x == (NX + 1) / 2 - 1))
					{
						sseismo0[i_steps - 1] = cvz[offset];
					}


					if ((y == (a0_pos)) && (x == (NX + 1) / 2 - 1))
					{
						sseismo2[i_steps - 1] = cvz[offset];
					}

					if ((y == (a1_pos)) && (x == (NX + 1) / 2 - 1))
					{
						sseismo3[i_steps - 1] = cvz[offset];
					}

					if ((y == JSOURCE) && (x == (NX + 1) / 2 - 1))
					{
						sseismo4[i_steps - 1] = cvz[offset];
					}
				}
			}

			//stress2
			for (int y = 1; y < NZ - 1; y++) {
				for (int x = 2; x < NX; x++) {
					int offset = x + y*NX;
					int right = offset + 1;
					int left = offset - 1;  int left2 = offset - 2;
					int bottom = offset - (NX + 1);
					int top = offset + (NX + 1); 
					int top2 = top + (NX + 1);
					float c33_half_z = 2.0 / (1.0 / crmu[offset] + 1.0 / crmu[top]);
					c33_half_z = crmu[top];

					float value_dx_sigmaxz = (27.0*cvz[offset] - 27.0*cvz[left] - cvz[right] + cvz[left2]) / DELTAX / 24.0;
					float value_dz_sigmaxz = (27.0*cvx[top] - 27.0*cvx[offset] - cvx[top2] + cvx[bottom]) / DELTAZ / 24.0;

					cmemory_dx_sigmaxz[offset] = cb_x[x] * cmemory_dx_sigmaxz[offset] + ca_x[x] * value_dx_sigmaxz;
					cmemory_dz_sigmaxz[offset] = cb_z_half_z[y] * cmemory_dz_sigmaxz[offset] + ca_z_half_z[y] * value_dz_sigmaxz;

					csigmaxz[offset] = csigmaxz[offset] + c33_half_z / 1.0 * (value_dx_sigmaxz / cK_x[x] + cmemory_dx_sigmaxz[offset] + value_dz_sigmaxz / cK_z[y] + cmemory_dz_sigmaxz[offset]) * DELTAT;
				}
			}

			//stress3
			for (int y = 2; y < NZ; y++) {
				for (int x = 1; x < NX - 1; x++) {
					int offset = x + y*NX;

					csigmaxx[offset] = (crlambdac[offset] + 2.0*crmu[offset])*cgamma11[offset] + crlambdac[offset] * cgamma22[offset] - calpha[offset] * csigma2[offset];
					csigmazz[offset] = crlambdac[offset] * cgamma11[offset] + (crlambdac[offset] + 2.0*crmu[offset])*cgamma22[offset] - calpha[offset] * csigma2[offset];
				}
			}

			//vx
			for (int y = 2; y < NZ; y++) {
				for (int x = 2; x < NX; x++) {
					int offset = x + y*NX;
					int right = offset + 1;
					int left = offset - 1;  
					int left2 = offset - 2;
					int bottom = offset - (NX + 1); 
					int bottom2 = bottom - (NX + 1);
					int top = offset + (NX + 1);

					float co = (crho[offset] * crsm[offset] - crhof[offset] * crhof[offset]) / DELTAT;
					float c1 = co + crho[offset] * cetaokappa[offset] * 0.50;
					float c2 = co - crho[offset] * cetaokappa[offset] * 0.50;
					float vtemp = cvxf[offset];
					float value_dx_vx1 = (27.0*csigmaxx[offset] - 27.0*csigmaxx[left] - csigmaxx[right] + csigmaxx[left2]) / DELTAX / 24.0;
					float value_dx_vx2 = (27.0*csigma2[offset] - 27.0*csigma2[left] - csigma2[right] + csigma2[left2]) / DELTAX / 24.0;
					float value_dz_vx = (27.0*csigmaxz[offset] - 27.0*csigmaxz[bottom] - csigmaxz[top] + csigmaxz[bottom2]) / DELTAZ / 24.0;

					cmemory_dx_vx1[offset] = cb_x[x] * cmemory_dx_vx1[offset] + ca_x[x] * value_dx_vx1;
					cmemory_dx_vx2[offset] = cb_x[x] * cmemory_dx_vx2[offset] + ca_x[x] * value_dx_vx2;
					cmemory_dz_vx[offset] = cb_z[y] * cmemory_dz_vx[offset] + ca_z[y] * value_dz_vx;

					cvxf[offset] = (c2*cvxf[offset] + (-crhof[offset] * (value_dx_vx1 / cK_x[x] + cmemory_dx_vx1[offset] + value_dz_vx / cK_z[y] + cmemory_dz_vx[offset]) - crho[offset] * (value_dx_vx2 / cK_x[x] + cmemory_dx_vx2[offset]))) / c1;

					vtemp = (vtemp + cvxf[offset])*0.50;

					cvx[offset] = cvx[offset] + (crsm[offset] * (value_dx_vx1 / cK_x[x] + cmemory_dx_vx1[offset] + value_dz_vx / cK_z[y] + cmemory_dz_vx[offset]) + crhof[offset] * (value_dx_vx2 / cK_x[x] + cmemory_dx_vx2[offset]) + crhof[offset] * cetaokappa[offset] * vtemp) / co;
				}
			}

			//vz
			for (int y = 1; y < NZ - 1; y++) {
				for (int x = 1; x < NX - 1; x++) {
					int offset = x + y*NX;
					int right = offset + 1; 
					int right2 = offset + 2;
					int left = offset - 1;
					int bottom = offset - (NX + 1);// int bottom2=bottom-NX;
					int top = offset + (NX + 1);
					int top2 = top + (NX + 1);

					float rho_half_x_half_z = crho[top];
					float rsm_half_x_half_z = crsm[top];
					float rhof_half_x_half_z = crhof[top];
					float etaokappa_half_x_half_z = cetaokappa[top];

					float co = (rho_half_x_half_z*rsm_half_x_half_z - rhof_half_x_half_z*rhof_half_x_half_z) / DELTAT;
					float c1 = co + rho_half_x_half_z*etaokappa_half_x_half_z*0.50;
					float c2 = co - rho_half_x_half_z*etaokappa_half_x_half_z*0.50;
					float vtemp = cvzf[offset];

					float value_dx_vz = (27.0*csigmaxz[right] - 27.0*csigmaxz[offset] - csigmaxz[right2] + csigmaxz[left]) / DELTAX / 24.0;
					float value_dz_vz1 = (27.0*csigmazz[top] - 27.0*csigmazz[offset] - csigmazz[top2] + csigmazz[bottom]) / DELTAZ / 24.0;
					float value_dz_vz2 = (27.0*csigma2[top] - 27.0*csigma2[offset] - csigma2[top2] + csigma2[bottom]) / DELTAZ / 24.0;

					cmemory_dx_vz[offset] = cb_x_half_x[x] * cmemory_dx_vz[offset] + ca_x_half_x[x] * value_dx_vz;
					cmemory_dz_vz1[offset] = cb_z_half_z[y] * cmemory_dz_vz1[offset] + ca_z_half_z[y] * value_dz_vz1;
					cmemory_dz_vz2[offset] = cb_z_half_z[y] * cmemory_dz_vz2[offset] + ca_z_half_z[y] * value_dz_vz2;

					cvzf[offset] = (c2*cvzf[offset] + (-rhof_half_x_half_z*(value_dx_vz / cK_x_half_x[x] + cmemory_dx_vz[offset] + value_dz_vz1 / cK_z_half_z[y] + cmemory_dz_vz1[offset]) - rho_half_x_half_z*(value_dz_vz2 / cK_z_half_z[y] + cmemory_dz_vz2[offset]))) / c1;
					vtemp = (vtemp + cvzf[offset])*0.50;

					cvz[offset] = cvz[offset] + (rsm_half_x_half_z*(value_dx_vz / cK_x_half_x[x] + cmemory_dx_vz[offset] + value_dz_vz1 / cK_z_half_z[y] + cmemory_dz_vz1[offset]) + rhof_half_x_half_z*(value_dz_vz2 / cK_z_half_z[y] + cmemory_dz_vz2[offset]) + rhof_half_x_half_z*etaokappa_half_x_half_z*vtemp) / co;
				}
			}
		}
		char nmfile[50];
		sprintf(nmfile, "vz%05i.vzo", i_steps);
		//std::ofstream outvz(nmfile); // output, normal file
		FILE* file1 = fopen(nmfile, "wb");
		for (int jj = 0; jj<NZ + 1; jj++)
		{
			for (int ii = 0; ii<NX + 1; ii++)
			{
				int ij = (NX + 1)*jj + ii;
				//outvz<<cvz[ij]<<" ";
				float f2 = cvz[ij];
				fwrite(&f2, sizeof(float), 1, file1);
			}
			//outvz<<"\n";
		}
		fclose(file1);
		char nmfile4[50];
		sprintf(nmfile4, "vx%05i.vxo", i_steps);
		//std::ofstream outvz2(nmfile2); // output, normal file
		FILE* file4 = fopen(nmfile4, "wb");
		for (int jj = 0; jj<NZ + 1; jj++)
		{
			for (int ii = 0; ii<NX + 1; ii++)
			{
				int ij = (NX + 1)*jj + ii;
				//outvz2<<cvx[ij]<<" ";
				float f2 = cvx[ij];
				fwrite(&f2, sizeof(float), 1, file4);
			}
			//outvz2<<"\n";
		}
		fclose(file4);

		char nmfile3[100]; char nmfile6[100]; char nmfile6d[100]; char nmfile7[100]; char nmfile7d[100];
		printf("\nmenyimpan file... \n");
		sprintf(nmfile6, "recvspvz%08i.vsp", int(f0));
		sprintf(nmfile6d, "recvspvx%08i.vsp", int(f0));
		sprintf(nmfile7, "rechorvz%08i.hor", int(f0));
		sprintf(nmfile7d, "rechorvx%08i.hor", int(f0));

		ofstream outvz3(nmfile6); // output, normal file
		for (int jj = 0; jj<NSTEP / samprek; jj++)
		{
			for (int ii = 0; ii<panj; ii++)
			{
				int ij = (panj)*jj + ii;
				outvz3 << sseismo[ij] << " ";
			}
			outvz3 << "\n";
		}
		ofstream outvz3d(nmfile6d); // output, normal file
		for (int jj = 0; jj<NSTEP / samprek; jj++)
		{
			for (int ii = 0; ii<panj; ii++)
			{
				int ij = (panj)*jj + ii;
				outvz3d << sdeismo[ij] << " ";
			}
			outvz3d << "\n";
		}
		ofstream outseya1(nmfile7); // output, normal file
		for (int jj = 0; jj<NSTEP / samprek; jj++)
		{
			for (int ii = 0; ii<panj2; ii++)
			{
				int ij = (panj2)*jj + ii;
				outseya1 << sseismo1[ij] << " ";
			}
			outseya1 << "\n";
		}
		ofstream outseya1d(nmfile7d); // output, normal file
		for (int jj = 0; jj<NSTEP / samprek; jj++)
		{
			for (int ii = 0; ii<panj2; ii++)
			{
				int ij = (panj2)*jj + ii;
				outseya1d << sdeismo1[ij] << " ";
			}
			outseya1d << "\n";
		}
	}
	return 0;
}

