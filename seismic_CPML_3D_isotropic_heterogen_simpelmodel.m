clear
clc

% total number of grid points in each direction of the grid
NX = 201;
NY = 201;
NZ = 101; % even number in order to cut along Z axis

% number of processes used in the MPI run
% and local number of points (for simplicity we cut the mesh along Z only)

% size of a grid cell
DELTAX = 10.d0;
ONE_OVER_DELTAX = 1.d0 / DELTAX;
DELTAY = DELTAX;
DELTAZ = DELTAX;
ONE_OVER_DELTAY = ONE_OVER_DELTAX; 
ONE_OVER_DELTAZ = ONE_OVER_DELTAX;

% P-velocity, S-velocity and density
model1=[3300 3300/1.732d0 2800];
model2=[2000 2000/1.732d0 2000];
vp=zeros(NX,NY,NZ) + model1(1);
vs=zeros(NX,NY,NZ) + model1(2);
rhos=zeros(NX,NY,NZ) + model1(3);
for k=1:NZ
    for j=1:NY
        for i=1:NX
            if j>=0
                if k>=50
                    vp(i,j,k)=model2(1);
                    vs(i,j,k)=model2(2);
                    rhos(i,j,k)=model2(3);
                end
            end
        end
    end
end

% total number of time steps
NSTEP = 350;

% time step in seconds
DELTAT = 1.6d-3;

% parameters for the source
f0 = 7.d0;
t0 = 1.20d0 / f0;
factor = 1.d7;

% thickness of the PML layer in grid points
NPOINTS_PML = 10;

% source
ISOURCE = NX/2;
JSOURCE = NY/2;
KSOURCE = 11;
xsource = (ISOURCE - 1) * DELTAX;
ysource = (JSOURCE - 1) * DELTAY;
zsource = (KSOURCE - 1) * DELTAZ;
% angle of source force clockwise with respect to vertical (Y) axis
ANGLE_FORCE = 135.d0;

% receivers
NREC = 2;
xdeb = xsource - 100.d0; % first receiver x in meters
ydeb = 2300.d0; % first receiver y in meters
xfin = xsource; % last receiver x in meters
yfin =  300.d0; % last receiver y in meters

% display information on the screen from time to time
IT_DISPLAY = 100;

% value of PI
PI = 3.141592653589793238462643d0;

% conversion from degrees to radians
DEGREES_TO_RADIANS = PI / 180.d0;

% zero
ZERO = 0.d0;

% large value for maximum
HUGEVAL = 1.d+30;

% velocity threshold above which we consider that the code became unstable
STABILITY_THRESHOLD = 1.d+25;

% power to compute d0 profile
NPOWER = 2.d0;

K_MAX_PML = 1.d0; % from Gedney page 8.11
ALPHA_MAX_PML = 2.d0*PI*(f0/2.d0); % from Festa and Vilotte

% arrays for the memory variables
% could declare these arrays in PML only to save a lot of memory, but proof of concept only here
memory_dvx_dx=zeros(NX,NY,NZ);
memory_dvx_dy=zeros(NX,NY,NZ);
memory_dvx_dz=zeros(NX,NY,NZ);
memory_dvy_dx=zeros(NX,NY,NZ);
memory_dvy_dy=zeros(NX,NY,NZ);
memory_dvy_dz=zeros(NX,NY,NZ);
memory_dvz_dx=zeros(NX,NY,NZ);
memory_dvz_dy=zeros(NX,NY,NZ);
memory_dvz_dz=zeros(NX,NY,NZ);
memory_dsigmaxx_dx=zeros(NX,NY,NZ);
memory_dsigmayy_dy=zeros(NX,NY,NZ);
memory_dsigmazz_dz=zeros(NX,NY,NZ);
memory_dsigmaxy_dx=zeros(NX,NY,NZ);
memory_dsigmaxy_dy=zeros(NX,NY,NZ);
memory_dsigmaxz_dx=zeros(NX,NY,NZ);
memory_dsigmaxz_dz=zeros(NX,NY,NZ);
memory_dsigmayz_dy=zeros(NX,NY,NZ);
memory_dsigmayz_dz=zeros(NX,NY,NZ);

% 1D arrays for the damping profiles
d_x=zeros(NX,1);
K_x=zeros(NX,1)+1;
alpha_x=zeros(NX,1);
a_x=zeros(NX,1);
b_x=zeros(NX,1);
d_x_half=zeros(NX,1);
K_x_half=zeros(NX,1)+1;
alpha_x_half=zeros(NX,1);
a_x_half=zeros(NX,1);
b_x_half=zeros(NX,1);

d_y=zeros(NY,1);
K_y=zeros(NY,1)+1;
alpha_y=zeros(NY,1);
a_y=zeros(NY,1);
b_y=zeros(NY,1);
d_y_half=zeros(NY,1);
K_y_half=zeros(NY,1)+1;
alpha_y_half=zeros(NY,1);
a_y_half=zeros(NY,1);
b_y_half=zeros(NY,1);

d_z=zeros(NZ,1);
K_z=zeros(NZ,1)+1;
alpha_z=zeros(NZ,1);
a_z=zeros(NZ,1);
b_z=zeros(NZ,1);
d_z_half=zeros(NZ,1);
K_z_half=zeros(NZ,1)+1;
alpha_z_half=zeros(NZ,1);
a_z_half=zeros(NZ,1);
b_z_half=zeros(NZ,1);

% change dimension of Z axis to add two planes for MPI
vx=zeros(NX,NY,NZ);
vy=zeros(NX,NY,NZ);
vz=zeros(NX,NY,NZ);
sigmaxx=zeros(NX,NY,NZ);
sigmayy=zeros(NX,NY,NZ);
sigmazz=zeros(NX,NY,NZ);
sigmaxy=zeros(NX,NY,NZ);
sigmaxz=zeros(NX,NY,NZ);
sigmayz=zeros(NX,NY,NZ);

% for receivers
ix_rec=zeros(NREC,1);
iy_rec=zeros(NREC,1);
xrec=zeros(NREC,1);
yrec=zeros(NREC,1);

% for seismograms
sisvx=zeros(NSTEP,NREC);
sisvy=zeros(NSTEP,NREC);

% for evolution of total energy in the medium
total_energy=zeros(NSTEP,1);

%---
%--- program starts here
%---

%--- define profile of absorption in PML region

% thickness of the PML layer in meters
thickness_PML_x = NPOINTS_PML * DELTAX;
thickness_PML_y = NPOINTS_PML * DELTAY;
thickness_PML_z = NPOINTS_PML * DELTAZ;

% reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
Rcoef = 0.001d0;

% compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
vpml=2700;
d0_x = - (NPOWER + 1) * vpml * log(Rcoef) / (2.d0 * thickness_PML_x);
d0_y = - (NPOWER + 1) * vpml * log(Rcoef) / (2.d0 * thickness_PML_y);
d0_z = - (NPOWER + 1) * vpml * log(Rcoef) / (2.d0 * thickness_PML_z);

% damping in the X direction

% origin of the PML layer (position of right edge minus thickness, in meters)
xoriginleft = thickness_PML_x;
xoriginright = (NX-1)*DELTAX - thickness_PML_x;

for i = 1:NX

% abscissa of current grid point along the damping profile
    xval = DELTAX *(i-1);

%---------- xmin edge

% define damping profile at the grid points
      abscissa_in_PML = xoriginleft - xval;
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_x;
        d_x(i) = d0_x * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = xoriginleft - (xval + DELTAX/2.d0);
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_x;
        d_x_half(i) = d0_x * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

%---------- xmax edge

% define damping profile at the grid points
      abscissa_in_PML = xval - xoriginright;
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_x;
        d_x(i) = d0_x * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = xval + DELTAX/2.d0 - xoriginright;
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_x;
        d_x_half(i) = d0_x * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% just in case, for -5 at the end
    if(alpha_x(i) < ZERO) 
        alpha_x(i) = ZERO;
    end
    if(alpha_x_half(i) < ZERO) 
        alpha_x_half(i) = ZERO;
    end

    b_x(i) = exp(- (d_x(i) / K_x(i) + alpha_x(i)) * DELTAT);
    b_x_half(i) = exp(- (d_x_half(i) / K_x_half(i) + alpha_x_half(i)) * DELTAT);

% this to avoid division by zero outside the PML
    if(abs(d_x(i)) > 1.d-6) 
        a_x(i) = d_x(i) * (b_x(i) - 1.d0) / (K_x(i) * (d_x(i) + K_x(i) * alpha_x(i)));
    end
    if(abs(d_x_half(i)) > 1.d-6) 
        a_x_half(i) = d_x_half(i) * ...
      (b_x_half(i) - 1.d0) / (K_x_half(i) * (d_x_half(i) + K_x_half(i) * alpha_x_half(i)));
    end

end

% damping in the Y direction

% origin of the PML layer (position of right edge minus thickness, in meters)
  yoriginbottom = thickness_PML_y;
  yorigintop = (NY-1)*DELTAY - thickness_PML_y;

for j = 1:NY

% abscissa of current grid point along the damping profile
    yval = DELTAY * (j-1);

%---------- ymin edge

% define damping profile at the grid points
      abscissa_in_PML = yoriginbottom - yval;
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_y;
        d_y(j) = d0_y * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.d0);
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_y;
        d_y_half(j) = d0_y * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

%---------- ymax edge

% define damping profile at the grid points
      abscissa_in_PML = yval - yorigintop;
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_y;
        d_y(j) = d0_y * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = yval + DELTAY/2.d0 - yorigintop;
      if(abscissa_in_PML >= ZERO)
        abscissa_normalized = abscissa_in_PML / thickness_PML_y;
        d_y_half(j) = d0_y * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

    b_y(j) = exp(- (d_y(j) / K_y(j) + alpha_y(j)) * DELTAT);
    b_y_half(j) = exp(- (d_y_half(j) / K_y_half(j) + alpha_y_half(j)) * DELTAT);

% this to avoid division by zero outside the PML
    if(abs(d_y(j)) > 1.d-6) 
        a_y(j) = d_y(j) * (b_y(j) - 1.d0) / (K_y(j) * (d_y(j) + K_y(j) * alpha_y(j)));
    end
    if(abs(d_y_half(j)) > 1.d-6) 
        a_y_half(j) = d_y_half(j) * ...
      (b_y_half(j) - 1.d0) / (K_y_half(j) * (d_y_half(j) + K_y_half(j) * alpha_y_half(j)));
    end
    
end

% damping in the Z direction

% origin of the PML layer (position of right edge minus thickness, in meters)
  zoriginbottom = thickness_PML_z;
  zorigintop = (NZ-1)*DELTAZ - thickness_PML_z;

for k = 1:NZ

% abscissa of current grid point along the damping profile
    zval = DELTAZ * (k-1);

%---------- zmin edge

% define damping profile at the grid points
      abscissa_in_PML = zoriginbottom - zval;
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_z;
        d_z(k) = d0_z * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_z(k) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_z(k) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = zoriginbottom - (zval + DELTAZ/2.d0);
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_z;
        d_z_half(k) = d0_z * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_z_half(k) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_z_half(k) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

%---------- zmax edge

% define damping profile at the grid points
      abscissa_in_PML = zval - zorigintop;
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_z;
        d_z(k) = d0_z * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_z(k) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_z(k) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end

% define damping profile at half the grid points
      abscissa_in_PML = zval + DELTAZ/2.d0 - zorigintop;
      if(abscissa_in_PML >= ZERO) 
        abscissa_normalized = abscissa_in_PML / thickness_PML_z;
        d_z_half(k) = d0_z * abscissa_normalized^NPOWER;
% this taken from Gedney page 8.2
        K_z_half(k) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized^NPOWER;
        alpha_z_half(k) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized) + 0.1d0 * ALPHA_MAX_PML;
      end


    b_z(k) = exp(- (d_z(k) / K_z(k) + alpha_z(k)) * DELTAT);
    b_z_half(k) = exp(- (d_z_half(k) / K_z_half(k) + alpha_z_half(k)) * DELTAT);

% this to avoid division by zero outside the PML
    if(abs(d_z(k)) > 1.d-6) 
        a_z(k) = d_z(k) * (b_z(k) - 1.d0) / (K_z(k) * (d_z(k) + K_z(k) * alpha_z(k)));
    end
    if(abs(d_z_half(k)) > 1.d-6) 
        a_z_half(k) = d_z_half(k) * ...
      (b_z_half(k) - 1.d0) / (K_z_half(k) * (d_z_half(k) + K_z_half(k) * alpha_z_half(k)));
    end
    
end


  xspacerec = (xfin-xdeb) / (NREC-1);
  yspacerec = (yfin-ydeb) / (NREC-1);
for irec=1:NREC
    xrec(irec) = xdeb + (irec-1)*xspacerec;
    yrec(irec) = ydeb + (irec-1)*yspacerec;
end

% find closest grid point for each receiver
for irec=1:NREC
    dist = HUGEVAL;
  for j = 1:NY
  for i = 1:NX
      distval = sqrt((DELTAX*(i-1) - xrec(irec))^2 + (DELTAY*(j-1) - yrec(irec))^2);
      if(distval < dist) 
        dist = distval;
        ix_rec(irec) = i;
        iy_rec(irec) = j;
      end
  end
  end
end

gx=100; gy=5; %num of gather x, y, z
gatx=floor(linspace(1+NPOINTS_PML,NX-(NPOINTS_PML+1),gx));
gaty=floor(linspace(1+NPOINTS_PML,NY-(NPOINTS_PML+1),gy));

gathersvx=zeros(gx,gy);
gathersvz=zeros(gx,gy);

% check the Courant stability condition for the explicit time scheme
% R. Courant et K. O. Friedrichs et H. Lewy (1928)
  Courant_number = vp(1,1,1) * DELTAT * sqrt(1.d0/DELTAX^2 + 1.d0/DELTAY^2 + 1.d0/DELTAZ^2);
  if(Courant_number > 1.d0) 
      'time step is too large, simulation will be unstable'
  end

k2begin = 2;
kminus1end = NZ - 1;

%---
%---  beginning of time loop
%---

sutaru=cputime;

for it = 1:NSTEP
%----------------------
% compute stress sigma
%----------------------
for k=k2begin:NZ
 for j=2:NY
   for i=1:NX-1
      cp=(3*vp(i,j,k)+vp(i+1,j,k)+vp(i,j-1,k)+vp(i,j,k-1))/6;
      cs=(3*vs(i,j,k)+vs(i+1,j,k)+vs(i,j-1,k)+vs(i,j,k-1))/6;
      rho=(3*rhos(i,j,k)+rhos(i+1,j,k)+rhos(i,j-1,k)+rhos(i,j,k-1))/2;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dvx_dx = (vx(i+1,j,k)-vx(i,j,k)) * ONE_OVER_DELTAX;
      value_dvy_dy = (vy(i,j,k)-vy(i,j-1,k)) * ONE_OVER_DELTAY;
      value_dvz_dz = (vz(i,j,k)-vz(i,j,k-1)) * ONE_OVER_DELTAZ;

      memory_dvx_dx(i,j,k) = b_x_half(i) * memory_dvx_dx(i,j,k) + a_x_half(i) * value_dvx_dx;
      memory_dvy_dy(i,j,k) = b_y(j) * memory_dvy_dy(i,j,k) + a_y(j) * value_dvy_dy;
      memory_dvz_dz(i,j,k) = b_z(k) * memory_dvz_dz(i,j,k) + a_z(k) * value_dvz_dz;

      value_dvx_dx = value_dvx_dx / K_x_half(i) + memory_dvx_dx(i,j,k);
      value_dvy_dy = value_dvy_dy / K_y(j) + memory_dvy_dy(i,j,k);
      value_dvz_dz = value_dvz_dz / K_z(k) + memory_dvz_dz(i,j,k);

      sigmaxx(i,j,k) = DELTAT_lambdaplus2mu*value_dvx_dx + ...
          DELTAT_lambda*(value_dvy_dy + value_dvz_dz) + sigmaxx(i,j,k);

      sigmayy(i,j,k) = DELTAT_lambda*(value_dvx_dx + value_dvz_dz) + ...
          DELTAT_lambdaplus2mu*value_dvy_dy + sigmayy(i,j,k);

      sigmazz(i,j,k) = DELTAT_lambda*(value_dvx_dx + value_dvy_dy) + DELTAT_lambdaplus2mu*value_dvz_dz + sigmazz(i,j,k);

   end
 end
end

for k=1:NZ
 for j=1:NY-1
   for i=2:NX
      cp=(2*vp(i,j,k)+vp(i-1,j,k)+vp(i,j+1,k))/4;
      cs=(2*vs(i,j,k)+vs(i-1,j,k)+vs(i,j+1,k))/4;
      rho=(2*rhos(i,j,k)+rhos(i-1,j,k)+rhos(i,j+1,k))/4;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dvy_dx = (vy(i,j,k)-vy(i-1,j,k)) * ONE_OVER_DELTAX;
      value_dvx_dy = (vx(i,j+1,k)-vx(i,j,k)) * ONE_OVER_DELTAY;

      memory_dvy_dx(i,j,k) = b_x(i) * memory_dvy_dx(i,j,k) + a_x(i) * value_dvy_dx;
      memory_dvx_dy(i,j,k) = b_y_half(j) * memory_dvx_dy(i,j,k) + a_y_half(j) * value_dvx_dy;

      value_dvy_dx = value_dvy_dx / K_x(i) + memory_dvy_dx(i,j,k);
      value_dvx_dy = value_dvx_dy / K_y_half(j) + memory_dvx_dy(i,j,k);

      sigmaxy(i,j,k) = DELTAT_mu*(value_dvy_dx + value_dvx_dy) + sigmaxy(i,j,k);

   end
 end
end

for k=1:kminus1end
 for j=1:NY
   for i=2:NX
      cp=(2*vp(i,j,k)+vp(i-1,j,k)+vp(i,j,k+1))/4;
      cs=(2*vs(i,j,k)+vs(i-1,j,k)+vs(i,j,k+1))/4;
      rho=(2*rhos(i,j,k)+rhos(i-1,j,k)+rhos(i,j,k+1))/4;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dvz_dx = (vz(i,j,k)-vz(i-1,j,k)) * ONE_OVER_DELTAX;
      value_dvx_dz = (vx(i,j,k+1)-vx(i,j,k)) * ONE_OVER_DELTAZ;

      memory_dvz_dx(i,j,k) = b_x(i) * memory_dvz_dx(i,j,k) + a_x(i) * value_dvz_dx;
      memory_dvx_dz(i,j,k) = b_z_half(k) * memory_dvx_dz(i,j,k) + a_z_half(k) * value_dvx_dz;

      value_dvz_dx = value_dvz_dx / K_x(i) + memory_dvz_dx(i,j,k);
      value_dvx_dz = value_dvx_dz / K_z_half(k) + memory_dvx_dz(i,j,k);

      sigmaxz(i,j,k) = DELTAT_mu*(value_dvz_dx + value_dvx_dz) + sigmaxz(i,j,k);

   end
 end

 for j=1:NY-1
   for i=1:NX
      cp=(2*vp(i,j,k)+vp(i,j+1,k)+vp(i,j,k+1))/4;
      cs=(2*vs(i,j,k)+vs(i,j+1,k)+vs(i,j,k+1))/4;
      rho=(2*rhos(i,j,k)+rhos(i,j+1,k)+rhos(i,j,k+1))/4;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dvz_dy = (vz(i,j+1,k)-vz(i,j,k)) * ONE_OVER_DELTAY;
      value_dvy_dz = (vy(i,j,k+1)-vy(i,j,k)) * ONE_OVER_DELTAZ;

      memory_dvz_dy(i,j,k) = b_y_half(j) * memory_dvz_dy(i,j,k) + a_y_half(j) * value_dvz_dy;
      memory_dvy_dz(i,j,k) = b_z_half(k) * memory_dvy_dz(i,j,k) + a_z_half(k) * value_dvy_dz;

      value_dvz_dy = value_dvz_dy / K_y_half(j) + memory_dvz_dy(i,j,k);
      value_dvy_dz = value_dvy_dz / K_z_half(k) + memory_dvy_dz(i,j,k);

      sigmayz(i,j,k) = DELTAT_mu*(value_dvz_dy + value_dvy_dz) + sigmayz(i,j,k);

   end
 end
end

for k=k2begin:NZ
 for j=2:NY
   for i=2:NX
      cp=(3*vp(i,j,k)+vp(i-1,j,k)+vp(i,j-1,k)+vp(i,j,k-1))/6;
      cs=(3*vs(i,j,k)+vs(i-1,j,k)+vs(i,j-1,k)+vs(i,j,k-1))/6;
      rho=(3*rhos(i,j,k)+rhos(i-1,j,k)+rhos(i,j-1,k)+rhos(i,j,k-1))/6;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dsigmaxx_dx = (sigmaxx(i,j,k)-sigmaxx(i-1,j,k)) * ONE_OVER_DELTAX;
      value_dsigmaxy_dy = (sigmaxy(i,j,k)-sigmaxy(i,j-1,k)) * ONE_OVER_DELTAY;
      value_dsigmaxz_dz = (sigmaxz(i,j,k)-sigmaxz(i,j,k-1)) * ONE_OVER_DELTAZ;

      memory_dsigmaxx_dx(i,j,k) = b_x(i) * memory_dsigmaxx_dx(i,j,k) + a_x(i) * value_dsigmaxx_dx;
      memory_dsigmaxy_dy(i,j,k) = b_y(j) * memory_dsigmaxy_dy(i,j,k) + a_y(j) * value_dsigmaxy_dy;
      memory_dsigmaxz_dz(i,j,k) = b_z(k) * memory_dsigmaxz_dz(i,j,k) + a_z(k) * value_dsigmaxz_dz;

      value_dsigmaxx_dx = value_dsigmaxx_dx / K_x(i) + memory_dsigmaxx_dx(i,j,k);
      value_dsigmaxy_dy = value_dsigmaxy_dy / K_y(j) + memory_dsigmaxy_dy(i,j,k);
      value_dsigmaxz_dz = value_dsigmaxz_dz / K_z(k) + memory_dsigmaxz_dz(i,j,k);

      vx(i,j,k) = DELTAT_over_rho*(value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + vx(i,j,k);

   end
 end

 for j=1:NY-1
   for i=1:NX-1
      cp=(3*vp(i,j,k)+vp(i+1,j,k)+vp(i,j+1,k)+vp(i,j,k-1))/6;
      cs=(3*vs(i,j,k)+vs(i+1,j,k)+vs(i,j+1,k)+vs(i,j,k-1))/6;
      rho=(3*rhos(i,j,k)+rhos(i+1,j,k)+rhos(i,j+1,k)+rhos(i,j,k-1))/6;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dsigmaxy_dx = (sigmaxy(i+1,j,k)-sigmaxy(i,j,k)) * ONE_OVER_DELTAX;
      value_dsigmayy_dy = (sigmayy(i,j+1,k)-sigmayy(i,j,k)) * ONE_OVER_DELTAY;
      value_dsigmayz_dz = (sigmayz(i,j,k)-sigmayz(i,j,k-1)) * ONE_OVER_DELTAZ;

      memory_dsigmaxy_dx(i,j,k) = b_x_half(i) * memory_dsigmaxy_dx(i,j,k) + a_x_half(i) * value_dsigmaxy_dx;
      memory_dsigmayy_dy(i,j,k) = b_y_half(j) * memory_dsigmayy_dy(i,j,k) + a_y_half(j) * value_dsigmayy_dy;
      memory_dsigmayz_dz(i,j,k) = b_z(k) * memory_dsigmayz_dz(i,j,k) + a_z(k) * value_dsigmayz_dz;

      value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half(i) + memory_dsigmaxy_dx(i,j,k);
      value_dsigmayy_dy = value_dsigmayy_dy / K_y_half(j) + memory_dsigmayy_dy(i,j,k);
      value_dsigmayz_dz = value_dsigmayz_dz / K_z(k) + memory_dsigmayz_dz(i,j,k);

      vy(i,j,k) = DELTAT_over_rho*(value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + vy(i,j,k);

   end
 end
end

for k=1:kminus1end
 for j=2:NY
   for i=1:NX-1
      cp=(3*vp(i,j,k)+vp(i+1,j,k)+vp(i,j-1,k)+vp(i,j,k+1))/2;
      cs=(3*vs(i,j,k)+vs(i+1,j,k)+vs(i,j-1,k)+vs(i,j,k+1))/2;
      rho=(3*rhos(i,j,k)+rhos(i+1,j,k)+rhos(i,j-1,k)+rhos(i,j,k+1))/2;
      
      mu = rho*cs*cs;
      lambda = rho*(cp*cp - 2.d0*cs*cs);
      lambdaplustwomu = rho*cp*cp;
      
      DELTAT_lambda = DELTAT*lambda;
      DELTAT_mu = DELTAT*mu;
      DELTAT_lambdaplus2mu = DELTAT*lambdaplustwomu;

      DELTAT_over_rho = DELTAT/rho;
      
      value_dsigmaxz_dx = (sigmaxz(i+1,j,k)-sigmaxz(i,j,k)) * ONE_OVER_DELTAX;
      value_dsigmayz_dy = (sigmayz(i,j,k)-sigmayz(i,j-1,k)) * ONE_OVER_DELTAY;
      value_dsigmazz_dz = (sigmazz(i,j,k+1)-sigmazz(i,j,k)) * ONE_OVER_DELTAZ;

      memory_dsigmaxz_dx(i,j,k) = b_x_half(i) * memory_dsigmaxz_dx(i,j,k) + a_x_half(i) * value_dsigmaxz_dx;
      memory_dsigmayz_dy(i,j,k) = b_y(j) * memory_dsigmayz_dy(i,j,k) + a_y(j) * value_dsigmayz_dy;
      memory_dsigmazz_dz(i,j,k) = b_z_half(k) * memory_dsigmazz_dz(i,j,k) + a_z_half(k) * value_dsigmazz_dz;

      value_dsigmaxz_dx = value_dsigmaxz_dx / K_x_half(i) + memory_dsigmaxz_dx(i,j,k);
      value_dsigmayz_dy = value_dsigmayz_dy / K_y(j) + memory_dsigmayz_dy(i,j,k);
      value_dsigmazz_dz = value_dsigmazz_dz / K_z_half(k) + memory_dsigmazz_dz(i,j,k);

      vz(i,j,k) = DELTAT_over_rho*(value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + vz(i,j,k);

   end
 end
end

% add the source (force vector located at a given grid point)
  a = pi*pi*f0*f0;
  t = (it-1)*DELTAT;

% Gaussian
% source_term = factor * exp(-a*(t-t0)**2)

% first derivative of a Gaussian
  source_term = - factor * 2.d0*a*(t-t0)*exp(-a*(t-t0^2));

% Ricker source time function (second derivative of a Gaussian)
% source_term = factor * (1.d0 - 2.d0*a*(t-t0)**2)*exp(-a*(t-t0)**2)

  force_x = sind(ANGLE_FORCE) * source_term;
  force_y = cosd(ANGLE_FORCE) * source_term;

% define location of the source
  i = floor(ISOURCE);
  j = floor(JSOURCE);
  k = floor(KSOURCE);

  vx(i,j,k) = vx(i,j,k) + force_x * DELTAT / rho;
  vy(i,j,k) = vy(i,j,k) + force_y * DELTAT / rho;

%{
% implement Dirichlet boundary conditions on the six edges of the grid
% xmin
  vx(1,:,:) = ZERO;
  vy(1,:,:) = ZERO;
  vz(1,:,:) = ZERO;
% xmax
  vx(NX,:,:) = ZERO;
  vy(NX,:,:) = ZERO;
  vz(NX,:,:) = ZERO;
% ymin
  vx(:,1,:) = ZERO;
  vy(:,1,:) = ZERO;
  vz(:,1,:) = ZERO;
% ymax
  vx(:,NY,:) = ZERO;
  vy(:,NY,:) = ZERO;
  vz(:,NY,:) = ZERO;
%$OMP END PARALLEL WORKSHARE
% zmin
  vx(:,:,1) = ZERO;
  vy(:,:,1) = ZERO;
  vz(:,:,1) = ZERO;
% zmax
  vx(:,:,NZ) = ZERO;
  vy(:,:,NZ) = ZERO;
  vz(:,:,NZ) = ZERO;
%}
% store seismograms
  for irec = 1:NREC
      sisvx(it,irec) = vx(ix_rec(irec),iy_rec(irec),NZ);
      sisvy(it,irec) = vy(ix_rec(irec),iy_rec(irec),NZ);
  end

% compute total energy in the medium (without the PML layers)
total_energy_kinetic = ZERO;
total_energy_potential = ZERO;

kmin = NPOINTS_PML+1;
kmax = NZ-NPOINTS_PML;

%$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i,j,k,epsilon_xx,epsilon_yy,epsilon_zz,epsilon_xy,epsilon_xz,epsilon_yz) ...
%$OMP SHARED(kmin,kmax,vx,vy,vz,sigmaxx,sigmayy,sigmazz, ...
%$OMP sigmaxy,sigmaxz,sigmayz) REDUCTION(+:total_energy_kinetic,total_energy_potential)
for k = kmin:kmax
  for j = NPOINTS_PML+1:NY-NPOINTS_PML
    for i = NPOINTS_PML+1:NX-NPOINTS_PML

% compute kinetic energy first, defined as 1/2 rho ||v||^2
% in principle we should use rho_half_x_half_y instead of rho for vy
% in order to interpolate density at the right location in the staggered grid cell
% but in a homogeneous medium we can safely ignore it
      total_energy_kinetic = total_energy_kinetic + 0.5d0 * rho*( ...
              vx(i,j,k)^2 + vy(i,j,k)^2 + vz(i,j,k)^2);

% add potential energy, defined as 1/2 epsilon_ij sigma_ij
% in principle we should interpolate the medium parameters at the right location
% in the staggered grid cell but in a homogeneous medium we can safely ignore it

% compute total field from split components
      epsilon_xx = (2.d0*(lambda + mu) * sigmaxx(i,j,k) - lambda * sigmayy(i,j,k) - ...
          lambda*sigmazz(i,j,k)) / (2.d0 * mu * (3.d0*lambda + 2.d0*mu));
      epsilon_yy = (2.d0*(lambda + mu) * sigmayy(i,j,k) - lambda * sigmaxx(i,j,k) - ...
          lambda*sigmazz(i,j,k)) / (2.d0 * mu * (3.d0*lambda + 2.d0*mu));
      epsilon_zz = (2.d0*(lambda + mu) * sigmazz(i,j,k) - lambda * sigmaxx(i,j,k) - ...
          lambda*sigmayy(i,j,k)) / (2.d0 * mu * (3.d0*lambda + 2.d0*mu));
      epsilon_xy = sigmaxy(i,j,k) / (2.d0 * mu);
      epsilon_xz = sigmaxz(i,j,k) / (2.d0 * mu);
      epsilon_yz = sigmayz(i,j,k) / (2.d0 * mu);

      total_energy_potential = total_energy_potential + ...
        0.5d0 * (epsilon_xx * sigmaxx(i,j,k) + epsilon_yy * sigmayy(i,j,k) + ...
        epsilon_yy * sigmayy(i,j,k)+ 2.d0 * epsilon_xy * sigmaxy(i,j,k) + ...
        2.d0*epsilon_xz * sigmaxz(i,j,k)+2.d0*epsilon_yz * sigmayz(i,j,k));

    end
  end
end


for i=1:gx
    for j=1:gy
        gathersvx(i,j)=vx(gatx(i),gaty(j),15);
        gathersvz(i,j)=vz(gatx(i),gaty(j),15);
    end
end

fid=fopen('gatvx.txt','at');
fprintf(fid,'%6.2f ',gathersvx);
fprintf(fid,'\n');
fclose(fid);

fid=fopen('gatvz.txt','at');
fprintf(fid,'%6.2f ',gathersvz);
fprintf(fid,'\n');
fclose(fid);
it
memory
if (mod(it,100)==0)
    %{
    a=zeros(NX,NZ); %slice y
    b=zeros(NX,NY); %slice z
    c=zeros(NY,NZ); %slice x
    
    for k=1:NZ
        for i=1:NX
            a(i,k)=vz(i,floor(JSOURCE),k);
        end
    end
    
    for k=1:NY
        for i=1:NX
            b(i,k)=vz(i,floor(KSOURCE),k);
        end
    end
    
    for k=1:NZ
        for i=1:NY
            c(i,k)=vz(i,floor(ISOURCE),k);
        end
    end
    %}
    a='vx'; b='vy'; c='vz'; e='time';
    d=num2str(it);
    dlmwrite(strcat(a,d),vx);
    dlmwrite(strcat(b,d),vy);
    dlmwrite(strcat(c,d),vz);
    
    
    endu=cputime;
    dlmwrite(strcat(e,d),endu-sutaru);
end
end
