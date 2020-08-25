!
! SEISMIC_CPML Version 1.1.3, September 2015.
!
! Copyright CNRS, France.
! Contributor: Dimitri Komatitsch, komatitsch aT lma DOT cnrs-mrs DOT fr
!
! This software is a computer program whose purpose is to solve
! the two-dimensional heterogeneous acoustic wave equation
! using a finite-difference method with Convolutional Perfectly Matched
! Layer (C-PML) conditions.
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".

  program seismic_CPML_2D_pressure

! 2D acoustic finite-difference code in pressure formulation
! with Convolutional-PML (C-PML) absorbing conditions for an heterogeneous acoustic medium

! Dimitri Komatitsch, CNRS, Marseille, September 2015.

! The pressure wave equation in an inviscid heterogeneous fluid is:
!
! 1/Kappa d2p / dt2 = div(grad(p) / rho) = d(1/rho dp/dx)/dx + d(1/rho dp/dy)/dy
!
! (see for instance Komatitsch and Tromp, Geophysical Journal International, vol. 149, p. 390-412 (2002), equations (19) and (21))
!
! The second-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used:
!
!            ^ y
!            |
!            |
!
!            +-------------------+
!            |                   |
!            |                   |
!            |                   |
!            |                   |
!            |                   |
!      dp/dy +---------+         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            +---------+---------+  ---> x
!            p       dp/dx

! pour afficher le resultat 2D sous forme d'image en couleurs :
!
!   " display image*.gif " ou " gimp image*.gif "
!
!   " montage -geometry +0+3 -tile 1x60 image*.gif allfiles.gif "
!   puis " display allfiles.gif " ou bien " gimp allfiles.gif "
!
! When compiling with Intel ifort, use " -assume byterecl " option to create temporary binary PNM images

  implicit none


! flags to add PML layers to the edges of the grid
  logical, parameter :: USE_PML_XMIN = .true.
  logical, parameter :: USE_PML_XMAX = .true. ! parameter attribute = a named constant (can't be altered)
  logical, parameter :: USE_PML_YMIN = .true.
  logical, parameter :: USE_PML_YMAX = .true.

! thickness of the PML layer in grid points
  integer :: NPOINTS_PML ![COMMAND LINE ARGUMENT] (sample points)   before: = 10

! total number of grid points in each direction of the grid
  integer :: NX ![COMMAND LINE ARGUMENT]  (samples)
  integer :: NY ![COMMAND LINE ARGUMENT]  (samples) (DEPTH AXIS)

! size of a grid cell
  double precision :: DELTAX ![COMMAND LINE ARGUMENT] (m)   before: = 12.5d0
  double precision :: DELTAY ![COMMAND LINE ARGUMENT] (m)   before: = DELTAX

! P-velocity, S-velocity and density
  double precision :: cp_value = 2500.d0 ! this is a double precision type  m/s
  double precision, parameter :: rho_value = 2200.d0  ! guessing this is kg/m3

! total number of time steps
  integer :: NSTEP ! [COMMAND LINE INPUT] parameter attribute means named constant

! time step in seconds
  double precision :: DELTAT ![COMMAND LINE ARGUMENT] (s)  before: = 2.d-3  

! rapport DELTAT / DELTAX
  double precision :: DELTAT_OVER_DELTAX ![CALCULATED BELOW]

! parameters for the source
! tres peu de dispersion si 8 Hz, correct si 10 Hz, dispersion sur S si 12 Hz
! mauvais si 16 Hz, tres mauvais si 25 Hz   
! [very little dispersion if 8 Hz, correct if 10 Hz, dispersion on S if 12 Hz
! bad if 16 Hz, very bad if 25 Hz ]
  double precision, parameter :: f0 = 20.d0
! decaler le Ricker en temps pour s'assurer qu'il est nul a t = 0
  double precision, parameter :: t0 = 1.2d0 / f0  ! shift the Ricker in time to make sure it is zero at t = 0
  double precision, parameter :: factor = 1.d4

! source in the medium
  double precision :: XSOURCE ![COMMAND LINE INPUT] (in meters)
  double precision :: YSOURCE ![COMMAND LINE INPUT] (in meters)

! receivers
  integer :: NREC !=11 [COMMAND LINE/ FILE INPUT]
  !double precision, parameter :: xdeb![COMMAND LINE/ FILE INPUT] = 100.d0   ! first receiver x in meters
  !double precision, parameter :: ydeb![COMMAND LINE/ FILE INPUT] = 2400.d0  ! first receiver y in meters
  !double precision, parameter :: xfin![COMMAND LINE/ FILE INPUT] = 3850.d0  ! last receiver x in meters
  !double precision, parameter :: yfin![COMMAND LINE/ FILE INPUT] = 2400.d0  ! last receiver y in meters

! affichage periodique d'informations a l'ecran
  integer, parameter :: IT_AFFICHE = 10  ! periodic display of information on the screen

! zero
  double precision, parameter :: ZERO = 0.d0

! large value for maximum
  double precision, parameter :: HUGEVAL = 1.d+30

! pressure threshold above which we consider the code became unstable
  double precision, parameter :: STABILITY_THRESHOLD = 1.d+25

! valeur de PI
  double precision, parameter :: PI = 3.141592653589793238462643d0

! main arrays
  double precision, allocatable :: xgrid(:,:),ygrid(:,:), & ! (NX, NY) DYNAMICALLY ALLOCATE THIS AFTER COMMAND LINE INPUT
      pressurepast(:,:),pressurepresent(:,:),pressurefuture(:,:), &
      pressure_xx(:,:),pressure_yy(:,:),dpressurexx_dx(:,:),dpressureyy_dy(:,:),kappa(:,:),cp(:,:),rho(:,:)

! for seismograms
  double precision, allocatable :: sispressure(:,:) !(NSTEP,NREC) DYNAMICALLY ALLOCATE THIS AFTER COMMAND LINE INPUT

! for source
  integer i_source,j_source
  double precision a,t,force_x,force_y

  integer, allocatable :: i_sources(:), j_sources(:) ![COMMAND LINE/ FILE INPUT]
  integer NSOURCES ![COMMAND LINE/ FILE INPUT]
  logical sourcefilecheck
  character(len=512) :: SOURCE_FILE ![COMMAND LINE ARGUMENT]

! for receivers
  !double precision xspacerec,yspacerec
  double precision distval,dist
  integer, allocatable :: ix_rec(:), iy_rec(:)
  !double precision, allocatable :: xrec(:), yrec(:)

  integer i,j,it,irec

  double precision nombre_Courant,pressure_max_all

! power to compute d0 profile
  double precision, parameter :: NPOWER = 2.d0

! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
  double precision, parameter :: K_MAX_PML = 1.d0
  double precision, parameter :: ALPHA_MAX_PML = 2.d0*PI*(f0/2.d0) ! from Festa and Vilotte

! arrays for the memory variables
! could declare these arrays in PML only to save a lot of memory, but proof of concept only here
  double precision, allocatable :: & ! (NX, NY) DYNAMICALLY ALLOCATE THIS AFTER COMMAND LINE INPUT
      memory_dpressure_dx(:,:), &
      memory_dpressure_dy(:,:), &
      memory_dpressurexx_dx(:,:), &
      memory_dpressureyy_dy(:,:)

  double precision :: &
      value_dpressure_dx, &
      value_dpressure_dy, &
      value_dpressurexx_dx, &
      value_dpressureyy_dy

! 1D arrays for the damping profiles
  double precision, allocatable :: d_x(:),K_x(:),alpha_prime_x(:),a_x(:),b_x(:),d_x_half(:),K_x_half(:), &
  alpha_prime_x_half(:),a_x_half(:),b_x_half(:)! (NX) DYNAMICALLY ALLOCATE THIS AFTER COMMAND LINE INPUT
  double precision, allocatable :: d_y(:),K_y(:),alpha_prime_y(:),a_y(:),b_y(:),d_y_half(:),K_y_half(:), &
  alpha_prime_y_half(:),a_y_half(:),b_y_half(:)! (NY) DYNAMICALLY ALLOCATE THIS AFTER COMMAND LINE INPUT

  double precision :: thickness_PML_x,thickness_PML_y,xoriginleft,xoriginright,yoriginbottom,yorigintop
  double precision :: Rcoef,d0_x,d0_y,xval,yval,abscissa_in_PML,abscissa_normalized

character(len=512) :: VELOCITY_FILE ![COMMAND LINE ARGUMENT]  m/s
character(len=512) :: OUT_DIR ![COMAND LINE ARGUMENT]
integer :: SIM_NUM ![COMMAND LINE ARGUMENT]
character(len=512) :: RECEIVER_FILE ![COMAND LINE ARGUMENT]
logical receiverfilecheck

character(len=512) :: command_arg ! for using command line inputs
integer :: filei, filej ! checks for importing velocity file

integer :: OUTPUT_WAVEFIELD ! [COMMAND LINE ARGUMENT]

!--- GET COMMAND LINE ARGUMENTS

if (command_argument_count()/=15) then
  print *, "ERROR: number of command line arguments must equal 15"

  print *, "NSTEP <int>",char(10),"&
  &NX <int>",char(10)," &
  &NY <int>",char(10)," &
  &DELTAX (m) <double precision>",char(10)," &
  &DELTAY (m) <double precision>",char(10)," &
  &DELTAT (s) <double precision>",char(10)," &
  &NPOINTS_PML <int>",char(10)," &
  &XSOURCE (m) <double precision>",char(10)," &
  &YSOURCE (m) <double precision>",char(10)," &
  &SOURCE_FILE <path>",char(10)," &
  &VELOCITY_FILE (m/s) <path>",char(10)," &
  &OUT_DIR <char>",char(10)," &
  &SIM_NUM <int>",char(10)," &
  &RECEIVER_FILE <path>",char(10)," &
  &OUTPUT_WAVEFIELD <0,1>",char(10)
  stop
end if

  do i = 1, command_argument_count()

    call get_command_argument(i, command_arg)

    if (i==1) then
    read (command_arg,*) NSTEP
    write(*,*) "NSTEP: ", NSTEP
    end if

    if (i==2) then
    read (command_arg,*) NX
    write(*,*) "NX: ", NX
    end if

    if (i==3) then
    read (command_arg,*) NY
    write(*,*) "NY: ", NY
    end if

    if (i==4) then
    read (command_arg,*) DELTAX
    write(*,*) "DELTAX: ", DELTAX
    end if

    if (i==5) then
    read (command_arg,*) DELTAY
    write(*,*) "DELTAY: ", DELTAY
    end if

    if (i==6) then
    read (command_arg,*) DELTAT
    write(*,*) "DELTAT: ", DELTAT
    end if

    if (i==7) then
    read (command_arg,*) NPOINTS_PML
    write(*,*) "NPOINTS_PML: ", NPOINTS_PML
    end if

    if (i==8) then
    read (command_arg,*) XSOURCE
    write(*,*) "XSOURCE: ", XSOURCE
    end if

    if (i==9) then
    read (command_arg,*) YSOURCE
    write(*,*) "YSOURCE: ", YSOURCE
    end if

    if (i==10) then
    SOURCE_FILE = command_arg
    write(*,*) "SOURCE_FILE: ", SOURCE_FILE
    end if

    if (i==11) then
    VELOCITY_FILE = command_arg
    write(*,*) "VELOCITY_FILE: ", VELOCITY_FILE
    end if

    if (i==12) then
    OUT_DIR = command_arg
    write(*,*) "OUT_DIR: ", OUT_DIR
    end if

    if (i==13) then
    read (command_arg,*) SIM_NUM
    write(*,*) "SIM_NUM: ", SIM_NUM
    end if

    if (i==14) then
    RECEIVER_FILE = command_arg
    write(*,*) "RECEIVER_FILE: ", RECEIVER_FILE
    end if

    if (i==15) then
    read (command_arg,*) OUTPUT_WAVEFIELD
    write(*,*) "OUTPUT_WAVEFIELD: ", OUTPUT_WAVEFIELD
    end if

  end do



! OPTIONAL ARGUMENTS

! READ SOURCE I,Js from FILE
inquire( file=SOURCE_FILE, exist=sourcefilecheck)
if (sourcefilecheck) then
  print *, "Importing source file.."
  open (77, file=SOURCE_FILE, status='old')! assume file exists (raise error otherwise)
  read(77,*) NSOURCES
  allocate(i_sources(NSOURCES))
  allocate(j_sources(NSOURCES))
  do i = 1,NSOURCES
    read(77,*) i_sources(i), j_sources(i)
    print *, i,": ",i_sources(i),",",j_sources(i)
  end do
  close(77)
else
  NSOURCES = 1
  allocate(i_sources(1))
  allocate(j_sources(1))
end if
print *,"NSOURCES: ",NSOURCES

! define location of receivers from file
inquire( file=RECEIVER_FILE, exist=receiverfilecheck)
if (receiverfilecheck) then
  print *, "Importing receiver file.."
  open (78, file=RECEIVER_FILE, status='old')! assume file exists (raise error otherwise)
  read(78,*) NREC
  allocate(ix_rec(NREC))
  allocate(iy_rec(NREC))
  do i = 1,NREC
    read(78,*) ix_rec(i), iy_rec(i)
    print *, i,": ",ix_rec(i),",",iy_rec(i)
  end do
  close(78)
else
  NREC = 1
  allocate(ix_rec(NREC))
  allocate(iy_rec(NREC))
  ix_rec(1) = 1
  iy_rec(1) = 1
end if
print *,"NREC: ",NREC

! dynamic allocations
allocate(sispressure(NSTEP,NREC))  ! allocate dynamic arrays

allocate(xgrid(NX,NY))! NB ALL WIDTH, HEIGHT
allocate(ygrid(NX,NY))
allocate(pressurepast(NX,NY))
allocate(pressurepresent(NX,NY))
allocate(pressurefuture(NX,NY))
allocate(pressure_xx(NX,NY))
allocate(pressure_yy(NX,NY))
allocate(dpressurexx_dx(NX,NY))
allocate(dpressureyy_dy(NX,NY))
allocate(kappa(NX,NY))
allocate(cp(NX,NY))
allocate(rho(NX,NY))

allocate(memory_dpressure_dx(NX,NY))
allocate(memory_dpressure_dy(NX,NY))
allocate(memory_dpressurexx_dx(NX,NY))
allocate(memory_dpressureyy_dy(NX,NY))

allocate(d_x(NX))
allocate(K_x(NX))
allocate(alpha_prime_x(NX))
allocate(a_x(NX))
allocate(b_x(NX))
allocate(d_x_half(NX))
allocate(K_x_half(NX))
allocate(alpha_prime_x_half(NX))
allocate(a_x_half(NX))
allocate(b_x_half(NX))

allocate(d_y(NY))
allocate(K_y(NY))
allocate(alpha_prime_y(NY))
allocate(a_y(NY))
allocate(b_y(NY))
allocate(d_y_half(NY))
allocate(K_y_half(NY))
allocate(alpha_prime_y_half(NY))
allocate(a_y_half(NY))
allocate(b_y_half(NY))

! dynamic variables
DELTAT_OVER_DELTAX = DELTAT / DELTAX

!---
!--- program starts here
!---
  

  print *
  print *,'2D acoustic finite-difference code in pressure formulation'
  print *

! create the mesh of grid points
  do j = 1,NY
    do i = 1,NX
      xgrid(i,j) = DELTAX * dble(i-1)
      ygrid(i,j) = DELTAY * dble(j-1)
    enddo
  enddo

! IMPORT VELOCITY MODEL
open (2, file=VELOCITY_FILE, status='old')! assume file exists (raise error otherwise)

! compute the Lame parameters and density
  do j = 1,NY
    do i = 1,NX

      ! IMPORT VELOCITY MODEL
      read(2,*) filei, filej, cp_value
      if (filei /= i .or. filej/= j) then
        close(2)
        print *, filei, i, filej, j, cp_value
        stop "ERROR: velocity model file does not match modelling params!"
      end if

! one can change the values of the density and P velocity model here to make it heterogeneous
      cp(i,j) = cp_value
      rho(i,j) = rho_value

      kappa(i,j) = rho(i,j)*cp(i,j)*cp(i,j)
    enddo
  enddo

close(2)

! find closest grid point for the source
  dist = HUGEVAL
  do j=1,NY
    do i=1,NX
      distval = sqrt((xgrid(i,j)-XSOURCE)**2 + (ygrid(i,j)-YSOURCE)**2)
      if(distval < dist) then
        dist = distval
        i_source = i
        j_source = j
      endif
    enddo
  enddo
  print *,'closest grid point for the source found at distance ',dist,' in i,j = ',i_source,j_source

! define location of receivers (done above..)
!  print *
!  print *,'There are ',NREC,' receivers'
!  print *
!  xspacerec = (xfin-xdeb) / dble(NREC-1)
!  yspacerec = (yfin-ydeb) / dble(NREC-1)
!  do irec=1,NREC
!    xrec(irec) = xdeb + dble(irec-1)*xspacerec ! i.e. straight line gather...
!    yrec(irec) = ydeb + dble(irec-1)*yspacerec
!  enddo

! find closest grid point for each receiver
!  do irec=1,NREC
!    dist = HUGEVAL
!    do j = 1,NY
!    do i = 1,NX
!      distval = sqrt((xgrid(i,j)-xrec(irec))**2 + (ygrid(i,j)-yrec(irec))**2)
!      if(distval < dist) then
!        dist = distval
!        ix_rec(irec) = i
!        iy_rec(irec) = j
!      endif
!    enddo
!    enddo
!    print *,'closest grid point for receiver ',irec,' found at distance ',dist,' in i,j = ',ix_rec(irec),iy_rec(irec)
!  enddo


! afficher la taille du modele   show the size of the model
  print *
  print *,'taille du modele suivant X = ',maxval(xgrid)
  print *,'taille du modele suivant Y = ',maxval(ygrid)
  print *

! verifier que la condition de stabilite de Courant est respectee
! verify that the current stability condition is met
! R. Courant et K. O. Friedrichs et H. Lewy (1928)
  nombre_Courant = maxval(cp) * DELTAT_OVER_DELTAX
  print *,'le nombre de Courant vaut ',nombre_Courant
  print *
  if(nombre_Courant > 1.d0/sqrt(2.d0)) then
    stop 'le pas de temps est trop grand, simulation instable'  ! the time step is too big, unstable simulation
  endif

! initialiser les tableaux
  pressurepresent(:,:) = ZERO
  pressurepast(:,:) = ZERO

! initialiser sismogrammes
  sispressure(:,:) = ZERO

! PML
  memory_dpressure_dx(:,:) = ZERO
  memory_dpressure_dy(:,:) = ZERO
  memory_dpressurexx_dx(:,:) = ZERO
  memory_dpressureyy_dy(:,:) = ZERO

!--- define profile of absorption in PML region

! thickness of the PML layer in meters
  thickness_PML_x = NPOINTS_PML * DELTAX
  thickness_PML_y = NPOINTS_PML * DELTAY

! reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
  Rcoef = 0.001d0

! check that NPOWER is okay
  if(NPOWER < 1) stop 'NPOWER must be greater than 1'

! compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
  d0_x = - (NPOWER + 1) * maxval(cp) * log(Rcoef) / (2.d0 * thickness_PML_x)
  d0_y = - (NPOWER + 1) * maxval(cp) * log(Rcoef) / (2.d0 * thickness_PML_y)

  print *,'d0_x = ',d0_x
  print *,'d0_y = ',d0_y
  print *

  d_x(:) = ZERO
  d_x_half(:) = ZERO
  K_x(:) = 1.d0
  K_x_half(:) = 1.d0
  alpha_prime_x(:) = ZERO
  alpha_prime_x_half(:) = ZERO
  a_x(:) = ZERO
  a_x_half(:) = ZERO

  d_y(:) = ZERO
  d_y_half(:) = ZERO
  K_y(:) = 1.d0
  K_y_half(:) = 1.d0
  alpha_prime_y(:) = ZERO
  alpha_prime_y_half(:) = ZERO
  a_y(:) = ZERO
  a_y_half(:) = ZERO

! damping in the X direction

! origin of the PML layer (position of right edge minus thickness, in meters)
  xoriginleft = thickness_PML_x
  xoriginright = (NX-1)*DELTAX - thickness_PML_x

  do i = 1,NX

! abscissa of current grid point along the damping profile
    xval = DELTAX * dble(i-1)

!---------- left edge
    if(USE_PML_XMIN) then

! define damping profile at the grid points
      abscissa_in_PML = xoriginleft - xval
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = xoriginleft - (xval + DELTAX/2.d0)
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x_half(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

!---------- right edge
    if(USE_PML_XMAX) then

! define damping profile at the grid points
      abscissa_in_PML = xval - xoriginright
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_x(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = xval + DELTAX/2.d0 - xoriginright
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_x
        d_x_half(i) = d0_x * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_x_half(i) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_x_half(i) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

! just in case, for -5 at the end
    if(alpha_prime_x(i) < ZERO) alpha_prime_x(i) = ZERO
    if(alpha_prime_x_half(i) < ZERO) alpha_prime_x_half(i) = ZERO

    b_x(i) = exp(- (d_x(i) / K_x(i) + alpha_prime_x(i)) * DELTAT)
    b_x_half(i) = exp(- (d_x_half(i) / K_x_half(i) + alpha_prime_x_half(i)) * DELTAT)

! this to avoid division by zero outside the PML
    if(abs(d_x(i)) > 1.d-6) a_x(i) = d_x(i) * (b_x(i) - 1.d0) / (K_x(i) * (d_x(i) + K_x(i) * alpha_prime_x(i)))
    if(abs(d_x_half(i)) > 1.d-6) a_x_half(i) = d_x_half(i) * &
      (b_x_half(i) - 1.d0) / (K_x_half(i) * (d_x_half(i) + K_x_half(i) * alpha_prime_x_half(i)))

  enddo

! damping in the Y direction

! origin of the PML layer (position of right edge minus thickness, in meters)
  yoriginbottom = thickness_PML_y
  yorigintop = (NY-1)*DELTAY - thickness_PML_y

  do j = 1,NY

! abscissa of current grid point along the damping profile
    yval = DELTAY * dble(j-1)

!---------- bottom edge
    if(USE_PML_YMIN) then

! define damping profile at the grid points
      abscissa_in_PML = yoriginbottom - yval
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.d0)
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y_half(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

!---------- top edge
    if(USE_PML_YMAX) then

! define damping profile at the grid points
      abscissa_in_PML = yval - yorigintop
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_y(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

! define damping profile at half the grid points
      abscissa_in_PML = yval + DELTAY/2.d0 - yorigintop
      if(abscissa_in_PML >= ZERO) then
        abscissa_normalized = abscissa_in_PML / thickness_PML_y
        d_y_half(j) = d0_y * abscissa_normalized**NPOWER
! from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
        K_y_half(j) = 1.d0 + (K_MAX_PML - 1.d0) * abscissa_normalized**NPOWER
        alpha_prime_y_half(j) = ALPHA_MAX_PML * (1.d0 - abscissa_normalized)
      endif

    endif

    b_y(j) = exp(- (d_y(j) / K_y(j) + alpha_prime_y(j)) * DELTAT)
    b_y_half(j) = exp(- (d_y_half(j) / K_y_half(j) + alpha_prime_y_half(j)) * DELTAT)

! this to avoid division by zero outside the PML
    if(abs(d_y(j)) > 1.d-6) a_y(j) = d_y(j) * (b_y(j) - 1.d0) / (K_y(j) * (d_y(j) + K_y(j) * alpha_prime_y(j)))
    if(abs(d_y_half(j)) > 1.d-6) a_y_half(j) = d_y_half(j) * &
      (b_y_half(j) - 1.d0) / (K_y_half(j) * (d_y_half(j) + K_y_half(j) * alpha_prime_y_half(j)))

  enddo

! beginning of the time loop
  do it = 1,NSTEP

! calculer les premieres derivees spatiales multipliees par les coefficients de Lame
! calculate the first spatial derivatives multiplied by the Lame coefficients
    do j = 1,NY
      do i = 1,NX-1
      value_dpressure_dx = (pressurepresent(i+1,j) - pressurepresent(i,j)) / DELTAX

      memory_dpressure_dx(i,j) = b_x_half(i) * memory_dpressure_dx(i,j) + a_x_half(i) * value_dpressure_dx

      value_dpressure_dx = value_dpressure_dx / K_x_half(i) + memory_dpressure_dx(i,j)

      pressure_xx(i,j) = value_dpressure_dx / rho(i,j)
      enddo
    enddo

    do j = 1,NY-1
      do i = 1,NX
      value_dpressure_dy = (pressurepresent(i,j+1) - pressurepresent(i,j)) / DELTAY

      memory_dpressure_dy(i,j) = b_y_half(j) * memory_dpressure_dy(i,j) + a_y_half(j) * value_dpressure_dy

      value_dpressure_dy = value_dpressure_dy / K_y_half(j) + memory_dpressure_dy(i,j)

      pressure_yy(i,j) = value_dpressure_dy / rho(i,j)
      enddo
    enddo

! calculer les deuxiemes derivees spatiales
! calculate the second spatial derivatives

! pour la mise a jour de pressure ci-dessous   for the update of pressure below
    do j = 1,NY
      do i = 2,NX
      value_dpressurexx_dx = (pressure_xx(i,j) - pressure_xx(i-1,j)) / DELTAX

      memory_dpressurexx_dx(i,j) = b_x(i) * memory_dpressurexx_dx(i,j) + a_x(i) * value_dpressurexx_dx

      value_dpressurexx_dx = value_dpressurexx_dx / K_x(i) + memory_dpressurexx_dx(i,j)

      dpressurexx_dx(i,j) = value_dpressurexx_dx
      enddo
    enddo

! pour la mise a jour ci-dessous      for the update below
    do j = 2,NY
      do i = 1,NX
      value_dpressureyy_dy = (pressure_yy(i,j) - pressure_yy(i,j-1)) / DELTAY

      memory_dpressureyy_dy(i,j) = b_y(j) * memory_dpressureyy_dy(i,j) + a_y(j) * value_dpressureyy_dy

      value_dpressureyy_dy = value_dpressureyy_dy / K_y(j) + memory_dpressureyy_dy(i,j)

      dpressureyy_dy(i,j) = value_dpressureyy_dy
      enddo
    enddo

! appliquer le schema d'evolution en temps
! on l'applique partout y compris sur certains points du bord qui n'ont pas ete calcules
! ci-dessus, ce qui est faux, mais ce n'est pas grave car on efface ces fausses valeurs
! juste apres en appliquant les conditions de Dirichlet ci-dessous

! apply the evolution scheme in time
! it is applied everywhere including on certain points of the edge which have not been calculated
! above, which is wrong, but it does not matter because we erase these false values
! just after applying the Dirichlet conditions below

  pressurefuture(:,:) = - pressurepast(:,:) + 2.d0 * pressurepresent(:,:) + &
                                  DELTAT*DELTAT * (dpressurexx_dx(:,:) + dpressureyy_dy(:,:)) * kappa(:,:)

! imposer les conditions de bord rigide de Dirichlet (pression nulle)
! impose rigid Dirichlet edge conditions (zero pressure)
! this applies Dirichlet at the bottom of the C-PML layers,
! which is the right condition to implement in order for C-PML to remain stable at long times

! bord de gauche
  pressurefuture(1,:) = ZERO

! bord de droite
  pressurefuture(NX,:) = ZERO

! bord du bas
  pressurefuture(:,1) = ZERO

! bord du haut
  pressurefuture(:,NY) = ZERO

! ajouter la source (Ricker) au point de la grille ou est situee la pression source
! add the source (Ricker) to the grid point where the source pressure is located


do i = 1, NSOURCES
  if (sourcefilecheck) then! override calculations above
    i_source = i_sources(i)
    j_source = j_sources(i)
  end if
    a = pi*pi*f0*f0
    t = dble(it-1)*DELTAT
    force_x = factor * (1.d0-2.d0*a*(t-t0)**2)*exp(-a*(t-t0)**2)
    force_y = factor * (1.d0-2.d0*a*(t-t0)**2)*exp(-a*(t-t0)**2)
    pressurefuture(i_source,j_source) = pressurefuture(i_source,j_source) + force_x / rho(i_source,j_source)
end do


! store seismograms
  do irec = 1,NREC
    sispressure(it,irec) = pressurepresent(ix_rec(irec),iy_rec(irec))! ie pressure field sampled at rec locations
  enddo


! ALWAYS write wavefield to binary (!)
  if (OUTPUT_WAVEFIELD == 1) then
    call write_wavefield(pressurepresent,NX,NY,it,OUT_DIR,SIM_NUM)
  end if

! output information every IT_AFFICHE time steps, and at time it=5
  if(mod(it,IT_AFFICHE) == 0 .or. it == 5) then

! print max absolute value of pressure
    pressure_max_all = maxval(abs(pressurepresent))
    !print *,'time step it, time t = ',it,dble(it-1)*DELTAT
    !print *,'max absolute value of pressure is ',pressure_max_all

! check stability of the code, exit if unstable
    if(pressure_max_all > STABILITY_THRESHOLD) then
        print *,'code became unstable and blew up'
        stop 1
    endif

! display de la pression sous forme d'image en couleur
    !call create_color_image(pressurepresent,NX,NY,it)  ! ie full pressure field  (it is iteration number)
    !print *,'image file written'
    !print *
    

  endif

! move new values to old values (the present becomes the past, the future becomes the present)
  pressurepast(:,:) = pressurepresent(:,:)
  pressurepresent(:,:) = pressurefuture(:,:)

  enddo   ! end of the time loop

! Always save seismograms
  call write_seismograms(sispressure,NSTEP,NREC,OUT_DIR,SIM_NUM)

! deallocate dynamic arrays

deallocate(ix_rec)
deallocate(iy_rec)

deallocate(i_sources)
deallocate(j_sources)

deallocate(sispressure)  

deallocate(xgrid)
deallocate(ygrid)
deallocate(pressurepast)
deallocate(pressurepresent)
deallocate(pressurefuture)
deallocate(pressure_xx)
deallocate(pressure_yy)
deallocate(dpressurexx_dx)
deallocate(dpressureyy_dy)
deallocate(kappa)
deallocate(cp)
deallocate(rho)

deallocate(memory_dpressure_dx)
deallocate(memory_dpressure_dy)
deallocate(memory_dpressurexx_dx)
deallocate(memory_dpressureyy_dy)

deallocate(d_x)
deallocate(K_x)
deallocate(alpha_prime_x)
deallocate(a_x)
deallocate(b_x)
deallocate(d_x_half)
deallocate(K_x_half)
deallocate(alpha_prime_x_half)
deallocate(a_x_half)
deallocate(b_x_half)

deallocate(d_y)
deallocate(K_y)
deallocate(alpha_prime_y)
deallocate(a_y)
deallocate(b_y)
deallocate(d_y_half)
deallocate(K_y_half)
deallocate(alpha_prime_y_half)
deallocate(a_y_half)
deallocate(b_y_half)

end program seismic_CPML_2D_pressure














!----
!----  routine de sauvegarde des sismogrammes
!----  backup routine of seismograms

subroutine write_wavefield(pressurepresent,NX,NY,it,OUT_DIR,SIM_NUM)

! save the full wavefield in ASCII format

implicit none

integer :: NX,NY,it
double precision, dimension(NX,NY) :: pressurepresent

integer :: ix,iy

character(len=*) :: OUT_DIR
character(len=512) :: file_name
integer :: record_len

integer :: SIM_NUM

!! WRITE AS A 64-bit BINARY file (for accuracy,speed,file size)

write(file_name,"('wavefield_',i8.8,'_',i8.8,'.bin')") SIM_NUM, it
file_name = trim(OUT_DIR)//trim(file_name)

! get record length (should be 8 bytes * NX for double precision)
inquire(iolength=record_len) pressurepresent(:,1)

! first delete the file (in case it is larger)
open(50, file=file_name, form='unformatted', access = 'direct', recl = record_len)
close(50, status='delete')
open(51, file=file_name, form='unformatted', access = 'direct', recl = record_len)
! write values
do iy=1,NY
  write(51, rec=iy) pressurepresent(:,iy)
  !write(51,"(i4.4,1x,i4.4,1x,f9.4)") ix,iy,pressurepresent(:,iy)! for text file output
enddo
close(51)

end subroutine write_wavefield




subroutine write_seismograms(sispressure,NSTEP,NREC,OUT_DIR,SIM_NUM)

! save the full wavefield in ASCII format

implicit none

integer :: NSTEP,NREC
double precision, dimension(NSTEP,NREC) :: sispressure

integer :: it,irec

character(len=*) :: OUT_DIR
character(len=512) :: file_name
integer :: record_len

integer :: SIM_NUM

!! WRITE AS A 64-bit BINARY file (for accuracy,speed,file size)

write(file_name,"('gather_',i8.8,'.bin')") SIM_NUM
file_name = trim(OUT_DIR)//trim(file_name)

! get record length (should be 8 bytes * NX for double precision)
inquire(iolength=record_len) sispressure(:,1)

! first delete the file (in case it is larger)
open(52, file=file_name, form='unformatted', access = 'direct', recl = record_len)
close(52, status='delete')
open(53, file=file_name, form='unformatted', access = 'direct', recl = record_len)
! write values
do irec=1,NREC
  write(53, rec=irec) sispressure(:,irec)
  !write(53,"(i4.4,1x,i4.4,1x,f9.4)") ix,iy,pressurepresent(:,iy)! for text file output
enddo
close(53)

end subroutine write_seismograms
