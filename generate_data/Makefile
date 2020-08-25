#
# Makefile for SEISMIC_CPML Version 1.2, April 2015.
# Dimitri Komatitsch, CNRS, France
#
SHELL=/bin/sh

O = obj

# the MEDIUM_MEMORY flag is for large 3D runs, which need more than 2 GB of memory

# GNU gfortran
F90 = gfortran
MPIF90 = mpif90
FLAGS = -std=f2003 -fimplicit-none -frange-check -O3 -fmax-errors=10 -pedantic -pedantic-errors -Waliasing -Wampersand -Wcharacter-truncation -Wline-truncation -Wsurprising -Wno-tabs -Wunderflow
MEDIUM_MEMORY = -mcmodel=medium
#OPEN_MP = -fopenmp

default: clean modified_seismic_CPML_2D_pressure_second_order

all: default

clean:
	/bin/rm -f *.o xmodified_seismic_CPML_2D_pressure_second_order

modified_seismic_CPML_2D_pressure_second_order:
	$(F90) $(FLAGS) -o xmodified_seismic_CPML_2D_pressure_second_order modified_seismic_CPML_2D_pressure_second_order.f90