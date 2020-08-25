#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:38:15 2019

@author: bmoseley
"""


# This script measures the average time taken to generate 100 forward simulations using the
# SEISMIC CPML library on a single CPU core.


import sys
import numpy as np
import time
sys.path.insert(0, '../shared_modules/')
import io_utils
from helper import run_command


# 73.0480 +/- 1.0574


## NB
# tried this without the i/o in fortran code, very similar speeds.

#### PARAMETERS

VEL_RUN = "layers_validate"
SIM_RUN = "temp"

ROOT_DIR = ""

N_VELS = 100


## For WAVENET
RANDOMSOURCE = False
N_REC = 11
DELTAREC = 50.0 # receiver spacing (m)

##
SOURCE_Yi = 14# source depth
##
NX = 128
NY = 128
DELTAX = 5. # grid spacing (m)
DELTAY = 5. # grid spacing (m)
NPOINTS_PML = 10 # number of PML points
NSTEPS = 512*4
DELTAT = 0.0005 # sample rate for FD modelling (s)
ds = 4# downsample factor (for pre-processing)
##


####


source_is = np.concatenate([(NX//2)*np.ones((N_VELS, 1, 1), dtype=int), SOURCE_Yi*np.ones((N_VELS, 1, 1), dtype=int)], axis=2)

receiver_is = np.array([int(np.floor( (DELTAX*NX/2. -(N_REC-1)*DELTAREC/2. + i*DELTAREC) / DELTAX) ) for i in range(N_REC)])
receiver_is = np.concatenate([receiver_is.reshape((N_REC,1)), SOURCE_Yi*np.ones((N_REC, 1), dtype=int)], axis=1)

print(source_is)
print()
print(receiver_is)
print()



VEL_DIR = ROOT_DIR + "velocity/" + VEL_RUN + "/"
OUT_SIM_DIR = ROOT_DIR + "gather/" + SIM_RUN + "/"
io_utils.get_dir(OUT_SIM_DIR)

ivel = 0

SIM_NUM = ivel
VEL_FILE = VEL_DIR + "velocity_%.8i.txt"%(ivel) 

OUTPUT_WAVEFIELD = 0# whether to output wavefield (I/O heavy!)

# run a separate simulation for each source
source_i = source_is[SIM_NUM,0]

# create a temporary directory for simulation output (prevent I/O clash between processes)
TEMP_OUT_SIM_DIR = OUT_SIM_DIR + str(SIM_NUM) + "/"
io_utils.get_dir(TEMP_OUT_SIM_DIR)

# create receiver file
RECEIVER_FILE = TEMP_OUT_SIM_DIR + "receiver_ijs_%s_%i.txt"%(SIM_RUN,SIM_NUM)
with open(RECEIVER_FILE,'w') as f:
    f.write("%i\n"%(N_REC))
    for rec_i in receiver_is: f.write("%i %i\n"%(rec_i[0]+1, rec_i[1]+1))# SEISMIC CPML uses indices starting at 1
 
# create source file (single source)
SOURCE_FILE = TEMP_OUT_SIM_DIR + "source_ijs_%s_%i.txt"%(SIM_RUN,SIM_NUM)
with open(SOURCE_FILE,'w') as f:
    f.write("%i\n"%(1))
    f.write("%i %i\n"%(source_i[0]+1, source_i[1]+1))# SEISMIC CPML uses indices starting at 1


# RUN FORWARD SIMULATION

cmd = "./xben_seismic_CPML_2D_pressure_second_order " + \
    "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"%(
    NSTEPS,
    NX,
    NY,
    DELTAX,
    DELTAY,
    DELTAT,
    NPOINTS_PML,
    0,# SOURCE_X (m)
    0,# SOURCE_Z (m)
    SOURCE_FILE,
    VEL_FILE,
    TEMP_OUT_SIM_DIR,
    SIM_NUM,
    RECEIVER_FILE,
    OUTPUT_WAVEFIELD)

# Inference
times = []
for i in range(25):
    start = time.time()
    for i in range(100):
        return_code = run_command(cmd.split(" "),verbose=False) # run
    end = time.time()
    times.append(end-start)
    print(end-start)
    
    if return_code != 0:
        print("ERROR: Simulation broke, check stderr")
    
times = np.array(times)[5:]
print(times)
print("Total time: %.4f +/- %.4f"%(times.mean(), times.std()))


io_utils.remove_dir(TEMP_OUT_SIM_DIR)


    
