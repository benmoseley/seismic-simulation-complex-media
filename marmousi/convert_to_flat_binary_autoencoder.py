#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:36:42 2018

@author: bmoseley
"""


# This script converts the marmousi velocity models, their simulated receiver gathers 
# and the source location for each gather into a flat, fixed record length binary file
# for efficient data loading when testing the conditional autoencoder network.
# The output binary file is stored in data/.


import os
import sys
import time
import numpy as np


class C:
    
    ROOT_DIR = ""
    
    VELOCITY_DIR = ROOT_DIR + "velocity/marmousi/"
    GATHER_DIR = ROOT_DIR + "gather/marmousi_2ms/"
    DATA_PATH = ROOT_DIR + "data/marmousi_2ms.bin"
    
    N_VELS = 100
    N_SOURCES = 3

c = C()


# convert .npy datasets into a single binary file

# load source positions
source_is = np.load(c.GATHER_DIR + "source_is.npy")

with open(c.DATA_PATH, 'wb') as f_data:
    
    # parse to flat binary
    for ivel in range(c.N_VELS):
    
        # load velocity data from .npy
        velocity = np.load(c.VELOCITY_DIR + "velocity_%.8i.npy"%(ivel))# shape: (NX, NY)
        
        for isource in range(c.N_SOURCES):
    
            gather=np.load(c.GATHER_DIR + "gather_%.8i_%.8i.npy"%(ivel,isource))# shape: (NREC, NSTEPS)
            
            source_i = source_is[ivel, isource].astype(np.float32)
            
            # write individual examples to file
            # (sacrifices size for readability (duplicates velocity model))
            velocity = velocity.astype("<f4")# ensure little endian float32
            gather = gather.astype("<f4")
            source_i = source_i.astype("<f4")
            f_data.write(velocity.tobytes(order="C"))# ensure C order
            f_data.write(gather.tobytes(order="C"))
            f_data.write(source_i.tobytes(order="C"))
            
        # periodically flush the buffer, just in case
        if (ivel+1)%1000 == 0:
            
            print(velocity.shape, gather.shape, source_i)
            print("%i of %i %s"%(ivel+1, c.N_VELS, time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())))
            f_data.flush()
            
print(os.path.getsize(c.DATA_PATH))
