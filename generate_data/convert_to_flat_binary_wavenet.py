#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:36:42 2018

@author: bmoseley
"""

import os
import sys
import time
import numpy as np


class C:
    
    if 'linux' in sys.platform.lower(): ROOT_DIR = "/data/greypartridge/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/"
    else: ROOT_DIR = ""

    VELOCITY_DIR = ROOT_DIR + "velocity/layers/"
    GATHER_DIR = ROOT_DIR + "gather/layers_2ms/"
    DATA_PATH = ROOT_DIR + "data/layers_2ms.bin"

    N_VELS = 50000
    N_SOURCES = 1
    
    SOURCE_Zi = 14 # depth of source (samples)

c = C()


# convert .npy datasets into a single binary file

# load source positions
source_is = np.load(c.GATHER_DIR + "source_is.npy")

with open(c.DATA_PATH, 'wb') as f_data:
    
    # parse to flat binary
    for ivel in range(c.N_VELS):
    
        # load velocity data from .npy
        velocity = np.load(c.VELOCITY_DIR + "velocity_%.8i.npy"%(ivel))# shape: (NX, NY)
        reflectivity = np.load(c.VELOCITY_DIR + "reflectivity_%.8i.npy"%(ivel))# shape: (1, NSTEPS)
        
        # do some reshaping
        # only take first trace, starting from source depth
        velocity = velocity[0:1,c.SOURCE_Zi:]
        velocity = velocity.transpose((1,0))# swap to NWC format
        reflectivity = reflectivity.transpose((1,0))
        
        for isource in range(c.N_SOURCES):
    
            gather=np.load(c.GATHER_DIR + "gather_%.8i_%.8i.npy"%(ivel,isource))# shape: (NREC, NSTEPS)
            source_i = source_is[ivel, isource].astype(np.float32)
            
            # do some reshaping
            gather = gather.transpose((1,0))
        
            # write individual examples to file
            # (sacrifices size for readability (duplicates velocity model))
            velocity = velocity.astype("<f4")# ensure little endian float32
            reflectivity = reflectivity.astype("<f4")
            gather = gather.astype("<f4")
            f_data.write(velocity.tobytes(order="C"))# ensure C order
            f_data.write(reflectivity.tobytes(order="C"))
            f_data.write(gather.tobytes(order="C"))
            
        # periodically flush the buffer, just in case
        if (ivel+1)%1000 == 0:
            
            print(velocity.shape, reflectivity.shape, gather.shape, source_i)
            print("%i of %i %s"%(ivel+1, c.N_VELS, time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())))
            f_data.flush()
            
print(os.path.getsize(c.DATA_PATH))
