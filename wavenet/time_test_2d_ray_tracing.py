#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:36:52 2019

@author: bmoseley
"""


# This script measures the average time taken to generate 100 forward simulations using the
# 2D ray tracing code in ../pyray/ on a single CPU core.



# 2.2086 +/- 0.0851


import numpy as np
import tensorflow as tf
from analysis import load_model, load_testdataset
import time

import sys
sys.path.insert(0, '../pyray')
from model_gather import model_gather
sys.path.insert(0, '../shared_modules')
import processing_utils


# Load model and dataset
tf.reset_default_graph()
model, c_dict, input_features, sess = load_model("new_forward_final", config=None, verbose=False)
d = load_testdataset("layers_2ms_validate.bin", N_EXAMPLES=1000, c_dict=c_dict, verbose=False)
#for k in c_dict: print("%s: %s"%(k, c_dict[k]))

# Get batches of test data
velocity_array, reflectivity_array, gather_array = d[:100]
print(velocity_array.shape, reflectivity_array.shape, gather_array.shape)



# set up
DT = 0.002
DX = 5.
DZ = 5.
NSTEPS = 512
NZ = 128-14 # velocity traces have been pre-processed
NREC = 11
DELTARECi=10
x_receivers = DX*(DELTARECi*(np.arange(NREC)-NREC//2))
source = np.load("../generate_data/gather/source/gather_00000000_00000000.npy")[5]

# Inference
times = []
for i in range(25):
    start = time.time()
    
    ## RAY TRACING
    
    conv2d = np.zeros(gather_array.shape)
    for ib in range(reflectivity_array.shape[0]):
        
        v0 = velocity_array[ib,:,0]
        ilayers = np.argwhere(np.diff(v0))[:,0]# ilayers stores left indices of interface (lower velocity value)
        v = np.concatenate([v0[ilayers],np.array([v0[-1]])],axis=0)
        z = np.concatenate([np.array([0]),DZ*(ilayers+1),np.array([DZ*NZ])], axis=0)# use right edges to define layer height
        rho = 2200*np.ones_like(v)
        dz = np.diff(z)
        assert len(dz) == len(v)
        
        if len(v): xs,dzs,vs,ps,rs = model_gather(dz,v,rho,0,0,x_receivers,DT,NSTEPS)
        else: rs = np.zeros((NREC,NSTEPS), dtype=float)
    
        conv2d[ib] = (rs.T).copy()
    
    # SOURCE CONVOLUTION
    
    for ib in range(velocity_array.shape[0]):
        for ir in range(NREC):
            conv2d[ib,:,ir] = 1.5*processing_utils.convolve_source(conv2d[ib,:,ir], source)

    end = time.time()
    times.append(end-start)
    
times = np.array(times)[5:]
print(times)
print("Total time: %.4f +/- %.4f"%(times.mean(), times.std()))
print(conv2d.shape)