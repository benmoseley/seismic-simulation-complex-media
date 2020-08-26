#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:28:31 2018

@author: bmoseley
"""

import sys
import matplotlib
if 'linux' in sys.platform.lower(): matplotlib.use('Agg')# use a non-interactive backend (ie plotting without windows)
import matplotlib.pyplot as plt

import shutil
from multiprocessing import Pool
import numpy as np

from generate_1D_traces import generate_1D_random_velocity_trace
from add_fault import add_fault

sys.path.insert(0, '../shared_modules/')
from distributions import lnLtoN
import io_utils


# INPUT VARIABLES

class Constants:
    
    def __init__(self):
        
        
        #####
        
        # OUTPUT NAME, NUMBER OF EXAMPLES
    
        #self.RUN = "layers"
        self.RUN = "fault"
        
        #self.n_examples = 50000
        self.n_examples = 100000
        
        self.random_seed = 123# DON'T FORGET TO CHANGE THIS FOR TEST DATA !
        # 123 for train/ validation
        # 1234 for test
        
        self.n_processes = 40
        
        if 'linux' in sys.platform.lower(): self.ROOT_DIR = "/data/greypartridge/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/"
        else: self.ROOT_DIR = ""
        #####
        
        
        self.VEL_DIR = self.ROOT_DIR + "velocity/" + self.RUN + "/"
    
        
        ## VELOCITY MODEL DIMENSIONS
        
        self.NPOINTS_PML = 10# for cropping
        self.NX = 108 + 2*self.NPOINTS_PML
        self.NZ = 108 + 2*self.NPOINTS_PML
        self.vm_ns = {"x":self.NX,"z":self.NZ} # size of velocity model (x,y,z, samples)
        self.vm_ds = {"dx":5.,"dz":5.} # delta of dimensions (m)
        
    
        ## 1D VELOCITY MODEL PARAMETERS
        
        self.vm_v_start = 1500 # fixed velocity value for top layer (i.e. constant initial source wavefield) (m/s)
        self.vm_thickness_start = 15 # first layer should be at least this thick (samples)
        
        # Normal distributions
        self.vm_velocity_fluctuation_mu_N, self.vm_velocity_fluctuation_sigma_N = 0.,400. # velocity fluctutation (m/s)
        
        # Log-normal distribution parameters
        self.vm_gradient_mu_L, self.vm_gradient_sigma_L = 2.2, 0.5 # velocity gradient (m/s per m)
        self.vm_v0_mu_L, self.vm_v0_sigma_L = 1800., 50. # starting velocity (m/s)
        self.vm_bed_thickness_mu_L, self.vm_bed_thickness_sigma_L = 100., 80. # bed thickness (m)  ! RECONSIDER THIS DISTRIBUTION
        
    

        ## FAULT MODEL PARAMETERS
        
        self.fm_m_range = (1.5,8)# range for fault gradient
        self.fm_x0_range = (self.vm_ns["x"]//4, 3*(self.vm_ns["x"]//4))
        self.fm_y0_range = (self.vm_ns["z"]//4, 3*(self.vm_ns["z"]//4))
        self.fm_y_start_range = (0, 2*self.vm_ns["z"]//3)
        self.fm_y_shift_range = (5, self.vm_ns["z"]//6)
        
        self.fm_fault_width = 1
        self.fm_fault_velocity = 1500
        
        
        
        # DERIVED PARAMETERS
        
        # log-normal conversions
        self.vm_gradient_mu_N, self.vm_gradient_sigma_N = lnLtoN(self.vm_gradient_mu_L, self.vm_gradient_sigma_L)
        self.vm_v0_mu_N, self.vm_v0_sigma_N = lnLtoN(self.vm_v0_mu_L, self.vm_v0_sigma_L)
        self.vm_bed_thickness_mu_N, self.vm_bed_thickness_sigma_N = lnLtoN(self.vm_bed_thickness_mu_L, self.vm_bed_thickness_sigma_L)

        
if __name__ == "__main__":
    
    # DEFINE PARAMETERS
    
    c = Constants()
        
    # SET UP DIRECTORIES
    
    io_utils.get_dir(c.VEL_DIR)
    #io_utils.clear_dir(c.VEL_DIR)  ### CAREFUL: DELETES ALL CONTENTS OF DIRECTORY RECURSIVELY
    
    shutil.copyfile('generate_velocity_models.py', c.VEL_DIR + 'generate_velocity_models_%s.py'%(c.RUN))

    
    # GENERATE 1D MODELS
    
    np.random.seed(c.random_seed)# for reproducibility
    vm_s = [generate_1D_random_velocity_trace(c) for _ in range(c.n_examples)]
    
    
    # CONVERT TO 2D, ADD FAULTS (MULTIPROCESSING)

    batches = np.array_split(np.arange(c.n_examples), np.max([1, c.n_examples // 100]))
    print("%i batches created"%(len(batches)))
    
    def generate_examples(example_indices):
        
        # FOR EACH VELOCITY MODEL
        print(example_indices)
        
        ns = np.arange(c.vm_ns["z"], dtype=float) # sample axis
        for i in example_indices:
            
            # GENERATE 1D VELOCITY PROFILE
            
            m = vm_s[i]# generated model
            vs = []# velocity axis
            cum_ln_s, vi = np.cumsum(m[0]), 0# cumulative counters
            for n in ns:
                if n > cum_ln_s[vi]: vi += 1
                vs.append(m[1][vi])
            vs = np.array(vs, dtype=float)
    
            # GENERATE 2D VELOCITY MODEL (X,Z only)
            
            velocity = np.zeros((c.vm_ns["x"],c.vm_ns["z"]))
            for ix in np.arange(c.vm_ns["x"]): velocity[ix,:] = vs
            
            ## ADD FAULT
            
            np.random.seed(c.random_seed+i)# set seed (distributed!)
            velocity = add_fault(c, velocity, fill_fault=True, plot=False)
            
            
            velocity[velocity > 4400.] = 4400.
            
            ### SAVE VELOCITY MODELS 
            velocity = velocity.astype(np.float32)
           
            # SAVE TO .NPY FILE (for QC)
            np.save(c.VEL_DIR+"velocity_%.8i.npy"%(i), velocity) # for fast retrival

            # SAVE TO .TXT FILE (for SEISMIC_CPML simulation)
            with open(c.VEL_DIR+"velocity_%.8i.txt"%(i),'w') as f:
                for iy in np.arange(c.vm_ns["z"]):
                    for ix in np.arange(c.vm_ns["x"]):
                        f.write("%i %i %.2f\n"%(ix+1, iy+1, velocity[ix, iy]))# SEISMIC CPML uses indices starting at 1

            
    # plot distributions
    l = np.concatenate([vm[0][1:] for vm in vm_s])*c.vm_ds["dz"]
    v = np.concatenate([vm[1][1:] for vm in vm_s])
    print(l.mean(), l.std(), v.mean(), v.std())
    
    plt.figure(figsize=(6,2), dpi=300)
    plt.subplot(1,2,1)
    plt.hist(v, bins=111)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Count")
    plt.subplot(1,2,2)
    plt.hist(l, bins=111)
    plt.xlabel("Layer thickness (m)")
    plt.ylabel("Count")
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(c.VEL_DIR+"distribution.pdf", bbox_inches='tight', pad_inches=0.01) # for fast retrival
    
    pool = Pool(processes=c.n_processes)
    result = pool.map(generate_examples, batches) # run in parallel

    
