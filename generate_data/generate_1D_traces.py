#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:34:35 2018

@author: bmoseley
"""


# This module generates random 1D profiles of horizontally layered velocity models,
# and is called by generate_velocity_models.py


import numpy as np

def generate_1D_random_velocity_trace(c):
    # Use probability distributions to create a 1D velocity trace
    ## OUTPUTS: (ln_s, v_s): tuple (list of layer lengths (samples), list of velocity of layers (m/s))
    
    # 1. generate a random list of bed thicknesses (m)
    lz_s = []
    while np.sum(lz_s) < c.vm_ns["z"]*c.vm_ds["dz"]:
        lz_s.append(np.random.lognormal(c.vm_bed_thickness_mu_N, c.vm_bed_thickness_sigma_N))# randomly sample thickness
        # assert top layer to be greater than c.vm_thickness_start
        if len(lz_s) == 1: lz_s[0] += c.vm_thickness_start*c.vm_ds["dz"]
            
    ln_s = np.array(lz_s, dtype=float) / c.vm_ds["dz"]
    ln_s = np.array([np.round(x) for x in ln_s], dtype=int)# round to ints
    ln_s[-1] = ln_s[-1] - (np.sum(ln_s) - c.vm_ns["z"])# cut to model size
    
    
    # 2. generate a random list of velocities for each bed (m/s)
    v_s = []
    v0 = np.random.lognormal(c.vm_v0_mu_N, c.vm_v0_sigma_N)# randomly sample starting velocity (m/s)
    k = np.random.lognormal(c.vm_gradient_mu_N, c.vm_gradient_sigma_N)# randomly sample velocity gradient (m/s per m)
    for i in np.arange(0,len(ln_s)):
        z = np.cumsum(ln_s*c.vm_ds["dz"])[i] - ((ln_s*c.vm_ds["dz"])[i])/2.# get depth of bed
        v = -1
        while v<0:
            # randomly sample velocity fluction (m/s)
            vf = np.random.normal(c.vm_velocity_fluctuation_mu_N, c.vm_velocity_fluctuation_sigma_N)
            v = v0 + k*z + vf
            if v > 4400.0: v = 4000. + 400.*np.random.random()# soft clip so forward modelling is stable
            if v < 1000.0: v = 1000 + 400*np.random.random()
        v_s.append(v)
        #print(z, ln_s[i]*vm_ds["dz"], np.sum(ln_s), v0, k, vf, v)
    v_s = np.array(v_s)
    if c.vm_v_start != None: v_s[0] = c.vm_v_start # assert starting velocity is c.vm_v_start m/s
        
    #print(ln_s, v_s)
    
    return (ln_s[:], v_s[:])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from main import Constants
    c = Constants()
    c.n_examples = 50000
    
    ## GENERATE TRAINING DATA
    
    np.random.seed(c.random_seed)# for reproducibility
    vm_s = [generate_1D_random_velocity_trace(c) for _ in range(c.n_examples)]
    
    
    ## PLOT RESULTS
    
    ## 1. plot 1d profiles
    ns = np.arange(0, c.vm_ns["z"], dtype=float) # sample axis
    for i in np.arange(0, c.n_examples)[:6]:
        m = vm_s[i]# generated model
        vs = []# velocity axis
        cum_ln_s, vi = np.cumsum(m[0]), 0
        for n in ns:
            if n > cum_ln_s[vi]: vi += 1
            vs.append(m[1][vi])
        vs = np.array(vs, dtype=float)
        
        plt.figure()
        plt.plot(vs,ns)
        plt.xlim(0,7000)
        plt.ylim(c.vm_ns["z"],0)
    
    # 2. plot histograms
    lz_s_all = [ln_s[i]*c.vm_ds["dz"] for ln_s,_ in vm_s for i in np.arange(len(ln_s)) if i != 0]
    v_s_all = [v_s[i] for _,v_s in vm_s for i in np.arange(len(v_s)) if i != 0]
    
    x = np.arange(0.1,5000,0.1)
    plt.figure(figsize=(5,2), dpi=300)
    plt.subplot(1,2,1)
    plt.hist(v_s_all, 51, density=False)
    plt.ylabel("Count")
    plt.xlabel("Layer velocity ($\mathrm{ms}^{-1}$)")
    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.ylim(0,18000)
    
    plt.subplot(1,2,2)
    plt.hist(lz_s_all, 129, density=False)
    plt.xlim(0,c.vm_bed_thickness_mu_L + 4* c.vm_bed_thickness_sigma_L)
    #plt.ylabel("Count")
    plt.yticks([])
    plt.ylim(0,18000)
    plt.xlabel("Layer thickness (m)")
    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.subplots_adjust(left=0.1, right=1., bottom=0.1, top=1., hspace=0.02, wspace=0.1)
    #plt.savefig("../../../report_plots/distributions.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()