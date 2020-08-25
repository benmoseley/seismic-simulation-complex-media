#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""

import numpy as np
import matplotlib.pyplot as plt

def get_1d_reflectivity_model(v_d, qc=False, srate=0.004, DZ=12.5, NSTEPS=750):
    """Depth to time convert a velocity trace's reflectivity series.
    Return r_t."""

    v_d = np.copy(v_d)
    assert len(v_d.shape) == 1 # only handle 1d arrays
    
    ## get reflectivity series
    r_d = np.pad(np.diff(v_d),(1,0), mode="constant")
    for i,val in enumerate(r_d):
        r_d[i] = val / (v_d[i-1]+v_d[i])# reflectivity constant
    
    if qc:
        print([val for i,val in enumerate(r_d) if val != 0.0])
        print([i for i,val in enumerate(r_d) if val != 0.0])
        plt.figure()
        plt.plot(v_d)
        plt.plot(2000+4000*r_d)
        #plt.xlim(20,40)
    
    ## depth to time (regular-> irregular)
    dt_d = DZ / v_d
    t_d = np.cumsum(dt_d)
    t_d = np.pad(t_d,(1,0), mode="constant")[:-1]# == time taken starting from index 0 to get to index i
    twt_d = 2*t_d # TWT values of each velocity value
    
    
    ## interpolation (irregular -> regular)
    t = np.arange(1,NSTEPS+1)*srate# time array to interpolate to. Note the output of seismic CPML => index 0 of gathers = output AFTER 1 timestep
    twt_d = twt_d + t[0]#assumes source starts in 0 index in vel model at time index 0 of gather

    if qc:
        print("Exact TWTs: ",[twt_d[i] for i,val in enumerate(r_d) if val != 0.0])
        
    ## sampling rate check
    if qc:
        plt.figure()
        plt.scatter(twt_d,np.ones(twt_d.size),label="twt_d")
        plt.scatter(t,np.ones(t.size),label="t")
        plt.xlim(0.5,0.6)
        plt.legend()
        print(t.size, twt_d.size)
    
    if np.min(np.diff(twt_d))<srate: print("WARNING: srate greater than input TWT sample rate, risks aliasing")
    
    # interpolate velocity?
    if qc:
        # Could use linear interpolation. Not great for discrete layers as spills velocity across boundary (causing earlier arrival times)
        v_tinterp = np.interp(t, twt_d, v_d)# (linear interpolation)
        
        # could find two neighbouring points and take leftmost point. This preserves discrete layer boundaries but not quite precise enough for preserving reflectivity spike locations
        v_tleft = np.zeros(NSTEPS)
        for it in range(NSTEPS):
            i_d = (np.abs(t[it]-twt_d)).argmin()
            if t[it]-twt_d[i_d] > 0: v_tleft[it] = v_d[i_d]
            else: v_tleft[it] = v_d[i_d-1]# take left-most point
        plt.figure()
        plt.scatter(t, v_tinterp,label="time, v_tinterp",s=70)
        plt.scatter(t, v_tleft,label="time, v_tleft",s=50)
        plt.scatter(twt_d, v_d, label="depth")
        plt.xlim(0.48, 0.55)
        plt.legend()
    
    # instead: DIRECTLY INSERT REFLECTIVITY VALUES
    # means r_t will be sparse, implying blocky vel model
    # this way maintains same reflectivity/velocity resolution as velocity model/simulation, and conserves total sum of reflectivity
    r_t = np.zeros(NSTEPS)
    for ir in range(len(r_d)):#[i for i,val in enumerate(r_d) if val !=0]:
        if twt_d[ir] < np.max(t): # as long as interpolation axis greater than r point
            it = (np.abs(t-twt_d[ir])).argmin()
            r_t[it] += r_d[ir]
    
    if qc:
        plt.figure()
        plt.scatter(t, r_t, label="r_t",s=70)
        plt.scatter(twt_d, r_d, label="r_d")
        #plt.xlim(0.48, 0.55)
        plt.legend()
        print([val for i,val in enumerate(r_t) if val != 0.0])
        print([i for i,val in enumerate(r_t) if val != 0.0])
        print("Interploated TWTs: ",[t[i] for i,val in enumerate(r_t) if val != 0.0])
        print("diff: ",np.array([twt_d[i] for i,val in enumerate(r_d) if val != 0.0]) - 
              np.array([t[i] for i,val in enumerate(r_t) if val != 0.0]))
        
    return r_t

def convolve_source(r_t, source):
    "Convolve reflectivity series with source signature"
    conv = np.convolve(r_t, source)[:-(len(source)-1)]
    assert conv.shape == r_t.shape
    return conv


def get_velocity_trace(r_t, qc=False, srate=0.004, DZ=12.5, NZ=236, v0=1500.):
    """Convert a r_t trace to velocity and time to depth convert
        Return v_d."""
    r_t = np.copy(r_t)
    assert len(r_t.shape) == 1 # only handle 1d arrays
    
    ## get velocity values
    v_t = np.zeros(r_t.shape)
    v_t[0] = v0
    for i,val in enumerate(r_t):
        if i > 0:
            v_t[i] = v_t[i-1]*(1+val)/(1-val)
    
    if qc:
        plt.figure()
        plt.plot(v_t)
        plt.plot(2000+4000*r_t)
    
    ## time to depth (regular-> irregular)
    dd_t = srate * v_t /2. # convert from TWT
    d_t = np.cumsum(dd_t)
    d_t = np.pad(d_t,(1,0), mode="constant")[:-1]# == distance travelled starting from index 0 to get to index i
    
    
    ## interpolation (irregular -> regular)
    d = np.arange(0,NZ)*DZ# distance array to interpolate too
    d_t = d_t + d[0]#assumes source starts in 0 index in vel model at time index 0 of gather

    ## sampling rate check
    #if np.min(np.diff(d_t))<DZ: print("WARNING: DZ greater than input sample rate, risks aliasing")
    # probably ok to have overlap of r values in interp below
    
    # interpolate velocity?
    if qc:
        # Could use linear interpolation. Not great for discrete layers as spills velocity across boundary (causing earlier arrival times)
        v_dinterp = np.interp(d, d_t, v_t)# (linear interpolation)

        plt.figure()
        plt.scatter(d, v_dinterp,label="depth, v_dinterp",s=70)
        plt.scatter(d_t, v_t, label="time")
        plt.legend()
    
    # instead: DIRECTLY INSERT REFLECTIVITY VALUES
    # means r_t will be sparse, implying blocky vel model
    # this way maintains same reflectivity/velocity resolution as velocity model/simulation, and conserves total sum of reflectivity
    r_d = np.zeros(d.shape)
    for ir in range(len(r_t)):#[i for i,val in enumerate(r_d) if val !=0]:
        if d_t[ir] < np.max(d):# as long as interpolation axis greater than r point
            i_d = (np.abs(d-d_t[ir])).argmin()
            r_d[i_d] += r_t[ir]
    
    if qc:
        plt.figure()
        plt.scatter(d, r_d, label="r_d",s=70)
        plt.scatter(d_t, r_t, label="r_t")
        #plt.xlim(0.48, 0.55)
        plt.legend()

    ## get velocity values (in depth this time)
    v_d = np.zeros(d.shape)
    v_d[0] = v0
    for i,val in enumerate(r_d):
        if i > 0:
            v_d[i] = v_d[i-1]*(1+val)/(1-val)
            
    return v_d

    
    
if __name__ == "__main__":
    
    from datasets import SeismicDataset
    from constants import Constants
    
    c = Constants()
    d = SeismicDataset(c)
    
    velocity, _, gather = d[0]
    velocity_trace = velocity[:,0]
    
    r_t = get_1d_reflectivity_model(velocity_trace, qc=True, srate=0.008, DZ=12.5, NSTEPS=600)
    
    source = np.array([0.1,0.5,1,0.5,0.1], dtype=float)# can change to another source signature
    
    # convolve with source wavelet
    conv = convolve_source(r_t, source)
    
    plt.figure(figsize=(15,6))
    plt.subplot(2,1,1)
    plt.plot(conv, label="1d")
    plt.plot(r_t, label="r_t")
    plt.legend()
    plt.show()
    
    print("*"*60)
    
    v_d_new = get_velocity_trace(r_t, qc=True, srate=0.008, DZ=12.5, NZ=236, v0=1500.)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(velocity_trace)
    plt.plot(v_d_new)
    plt.subplot(2,1,2)
    plt.plot(velocity_trace)
    plt.plot(v_d_new)
    plt.xlim(150,236)