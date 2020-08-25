#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:52:08 2020

@author: bmoseley
"""


# This module calculates the Zoeppritz amplitude coefficients for a P-wave incident on an acoustic
# interface, and is called by model_gather.py.


import numpy as np

def zoeppritz_acoustic(rho1,vp1,rho2,vp2,p):
    """Get reflection and transmission coefficients for a P-wave at an acoustic (fluid-fluid) interface"""
    
    cos1 = np.sqrt(1.-(p*vp1)**2)
    cos2 = np.sqrt(1+0j-(p*vp2)**2)# use complex numbers to deal with critical angle
    
    Rr = (rho2*vp2*cos1-rho1*vp1*cos2)/(rho2*vp2*cos1+rho1*vp1*cos2)
    Rt = 1+Rr
    
    return np.stack([np.abs(Rr), np.abs(Rt), np.angle(Rr), np.angle(Rt)],axis=-1)



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    angles = np.pi*np.arange(0,90,0.1)/180
    
    # speed up
    vp1,vp2 = 1000,2000
    rho1,rho2 = 1000,1200
    p = np.sin(angles)/vp1

    r = zoeppritz_acoustic(rho1,vp1,rho2,vp2,p)
    
    plt.figure()
    plt.plot(180*angles/np.pi, r[:,0], label="R")
    plt.plot(180*angles/np.pi, r[:,1], label="T")
    plt.plot(180*angles/np.pi, r[:,2], label="Rp")
    plt.plot(180*angles/np.pi, r[:,3], label="Tp")
    plt.legend()
    
    # inversion
    vp1,vp2 = 2000,1000
    rho1,rho2 = 1000,1200
    p = np.sin(angles)/vp1
    
    r = zoeppritz_acoustic(rho1,vp1,rho2,vp2,p)
    
    plt.figure()
    plt.plot(180*angles/np.pi, r[:,0], label="R")
    plt.plot(180*angles/np.pi, r[:,1], label="T")
    plt.plot(180*angles/np.pi, r[:,2], label="Rp")
    plt.plot(180*angles/np.pi, r[:,3], label="Tp")
    plt.legend()
    
    # other
    vp1,vp2 = 1500,2600
    rho1,rho2 = 2200,2200
    p = np.sin(angles)/vp1
    
    r = zoeppritz_acoustic(rho1,vp1,rho2,vp2,p)
    
    plt.figure()
    plt.plot(180*angles/np.pi, r[:,0], label="R")
    plt.plot(180*angles/np.pi, r[:,1], label="T")
    plt.plot(180*angles/np.pi, r[:,2], label="Rp")
    plt.plot(180*angles/np.pi, r[:,3], label="Tp")
    plt.legend()
    