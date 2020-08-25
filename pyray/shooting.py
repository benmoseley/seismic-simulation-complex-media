#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:26:15 2020

@author: bmoseley
"""


# This module solves for the p value of a transmitted (non-reflected) ray through a stratified 2D V(z) model,
# given the final offset of the transmitted ray, using a bisection algorithm.
# This module is called by model_gather.py.


import numpy as np



def shoot_rays(dz, v, offset, xc=0.1):
    "Use bisection algorithm to get 2D travel times through a stratified 1D model for a given offset"
    
    #	1. find p interval which brackets offset
    #	2. shoot a finer fan of rays within this p interval
    #	3. repeat 1 and 2 until error small enough
    
    assert len(dz) == len(v)
    
    dz = np.abs(dz.astype(float))# use absolute values (convert any upgoing to all downgoing for purpose of p estimation)
    v = v.astype(float)
    
    ITER_MAX = 100
    N_FANS = 200
    N_LAYERS = len(v)
    
    if offset < 0: sign = -1# if negative offset, output final ray in opposite direction
    else: sign = 1
    offset = np.abs(float(offset))
    
    if offset == 0.:# deal with 0 offset case
        x = np.zeros_like(dz)# horizontal distances travelled by ray
        t = dz/v# times travelled by ray
        return x,t,0.
            
    pmax = 1/np.max(v)
    dzz = dz.reshape((N_LAYERS,1))*np.ones((1,N_FANS))# broadcasted layer height (N_layers,N_fans)
    p1,p2 = 0,pmax
    i_iter = 0
    while i_iter < ITER_MAX:
        
        i_iter+=1
        
        # p is the ray parameter which is invariant for each ray as it travels through layers, p = sin(theta) / v, thus it defines a ray
        # (because of Snell's law)
        # maximum ray parameter we should allow is 1/(v_max) otherwise Snell's law is ill-defined (reached critical angle in fastest layer)
        # for each layer one can derive time and distance travelled
        
        pfan = np.linspace(p1, p2, N_FANS, endpoint=False)# define fan of rays uniformly sampled in p (N_fans,)
        
        sin = v.reshape((N_LAYERS,1))*pfan.reshape((1,N_FANS))# broadcasted sin theta for each layer (N_layers,N_fans)
        x = np.sum((dzz*sin)/np.sqrt(1-sin**2), axis=0)# total horizontal distance travelled by ray (N_fans,)
        
        # check where the fan falls within the target offset
        # this finds indices where elements should be inserted to maintain order. Note x assumed to be sorted ascending
        i = np.searchsorted(x, offset, side="left")
        in_fan = False
        if i == N_FANS:# deal with edge cases
            p1,p2 = pfan[-1],pmax
            x1,x2 = 0,0
            #print("large", x)
        elif i == 0:# deal with edge cases
            p1,p2 = 0,pfan[0]
            x1,x2 = 0,0
            #print("small", x)
        else:
            p1,p2 = pfan[i-1],pfan[i]
            x1,x2 = x[i-1],x[i]
            assert x1 <= offset and offset <= x2
            in_fan = True
        
        # if close enough to target offset return
        if (np.abs(offset-x1)<xc or np.abs(offset-x2)<xc) and in_fan:
        
            # final linear interpolation
            p_final = p1 + (offset-x1)*(p2-p1)/(x2-x1)
        
            # sort out direction
            p_final *= sign
            
            # Compute final travel time, angle
            sin = v*p_final
            cos = np.sqrt(1-sin**2)
            tan = sin/cos
            x = dz*tan# horizontal distances travelled by ray
            t = dz/(v*cos)# times travelled by ray
            
            return x,t,p_final
    
    
    print(x1, x2, in_fan, x)
    raise Exception("No convergence")

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import time
    
    z = np.array([5,10,20,34,40], dtype=float)
    v = np.array([1800,1500,2200,1700], dtype=float)
    dz = np.diff(z)
    

    V = np.expand_dims(np.concatenate([v[i]*np.ones(int(dz[i])) for i in range(len(dz))]),-1)
    plt.figure()
    plt.imshow(V, interpolation="nearest",extent=(-10,int(1.1*100),V.shape[0],0))
    
    
    start = time.time()
    for offset in np.arange(0,100,10):
        x,t,p = shoot_rays(dz, v, offset)
        plt.plot(np.cumsum(np.concatenate([np.zeros(1),x])),z-z[0],color="k")
        print(180*np.arcsin(p*v[0])/np.pi, np.sum(x), np.sum(t))
        
    print("%.5f s"%(time.time()-start))
    