#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:27:19 2018

@author: bmoseley
"""


# This module adds random faults to 2D velocity models, and is called by
# generate_velocity_models.py


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def add_fault(c, velocity, fill_fault=True, plot=False):
    "add a fault to a velocity model"
    
    # fault parameters
    
    m_range = c.fm_m_range
    x0_range = c.fm_x0_range
    y0_range = c.fm_y0_range
    y_start_range = c.fm_y_start_range
    y_shift_range = c.fm_y_shift_range
    
    fault_width = c.fm_fault_width
    fault_velocity = c.fm_fault_velocity
    
    # sample distributions
    
    m = np.random.uniform(m_range[0], m_range[1])
    if np.random.random() > 0.5: m *= -1
    
    x0,y0 = (np.random.randint(x0_range[0],x0_range[1]),
             np.random.randint(y0_range[0],y0_range[1]))
    y_start = np.random.randint(y_start_range[0],y_start_range[1])
    y_shift = -np.random.randint(y_shift_range[0],y_shift_range[1])
    
    y_range = (y_start, y_start + velocity.shape[1])
    if plot: print(m, (x0,y0), y_range, y_shift)
    
    # derived parameters
    c = y0 - m*x0# for selecting slab masks
    delta_y = int(y_shift)# for actually shifting the fault (rounded)
    delta_x = int(np.round(y_shift/m))

    # pad y by delta_y
    # pad x by delta_x     # both these may over-pad (for code simplicity)
    pad_x, pad_y = np.abs(delta_x), np.abs(delta_y)
    v_pad = np.pad(np.copy(velocity), [(pad_x, pad_x), (pad_y, pad_y)], 'edge')

    # get fault block masks
    slab_mask = np.zeros(v_pad.shape)
    empty_mask = np.zeros(v_pad.shape)
    reverse = False
    if np.random.random() > 0.5: reverse = True
    for xval,col in enumerate(v_pad):
        for yval,_ in enumerate(col):
            if ((m*(xval-pad_x)+c < (yval-pad_y) and reverse) or (m*(xval-pad_x)+c > (yval-pad_y) and not reverse)):
                if (y_range[0]<(yval-pad_y)<=y_range[1]):
                    slab_mask[xval,yval] = 1# just fills slab
                if ((y_range[0]-pad_y)<(yval-pad_y)<=y_range[1]):
                    empty_mask[xval,yval] = 1# fills upwards to include delta_y
               
    # shift blocks
    slab = v_pad*slab_mask
    #order ensures no interpolation, origin is origin in original image (which is shifted to 0,0 in output)
    slab_shift = scipy.ndimage.affine_transform(slab,np.diag((1,1)),(delta_x,delta_y), order=0)
    empty_mask_shift = scipy.ndimage.affine_transform(empty_mask,np.diag((1,1)),(delta_x,delta_y), order=0)
    v_pad_shift = v_pad*(1-empty_mask_shift)+slab_shift
        
    
    # fill zeros, by whatever is above slab
    found=False
    fill_value = np.mean(velocity)
    for yval in np.arange(v_pad.shape[1]):
        row = slab_mask[:,yval]
        for xval,val in enumerate(row):
            if val == 1 and yval != 0:
                fill_value = v_pad[xval,yval-1]
                found=True
                break
        if found: break

    v_pad_fill = np.copy(v_pad_shift)
    for xval,col in enumerate(v_pad_shift):
        for yval,vval in enumerate(col):
            if vval == 0:
                v_pad_fill[xval,yval] = fill_value
            
            # add fault block
            x_fault = ((yval-pad_y-c)/m)
            if (fill_fault and
                y_range[0]<(yval-pad_y)<=(y_range[1]+pad_y) and
                (x_fault-fault_width<(xval-pad_x)<=x_fault+fault_width)):
                v_pad_fill[xval,yval] = fault_velocity

    # crop
    v_fill = v_pad_fill[pad_x:-pad_x,pad_y:-pad_y]
        
    # optionally plot
    if plot:
        x = np.arange(velocity.shape[0])
        
        plt.figure()
        plt.imshow(velocity.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x, m*x+c)
        plt.scatter(x0,y0)
        plt.scatter((y_range[0]-c)/m,y_range[0])
        plt.scatter((y_range[1]-y_shift-c)/m,y_range[1]-y_shift)
        plt.xlim(0,velocity.shape[0])
        plt.ylim(velocity.shape[1],0)
        
        x_pad = np.arange(v_pad.shape[0])
        
        plt.figure()
        plt.imshow(v_pad.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(slab_mask.T)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(empty_mask.T)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(slab_shift.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(empty_mask_shift.T)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(v_pad_shift.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(v_pad_fill.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x_pad, m*(x_pad-pad_x)+c+pad_y)
        plt.scatter(x0+pad_x,y0+pad_y)
        plt.xlim(0,v_pad.shape[0])
        plt.ylim(v_pad.shape[1],0)

        plt.figure()
        plt.imshow(v_fill.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.plot(x, m*x+c)
        plt.scatter(x0,y0)
        plt.scatter((y_range[0]-c)/m,y_range[0])
        plt.scatter((y_range[1]-y_shift-c)/m,y_range[1]-y_shift)
        plt.xlim(0,velocity.shape[0])
        plt.ylim(velocity.shape[1],0)
        
    return v_fill


if __name__ == "__main__":

    from main import Constants
    c = Constants()
    
    velocity = np.load(c.VEL_DIR+"velocity_000000.npy")
    

    np.random.seed(c.random_seed)# for reproducibility
    
    for _ in range(100):
        velocity_fill = add_fault(c, velocity, fill_fault=True, plot=False)
        
        plt.figure(figsize=(10,4))#, dpi=300)
        plt.subplot(1,2,1)
        plt.imshow(velocity.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.xlim(0,velocity.shape[0])
        plt.ylim(velocity.shape[1],0)
    
        plt.subplot(1,2,2)
        plt.imshow(velocity_fill.T, vmin=0, vmax=3500)
        plt.colorbar()
        plt.xlim(0,velocity.shape[0])
        plt.ylim(velocity.shape[1],0)
    
    
    