#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:28:13 2020

@author: bmoseley
"""


# This script segments the marmousi model into many 128 x 128 chunks, used to
# test the generalisation ability of the conditional autoencoder network.
# The marmousi model is taken from here: https://wiki.seg.org/wiki/AGL_Elastic_Marmousi


import numpy as np
import obspy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, '../shared_modules')
import io_utils




# 1. load raw model (1.25 x 1.25 m)

a = obspy.read("MODEL_P-WAVE_VELOCITY_1.25m.segy")

data = np.array([tr.data for tr in a.traces])

print(data.shape)

plt.figure(figsize=(20,6))
plt.imshow(data.T)
plt.colorbar()
plt.show()



# 2. decimate onto 5x5 m grid, and save

data = data[::4,::4]
#np.save("marmousi_vp.npy", data)

print(data.shape)

plt.figure(figsize=(20,6))
plt.imshow(data.T)
plt.colorbar()
plt.show()



# 3. cut out interesting AOIs

np.random.seed(1234)
LX = 128
N_BOXES = 100
boxes = LX*np.ones((N_BOXES,4),dtype=int)
boxes[:,0] = np.random.randint(1500,2200,N_BOXES)
boxes[:,1] = np.random.randint(60,200,N_BOXES)

io_utils.get_dir("velocity/marmousi/")
for ibox,box in enumerate(boxes):
    aoi = data[box[0]:box[0]+box[2],box[1]:box[1]+box[3]]
    np.save("velocity/marmousi/velocity_%.8i.npy"%(ibox), aoi)
    print(aoi.shape)

    # SAVE TO .TXT FILE (for SEISMIC_CPML simulation)
    with open("velocity/marmousi/velocity_%.8i.txt"%(ibox),'w') as f:
        for iy in range(LX):
            for ix in range(LX):
                f.write("%i %i %.2f\n"%(ix+1, iy+1, aoi[ix, iy]))# SEISMIC CPML uses indices starting at 1
                
    
plt.figure(figsize=(20,6))
plt.imshow(data.T)
plt.colorbar()
ax = plt.gca()
for box in boxes:
    ax.add_patch(patches.Rectangle(
            (box[0],box[1]),box[3],box[2],
            fill=False, edgecolor="black", linewidth=1))
plt.show()

plt.figure()
plt.imshow(aoi.T)
plt.colorbar()
plt.show()