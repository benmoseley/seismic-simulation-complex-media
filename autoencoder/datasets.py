#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:12:01 2018

@author: bmoseley
"""


# This module defines a SeismicDataset class for easily loading the binary data
# stored in ../generate_data/ as pytorch tensors suitable for training and testing
# the conditional autoencoder network in main.py.
# This class is used in main.py to load training and test data, and its hyperparameters are provided
# by constants.py.


import sys
import matplotlib
if 'linux' in sys.platform.lower(): matplotlib.use('Agg')# use a non-interactive backend (ie plotting without windows)
import matplotlib.pyplot as plt

import os
import copy
import torch
from torch.utils.data import Dataset
import numpy as np


# DEFINE DATASET

class SeismicDataset(Dataset):

    def __init__(self,
                 c,
                 irange=None,
                 verbose=True):
        
        self.c = c
        self.verbose = verbose
        
        if type(irange) == type(None):
            self.n_examples = c.N_EXAMPLES
            self.irange = np.arange(self.n_examples)
        else:
            self.n_examples = len(irange)
            self.irange = np.array(irange)
        if self.verbose:
            print("%i examples"%(self.n_examples))
            print(self.irange)
    
    def __len__(self):# REQUIRED
        return self.n_examples
    
    def _preprocess(self, gather, velocity, source_i, i):
        
        ## PRE-PROCESS HERE
        gather = (gather - self.c.GATHER_MU) / self.c.GATHER_SIGMA
        velocity = (velocity - self.c.VELOCITY_MU) / self.c.VELOCITY_SIGMA
        source_i = source_i/self.c.VELOCITY_SHAPE[1]# /NX
        
        sample = {'inputs': [torch.from_numpy(velocity), torch.from_numpy(source_i)],
                  'labels': [torch.from_numpy(gather),],
                  'i': i}
        
        return sample
    
    
class SeismicBinaryDBDataset(SeismicDataset):
    """DBSeismic dataset, for use with mydataloader"""

    def __init__(self, 
                 c,
                 irange=None,
                 verbose=True):
        super().__init__(c, irange, verbose)
        
        # check dataset exists before passing self to mydataloader
        if not os.path.isfile(self.c.DATA_PATH): raise OSError("Unable to locate file: %s"%(self.c.DATA_PATH))
        
        # check expected file sizes
        self.velocity_nbytes = (np.prod(self.c.VELOCITY_SHAPE)*32)//8# 32 bit floating point numbers
        self.gather_nbytes = (np.prod(self.c.GATHER_SHAPE)*32)//8# 32 bit floating point numbers
        self.source_i_nbytes = (np.prod(self.c.SOURCE_SHAPE)*32)//8# 32 bit floating point numbers
        self.total_nbytes = self.velocity_nbytes + self.gather_nbytes + self.source_i_nbytes
        total_size, file_size = self.c.N_EXAMPLES*self.total_nbytes, os.path.getsize(self.c.DATA_PATH)
        if file_size < total_size: raise Exception("ERROR: file size < expected size: %s (%i < %i)"%(self.c.DATA_PATH, file_size, total_size))
        if file_size > total_size: print("WARNING: file size > expected size: %s (%i != %i)"%(self.c.DATA_PATH, file_size, total_size))

    def open_file_reader(self):
        "Open database file reader"
        # WARNING: do not open this file descriptor as self attribute in main thread if using 
        # multiprocessing in DataLoader: this risks sharing file descriptors across worker processes in DataLoader
        self.reader = open(self.c.DATA_PATH, 'rb')
        
    def close_file_reader(self):
        "Close database file reader"
        self.reader.close()
        
    def initialise_worker_fn(self, *args):
        "Intialise worker for multiprocessing dataloading"
        # copy some of the attributes in the worker process memory to avoid this referencing issue:
        # https://github.com/pytorch/pytorch/issues/13246
        
        self.open_file_reader()# just in case
        self.irange = np.copy(self.irange)# just in case
        self.c = copy.deepcopy(self.c)# just in case

    def __del__(self):
        if hasattr(self, 'reader'): self.close_file_reader()# just in case
        
    def __getitem__(self, i):
        # load ith data sample, given open files
        # return data sample as dictionary
        
        # read one big chunk
        self.reader.seek(self.irange[i]*self.total_nbytes)
        buf = self.reader.read(self.total_nbytes)# read raw bytes
        array = np.frombuffer(buf, dtype="<f4")# 32 bit floating point, little endian byte ordering
        
        # parse
        offset, delta = 0, np.prod(self.c.VELOCITY_SHAPE)
        velocity = array[offset:offset+delta]
        offset += delta; delta = np.prod(self.c.GATHER_SHAPE)
        gather = array[offset:offset+delta]
        offset += delta; delta = np.prod(self.c.SOURCE_SHAPE)
        source_i = array[offset:offset+delta]
        
        # values are in 'C' order, NCWH format
        velocity = velocity.reshape(self.c.VELOCITY_SHAPE)
        gather = gather.reshape(self.c.GATHER_SHAPE)
        source_i = source_i.reshape(self.c.SOURCE_SHAPE)
        
        return self._preprocess(gather, velocity, source_i, self.irange[i])


if __name__ == "__main__":
    
    from constants import Constants
    
    #from torch.utils.data import DataLoader
    from mydataloader import DataLoader
    
    c = Constants()
    print(c)
    
    torch.manual_seed(123)
    
    Dataset = SeismicBinaryDBDataset
    
    
    traindataset = Dataset(c,
                             irange=np.arange(0,7*c.N_EXAMPLES//10),
                             verbose=True)
    
    testdataset = Dataset(c,
                             irange=np.arange(7*c.N_EXAMPLES//10,c.N_EXAMPLES),
                             verbose=True)
    
    assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0
    
    trainloader = DataLoader(traindataset,
                            batch_size=c.BATCH_SIZE,
                            shuffle=True, # reshuffles data at every epoch
                            num_workers=5,# num_workers = number of multiprocessing workers
                            drop_last=True)# so that each batch is complete
    trainloader_iter = iter(trainloader)
    
    if not traindataset.isDB:# DB datasets don't support direct indexing
        print("TRAIN dataset:")
        for i in range(10):
            sample = traindataset[i]# data sample is read on the fly
            #print(sample)
            print(i, sample['inputs'][0].size(), sample['labels'][0].size(), sample['i'])
        print(sample['inputs'][0].dtype, sample['labels'][0].dtype)
        
    if not testdataset.isDB:
        print("TEST dataset:")
        for i in range(10):
            sample = testdataset[i]# data sample is read on the fly
            #print(sample)
            print(i, sample['inputs'][0].size(), sample['labels'][0].size(), sample['i'])
        print(sample['inputs'][0].dtype, sample['labels'][0].dtype)
    
    print("BATCHED dataset:")
    for i_batch in range(10):
        sample_batch = next(trainloader_iter)
        print(i_batch, sample_batch['inputs'][0].size(), sample_batch['labels'][0].size(), sample_batch['i'])

    # plot last batch
    for ib in range(5):
        
        plt.figure(figsize=(11,5))
        plt.subplot(1,2,1)
        plt.imshow(sample_batch["inputs"][0][ib,0,:,:].numpy().T, vmin=-2, vmax=2)
        plt.colorbar()
        plt.title(sample_batch["i"][ib].numpy())
        plt.subplot(1,2,2)
        plt.imshow(sample_batch["labels"][0][ib,0,:,:].numpy().T,
                   aspect=0.2, cmap="Greys", vmin=-2, vmax=2)
        plt.colorbar()
        plt.title(str(sample_batch["inputs"][1][ib,0,0,0].numpy()))
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
    
    
    