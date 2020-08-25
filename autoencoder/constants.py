#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:43:14 2018

@author: bmoseley
"""


# This module defines the Constants object, which defines all of the hyperparameters 
# used when defining and training the conditional autoencoder network.
# It also defines helper functions for easily saving the hyperparameter values 
# for each training run. 
# This class is used by main.py when training the autoencoder model, and is passed to
# datasets.py and models.py too.


import socket

import torch.nn.functional as F
import models
import losses

from datasets import SeismicBinaryDBDataset

import sys
sys.path.insert(0, '../shared_modules/')
from constantsBase import ConstantsBase



class Constants(ConstantsBase):
    
    def __init__(self, **kwargs):
        "Define default parameters"
        
        ######################################
        ##### GLOBAL CONSTANTS FOR MODEL
        ######################################
        

        self.RUN = "fault_constant8_vvdeep_small_final"

        self.DATA = "fault_2ms_r.bin"
        
        # GPU parameters
        self.DEVICE = 2# cuda device
        
        # Model parameters
        self.MODEL = models.AE_r
        
        self.MODEL_LOAD_PATH = None
        #self.MODEL_LOAD_PATH = "server/models/layers_new_lr1e4_b100_constant8_vvdeep_r_l1/model_03000000.torch"
        
        self.DROPOUT_RATE = 0.0# probability to drop
        self.ACTIVATION = F.relu
        
        # Optimisation parameters
        self.LOSS = losses.l1_mean_loss_gain
        self.BATCH_SIZE = 100
        self.LRATE = 1e-4
        self.WEIGHT_DECAY = 0 # L2 weight decay parameter
        
        # seed
        self.SEED = 123
        
        # training length
        self.N_STEPS = 3000000
        
        # CPU parameters
        self.N_CPU_WORKERS = 1# number of multiprocessing workers for DataLoader
        
        self.DATASET = SeismicBinaryDBDataset
        
        # input dataset properties
        self.N_EXAMPLES = 300000
        self.VELOCITY_SHAPE = (1, 128, 128)# 1, NX, NZ
        self.GATHER_SHAPE = (1, 32, 512)# 1, NREC, NSTEPS
        self.SOURCE_SHAPE = (2, 1, 1)# 2, 1, 1
        
        # pre-processing
        self.T_GAIN = 2.5# gain on gather
        self.VELOCITY_MU = 2700.0 # m/s , for normalising the velocity models in pre-processing
        self.VELOCITY_SIGMA = 560.0 # m/s , for normalising the velocity models in pre-processing        
        self.GATHER_MU = 0.
        self.GATHER_SIGMA = 1.0
        
        ## 3. SUMMARY OUTPUT FREQUENCIES
        self.SUMMARY_FREQ    = 1000    # how often to save the summaries, in # steps
        self.TEST_FREQ       = 2000    # how often to test the model on test data, in # steps
        self.MODEL_SAVE_FREQ = 250000    # how often to save the model, in # steps
        
        #self.SUMMARY_FREQ    = 2    # how often to save the summaries, in # steps
        #self.TEST_FREQ       = 4    # how often to test the model on test data, in # steps
        #self.MODEL_SAVE_FREQ = 5

        
        ########
        
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]
        
        self.SUMMARY_OUT_DIR = "results/summaries/%s/"%(self.RUN)
        self.MODEL_OUT_DIR = "results/models/%s/"%(self.RUN)
    
        self.HOSTNAME = socket.gethostname().lower()
        if 'greyostrich' in self.HOSTNAME:
            self.DATA_PATH = "/data/greyostrich/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/data/"+self.DATA
        elif 'greypartridge' in self.HOSTNAME:
            self.DATA_PATH = "/data/greypartridge/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/data/"+self.DATA
        else:
            self.DATA_PATH = "../generate_data/data/"+self.DATA
    



