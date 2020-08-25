#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


# This module defines the Constants object, which defines all of the hyperparameters 
# used when defining and training the WaveNet (and inverse WaveNet) network.
# It also defines helper functions for easily saving the hyperparameter values 
# for each training run. 
# This class is used by main.py when training the WaveNet model, and is passed to
# datasets.py and models.py too.


import sys
import os
import pickle
import tensorflow as tf
from models import SeismicWavenet
from models_compare import SeismicCNN

sys.path.insert(0, '../shared_modules/')
import io_utils

class Constants:
    
    def __init__(self, **kwargs):
        "Define some default parameters"
        
        ######################################
        ##### GLOBAL CONSTANTS
        ######################################
        
        # DEFAULT VALUES:
        
        # Run name
        self.MODEL_RUN = "new_inverse_final_cnn_dilated"# name of run
        self.MODEL_LOAD_PATH = None# load previous model weights
        
        # Data directories
        self.DATA_PATH = "/data/greyostrich/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/data/layers_2ms.bin"
        #self.DATA_PATH = "../generate_data/data/layers_2ms.bin"
        self.N_EXAMPLES = 50000

        # Input dimensions
        self.VELOCITY_SHAPE = (114, 1)# NZ, 1
        self.REFLECTIVITY_SHAPE = (512, 1)# NSTEPS, 1
        self.GATHER_SHAPE = (512, 11)# NSTEPS, NREC

        # pre-processing  
        self.GATHER_MU = 0. # m/s, for normalising the gather in pre-processing
        self.GATHER_SIGMA = 1.0 # m/s
        
        # Wavenet parameters
        #self.MODEL = SeismicWavenet
        #self.INVERSE = True # whether to learn forward or inverse problem
        #self.WAVE_HIDDEN_CHANNELS = 128# number of hidden channels
        #self.NUM_WAVE_BLOCKS = 1# number of wavenet blocks
        #self.WAVE_RATES = [1,2,4,8,16,32,64,128,256]# defines number of hidden layers
        #self.WAVE_BIASES = False# whether to use biases in wavenet
        #self.WAVE_ACTIVATION = tf.nn.relu# activation function
        # Conv output layer parameters
        #self.CONV_FILTER_LENGTH = 101# filter length of convolutional layer
        
        # alternative CNN parameters
        self.MODEL = SeismicCNN
        self.INVERSE = True # whether to learn forward or inverse problem
        self.CNN_HIDDEN_CHANNELS = 128
        self.CNN_CONV_FILTER_LENGTH = 3# must be odd to be symmetric
        self.CNN_RATES = [1,2,4,8,16,32,64,128,256]
        #self.CNN_RATES = [1,]*9
        
        # Optimisation parameters
        self.BATCH_SIZE = 20# batch size
        self.LRATE = 1e-5# learning rate
        self.T_GAIN = 2.5# exponent of loss gain function
        self.L_NUM = 2# lp loss
        
        # Training length
        self.N_STEPS = 500000# number of training steps
        
        # Seed
        self.SEED = 123# starting seed, for repeatability
        
        # GPU parameters
        self.DEVICE = 0# cuda device
        
        ## Output reporting frequencies
        self.SUMMARY_FREQ    = 1000# how often to save the summaries, in # steps
        self.TEST_FREQ       = 2000# how often to test the model on test data, in # steps
        self.MODEL_SAVE_FREQ = 50000# how often to save the model, in # steps
        self.PLOT_FREQ       = 4000# how often to save plot summaries, in # steps
        # Note: PLOT_FREQ should be a multiple of SUMMARY_FREQ and TEST_FREQ

        ########
        
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]
        
        
    # note can set members freely, below only for index assignment
    def __getitem__(self, key):
        if key not in self.__dict__.keys(): raise Exception('key "%s" not in self.__dict__'%(key))
        return self.__dict__[key]
    def __setitem__(self, key, item):
        if key not in self.__dict__.keys(): raise Exception('key "%s" not in self.__dict__'%(key))
        if key == "MODEL_RUN" and self.__dict__[key] != item:
            for name in ["OUT_DIR", "MODEL_OUT_DIR", "SUMMARY_OUT_DIR"]:# clear out dir paths
                if hasattr(self, name): delattr(self, name)
        self.__dict__[key] = item
            
        
    ## OTHER HELPER FUNCTIONS
    def get_out_dirs(self):
        '''
        Gets output directories for model
        '''
        self.OUT_DIR = io_utils.get_dir('./results/')
        # directory for saved models
        self.MODEL_OUT_DIR = io_utils.get_dir(os.path.join(self.OUT_DIR, 'models/', self.MODEL_RUN+"/"))
        # directory for saved summaries
        self.SUMMARY_OUT_DIR = io_utils.get_dir(os.path.join(self.OUT_DIR, 'summaries/', self.MODEL_RUN+"/"))
    
    def clear_out_dirs(self):
        '''
        Clears all output content for current MODEL_RUN
        '''
        if "OUT_DIR" in self.__dict__.keys():
            io_utils.clear_dir(self.MODEL_OUT_DIR)
            io_utils.clear_dir(self.SUMMARY_OUT_DIR)
        
    def save_constants_file(self):
        "Save a constants to file in self.SUMMARY_OUT_DIR"
        # Note: pickling only saves functions/ classes by name reference so
        # the unpickling environment needs access to the source code
        # https://docs.python.org/3.7/library/pickle.html#what-can-be-pickled-and-unpickled
        if "SUMMARY_OUT_DIR" not in self.__dict__.keys(): raise Exception("ERROR: SUMMARY_OUT_DIR not defined")
        with open(self.SUMMARY_OUT_DIR + "constants_%s.txt"%(self.MODEL_RUN), 'w') as f:
            for k in self.__dict__: f.write("%s: %s\n"%(k,self[k]))
        with open(self.SUMMARY_OUT_DIR + "constants_%s.pickle"%(self.MODEL_RUN), 'wb') as f:
            pickle.dump(self.__dict__, f)
            
    def __str__(self):
        s = repr(self) + '\n'
        for k in self.__dict__: s+="%s: %s\n"%(k,self[k])
        return s
    
    
def load_constants_dict(filepath, verbose=False):
    """load a saved constants dict from filepath"""
    c_dict = pickle.load(open(filepath, "rb"))
    for key in ["OUT_DIR", "MODEL_OUT_DIR","SUMMARY_OUT_DIR"]:
        if key in c_dict: c_dict.pop(key)
    return c_dict

if __name__ == "__main__":
    
    c = Constants(MODEL_RUN="test")
    c.get_out_dirs()
    c.clear_out_dirs()
    c.save_constants_file()
    print(c)
    c["MODEL_RUN"] = "new"
    print(c)
