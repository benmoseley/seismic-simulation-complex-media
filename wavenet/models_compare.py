#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:34:51 2019

@author: bmoseley
"""


# This module defines alternative network designs used for testing the sensitivity
# of the accuracy of our approach to different network architectures. We define a
# standard CNN, with the option of adding dilations to its convolutional layers. This
# model is selected in constants.py and is a child class of SeismicWavenet from models.py.


import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
from tf_utils import w, b

from models import SeismicWavenet


class SeismicCNN(SeismicWavenet):
    """Defines a flat CNN class to compare the wavenet to.
    Inherets all of the boilerplate code from SeismicWavenet, just changes the model graph.
    """
    
    # 16 layers = 65 width
    # 8 layers = 129 width
    # 32 layers = 33 width
    
    def define_graph(self):
        """
        Define model graph.
        """
    
        if self.verbose: print("Defining graph...")
        
        ##
        # DEFINE VARIABLES
        ##
        
        self.N_LAYERS = len(self.c.CNN_RATES)
        self.weights, self.biases = {}, {}
        with tf.name_scope('conv_params'):
            for i in range(self.N_LAYERS):
                in_chans = out_chans = self.c.CNN_HIDDEN_CHANNELS
                if i==0: in_chans = self.x_shape[2]
                if i==self.N_LAYERS - 1: out_chans = self.y_true_shape[2]
                
                conv_kernel = [self.c.CNN_CONV_FILTER_LENGTH,in_chans,out_chans]
                stddev = np.sqrt(1) / np.sqrt(np.prod(conv_kernel[:2]))
                weights = w(conv_kernel, mean=0., stddev=stddev, name="weights")
                biases = b(conv_kernel[2:], const=0.1, name="biases")
                self.weights["conv1d_%i"%(i)] = weights
                self.biases["conv1d_%i"%(i)] = biases
        
        ##
        # DEFINE GRAPH
        ##
        
        def construct_layers(x):
            
            if self.verbose: 
                print("y_true: ",self.y_true.shape)
                print("x: ",x.shape)
            
            if self.inverse: x = x[:,::-1,:]# FLIP DATA TO REMAIN CAUSAL
            
            # CONVOLUTIONAL LAYERS
            for i in range(self.N_LAYERS):
                with tf.name_scope("conv1d_%i"%(i)):
                    # padded dilated convolution
                    x = tf.nn.convolution(x, filter=self.weights["conv1d_%i"%(i)],
                                                                 strides=[1],
                                                                 dilation_rate=[self.c.CNN_RATES[i]],
                                                                 padding="SAME", 
                                                                 data_format="NWC")
                    x = x + self.biases["conv1d_%i"%(i)]
                    # activation
                    if i!=self.N_LAYERS - 1: x = tf.nn.relu(x)
                    if self.verbose: print("conv1d_%i: "%(i),x.shape)
            
            if self.inverse: x = x[:,::-1,:]# FLIP DATA TO REMAIN CAUSAL
            
            return x
        
        ## initialise network
        self.y = construct_layers(self.x)
        
        assert self.y.shape.as_list() == self.y_true.shape.as_list()

         # print out number of weights
        self.num_weights = np.sum([self.weights[tensor].shape.num_elements() for tensor in self.weights])
        self.num_biases = np.sum([self.biases[tensor].shape.num_elements() for tensor in self.biases])
        self.total_num_trainable_params = self.num_weights+self.num_biases
        if self.verbose: print(self)
        
        # check no more trainable variables introduced
        assert self.total_num_trainable_params == np.sum([tensor.shape.num_elements() for tensor in tf.trainable_variables()])

    def __str__(self):
        if hasattr(self, "total_num_trainable_params"):
            s = "\nConv1d:\n\tNumber of weights: %i\n\tNumber of biases: %i"%(self.num_weights, self.num_biases)
            s += "\nTotal number of trainable parameters: %i"%(self.total_num_trainable_params)
            return s
        
        

if __name__ == "__main__":
    
    from constants import Constants
    from datasets import SeismicDataset
    
    c = Constants()
    
    tf.reset_default_graph()
    tf.set_random_seed(123)
    
    d = SeismicDataset(c)
    d.define_graph()
    train_features, test_features = d.train_features, d.test_features
    
    model = SeismicCNN(c, train_features, inverse=False, verbose=True)
    model.define_graph()
    model.define_loss()
    model.define_summaries()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        model.test_step(sess, summary_writer=None, show_plot=True)