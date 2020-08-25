#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 12:23:37 2018

@author: bmoseley
"""


'''

Simple script for defining a WaveNet model using tensorflow and python - in 150 lines!

Example use:

# define input tensor
inputs = tf.placeholder(tf.float32, shape=(<batch_size>, <num_time_samples>, <num_input_channels>))

# define wavenet model
W = Wavenet1D(num_input_channels)
W.define_variables()

# get output tensor
outputs = W.define_graph(inputs)

# output shape = (<batch_size>, <num_time_samples>, <num_hidden_channels>)

'''


import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '../shared_modules/')
from tf_utils import w,b

class Wavenet1D:
    """
    Class to define a 1D Wavenet model.
    
    Expects:
    inputs shape (2D): (?, num_time_samples, in_channels) *Assumes input is NWC format*
    
    Outputs:
    outputs shape (2D): (?, num_time_samples, hidden_channels)
    """
        
    def __init__(self, in_channels, filter_width=2, num_blocks=2, num_layers=14, hidden_channels=128, rates=None, activation=tf.nn.relu, biases=True, verbose=True):
        "Initialise class, define Wavenet structure"
        
        self.in_channels = in_channels
        self.filter_width = filter_width
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.biases = biases
        self.verbose = verbose
        
        if rates != None:
            assert len(rates) == self.num_layers
            self.rates = rates
        else:
            self.rates = [2**layer_num for layer_num in range(self.num_layers)]# default exponential dilation
            
        self.layer_weights = self.layer_biases = None
        self.num_weights = self.num_biases = None
        
    def define_variables(self, layer_variables=None):
        "Define trainable variables"

        if self.verbose: print("Defining Wavenet layer variables..")
        
        if layer_variables == None:
            with tf.name_scope("wavenet_params"):
                
                ## DEFINE VARIABLES
                in_channels = self.in_channels
                layer_weights, layer_biases = [], []
                for block_num in range(self.num_blocks):# FOR EACH BLOCK
                    for layer_num in range(self.num_layers):# FOR EACH LAYER
                        
                        name = "block_%i-layer_%i"%(block_num, layer_num)
                        with tf.name_scope(name):
                            
                            # WEIGHT INITIALISATION (==sqrt(k / number of inputs per neuron)), k=2 for relu, k=1 for tanh
                            if self.activation == tf.nn.relu:
                                stddev = np.sqrt(2) / np.sqrt(self.filter_width * in_channels)
                            elif self.activation == tf.nn.tanh:
                                stddev = np.sqrt(1) / np.sqrt(self.filter_width * in_channels)
                            else:
                                stddev = 0.01
                            weights = w(shape=(self.filter_width, in_channels, self.hidden_channels), mean=0., stddev=stddev, name="weights")
                            #weights = w(shape=(self.filter_width, in_channels, self.hidden_channels), mean=1, stddev=0, name="weights")# for testing purposes only
                            layer_weights.append(weights)
                            
                            if self.biases:
                                if self.activation == tf.nn.relu: const = 0.1
                                else: const = 0.0
                                biases = b(shape=(1, 1, self.hidden_channels), const=const, name="biases")
                                layer_biases.append(biases)
                            
                            in_channels = self.hidden_channels
                                
        else:
            layer_weights=layer_variables[0]
            layer_biases=layer_variables[1]
        
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases
        self.num_weights = np.sum([weights.shape.num_elements() for weights in self.layer_weights])
        self.num_biases = np.sum([biases.shape.num_elements() for biases in self.layer_biases])
        
        return (layer_weights, layer_biases)
            
    def define_graph(self, inputs):
        "Define tensorflow graph using inputs tensor as input"
        
        # DEFINE GRAPH
        if self.verbose: print("Defining Wavenet..")
        
        with tf.name_scope("Wavenet"):
            i=0
            hiddens = inputs
            layer_hiddens = []
            for block_num in range(self.num_blocks):# FOR EACH BLOCK
                for layer_num in range(self.num_layers):# FOR EACH LAYER
                    
                    ## DEFINE NAME/ DILATION RATE
                    rate = self.rates[layer_num] # dilation rate
                    name = "block_%i-rate_%i"%(block_num, rate)
                    
                    ## PERFORM DILATED 1D CONVOLUTION
                    if self.biases: biases = self.layer_biases[i]
                    else: biases = None
                    hiddens = self._dilated_conv1d(hiddens, self.layer_weights[i], biases=biases,
                                                  filter_width=self.filter_width, rate=rate,
                                                  name=name, activation=self.activation)
                    
                    layer_hiddens.append(hiddens)
                    i+=1
        
        self.outputs = hiddens
        
        return hiddens
        ## TODO: USE DYNAMIC PROGRAMMING FOR FAST RECURSIVE INFERENCE (O(L) instead of (O(2^L))) (see: https://github.com/tomlepaine/fast-wavenet)


    def _dilated_conv1d(self, inputs, weights, biases=None, filter_width=2, rate=1,
                       name=None, activation=tf.nn.relu):
        "Perform dillated 1d convolution. *Assumes input is NWC format* *uses a stride of 1*"
        assert name
        
        with tf.name_scope(name):
            
            # pad inputs appropriately (to remain causal & same output length)
            with tf.name_scope("pad_left"):
                inputs = tf.pad(inputs, [[0, 0], [rate*(filter_width-1), 0], [0, 0]])# pad appropriate zeros on input
            
            # perform standard 1D convolution!
            outputs =  tf.nn.convolution(inputs, 
                                         weights, 
                                         strides=[1],
                                         dilation_rate=[rate],
                                         padding="VALID", # PADDING=VALID important so that it remains causal (see: https://www.tensorflow.org/api_guides/python/nn#Convolution)
                                         data_format="NWC")
            if biases != None: outputs = outputs + biases
            if activation != None: outputs = activation(outputs)
            
        return outputs


if __name__ == "__main__":
    
    # Test code
    
    # Define a simple WaveNet graph, intialise its variables, pass an array through it and print outputs
    
    tf.reset_default_graph()
    tf.set_random_seed(123)

    # define inputs
    inputs = tf.placeholder(tf.float32, shape=(None, 5, 2))
    
    # define WaveNet graph and outputs
    W = Wavenet1D(in_channels=2, filter_width=2, num_blocks=1, num_layers=2, hidden_channels=2, activation=tf.nn.relu, biases=True)
    W.define_variables()
    print(W.rates)
    outputs = W.define_graph(inputs)
    
    # save summaries to view the graph in tensorboard
    summary_writer = tf.summary.FileWriter(".")
    summary_writer.add_graph(tf.get_default_graph())
    
    # initialise wavenet variables, pass array through it
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # print out weight/ biase values
        for block_num in range(W.num_blocks):# FOR EACH BLOCK
            for layer_num in range(W.num_layers):# FOR EACH LAYER
                print("b%i, l%i"%(block_num, layer_num))
                print(sess.run(W.layer_weights[block_num*W.num_layers+layer_num]))
                print(sess.run(W.layer_biases[block_num*W.num_layers+layer_num]))
        
        input_array = np.ones(shape=(1,5,2))
        feed_dict={inputs:input_array}
        output_array = sess.run(outputs, feed_dict=feed_dict)
        print("In:\n", input_array)
        print("Out:\n", output_array)
        