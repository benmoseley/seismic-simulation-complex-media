#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


# This module defines the WaveNet (and inverse WaveNet) network, which is defined 
# through a tensorflow graph stored in the SeismicWavenet class.
# This class also defines the loss function, optimiser and training operations used by
# main.py to train the network, as well as the summary statistics used for displaying the 
# results in TensorBoard.
# The SeismicWavenet class is selected in constants.py and its methods are called
# inside the training loop of main.py. Its hyperparameters are provided by constants.py.


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from losses import lp_mean_loss
from tf_utils import w, b
from wavenet import Wavenet1D

import sys
sys.path.insert(0, '../shared_modules/')
import plot_utils
import processing_utils


class SeismicWavenet:
    def __init__(self, c, input_features, inverse=None, verbose=False):
        """
        Initialize a SeismicWavenet model.
        
        inverse=True tries to learn seismic inversion
        inverse=False tries to learn forward modelling
        
        inverse takes default value from constants object.
        
        Input should follow NWC format:
            
            input_features = {  "velocity": (num_batches, NZ, 1)
                                "reflectivity":(num_batches, NSTEPS, 1)
                                "gather": (num_batches, NSTEPS, NREC)
                             }
        
        """
        if inverse == None: inverse = c.INVERSE
        
        self.c = c# model hyperparameters
        self.input_features = input_features# dictionary of tensors used as input to the model (can be placeholders or tensors)
        self.inverse = inverse
        self.verbose = verbose

        ##
        # DEFINE INPUT DATA
        ##
        
        if self.inverse:
            self.x = self.input_features["gather"]
            self.y_true = self.input_features["reflectivity"]
        else:
            self.x = self.input_features["reflectivity"]
            self.y_true = self.input_features["gather"]
        self.velocity = self.input_features["velocity"]
        
        # INPUT/OUTPUT HAS SHAPE NWC
        self.x_shape = self.x.shape.as_list()
        self.y_true_shape = self.y_true.shape.as_list()
        

    def define_graph(self):
        """
        Define model graph.
        """
    
        if self.verbose: print("Defining graph...")
        
        ##
        # DEFINE VARIABLES
        ##
        
        # define weights for wavenet
        self.W = Wavenet1D(in_channels=self.x_shape[2],
                           filter_width=2,
                           num_blocks=self.c.NUM_WAVE_BLOCKS,
                           num_layers=len(self.c.WAVE_RATES), 
                           hidden_channels=self.c.WAVE_HIDDEN_CHANNELS, 
                           rates=self.c.WAVE_RATES, 
                           activation=self.c.WAVE_ACTIVATION, 
                           biases=self.c.WAVE_BIASES, 
                           verbose=False)
        self.W.define_variables()
            
        # define weights for final convolutional layer
        self.CONV_KERNEL = [self.c.CONV_FILTER_LENGTH,self.c.WAVE_HIDDEN_CHANNELS,self.y_true_shape[2]]
        self.weights, self.biases = {}, {}
        with tf.name_scope('conv1d_params'):
            stddev = np.sqrt(1) / np.sqrt(np.prod(self.CONV_KERNEL[:2]))
            weights = w(self.CONV_KERNEL, mean=0., stddev=stddev, name="weights")
            biases = b(self.CONV_KERNEL[2:], const=0.0, name="biases")
            self.weights["conv1d"] = weights
            self.biases["conv1d"] = biases
            
        ##
        # DEFINE GRAPH
        ##
        
        def construct_layers(x):
            
            if self.verbose: 
                print("y_true: ",self.y_true.shape)
                print("x: ",x.shape)
            
            if self.inverse: x = x[:,::-1,:]# FLIP DATA TO REMAIN CAUSAL
                
            # WAVENET
            x = self.W.define_graph(x)
            if self.verbose: print("wavenet: ",x.shape)
                    
            # CONVOLUTION
            with tf.name_scope("conv1d"):
                # causal convolution
                with tf.name_scope("pad_left"):
                    x = tf.pad(x, [[0, 0], [(self.CONV_KERNEL[0]-1), 0], [0, 0]])# pad appropriate zeros on input
                x = tf.nn.convolution(x, filter=self.weights["conv1d"], strides=[1], padding="VALID", data_format="NWC")
                x = x + self.biases["conv1d"]
                if self.verbose: print("conv1d: ",x.shape)
            
            if self.inverse: x = x[:,::-1,:]# FLIP DATA TO REMAIN CAUSAL
            
            return x
        
        ## initialise network
        self.y = construct_layers(self.x)
        
        assert self.y.shape.as_list() == self.y_true.shape.as_list()

         # print out number of weights
        self.num_weights = np.sum([self.weights[tensor].shape.num_elements() for tensor in self.weights])
        self.num_biases = np.sum([self.biases[tensor].shape.num_elements() for tensor in self.biases])
        self.total_num_trainable_params = self.num_weights+self.num_biases+self.W.num_weights+self.W.num_biases
        if self.verbose: print(self)
        
        # check no more trainable variables introduced
        assert self.total_num_trainable_params == np.sum([tensor.shape.num_elements() for tensor in tf.trainable_variables()])
        
    def define_loss(self):
        """
        Define model loss and optimizer ops
        """
        
        ##
        # DEFINE LOSS, OPTIMIZER, TRAIN OP
        ##
        if self.verbose: print("Defining loss, optimizer and train op...")
        
        with tf.name_scope('loss'):
            
            # LP loss
                
            # define gain profile for forward loss
            if not self.inverse:
                gain = np.arange(0,self.c.REFLECTIVITY_SHAPE[0])**self.c.T_GAIN
                gain = gain / np.median(gain)# normalise gain profile
                gain = np.expand_dims(gain, -1)
                gain = np.pad(gain, [(0,0),(self.c.GATHER_SHAPE[1]-1,0)],mode='edge')
                gain = tf.constant(gain, dtype=tf.float32, name='gain')
                if self.verbose:
                    print("gain: ",gain.shape)
                    #print((gain*self.y).shape, (gain*self.y_true).shape)
            else:
                gain = 1.
            
            # use the same graph to evaluate train and test loss
            self.loss_train = self.loss_test = lp_mean_loss(gain*self.y, gain*self.y_true, l_num=self.c.L_NUM)
            
        with tf.name_scope('optimiser'):
            self.global_step = tf.Variable(0, trainable=False)# initialise global step variable
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.c.LRATE, name='optimizer')
            self.train_op = self.optimizer.minimize(self.loss_train, global_step=self.global_step, name='train_op')
            
    def define_summaries(self):
        """
        Define tensorboard summaries for model
        """
        
        ##
        # DEFINE TENSORBOARD SUMMARIES
        ##
        if self.verbose: print("Defining tensorboard summaries...")
        
        self.summaries_train = []   # list of train summaries
        self.summaries_test = []    # list of test summaries
        
        # weights/ biases summary (histogram)
        for tensor in self.weights:
            summary_weights_histogram = tf.summary.histogram('conv1d_weights/%s'%(tensor), self.weights[tensor])
            self.summaries_train.append(summary_weights_histogram)
        for tensor in self.biases:
            summary_biases_histogram = tf.summary.histogram('conv1d_biases/%s'%(tensor), self.biases[tensor])
            self.summaries_train.append(summary_biases_histogram)

        with tf.name_scope('accuracy'):

            # train loss summary (scalar)
            summary_loss_train = tf.summary.scalar('loss/train_loss', self.loss_train)
            self.summaries_train.append(summary_loss_train)
            
            # test loss summary (scalar)
            summary_loss_test = tf.summary.scalar('loss/test_loss', self.loss_test)
            self.summaries_test.append(summary_loss_test)

            # summary images (test and train)
            self.train_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
            self.summary_train_image = tf.summary.image('train/predictions', self.train_image, max_outputs=20)# tensor type
            
            self.test_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
            self.summary_test_image = tf.summary.image('test/predictions', self.test_image, max_outputs=20)# tensor type
            
        # training rate statistic (updated globably)
        self.trate = tf.placeholder(tf.float32, shape=(), name="trate")
        self.summary_trate = tf.summary.scalar('steps_sec', self.trate)
            
        # merge all summaries
        self.summaries_train = tf.summary.merge(self.summaries_train)
        self.summaries_test = tf.summary.merge(self.summaries_test)


    def train_step(self, sess, summary_writer=None, handle_dict=None, show_plot=False):
        """
        Runs a training step.
        """
        # training step
        _, global_step = sess.run([self.train_op,
                                   self.global_step],
                                  feed_dict=handle_dict)
                    
        # training statistics
        if global_step % self.c.SUMMARY_FREQ == 0:
            
            # generate output
            output = sess.run([self.x, self.y, self.y_true, self.velocity,
                               self.loss_train, self.summaries_train],
                              feed_dict=handle_dict)
            
            # add summaries
            if summary_writer: summary_writer.add_summary(output[5], global_step)
            
            # add plot summaries
            if global_step % self.c.PLOT_FREQ == 0 or show_plot:
                fig = self._plot_results(*output[0:4], name="train")
                if show_plot: plt.show()
                feed_dict = {self.train_image:plot_utils.fig2rgb_array(fig, expand=True)}
                if not show_plot: plt.close()
                summary_train_image = sess.run(self.summary_train_image, feed_dict=feed_dict)
                if summary_writer: summary_writer.add_summary(summary_train_image, global_step)

            print("%i Loss (train): %.4f"%(global_step, output[4]))
            
        return global_step

    def test_step(self, sess, summary_writer=None, handle_dict=None, show_plot=False):
        """
        Runs a testing step.
        """
        global_step = sess.run(self.global_step)
        
        # generate output
        output = sess.run([self.x, self.y, self.y_true, self.velocity,
                           self.loss_test, self.summaries_test],
                          feed_dict=handle_dict)
        
        # add summaries
        if summary_writer: summary_writer.add_summary(output[5], global_step)
        
        # add plot summaries
        if global_step % self.c.PLOT_FREQ == 0 or show_plot:
            fig = self._plot_results(*output[0:4], name="test")
            if show_plot: plt.show()
            feed_dict = {self.test_image:plot_utils.fig2rgb_array(fig, expand=True)}
            if not show_plot: plt.close()
            summary_test_image = sess.run(self.summary_test_image, feed_dict=feed_dict)
            if summary_writer: summary_writer.add_summary(summary_test_image, global_step)

        print("%i Loss (test): %.4f"%(global_step, output[4]))
        

    def _plot_results(self, x, y, y_true, velocity, name=""):
        """
        Plot test/train results to output
        """
        
        n_show = np.min([4, x.shape[0]])
        fig = plt.figure(figsize=(12,10))# width, height
        
        NSTEPS = self.c.REFLECTIVITY_SHAPE[0]
        NREC = self.c.GATHER_SHAPE[1]
        NZ = self.c.VELOCITY_SHAPE[0]
        
        if self.inverse: gather, reflectivity = x,y_true
        else: reflectivity, gather = x,y_true
        
        # get gain profile & limit
        gain = np.arange(0,NSTEPS)**self.c.T_GAIN
        gain = gain / np.median(gain)# normalise gain profile
        lim = 1.5
        
        for ib in range(n_show):
            irow = ib//2
            
            # PLOT VELOCITY PROFILE (DEPTH)

            if not self.inverse:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+0))
                label = "velocity"
            else:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+3))
                label="true velocity"
                
            plt.plot(velocity[ib,:,0], np.arange(NZ), color="tab:red", label=label)
            if self.inverse:
                v = processing_utils.get_velocity_trace(y[ib,:,0], srate=0.002, DZ=5., NZ=NZ, v0=velocity[ib,0,0])
                plt.plot(v, np.arange(NZ), color="tab:green", label="predicted velocity")
            if ib==2:
                plt.xlabel("Velocity (m/s)")
                plt.xticks([1000, 2000, 3000])
                plt.ylabel("Depth (samples)")
            else:
                plt.yticks([])
                plt.xticks([])
            plt.xlim(1000, 4000)
            plt.ylim(NZ, 0)
            plt.legend(loc=1)
        
            # PLOT GATHER (TIME)

            if not self.inverse:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+2), colspan=2)
            else:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+0), colspan=2)
            
            for ir in range(NREC):
                if ir == 0:
                    if not self.inverse:
                        label1="true gather"
                        label2="predicted gather"
                    else:
                        label1="gather"
                else: label1=label2=None
    
                plt.plot(ir+gain*gather[ib,:,ir]/lim, np.arange(NSTEPS), color='tab:red', label=label1)
                if not self.inverse: plt.plot(ir+gain*y[ib,:,ir]/lim, np.arange(NSTEPS), color='tab:green', label=label2)
            
            plt.xlim(-1, (NREC-1)+1)
            plt.yticks([])
            plt.xticks([])
            plt.ylim(NSTEPS,0)
            plt.legend(loc=1)
            
            # PLOT REFFLECTIVITY SERIES (TIME)
            
            if not self.inverse:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+1))
                label = "reflectivity"
            else:
                plt.subplot2grid((2, 8), (irow, 4*ib-8*irow+2))
                label="true reflectivity"
                
            plt.plot(reflectivity[ib,:,0], np.arange(NSTEPS), color='tab:red', label=label)
            if self.inverse: plt.plot(y[ib,:,0], np.arange(NSTEPS), color='tab:green', label="predicted reflectivity")

            if ib==2:
                plt.xlabel("Reflectivity")
                plt.ylabel("TWT (samples)")
                plt.yticks([100,200,300,400,500])
            else:
                plt.yticks([])
                plt.xticks([])
            plt.xlim(-0.4,0.4)
            plt.ylim(NSTEPS,0)
            plt.legend(loc=3)
            
        plt.subplots_adjust(left=0.05, bottom=0.05, right=1, top=1,
                wspace=0.2, hspace=0.0)
        return fig

    ## HELPER METHODS
    
    def __str__(self):
        if hasattr(self, "total_num_trainable_params"):
            s = "Wavenet:\n\tNumber of weights: %i\n\tNumber of biases: %i"%(self.W.num_weights, self.W.num_biases)
            s += "\nConv1d:\n\tNumber of weights: %i\n\tNumber of biases: %i"%(self.num_weights, self.num_biases)
            s += "\nTotal number of trainable parameters: %i"%(self.total_num_trainable_params)
            #for tensor in tf.trainable_variables(): print(tensor.name)
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
    
    model = SeismicWavenet(c, train_features, inverse=False, verbose=True)
    model.define_graph()
    model.define_loss()
    model.define_summaries()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        model.test_step(sess, summary_writer=None, show_plot=True)

    
    