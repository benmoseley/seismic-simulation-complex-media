#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


# This script trains a SeismicWaveNet model, given a Constants object which contains
# all of the training hyperparameters of the model. It also sets up a summary writer
# used for saving the training statistics for TensorBoard. This script is
# the main entry point for training the WaveNet (and inverse WaveNet) networks.


import sys
import matplotlib
if 'linux' in sys.platform.lower(): matplotlib.use('Agg')# use a non-interactive backend if on remote server (plotting without windows)
import tensorflow as tf
import numpy as np

from constants import Constants
from datasets import SeismicDataset
import time

class Trainer:
    """
    Trains a model
    """
    def __init__(self, c):
        """
            Define model.
        """
        # constants object
        self.c = c
        
        # clear previous output directories/ create new output directories
        self.c.get_out_dirs()# sets up output directories
        self.c.clear_out_dirs()# (!) DELETES ALL CONTENTS OF OUTPUT DIRECTORIES!
        self.c.save_constants_file()# saves constants to file
        
        print(self.c)
        
        # make a summary writer
        self.summary_writer = tf.summary.FileWriter(self.c.SUMMARY_OUT_DIR)

        # clear previous graphs
        tf.reset_default_graph()
        
        # set random seed (for reproducibility)
        np.random.seed(self.c.SEED)
        tf.set_random_seed(self.c.SEED)# only affects current graph, use after tf.reset_default_graph()
        
        # define input data tensors
        with tf.name_scope("SeismicDataset"):
            self.dataset = SeismicDataset(self.c)
            self.dataset.define_graph()
            
        # define model
        self.model = c.MODEL(c=self.c,
                            input_features=self.dataset.features,# feedable iterator
                            inverse=self.c.INVERSE,
                            verbose=True)
        self.model.define_graph()
        self.model.define_loss()
        self.model.define_summaries()

        # make a model saver
        self.saver = tf.train.Saver(max_to_keep=5)# for saving model
        
        # startup a session
        config = tf.ConfigProto(log_device_placement=False,
				intra_op_parallelism_threads=0,# system picks appropriate number
				inter_op_parallelism_threads=0)
        config.gpu_options.visible_device_list = str(self.c.DEVICE)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        self.sess = tf.Session(config=config)
        
        # get training/ test handles
        self.train_handle = self.sess.run(self.dataset.train_iterator.string_handle())
        self.test_handle = self.sess.run(self.dataset.test_iterator.string_handle())
        
        # add graph to summary writer
        self.summary_writer.add_graph(self.sess.graph)
        
        # initialise & finalise graph
        self.sess.run(tf.global_variables_initializer())
        print("Uninitialised variables check: ", self.sess.run(tf.report_uninitialized_variables()))
        self.sess.graph.finalize()
        
        # load previous model
        if self.c.MODEL_LOAD_PATH is not None:
             # Restore (all) variables
            self.saver.restore(self.sess, self.c.MODEL_LOAD_PATH)
            print('Model parameters restored from: ' + self.c.MODEL_LOAD_PATH)
        
    def train(self):
        """
        Runs a training loop on the model networks.
        """
        
        print("Training..")
        print()
        
        start = start0 = time.time()
        for _ in np.arange(self.c.N_STEPS):
            
            # Always run train step - includes train statistics
            self.global_step = self.model.train_step(self.sess, 
                                                     summary_writer=self.summary_writer, 
                                                     handle_dict={self.dataset.handle:self.train_handle},
                                                     show_plot=False)

            # test statistics
            if self.global_step % self.c.TEST_FREQ == 0:
                self.test()
                
            # steps/ sec statistics
            if (self.global_step % self.c.SUMMARY_FREQ == 0):
                trate = self.c.SUMMARY_FREQ/(time.time()-start)
                summary_trate = self.sess.run(self.model.summary_trate, feed_dict={self.model.trate:trate})
                self.summary_writer.add_summary(summary_trate, self.global_step)
                print('%i Training steps/ second: %.2f'%(self.global_step, trate))
                print('%i Total run time: %.2f hrs'%(self.global_step, (time.time()-start0)/(60.*60.)))
                print('%i %s'%(self.global_step, time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())))
                start = time.time()
            
            # save the models
            if (self.global_step % self.c.MODEL_SAVE_FREQ == 0):
                print('-' * 70)
                print('%i Saving model...'%(self.global_step))
                save_file = self.saver.save(self.sess,
                                self.c.MODEL_OUT_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print('%i Model saved to: %s'%(self.global_step, save_file))
                print('-' * 70)
        
    def test(self):
        """
        Runs one test step on the generator network.
        """

        print('-' * 70)
        print('Testing...')
        self.model.test_step(self.sess, 
                             summary_writer=self.summary_writer, 
                             handle_dict={self.dataset.handle:self.test_handle},
                             show_plot=False)
        print('-' * 70)

    def close(self):
        ' Run closing operations'
        
        self.summary_writer.close()
        self.sess.close()
    
    
def main():
    
    # RUN TRAINING LOOP
    
    c = Constants()# run with default params
    
    run = Trainer(c)
    
    run.test()
    run.train()
    run.close()
    
if __name__ == '__main__':
        
    main()
    
