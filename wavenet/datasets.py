#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


# This module defines a SeismicDataset class for easily loading the binary data
# stored in ../generate_data/ as tensorflow tensors (via TensorFlow Datasets) suitable for training and testing
# the WaveNet (and inverse WaveNet) network in main.py. This class can also be sliced to output normal numpy tensors too.
# This class is used in main.py to load training and test data, and its hyperparameters are provided
# by constants.py.


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SeismicDataset:
    
    def __init__(self, c):
        "Define a seismic dataset"
        
        self.c = c
        
        # expects 32 bit floating point numbers
        self.velocity_n_bytes = int(np.prod(self.c.VELOCITY_SHAPE)*32/8)
        self.reflectivity_n_bytes = int(np.prod(self.c.REFLECTIVITY_SHAPE)*32/8)
        self.gather_n_bytes = int(np.prod(self.c.GATHER_SHAPE)*32/8)
        self.total_nbytes = self.velocity_n_bytes + self.reflectivity_n_bytes + self.gather_n_bytes
        
    def define_graph(self):
        "Define the tensorflow graph for loading this dataset"
        
        dataset = tf.data.FixedLengthRecordDataset(filenames=self.c.DATA_PATH,
                                                   record_bytes=self.total_nbytes)
        
        dataset = dataset.map(self._parse_record)
        
        # split 80/20 train / test
        train_dataset = dataset.take(80*self.c.N_EXAMPLES//100)
        test_dataset = dataset.skip(80*self.c.N_EXAMPLES//100) 

        # batch training examples
        shuffle_size = (80*self.c.N_EXAMPLES//100)//self.c.BATCH_SIZE
        train_dataset = train_dataset.repeat().shuffle(shuffle_size).batch(batch_size=self.c.BATCH_SIZE, drop_remainder=True)
        train_dataset = train_dataset.prefetch(1)
        train_iterator = train_dataset.make_one_shot_iterator()
        
        # batch test examples
        shuffle_size = (20*self.c.N_EXAMPLES//100)//self.c.BATCH_SIZE
        test_dataset = test_dataset.repeat().shuffle(shuffle_size).batch(batch_size=self.c.BATCH_SIZE, drop_remainder=True)
        test_iterator = test_dataset.make_one_shot_iterator()
        
        # for feedable iterator
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.train_features = train_iterator.get_next()
        self.test_features = test_iterator.get_next()
    
        self.handle = handle
        self.iterator = iterator
        self.features = iterator.get_next()
        
    def _parse_record(self, record):
        "parse an example from a raw binary record"
        
        tensor = tf.decode_raw(record,
                           out_type=tf.float32,# 32 bit floating point numbers
                           little_endian=True)# little endian byte ordering
        
        velocity, reflectivity, gather = self._parse_flat(tensor)
        
        # values are in 'C' order, NWC format
        velocity = tf.reshape(velocity, self.c.VELOCITY_SHAPE)
        reflectivity = tf.reshape(reflectivity, self.c.REFLECTIVITY_SHAPE)
        gather = tf.reshape(gather, self.c.GATHER_SHAPE)
        
        # pre-processing
        gather = (gather - self.c.GATHER_MU) / self.c.GATHER_SIGMA
        
        return {"velocity":velocity, "reflectivity":reflectivity, "gather":gather}
    
    def _parse_flat(self, array_like):
        
        offset, delta = 0, np.prod(self.c.VELOCITY_SHAPE)
        velocity = array_like[offset:offset+delta]
        offset += delta; delta = np.prod(self.c.REFLECTIVITY_SHAPE)
        reflectivity = array_like[offset:offset+delta]
        offset += delta; delta = np.prod(self.c.GATHER_SHAPE)
        gather = array_like[offset:offset+delta]
        return velocity, reflectivity, gather
    
    ## HELPER METHODS
    
    def __getitem__(self, s):
        """Helper method, load a single example from the binary file directly to a numpy array.
        If a slice is given, return a batch of examples."""
        
        # parse slice
        if type(s) == tuple:
            r = s
        else:
            if type(s) != slice: s = slice(s, s+1, 1)# only integer provided
            start, stop, step = s.start, s.stop, s.step
            if start == None: start = 0
            if stop == None: stop = len(self)
            if step == None: step = 1
            if step < 0: r = range(stop, start, step)
            else: r = range(start, stop, step)
        n_batches = len(r)
        #print(r)
        
        velocities = np.zeros((n_batches,)+self.c.VELOCITY_SHAPE)
        reflectivities = np.zeros((n_batches,)+self.c.REFLECTIVITY_SHAPE)
        gathers = np.zeros((n_batches,)+self.c.GATHER_SHAPE)
    
        for ib,i in enumerate(r):
            with open(self.c.DATA_PATH, "rb") as f:
                f.seek(i*self.total_nbytes)
                buf = f.read(self.total_nbytes)
            array = np.frombuffer(buf, dtype="<f4")# 32 bit floating point, little endian byte ordering
            
            velocity, reflectivity, gather = self._parse_flat(array)
            
            # values are in 'C' order, NWC format
            velocity = np.reshape(velocity, self.c.VELOCITY_SHAPE)
            reflectivity = np.reshape(reflectivity, self.c.REFLECTIVITY_SHAPE)
            gather = np.reshape(gather, self.c.GATHER_SHAPE)
            
            # pre-processing
            gather = (gather - self.c.GATHER_MU) / self.c.GATHER_SIGMA
            
            velocities[ib] = np.copy(velocity)
            reflectivities[ib] = np.copy(reflectivity)
            gathers[ib] = np.copy(gather)
        
        if n_batches == 1:
            return velocity, reflectivity, gather
        else:
            return velocities, reflectivities, gathers
    
    def __len__(self):
        return self.c.N_EXAMPLES


## HELPER FUNCTIONS
        
def plot_example(velocity, reflectivity, gather, t_gain=2.):
    
    NZ, NSTEPS, NREC = velocity.shape[0], gather.shape[0], gather.shape[1]

    fig = plt.figure(figsize=(8,5))
    plt.subplot2grid((1,4),(0,0))
    plt.plot(velocity[:,0], np.arange(NZ), color="tab:red", label="velocity")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Depth (samples)")
    plt.ylim(NZ, 0)
    
    gain = np.arange(0,NSTEPS)**t_gain# t^2 gain
    gain = gain / np.median(gain)# normalise gain profile
    lim = 1.5
    plt.subplot2grid((1, 4), (0,2), colspan=2)
    for ir in range(NREC):
        if ir == 0: label = "gather"
        else: label = None
        plt.plot(ir+gain*gather[:,ir]/lim, np.arange(NSTEPS), color='tab:red', label=label)   
    plt.xlim(-1, (NREC-1)+1)
    plt.xlabel("Receiver index")
    plt.ylabel("TWT (samples)")
    plt.ylim(NSTEPS,0)
                     
    plt.subplot2grid((1, 4), (0,1))
    plt.plot(reflectivity[:,0], np.arange(NSTEPS), color='tab:red', label="reflectivity")
    plt.xlabel("Reflectivity")
    plt.ylabel("TWT (samples)")
    plt.xlim(-0.4,0.4)
    plt.ylim(NSTEPS,0)
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=1, top=1, wspace=0.5, hspace=0.0)
    
    return fig

if __name__ == "__main__":

    import time
    from constants import Constants
    
    c = Constants()
    
    tf.reset_default_graph()
    
    tf.set_random_seed(c.SEED)
    np.random.seed(c.SEED)
    
    d = SeismicDataset(c)
    d.define_graph()
    train_features, test_features = d.train_features, d.test_features
    handle, features = d.handle, d.features
                
    with tf.Session() as sess:
        
        train_handle = sess.run(d.train_iterator.string_handle())
        test_handle = sess.run(d.test_iterator.string_handle())

        start = time.time()
        for i in range(5000):
            
            output = sess.run(features, feed_dict={handle:train_handle})
            #output = sess.run(features, feed_dict={handle:test_handle})
            #output = sess.run(train_features)
            #output = sess.run(test_features)
            
            velocity_array = output["velocity"]
            reflectivity_array = output["reflectivity"]
            gather_array = output["gather"]

            if (i+1) % 1000 == 0:
                rate = (time.time()-start)/1000
                print("Rate: %.5f s/batch"%(rate))
                
                plot_example(velocity_array[0], reflectivity_array[0], gather_array[0])
                plt.show()
                start = time.time()
                
    print("Testing indexing..")
    
    plot_example(*d[0])
    plt.show()
    plot_example(*d[1001])
    plt.show()
    
    velocity_array, reflectivity_array, gather_array = d[:10:3]
    for ib in range(velocity_array.shape[0]):
        plot_example(velocity_array[ib], reflectivity_array[ib], gather_array[ib])
