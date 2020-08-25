#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:36:52 2019

@author: bmoseley
"""


# This script measures the average time taken to generate 100 forward simulations using the
# WaveNet network, either on a single CPU core or a single GPU. It also measures the
# average time taken to generate 100 velocity predictions using the inverse WaveNet.



# 3.7902 +/- 0.0347
# 1.2691 +/- 0.0170

# 0.1335 +/- 0.0008
# 0.0508 +/- 0.0007

import socket
import numpy as np
import tensorflow as tf
from analysis import load_model, load_testdataset
import time

DEVICE = 1

# use GPU or CPU
if 'greyostrich' in socket.gethostname().lower():
    config = tf.ConfigProto(log_device_placement=False,
    				intra_op_parallelism_threads=0,# system picks appropriate number
    				inter_op_parallelism_threads=0)
    config.gpu_options.visible_device_list = str(DEVICE)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
else:
    config = tf.ConfigProto()
    config = tf.ConfigProto(device_count = {'CPU': 1})
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
print(config)


# Load model and dataset
tf.reset_default_graph()
model, c_dict, input_features, sess = load_model("new_forward_final", config=config, verbose=False)
d = load_testdataset("layers_2ms_validate.bin", N_EXAMPLES=1000, c_dict=c_dict, verbose=False)
#for k in c_dict: print("%s: %s"%(k, c_dict[k]))

# Get batches of test data
velocity_array, reflectivity_array, gather_array = d[:100]
print(velocity_array.shape, reflectivity_array.shape, gather_array.shape)

# Inference
times = []
for i in range(25):
    start = time.time()
    gather_prediction_array = sess.run(model.y, feed_dict={input_features["reflectivity"]: reflectivity_array})
    end = time.time()
    times.append(end-start)
times = np.array(times)[5:]
print(times)
print("Total time: %.4f +/- %.4f"%(times.mean(), times.std()))
print(gather_prediction_array.shape)








# Load model and dataset
tf.reset_default_graph()
model, c_dict, input_features, sess = load_model("new_inverse_final", config=config, verbose=False)
d = load_testdataset("layers_2ms_validate.bin", N_EXAMPLES=1000, c_dict=c_dict, verbose=False)
#for k in c_dict: print("%s: %s"%(k, c_dict[k]))

# Get batches of test data
velocity_array, reflectivity_array, gather_array = d[:100]
print(velocity_array.shape, reflectivity_array.shape, gather_array.shape)

# Inference
times = []
for i in range(25):
    start = time.time()
    reflectivity_prediction_array = sess.run(model.y, feed_dict={input_features["gather"]: gather_array})
    end = time.time()
    times.append(end-start)
times = np.array(times)[5:]
print(times)
print("Total time: %.4f +/- %.4f"%(times.mean(), times.std()))
print(reflectivity_prediction_array.shape)
