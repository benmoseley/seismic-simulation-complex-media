#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:30:40 2019

@author: bmoseley
"""


# This script measures the average time taken to generate 100 forward simulations using the
# conditional autoencoder network, either on a single CPU core or a single GPU.


# 3.2761 +/- 0.0529

# 0.1762 +/- 0.0026

import numpy as np
import torch
from analysis import load_model, load_testdataset
import time

DEVICE = 1

# only use 1 core for CPU
torch.set_num_threads(1)

# Load model and dataset
model, c_dict = load_model("fault_cae", verbose=False)
d = load_testdataset("fault_2ms_r_validate.bin", N_EXAMPLES=3000, c_dict=c_dict, verbose=False)

# Get batches of test data
irange = np.arange(100)

d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# put data on GPU or CPU
device = torch.device("cuda:%i"%(DEVICE) if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
inputs = [input.to(device) for input in inputs]

# Inference
times = []
for i in range(30):
    start = time.time()
    with torch.no_grad():# faster inference without tracking
        model.eval()
        outputs = model(*inputs)
    end = time.time()
    times.append(end-start)
times = np.array(times)[10:]
print(times)
print("Total time: %.4f +/- %.4f"%(times.mean(), times.std()))
print(outputs[0].shape)