#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:04:43 2018

@author: bmoseley
"""


# This script trains a conditional autoencoder model, given a Constants object which contains
# all of the training hyperparameters of the model.
# It defines the loss function, optimiser and training operations used to train the network, 
# as well as the summary statistics used for displaying the results in TensorBoard.
# This script is the main entry point for training the conditional autoencoder network.


import sys
import os
import time

import matplotlib
if 'linux' in sys.platform.lower(): matplotlib.use('Agg')# use a non-interactive backend (ie plotting without windows)
import matplotlib.pyplot as plt

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
from torch.utils.data import RandomSampler, DataLoader
from mysampler import BatchSampler
from constants import Constants
import losses
from torch_utils import get_weights, get_weights_update_percent



## This needs to be specified - problem dependent
def plot_result(inputs_array, outputs_array, labels_array, sample_batch, ib=0, isource=0,
                aspect=0.2):
    "Plot a network prediction, compare to ground truth and input"
    f = plt.figure(figsize=(12,5))
    
    # define gain profile for display
    t_gain = np.arange(outputs_array.shape[-1], dtype=np.float32)**2.5
    t_gain = t_gain/np.median(t_gain)
    t_gain = t_gain.reshape((1,1,1,outputs_array.shape[-1]))# along NSTEPS
    
    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    plt.imshow(inputs_array[ib,0,:,:].T, vmin=-1, vmax=1)
    plt.colorbar()
    
    plt.subplot2grid((1, 4), (0, 2), colspan=1)
    plt.imshow((t_gain*outputs_array)[ib,isource,:,:].T,
               aspect=aspect, cmap="Greys", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("%f, %f"%(np.min(outputs_array),np.max(outputs_array)))
    
    plt.subplot2grid((1, 4), (0, 3), colspan=1)
    plt.imshow((t_gain*labels_array)[ib,isource,:,:].T,
               aspect=aspect, cmap="Greys", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("%s"%(sample_batch["inputs"][1].detach().cpu().numpy().copy()[ib,:,0,0]))# label with source position
    
    return f
        


class Trainer:
    "Generic model trainer class"
    
    def __init__(self, c):
        "Initialise torch, output directories, training dataset and model"
        
        
        ## INITIALISE
        
        # set seed
        if c.SEED == None: c.SEED = torch.initial_seed()
        else: torch.manual_seed(c.SEED)# likely independent of numpy
        np.random.seed(c.SEED)
                       
        # clear directories
        c.get_outdirs()
        c.save_constants_file()# saves torch seed too
        print(c)
        
        # set device/ threads
        device = torch.device("cuda:%i"%(c.DEVICE) if torch.cuda.is_available() else "cpu")
        print("Device: %s"%(device))
        torch.backends.cudnn.benchmark = False#let cudnn find the best algorithm to use for your hardware (not good for dynamic nets)
        torch.set_num_threads(1)# for main inference
        
        print("Main thread ID: %i"%os.getpid())
        print("Number of CPU threads: ", torch.get_num_threads())
        print("Torch seed: ", torch.initial_seed())
        
        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)


        ### DEFINE TRAIN/TEST DATASETS
        
        # split dataset 80:20
        irange = np.arange(0, c.N_EXAMPLES)
        np.random.shuffle(irange)# randomly shuffle the indicies (in place) before splitting. To get diversity in train/test split.
        traindataset = c.DATASET(c,
                                 irange=irange[0:(8*c.N_EXAMPLES//10)],
                                 verbose=True)
        testdataset = c.DATASET(c,
                                irange=irange[(8*c.N_EXAMPLES//10):c.N_EXAMPLES],
                                verbose=True)
        assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0# make sure examples aren't shared!
        
        #### DEFINE MODEL
        
        model = c.MODEL(c)
        
        # load previous weights
        if c.MODEL_LOAD_PATH != None:
            cp = torch.load(c.MODEL_LOAD_PATH,
                            map_location=torch.device('cpu'))# remap tensors from gpu to cpu if needed
            model.load_state_dict(cp['model_state_dict'])
            ioffset = cp["i"]
            print("Loaded model weights from: %s"%(c.MODEL_LOAD_PATH))
        else: ioffset = 0
        
        # print out parameters
        #writer.add_graph(model, torch.zeros((1,)+c.VELOCITY_SHAPE))# write graph before placing on GPU
        print()
        print("Model: %s"%(model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i"%(total_params))
        print("Total number of trainable parameters: %i"%(total_trainable_params))
        #for p in model.parameters(): print(p.size(), p.numel())
        
        model.to(device)

        self.c, self.device, self.writer = c, device, writer
        self.traindataset, self.testdataset = traindataset, testdataset
        self.model, self.ioffset = model, ioffset

    def train(self):
        "train model"
        
        c, device, writer = self.c, self.device, self.writer
        traindataset, testdataset = self.traindataset, self.testdataset
        model, ioffset = self.model, self.ioffset
        
        ### TRAIN
        
        print()
        print("Training..")
        
        N_BATCHES = len(traindataset)//c.BATCH_SIZE
        N_EPOCHS = int(np.ceil(c.N_STEPS/N_BATCHES))
        
        # below uses my own batch sampler so that dataloader iterators run over n_epochs
        # also uses dataset.initialise_file_reader method to open a file handle in each worker process, instead of a shared one on the main thread
        # DataLoader essentially iterates through iter(batch_sampler) or iter(sampler) depending on inputs
        # calling worker_init in each worker process
        trainloader = DataLoader(traindataset,
                        batch_sampler=BatchSampler(RandomSampler(traindataset, replacement=True),# randomly sample with replacement
                                                   batch_size=c.BATCH_SIZE,
                                                   drop_last=True,
                                                   n_epochs=1),
                        worker_init_fn=traindataset.initialise_worker_fn,
                        num_workers=c.N_CPU_WORKERS,# num_workers = spawns multiprocessing subprocess workers
                        timeout=300)# timeout after 5 mins of no data loading
        
        testloader = DataLoader(testdataset,
                        batch_sampler=BatchSampler(RandomSampler(testdataset, replacement=True),# randomly sample with replacement
                                                   batch_size=c.BATCH_SIZE,
                                                   drop_last=True,
                                                   n_epochs=N_EPOCHS),
                        worker_init_fn=testdataset.initialise_worker_fn,
                        num_workers=1,# num_workers = spawns multiprocessing subprocess workers
                        timeout=300)# timeout after 5 mins of no data loading
        
        testloader_iterator = iter(testloader)
        trainloader_iterator = iter(trainloader)
        #assert len(trainloader_iterator) == N_EPOCHS * N_BATCHES
        
        #optimizer = torch.optim.SGD(model.parameters(), lr=c.LRATE, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE, weight_decay=c.WEIGHT_DECAY)
        
        start0 = start1 = time.time(); w1 = get_weights(model)
        for ie in range(N_EPOCHS):  # loop over the dataset multiple times
            
            wait_start, wait_time, gpu_time, gpu_utilisation = time.time(), 0., 0., 0.
            for ib in range(N_BATCHES):
                i = ioffset + ie*N_BATCHES+ib
                
                try:# get next sample_batch
                    sample_batch = next(trainloader_iterator)
                except StopIteration:# restart iterator
                    del trainloader_iterator
                    trainloader_iterator = iter(trainloader)# re-initiates batch/sampler iterators, with new random starts
                    sample_batch = next(trainloader_iterator)
                    
                #sample_batch = next(trainloader_iterator)
                #if ib == 0: print(sample_batch["i"])# check
                   
                wait_time += time.time()-wait_start
                
                
                ## TRAIN
                
                gpu_start = time.time()
                
                model.train()# switch to train mode (for dropout/ batch norm layers)
                
                # get the data
                inputs = sample_batch["inputs"]# expects list of inputs
                labels = sample_batch["labels"]# expects list of labels
                inputs = [inp.to(device) for inp in inputs]
                labels = [lab.to(device) for lab in labels]
                
                # zero the parameter gradients  AT EACH STEP
                optimizer.zero_grad()# zeros all parameter gradient buffers
        
                # forward + backward + optimize
                outputs = model(*inputs)# expect tuple of outputs
                loss = c.LOSS(*labels, *outputs, c)# note loss is on cuda if labels/ outputs on cuda
                loss.backward()# updates all gradients in model
                optimizer.step()# updates all parameters using their gradients
                
                gpu_time += time.time()-gpu_start
                
                ## TRAIN STATISTICS
                
                if (i + 1) % 100 == 0:
                    gpu_utilisation = 100*gpu_time/(wait_time+gpu_time)
                    print("Wait time average: %.4f s GPU time average: %.4f s GPU util: %.2f %% device: %i"%(wait_time/100, gpu_time/100, gpu_utilisation, c.DEVICE))
                    gpu_time, wait_time = 0.,0.
                    
                if (i + 1) % c.SUMMARY_FREQ == 0:
                    
                    rate = c.SUMMARY_FREQ/(time.time()-start1)
                    
                    with torch.no_grad():# faster inference without tracking
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"]# expects list of inputs
                        labels = sample_batch["labels"]# expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)
                        
                        l1loss = losses.l1_mean_loss(labels[0], outputs[0]).item()
                        l2loss = losses.l2_mean_loss(labels[0], outputs[0]).item()

                        writer.add_scalar("loss/l1_loss/train", l1loss, i + 1)
                        writer.add_scalar("loss/l2_loss/train", l2loss, i + 1)
                
                        inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
                        outputs_array = outputs[0].detach().cpu().numpy().copy()
                        labels_array = labels[0].detach().cpu().numpy().copy()
                        if (i + 1) % (10 * c.SUMMARY_FREQ) == 0:
                            f = plot_result(inputs_array, outputs_array, labels_array, sample_batch)
                            writer.add_figure("compare/train", f, i + 1, close=True)
                        
                        # check weight updates from previous summary
                        w2 = get_weights(model)
                        mu, _, av = get_weights_update_percent(w1, w2)
                        s = "Weight updates (%.1f %% average): "%(100*av)
                        for m in mu: s+="%.1f "%(100*m)
                        print(s)
                        del w1; w1 = w2

                        # add run statistics
                        writer.add_scalar("stats/epoch", ie, i + 1)
                        writer.add_scalar("stats/rate/batch", rate, i + 1)
                        writer.add_scalar("stats/rate/gpu_utilisation", gpu_utilisation, i + 1)
                        
                        # output to screen
                        print('[epoch: %i/%i, batch: %i/%i i: %i] l2loss: %.4f rate: %.1f elapsed: %.2f hr %s %s' % (
                               ie + 1,
                               N_EPOCHS,
                               ib + 1, 
                               N_BATCHES, 
                               i + 1,
                               l2loss,
                               rate,
                               (time.time()-start0)/(60*60),
                               time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),
                               c.RUN
                                ))
                        
                    start1 = time.time()
                
                ## TEST STATISTICS
                
                if (i + 1) % c.TEST_FREQ == 0:
                    
                    with torch.no_grad():# faster inference without tracking
                        
                        try:# get next sample_batch
                            sample_batch = next(testloader_iterator)
                        except StopIteration:# restart iterator
                            del testloader_iterator
                            testloader_iterator = iter(testloader)# re-initiates batch/sampler iterators, with new random starts
                            sample_batch = next(testloader_iterator)
                            #print(sample_batch["i"])# check
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"]# expects list of inputs
                        labels = sample_batch["labels"]# expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)
                        
                        l1loss = losses.l1_mean_loss(labels[0], outputs[0]).item()
                        l2loss = losses.l2_mean_loss(labels[0], outputs[0]).item()
                        
                        writer.add_scalar("loss/l1_loss/test", l1loss, i + 1)
                        writer.add_scalar("loss/l2_loss/test", l2loss, i + 1)
                
                        inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
                        outputs_array = outputs[0].detach().cpu().numpy().copy()
                        labels_array = labels[0].detach().cpu().numpy().copy()
                        if (i + 1) % (10 * c.TEST_FREQ) == 0:
                            f = plot_result(inputs_array, outputs_array, labels_array, sample_batch)
                            writer.add_figure("compare/test", f, i + 1, close=True)
                
                ## SAVE
                
                if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                    
                    model.eval()
                    
                    model.to(torch.device('cpu'))# put model on cpu before saving
                    # to avoid out-of-memory error
                    
                    # save a checkpoint
                    torch.save({
                    'i': i + 1,
                    'model_state_dict': model.state_dict(),
                    }, c.MODEL_OUT_DIR+"model_%.8i.torch"%(i + 1))
                    
                    model.to(device)
    
                wait_start = time.time()
        
        
        del trainloader_iterator, testloader_iterator
            
        print('Finished Training (total runtime: %.1f hrs)'%(
                        (time.time()-start0)/(60*60)))
        
    def close(self):
        self.writer.close()
         

if __name__ == "__main__":

    
    import models
    
    #cs = [Constants(),]
    
    DEVICE = 7
    
    #cs = [Constants(RUN="fault_cae",
    #                DEVICE=DEVICE,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae2",
    #                DEVICE=DEVICE,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae_seed",
    #                DEVICE=DEVICE,
    #                SEED=1234,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae_l2",
    #                LOSS=losses.l2_mean_loss_gain,
    #                DEVICE=DEVICE,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae_gain0",
    #                T_GAIN=0,
    #                DEVICE=DEVICE,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae_gain5",
    #                T_GAIN=5,
    #                DEVICE=DEVICE,
    #                ),]
    
    #cs = [Constants(RUN="fault_cae_shallow",
    #                MODEL=models.AE_shallow_r,
    #                DEVICE=DEVICE,
    #                ),]
    
    cs = [Constants(RUN="fault_cae_narrow",
                    MODEL=models.AE_narrow_r,
                    DEVICE=DEVICE,
                    ),]
    '''
    cs = [Constants(RUN="fault_cae3",
                    DEVICE=DEVICE,
                    ),]
    '''
    for c in cs:
        run = Trainer(c)
        run.train()
        run.close()
    