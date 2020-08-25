#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:40:02 2019

@author: bmoseley
"""


# This module defines helper functions for loading a trained WaveNet model and
# for loading a test dataset for a trained model. It is called by the Jupyter 
# notebooks used to generate the final report plots.


import os
import pickle
import tensorflow as tf
import models

from datasets import SeismicDataset

import sys
sys.path.insert(0, '../shared_modules/')
from helper import DictToObj


def load_model(MODEL_LOAD_RUN, rootdir="server/", config=None, verbose=False):
    """load a model and its constants object from rootdir.
    MODEL_LOAD_RUN can be of form 'model' or 'model/model.ckpt-XX' """
    
    rootdir = rootdir.rstrip("/")+"/"
    
    # parse MODEL_RUN and MODEL_DIR from path
    MODEL_LOAD_RUN = MODEL_LOAD_RUN.rstrip("/").split("/")
    MODEL_RUN = MODEL_LOAD_RUN[0]
    MODEL_DIR = rootdir+"models/%s/"%(MODEL_RUN)
    
    # parse specific model file
    if len(MODEL_LOAD_RUN) == 2:# if model file specified, load that model
        MODEL = MODEL_LOAD_RUN[1]
    else:# else load the final model in directory
        model_files = sorted(os.listdir(MODEL_DIR))
        MODEL = ".".join(model_files[-1].split(".")[:-1])

    # load constants dicionary
    CONSTANTS = "constants_%s.pickle"%(MODEL_RUN)
    SUMMARY_DIR = rootdir+"summaries/%s/"%(MODEL_RUN)
    if verbose: print("Loading constants: %s"%(SUMMARY_DIR+CONSTANTS))
    c_dict = pickle.load(open(SUMMARY_DIR+CONSTANTS, "rb"))
    c = DictToObj(**c_dict, copy=True)# convert to object
    if verbose: print(c)
    
    # restore a model using model file and constants file
    print("Loading model from: %s"%(MODEL_DIR+MODEL))
    
    # define model input and output tensorflow placeholders
    velocity = tf.placeholder(shape=(None,)+c.VELOCITY_SHAPE, dtype=tf.float32, name="velocity")
    reflectivity = tf.placeholder(shape=(None,)+c.REFLECTIVITY_SHAPE, dtype=tf.float32, name="reflectivity")
    gather = tf.placeholder(shape=(None,)+c.GATHER_SHAPE, dtype=tf.float32, name="gather")
    input_features = {"velocity":velocity, "reflectivity": reflectivity, "gather": gather}
    
    # define and load model
    model = c.MODEL(c, input_features)
    model.define_graph()
    model.define_loss()
    saver = tf.train.Saver()# for loading model
    sess = tf.Session(config=config)
    sess.graph.finalize()
    saver.restore(sess, MODEL_DIR+MODEL)# restore weights
    #print("Uninitialised variables check: ", sess.run(tf.report_uninitialized_variables()))
    if verbose: print(model)
    
    return model, c_dict, input_features, sess
    

def load_testdataset(DATA_PATH,
                     N_EXAMPLES,
                     c_dict,
                     rootdir="../generate_data/data/",
                     verbose=False):
    """load a test dataset using constants from c_dict"""
    
    rootdir = rootdir.rstrip("/")+"/"
    
    if verbose: print("Loading testdataset from %s%s"%(rootdir,DATA_PATH))
    
    # build c_eval object
    c_eval = DictToObj(**c_dict)
    
    # override fields
    c_eval.N_EXAMPLES = N_EXAMPLES
    c_eval.DATA_PATH = rootdir+DATA_PATH
    
    # delete any output dirs for safety
    for name in ["OUT_DIR", "MODEL_OUT_DIR", "SUMMARY_OUT_DIR"]:# clear out dir paths
        if hasattr(c_eval, name): delattr(c_eval, name)
    if verbose: print(c_eval)
    
    # load a test dataset using this evaluation constants object
    # use SeismicDataset for simplicity
    
    testdataset = SeismicDataset(c_eval)
    # don't define the dataset graph
    
    return testdataset


if __name__ == "__main__":
    
    tf.reset_default_graph()
    
    _, c_dict, _, _ = load_model("forward2", verbose=True)
    
    testdataset = load_testdataset("layers_2ms_validate.bin", N_EXAMPLES=1000, c_dict=c_dict)
    