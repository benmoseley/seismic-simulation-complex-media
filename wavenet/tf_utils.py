#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


import tensorflow as tf

def w(shape, stddev=0.01, mean=0.0, name=None):
    """
    Returns a weight layer with the given shape and standard deviation. Initialized with a normal distribution.
    """
    return tf.Variable(tf.random_normal(shape, stddev=stddev, mean=mean), name=name)


def b(shape, const=0.1, name=None):
    """
    Returns a bias layer with the given shape. Initialised with a constant value.
    """
    return tf.Variable(tf.constant(const, shape=shape), name=name)