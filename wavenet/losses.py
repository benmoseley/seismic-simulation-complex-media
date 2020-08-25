#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""

import tensorflow as tf


def lp_mean_loss(y, y_true, l_num):
    """
    Computes an lp mean loss
    """
    loss = tf.reduce_mean(tf.abs(y-y_true)**l_num)
    return loss
