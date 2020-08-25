#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:39:53 2018

@author: bmoseley
"""


# This module defines various generic helper functions in pytorch.


import torch
import numpy as np


get_weights = lambda model: [p.detach().cpu().numpy()[:] for p in model.parameters()]

def get_weights_update_percent(weights1, weights2):
    assert len(weights1) == len(weights2)
    
    N = sum([w.size for w in weights1])
    
    mean, std, sum_all = [],[], 0
    for i in range(len(weights1)):
        w1, w2 = weights1[i], weights2[i]
        d = np.abs((w2 - w1)/np.mean(np.abs(w1)))
        mean.append(np.mean(d))
        std.append(np.std(d))
        sum_all += np.sum(d)
    return mean, std, sum_all/N







if __name__ == "__main__":
    
    torch.manual_seed(123)
    
    model = torch.nn.Conv2d(2,2,3)
    
    # get weights
    weights = get_weights(model)
    for x in weights: print(x.size)
    for x in weights[:5]: print(x.flatten())
    
    w1 = [np.arange(-10,10)]
    w2 = [np.arange(-9,11)*0.5]
    print(get_weights_update_percent(w1, w2))
