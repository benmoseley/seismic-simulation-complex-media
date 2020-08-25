#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:26:35 2019

@author: bmoseley
"""


# This module defines a custom pytorch batch sampler, which repeats the sampling of a
# training dataset over multiple epochs. This sampler is used in main.py.


from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


# This class batches up examples over n_epochs.

class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last, n_epochs=1):# CHANGED
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        #### ADDED
        n_epochs = int(n_epochs)
        if n_epochs < 1: raise Exception("n_epochs < 1: %i"%(n_epochs))
        self.n_epochs = n_epochs
        ####
        
    def __iter__(self):
        
        for ie in range(self.n_epochs):# ADDED
            
            batch = []
            for idx in self.sampler:# creates a new iter(sampler) every time for loop entered
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.n_epochs * ( len(self.sampler) // self.batch_size )# CHANGED
        else:
            return self.n_epochs * ( (len(self.sampler) + self.batch_size - 1) // self.batch_size )# CHANGED
        

        
        
        
if __name__ == "__main__":
    
    from torch.utils.data import Dataset, RandomSampler
    
    class D(Dataset):
        
        def __len__(self):
            return 10
    
    d = D()
    s = RandomSampler(d)
    b = BatchSampler(sampler=s, batch_size=3, drop_last=True, n_epochs=4)
    
    for x in b: print(x)
    
    
    print(len(b))