#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:00:57 2019

@author: bmoseley
"""


# This module defines the conditional autoencoder network, which is defined through a pytorch model,
# wrapped by the AE_r class. 

# This class is selected in constants.py and is called inside the training loop 
# of main.py. Its hyperparameters are provided by constants.py.

# This module also defines alternative network designs used for testing the sensitivity
# of the accuracy of our approach to different network architectures.


import torch
import torch.nn as nn  
import numpy as np
    



class AE_r(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.name = "AE_r"
        self.c = c
        
        ## DEFINE WEIGHTS

        ## ENCODER
        
        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True


        self.conv1a = nn.Conv2d(1, 8, (3,3), (1,1), (1,1))
        self.conv1a_bn = nn.BatchNorm2d(8)
        self.drop1a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv1 = nn.Conv2d(8, 16, (2,2), (2,2), 0)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout2d(c.DROPOUT_RATE)



        self.conv2a = nn.Conv2d(16, 16, (3,3), (1,1), (1,1))
        self.conv2a_bn = nn.BatchNorm2d(16)
        self.drop2a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv2 = nn.Conv2d(16, 32, (2,2), (2,2), 0)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.conv3a = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.conv3a_bn = nn.BatchNorm2d(32)
        self.drop3a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv3 = nn.Conv2d(32, 64, (2,2), (2,2), 0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.conv4 = nn.Conv2d(64, 128, (2,2), (2,2), 0)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv5 = nn.Conv2d(128, 256, (2,2), (2,2), 0)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv6 = nn.Conv2d(256, 512, (2,2), (2,2), 0)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.drop6 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv7 = nn.Conv2d(512, 1024, (2,2), (2,2), 0)
        self.conv7_bn = nn.BatchNorm2d(1024)
        self.drop7 = nn.Dropout2d(c.DROPOUT_RATE)
        
        ## DECODER
        
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
        self.convT1 = nn.ConvTranspose2d(1025, 1025, (2,2), (2,2), 0)
        self.convT1_bn = nn.BatchNorm2d(1025)
        self.dropT1 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT2 = nn.ConvTranspose2d(1025, 512, (2,4), (2,4), 0)
        self.convT2_bn = nn.BatchNorm2d(512)
        self.dropT2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT2a = nn.Conv2d(512, 512, (3,3), (1,1), (1,1))
        self.convT2a_bn = nn.BatchNorm2d(512)
        self.dropT2a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT2b = nn.Conv2d(512, 512, (3,3), (1,1), (1,1))
        self.convT2b_bn = nn.BatchNorm2d(512)
        self.dropT2b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT3 = nn.ConvTranspose2d(512, 256, (2,4), (2,4), 0)
        self.convT3_bn = nn.BatchNorm2d(256)
        self.dropT3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT3a = nn.Conv2d(256, 256, (3,3), (1,1), (1,1))
        self.convT3a_bn = nn.BatchNorm2d(256)
        self.dropT3a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT3b = nn.Conv2d(256, 256, (3,3), (1,1), (1,1))
        self.convT3b_bn = nn.BatchNorm2d(256)
        self.dropT3b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT4 = nn.ConvTranspose2d(256, 64, (2,4), (2,4), 0)
        self.convT4_bn = nn.BatchNorm2d(64)
        self.dropT4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT4a = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.convT4a_bn = nn.BatchNorm2d(64)
        self.dropT4a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT4b = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.convT4b_bn = nn.BatchNorm2d(64)
        self.dropT4b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT5 = nn.ConvTranspose2d(64, 8, (2,4), (2,4), 0)
        self.convT5_bn = nn.BatchNorm2d(8)
        self.dropT5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT5a = nn.Conv2d(8, 8, (3,3), (1,1), (1,1))
        self.convT5a_bn = nn.BatchNorm2d(8)
        self.dropT5a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT5b = nn.Conv2d(8, 8, (3,3), (1,1), (1,1))
        self.convT5b_bn = nn.BatchNorm2d(8)
        self.dropT5b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        self.convT6 = nn.Conv2d(8, 1, (1,1), (1,1), 0)# final linear layer

    def forward(self, x, s):

        ## DEFINE OPERATIONS
              
        x = self.drop1a(self.c.ACTIVATION(self.conv1a_bn(self.conv1a(x))))
        x = self.drop1(self.c.ACTIVATION(self.conv1_bn(self.conv1(x))))
        
        x = self.drop2a(self.c.ACTIVATION(self.conv2a_bn(self.conv2a(x))))
        x = self.drop2(self.c.ACTIVATION(self.conv2_bn(self.conv2(x))))
        
        x = self.drop3a(self.c.ACTIVATION(self.conv3a_bn(self.conv3a(x))))
        x = self.drop3(self.c.ACTIVATION(self.conv3_bn(self.conv3(x))))
        
        x = self.drop4(self.c.ACTIVATION(self.conv4_bn(self.conv4(x))))
        
        x = self.drop5(self.c.ACTIVATION(self.conv5_bn(self.conv5(x))))
        
        x = self.drop6(self.c.ACTIVATION(self.conv6_bn(self.conv6(x))))
        
        x = self.drop7(self.c.ACTIVATION(self.conv7_bn(self.conv7(x))))
        
        #print(x.shape)
        x = torch.cat((x, s[:,0:1,:,:]), dim=1)
        #print(x.shape)
        
        x = self.dropT1(self.c.ACTIVATION(self.convT1_bn(self.convT1(x))))
        
        x = self.dropT2(self.c.ACTIVATION(self.convT2_bn(self.convT2(x))))
        x = self.dropT2a(self.c.ACTIVATION(self.convT2a_bn(self.convT2a(x))))
        x = self.dropT2b(self.c.ACTIVATION(self.convT2b_bn(self.convT2b(x))))

        x = self.dropT3(self.c.ACTIVATION(self.convT3_bn(self.convT3(x))))
        x = self.dropT3a(self.c.ACTIVATION(self.convT3a_bn(self.convT3a(x))))
        x = self.dropT3b(self.c.ACTIVATION(self.convT3b_bn(self.convT3b(x))))
        
        x = self.dropT4(self.c.ACTIVATION(self.convT4_bn(self.convT4(x))))
        x = self.dropT4a(self.c.ACTIVATION(self.convT4a_bn(self.convT4a(x))))
        x = self.dropT4b(self.c.ACTIVATION(self.convT4b_bn(self.convT4b(x))))

        x = self.dropT5(self.c.ACTIVATION(self.convT5_bn(self.convT5(x))))
        x = self.dropT5a(self.c.ACTIVATION(self.convT5a_bn(self.convT5a(x))))
        x = self.dropT5b(self.c.ACTIVATION(self.convT5b_bn(self.convT5b(x))))
        
        x = self.convT6(x)# final linear layer
        
        return x,



class AE_narrow_r(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.name = "AE_narrow_r"
        self.c = c
        
        ## DEFINE WEIGHTS

        ## ENCODER
        
        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True


        self.conv1a = nn.Conv2d(1, 4, (3,3), (1,1), (1,1))
        self.conv1a_bn = nn.BatchNorm2d(4)
        self.drop1a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv1 = nn.Conv2d(4, 8, (2,2), (2,2), 0)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout2d(c.DROPOUT_RATE)



        self.conv2a = nn.Conv2d(8, 8, (3,3), (1,1), (1,1))
        self.conv2a_bn = nn.BatchNorm2d(8)
        self.drop2a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv2 = nn.Conv2d(8, 16, (2,2), (2,2), 0)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.conv3a = nn.Conv2d(16, 16, (3,3), (1,1), (1,1))
        self.conv3a_bn = nn.BatchNorm2d(16)
        self.drop3a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv3 = nn.Conv2d(16, 32, (2,2), (2,2), 0)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.conv4 = nn.Conv2d(32, 64, (2,2), (2,2), 0)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv5 = nn.Conv2d(64, 128, (2,2), (2,2), 0)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.drop5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv6 = nn.Conv2d(128, 256, (2,2), (2,2), 0)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.drop6 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv7 = nn.Conv2d(256, 512, (2,2), (2,2), 0)
        self.conv7_bn = nn.BatchNorm2d(512)
        self.drop7 = nn.Dropout2d(c.DROPOUT_RATE)
        
        ## DECODER
        
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
        self.convT1 = nn.ConvTranspose2d(513, 513, (2,2), (2,2), 0)
        self.convT1_bn = nn.BatchNorm2d(513)
        self.dropT1 = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT2 = nn.ConvTranspose2d(513, 256, (2,4), (2,4), 0)
        self.convT2_bn = nn.BatchNorm2d(256)
        self.dropT2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT2a = nn.Conv2d(256, 256, (3,3), (1,1), (1,1))
        self.convT2a_bn = nn.BatchNorm2d(256)
        self.dropT2a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT2b = nn.Conv2d(256, 256, (3,3), (1,1), (1,1))
        self.convT2b_bn = nn.BatchNorm2d(256)
        self.dropT2b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT3 = nn.ConvTranspose2d(256, 128, (2,4), (2,4), 0)
        self.convT3_bn = nn.BatchNorm2d(128)
        self.dropT3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT3a = nn.Conv2d(128, 128, (3,3), (1,1), (1,1))
        self.convT3a_bn = nn.BatchNorm2d(128)
        self.dropT3a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT3b = nn.Conv2d(128, 128, (3,3), (1,1), (1,1))
        self.convT3b_bn = nn.BatchNorm2d(128)
        self.dropT3b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT4 = nn.ConvTranspose2d(128, 32, (2,4), (2,4), 0)
        self.convT4_bn = nn.BatchNorm2d(32)
        self.dropT4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT4a = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.convT4a_bn = nn.BatchNorm2d(32)
        self.dropT4a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT4b = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.convT4b_bn = nn.BatchNorm2d(32)
        self.dropT4b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        
        self.convT5 = nn.ConvTranspose2d(32, 4, (2,4), (2,4), 0)
        self.convT5_bn = nn.BatchNorm2d(4)
        self.dropT5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT5a = nn.Conv2d(4, 4, (3,3), (1,1), (1,1))
        self.convT5a_bn = nn.BatchNorm2d(4)
        self.dropT5a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT5b = nn.Conv2d(4, 4, (3,3), (1,1), (1,1))
        self.convT5b_bn = nn.BatchNorm2d(4)
        self.dropT5b = nn.Dropout2d(c.DROPOUT_RATE)
        
        
        self.convT6 = nn.Conv2d(4, 1, (1,1), (1,1), 0)# final linear layer

    def forward(self, x, s):

        ## DEFINE OPERATIONS
              
        x = self.drop1a(self.c.ACTIVATION(self.conv1a_bn(self.conv1a(x))))
        x = self.drop1(self.c.ACTIVATION(self.conv1_bn(self.conv1(x))))
        
        x = self.drop2a(self.c.ACTIVATION(self.conv2a_bn(self.conv2a(x))))
        x = self.drop2(self.c.ACTIVATION(self.conv2_bn(self.conv2(x))))
        
        x = self.drop3a(self.c.ACTIVATION(self.conv3a_bn(self.conv3a(x))))
        x = self.drop3(self.c.ACTIVATION(self.conv3_bn(self.conv3(x))))
        
        x = self.drop4(self.c.ACTIVATION(self.conv4_bn(self.conv4(x))))
        
        x = self.drop5(self.c.ACTIVATION(self.conv5_bn(self.conv5(x))))
        
        x = self.drop6(self.c.ACTIVATION(self.conv6_bn(self.conv6(x))))
        
        x = self.drop7(self.c.ACTIVATION(self.conv7_bn(self.conv7(x))))
        
        #print(x.shape)
        x = torch.cat((x, s[:,0:1,:,:]), dim=1)
        #print(x.shape)
        
        x = self.dropT1(self.c.ACTIVATION(self.convT1_bn(self.convT1(x))))
        
        x = self.dropT2(self.c.ACTIVATION(self.convT2_bn(self.convT2(x))))
        x = self.dropT2a(self.c.ACTIVATION(self.convT2a_bn(self.convT2a(x))))
        x = self.dropT2b(self.c.ACTIVATION(self.convT2b_bn(self.convT2b(x))))

        x = self.dropT3(self.c.ACTIVATION(self.convT3_bn(self.convT3(x))))
        x = self.dropT3a(self.c.ACTIVATION(self.convT3a_bn(self.convT3a(x))))
        x = self.dropT3b(self.c.ACTIVATION(self.convT3b_bn(self.convT3b(x))))
        
        x = self.dropT4(self.c.ACTIVATION(self.convT4_bn(self.convT4(x))))
        x = self.dropT4a(self.c.ACTIVATION(self.convT4a_bn(self.convT4a(x))))
        x = self.dropT4b(self.c.ACTIVATION(self.convT4b_bn(self.convT4b(x))))

        x = self.dropT5(self.c.ACTIVATION(self.convT5_bn(self.convT5(x))))
        x = self.dropT5a(self.c.ACTIVATION(self.convT5a_bn(self.convT5a(x))))
        x = self.dropT5b(self.c.ACTIVATION(self.convT5b_bn(self.convT5b(x))))
        
        x = self.convT6(x)# final linear layer
        
        return x,
    



class AE_shallow_r(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.name = "AE_shallow_r"
        self.c = c
        
        ## DEFINE WEIGHTS

        ## ENCODER
        
        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True


        self.conv1a = nn.Conv2d(1, 8, (3,3), (1,1), (1,1))
        self.conv1a_bn = nn.BatchNorm2d(8)
        self.drop1a = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv1 = nn.Conv2d(8, 16, (2,2), (2,2), 0)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout2d(c.DROPOUT_RATE)

        self.conv2 = nn.Conv2d(16, 32, (2,2), (2,2), 0)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv3 = nn.Conv2d(32, 64, (2,2), (2,2), 0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv4 = nn.Conv2d(64, 128, (2,2), (2,2), 0)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv5 = nn.Conv2d(128, 256, (2,2), (2,2), 0)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv6 = nn.Conv2d(256, 512, (2,2), (2,2), 0)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.drop6 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.conv7 = nn.Conv2d(512, 1024, (2,2), (2,2), 0)
        self.conv7_bn = nn.BatchNorm2d(1024)
        self.drop7 = nn.Dropout2d(c.DROPOUT_RATE)
        
        ## DECODER
        
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
        self.convT1 = nn.ConvTranspose2d(1025, 1025, (2,2), (2,2), 0)
        self.convT1_bn = nn.BatchNorm2d(1025)
        self.dropT1 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT2 = nn.ConvTranspose2d(1025, 512, (2,4), (2,4), 0)
        self.convT2_bn = nn.BatchNorm2d(512)
        self.dropT2 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT3 = nn.ConvTranspose2d(512, 256, (2,4), (2,4), 0)
        self.convT3_bn = nn.BatchNorm2d(256)
        self.dropT3 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT4 = nn.ConvTranspose2d(256, 64, (2,4), (2,4), 0)
        self.convT4_bn = nn.BatchNorm2d(64)
        self.dropT4 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT5 = nn.ConvTranspose2d(64, 8, (2,4), (2,4), 0)
        self.convT5_bn = nn.BatchNorm2d(8)
        self.dropT5 = nn.Dropout2d(c.DROPOUT_RATE)
        
        self.convT6 = nn.Conv2d(8, 1, (1,1), (1,1), 0)# final linear layer

    def forward(self, x, s):

        ## DEFINE OPERATIONS
              
        x = self.drop1a(self.c.ACTIVATION(self.conv1a_bn(self.conv1a(x))))
        x = self.drop1(self.c.ACTIVATION(self.conv1_bn(self.conv1(x))))
        
        x = self.drop2(self.c.ACTIVATION(self.conv2_bn(self.conv2(x))))
        
        x = self.drop3(self.c.ACTIVATION(self.conv3_bn(self.conv3(x))))
        
        x = self.drop4(self.c.ACTIVATION(self.conv4_bn(self.conv4(x))))
        
        x = self.drop5(self.c.ACTIVATION(self.conv5_bn(self.conv5(x))))
        
        x = self.drop6(self.c.ACTIVATION(self.conv6_bn(self.conv6(x))))
        
        x = self.drop7(self.c.ACTIVATION(self.conv7_bn(self.conv7(x))))
        
        #print(x.shape)
        x = torch.cat((x, s[:,0:1,:,:]), dim=1)
        #print(x.shape)
        
        x = self.dropT1(self.c.ACTIVATION(self.convT1_bn(self.convT1(x))))
        
        x = self.dropT2(self.c.ACTIVATION(self.convT2_bn(self.convT2(x))))

        x = self.dropT3(self.c.ACTIVATION(self.convT3_bn(self.convT3(x))))
        
        x = self.dropT4(self.c.ACTIVATION(self.convT4_bn(self.convT4(x))))

        x = self.dropT5(self.c.ACTIVATION(self.convT5_bn(self.convT5(x))))
        
        x = self.convT6(x)# final linear layer
        
        return x,
    
    
    
    

def trace_static_model(model, x, s=None, verbose=False):
    "trace through a static model, printing layer shapes and output shapes"
    
    def _print(x, module):
        if verbose or ("BatchNorm" not in str(module) and "Dropout" not in str(module)):
            row = ["%s"%(list(x.size()),), "%s"%(x.numel(),)]
            pad = 1
            for p in module.parameters():
                row += ["%s"%(list(p.size()),), "%s"%(p.numel(),)]
                pad += 1
            print(("{:<22}{:<10}"*pad).format(*row))
    
    flag = False
    sizes = []
    row = ["%s"%(list(x.size()),), "%s"%(x.numel(),)]
    print(("{:<22}{:<10}"*1).format(*row))
    for module in list(model.modules())[1:]:# don't include main module
        
        ## ADDED for concat model
        if "ConvT" in str(module) and not flag and type(s)!=type(None):
            x = torch.cat((x, s[:,0:1,:,:]), dim=1)
            flag = True
        ##
        
        x = module(x)
        _print(x, module)
        if "Conv" in str(module): sizes.append(module.out_channels)

    row = ["%s"%(list(x.size()),), "%s"%(x.numel(),)]
    print(("{:<22}{:<10}"*1).format(*row))
    return np.array(sizes)


if __name__ == "__main__":
    
    from constants import Constants
    
    c = Constants()
    
    for Model in [AE_r, AE_narrow_r, AE_shallow_r]:
        model = Model(c)
        
        print("Model: %s"%(model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i"%(total_params))
        print("Total number of trainable parameters: %i"%(total_trainable_params))
        
        x = torch.zeros((c.BATCH_SIZE,)+c.VELOCITY_SHAPE)
        s = torch.zeros((c.BATCH_SIZE,)+c.SOURCE_SHAPE)
    
        sizes = trace_static_model(model, x, s, verbose=False)
        print(len(sizes))
        print(model(x,s)[0].shape)
        print()
    
    