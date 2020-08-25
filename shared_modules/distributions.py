#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:04:43 2018

@author: bmoseley
"""


# This module converts the moments of the lognormal distribution to the moments
# of its corresponding normal distribution, and is called by ../generate_data/generate_velocity_models.py.


import numpy as np

lnLtoN = lambda mu_L, sigma_L: (np.log(mu_L/np.sqrt(1.+(sigma_L/mu_L)**2.)), np.sqrt(np.log(1.+(sigma_L/mu_L)**2.)))
lognormal_pdf = lambda x, mu_N, sigma_N: (1./ (x * sigma_N * np.sqrt(2.*np.pi)))* (np.exp(-(np.log(x) - mu_N)**2. / (2. * sigma_N**2.)))
normal_pdf = lambda x, mu_N, sigma_N: (1./(sigma_N * np.sqrt(2.*np.pi))) * np.exp( -(x - mu_N)**2. / (2. * sigma_N**2.) )



if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    # QC (plot distributions)
    mu, sigma = 500., 1000. # mean and standard deviation
    mu, sigma = 0.5, 0.1 # mean and standard deviation

    s = np.random.lognormal(lnLtoN(mu, sigma)[0],lnLtoN(mu, sigma)[1], 1000); x = np.arange(0.1,20,0.01)
    plt.hist(s, 100, normed=True, align='mid')
    plt.plot(x, lognormal_pdf(x,lnLtoN(mu, sigma)[0],lnLtoN(mu, sigma)[1]), linewidth=2, color='r')
    plt.xlim(0,1)
    #plt.ylim(0,0.5)

    # QC (plot distributions)
    mu, sigma = 2., 3. # mean and standard deviation

    s = np.random.normal(mu, sigma, 1000); x = np.arange(-50,50,0.1 )
    plt.figure()
    plt.hist(s, 100, normed=True, align='mid')
    plt.plot(x, normal_pdf(x ,mu, sigma), linewidth=2, color='r')
    plt.xlim(-10,10)
    #plt.ylim(0,0.5)
    plt.show()

