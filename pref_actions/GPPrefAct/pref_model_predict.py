#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:20:13 2017

@author: nawalgao
"""

import numpy as np
from scipy.stats import norm


class predict(object):
    def __init__(self, X, m, samples):
        """
        Predicting the required posterior utlity function
        and/or preference probability  values.
        based on trained GPflow model described using:
            m: Trained gpflow model object (on training set)
            samples : posterior hyperparameters' values 
            X: normalized input feature values at which we want to make the predictions
        """
        self.X = X
        self.m = m
        self.samples = samples
        self.num_datapoints = X.shape[0]
        u_samples = []
        for s in samples:
            m.set_state(s)
            u_samples.append(m.predict_f_samples(X, 1))
        u_samples = np.vstack(u_samples)
        self.u = u_samples[:,:,0]
    
    def predict_p(self):
        """
        Predict the preference probability values for normalized duels which are given as
        a concatenated matrix 
        X_mat = [X_previous, X_current].T ;  size : 2(num_duels) x num_feat  
        Output:
          p : posterior preference probability matrix ; size : S x num_duels 
          p_mean : posterior preference probability mean ; size : 1 x num_duels
          p_var : posterior preference probability variance ; size : 1 x num_duels
        """
        u1 = self.u[:,:self.num_datapoints/2]
        u2 = self.u[:,self.num_datapoints/2:]
        u_diff = u2 - u1
        p = norm.cdf(u_diff)
        p_mean = np.mean(p, axis=0)
        p_std = np.std(p, axis = 0)
        p_var = p_std**2
        return p, p_mean, p_var
    
    def predict_u(self):
        """
        Predict utility function values at grid feature values X
        Outputs:
            u : posterior utility function values
            u_mean : mean of posterior utility function values' samples
            u_mean_max : maximum of utility function values
            
        """
        u_mean = np.mean(self.u, axis = 0)
        max_ind = self.u.argmax(axis = 1)
        #u_mean_max = np.max(u_mean)
        return self.u, u_mean, max_ind
        
        
        
        
        
        
    
        
    
    