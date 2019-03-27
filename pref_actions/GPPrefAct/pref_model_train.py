#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Trains a Preference model

"""


import numpy as np
import json # Needed for config file
import os
import gpflow 
import build_model



def HMC_train(X_mat, Y, config_file, mean_func):
    
    """
    Training a preference model with 2D or 1D input features
    Inputs:
    data : pairwise training dataset
    X_mat : previous feat columnwise contatenated with current feat
    X = [X_pre, X_current]  ; size : N x 2(num_features)
    config_file : configuration file (.json) for GP preference model
    Outputs:
    m :  GPflow preference model object
    samples : samples of HMC sampling
    """
    
    # Sanity checks
    assert os.path.isfile(config_file)
    # Read configuration file
    with open(config_file, 'r') as fd:
        config = json.loads(fd.read())
    
    # number of features
    num_feat = X_mat.shape[1]/2
    
    print num_feat
    
    # Concatenating previous features with current features
    X = np.vstack([X_mat[:,:num_feat], X_mat[:,num_feat:]])    
    Y = Y.astype(float)[:,None]
    
    
     # HMC sampling
    MAP_optimize_maxiter = config['MCMC']['MAP_optimize_maxiter']
    num_samples = config['MCMC']['num_samples']
    thin = config['MCMC']['thin']
    burn = config['MCMC']['burn']
    epsilon = config['MCMC']['epsilon']
    Lmax = config['MCMC']['Lmax']
    verbose = eval(config['MCMC']['verbose'])
    
    # Priors for covariance kernel hyperparameters
    lengthscale_prior = eval(config['Prior']['lengthscales'])
    variance_prior = eval(config['Prior']['variance'])
    
    # kernels
    k1 = gpflow.kernels.RBF(1, active_dims=[0])
    k2 = gpflow.kernels.RBF(1, active_dims=[1])
    
    
    # likelihood
    l = gpflow.likelihoods.Bernoulli()
    
    if num_feat == 1:
        k = k1
        #print 'aaaa'
    if num_feat == 2:
        k = k1*k2
        #print 'bbbb'
    if mean_func == 'linear':
        meanf = eval(config['MeanFunc']['linear'])
        # Priors for mean function hyperparameters
        A_prior = eval(config['Prior']['linear_mean_func_A'])
        b_prior = eval(config['Prior']['linear_mean_func_b'])
        m = build_model.GPPrefLearn(X, Y, k, l, meanf)
    
        if num_feat == 1:   
            m.kern.lengthscales.prior = lengthscale_prior
            m.kern.variance.prior = variance_prior
        if num_feat == 2:
            m.kern.rbf_1.lengthscales.prior = lengthscale_prior
            m.kern.rbf_2.lengthscales.prior = lengthscale_prior
            m.kern.rbf_1.variance.prior = variance_prior
        m.mean_function.A.prior = A_prior
        m.mean_function.b.prior = b_prior
    else:
        m = build_model.GPPrefLearn(X, Y, k, l)
        if num_feat == 1:   
            m.kern.lengthscales.prior = lengthscale_prior
            m.kern.variance.prior = variance_prior
        if num_feat == 2:
            m.kern.rbf_1.lengthscales.prior = lengthscale_prior
            m.kern.rbf_2.lengthscales.prior = lengthscale_prior
            m.kern.rbf_1.variance.prior = variance_prior
        
    
    m.optimize(maxiter= MAP_optimize_maxiter) # start n2ear MAP
    samples = m.sample(num_samples, verbose= verbose, epsilon= epsilon,
                       thin = thin, burn = burn, Lmax=Lmax)

    return m, samples

