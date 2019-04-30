#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:39:54 2019

@author: nimishawalgaonkar
"""

import gpflow
import numpy as np
from gpflow.model import Model
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder, ParamList
from gpflow._settings import settings
float_type = settings.dtypes.float_type


def total_all_actions(concat_cur_prev_feat_list):
    """
    Calculate the total number of actions (taken by all the occupants)
    """
    shape = np.array([])
    for l in concat_cur_prev_feat_list:
        n = l.shape[0]
        shape = np.append(shape, n)
    total_shape = np.sum(shape)
    return total_shape
    


class GPCollabPrefLearn(Model):
    """
    A base class for collaborative GPs based preference learning
    """
    
    def __init__(self, prev_ind_list, cur_ind_list, X_grid,
                 kerns_list, name = 'collaborative_pref_gps'):
        
        Model.__init__(self, name)
        
        total_shape = total_all_actions(prev_ind_list)
        
        Y = np.ones(total_shape)[:,None]
        self.Y = DataHolder(Y)
        
        # Introducing Paramlist to define kernels for latent GPs H
        self.kerns_list = ParamList(kerns_list)
        
        self.X_grid = DataHolder(X_grid[:,None])
        
        self.prev_ind_list = prev_ind_list
        self.cur_ind_list = cur_ind_list
        
        # define likelihood
        self.likelihood = gpflow.likelihoods.Bernoulli()
    
    @AutoFlow((float_type, [None, None]))
    def predict_h(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        
        return self.build_predict_h(Xnew)
        