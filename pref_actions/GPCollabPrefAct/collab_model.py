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


class GPCollabPrefLearn(Model):
    """
    A base class for collaborative GPs based preference learning
    """
    
    def __init__(self, concat_cur_prev_feat_mat, concat_ind_cur_prev_mat, X_grid,
                 kerns_list, name = 'collaborative_pref_gps'):
        
        Model.__init__(self, name)
        
        concat_ind_cur_prev_mat_shape = concat_ind_cur_prev_mat.shape
        total_u_diff_all_occ = concat_ind_cur_prev_mat_shape[0]*concat_ind_cur_prev_mat_shape[1]/2
        Y = np.ones(total_u_diff_all_occ)[:,None]
        self.Y = DataHolder(Y)
        
        # Introducing Paramlist to define kernels for latent GPs H
        self.kerns_list = ParamList(kerns_list)
        
        self.X_grid = DataHolder(X_grid[:,None])
        self.X = DataHolder(concat_cur_prev_feat_mat)
        self.rel_indices = tf.constant(concat_ind_cur_prev_mat)
        
        # define likelihood
        self.likelihood = gpflow.likelihoods.Bernoulli()
    
    @AutoFlow((float_type, [None, None]))
    def predict_h(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        
        return self.build_predict_h(Xnew)
        