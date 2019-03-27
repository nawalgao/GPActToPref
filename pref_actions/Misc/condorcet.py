#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 09:00:18 2018

@author: nimishawalgaonkar
"""

import numpy as np
import pickle


def read_pickle_object(pickle_file_name):
    """
    Reads gpflow object m saved as a pickle file
    """
    with open(pickle_file_name, 'rb') as input:
        m = pickle.load(input)
    return m

class syn_comfort_discomfort(object):
    def __init__(self, predict,
                 m, samples, grid_min, grid_max, num_grid):
        """
        Comfort and discomfort probabilities based on Copeland score
        num_feat : number of features to use
        cluster : 1, 2 or 3
        X_train_file : training features
        m_file : GPflow model object pickle file
        samples_file : GPflow samples saved file
        config_file : configuration file
        """
        
        self.predict = predict
        self.num_feat = 1
        self.samples = samples
        self.m = m
        
        # Setting up the grid (duels) on which to check our predicted preference probabilities

        self.Xt = np.linspace(grid_min, grid_max, 2*num_grid)
        self.Xt_landmark = np.linspace(grid_min, grid_max, 0.5*num_grid)
    
    def cope_comfort(self, X_state):
        """
        Normalized soft Copeland score to evaluate the comfort, brighter, darker probabilities
        """
        # For each state in Xt calculate the soft copeland score
        copeland = np.zeros(shape = (self.samples.shape[0], X_state.shape[0]))
        for i,x in enumerate(X_state):
            x_vec = np.repeat(x, self.Xt_landmark.shape[0])
            x_mat = np.hstack([x_vec[:,None],self.Xt_landmark[:,None]])
            x_mat_norm = (x_mat - 0.5)/0.5
            x_mat_norm_concat = np.vstack([x_mat_norm[:,self.num_feat:], x_mat_norm[:,:self.num_feat]])
            grid_pred_object = self.predict(x_mat_norm_concat, self.m, self.samples)
            p_grid_winner, p_mean_grid_winner, p_var_grid_winner = grid_pred_object.predict_p()
            sum_prob = np.sum(p_grid_winner, axis = 1)
            copeland[:,i] = sum_prob 
        cope_comfort = copeland/self.Xt_landmark.shape[0]
        return cope_comfort
        
    def cope_comfort_brighter_darker(self):
        """
        Normalized soft Copeland score to evaluate the comfort, brighter, darker probabilities
        """
        
        # For each state in Xt calculate the soft copeland score
        copeland = np.zeros(shape = (self.samples.shape[0], self.Xt.shape[0]))
        for i,x in enumerate(self.Xt):
            x_vec = np.repeat(x, self.Xt_landmark.shape[0])
            x_mat = np.hstack([x_vec[:,None],self.Xt_landmark[:,None]])
            x_mat_norm = (x_mat - 0.5)/0.5
            
            x_mat_norm_concat = np.vstack([x_mat_norm[:,self.num_feat:], x_mat_norm[:,:self.num_feat]])
            grid_pred_object = self.predict(x_mat_norm_concat, self.m, self.samples)
            p_grid_winner, p_mean_grid_winner, p_var_grid_winner = grid_pred_object.predict_p()
            sum_prob = np.sum(p_grid_winner, axis = 1)
            copeland[:,i] = sum_prob 
        cope_comfort = copeland/self.Xt_landmark.shape[0]
        b1 = np.zeros(shape = cope_comfort.shape)
        b2 = np.zeros(shape = cope_comfort.shape)
        
        for i in np.arange(cope_comfort.shape[1]):
            b1[:,i] = (1. - cope_comfort[:,i])*np.sum(cope_comfort[:,i:], axis = 1)
            b2[:,i] = (1. - cope_comfort[:,i])*np.sum(cope_comfort[:,:i], axis = 1)
        prob_brighter = b1/(2*self.Xt_landmark.shape[0])
        prob_darker = b2/(2*self.Xt_landmark.shape[0])
        return cope_comfort, prob_brighter, prob_darker
#    def cope_comfort_brighter_darker_2D(self, SP_fixed_value):
#        num_feat = 2
#        copeland = np.zeros(shape = (self.samples.shape[0], self.Xt.shape[0]))
#        SP_vec = np.repeat(SP_fixed_value, self.Xt_landmark.shape[0])
#        for i,x in enumerate(self.Xt):
#            x_vec = np.repeat(x, self.Xt_landmark.shape[0])
#            x_mat = np.hstack([x_vec[:,None],SP_vec[:,None],
#                               self.Xt_landmark[:,None], SP_vec[:,None]])
#            x_mat_norm = data_generation.normalize_data(x_mat).min_max_normalize2D(self.config_file)
#            x_mat_norm_concat = np.vstack([x_mat_norm[:,:num_feat], x_mat_norm[:,num_feat:]])
#            grid_pred_object = pref_model_predict.predict(x_mat_norm_concat, self.m, self.samples)
#            p_grid_winner, p_mean_grid_winner, p_var_grid_winner = grid_pred_object.predict_p()
#            sum_prob = np.sum(p_grid_winner, axis = 1)
#            copeland[:,i] = sum_prob 
#        cope_comfort = copeland/self.Xt_landmark.shape[0]
#        b1 = np.zeros(shape = cope_comfort.shape)
#        b2 = np.zeros(shape = cope_comfort.shape)
#        
#        for i in np.arange(cope_comfort.shape[1]):
#            b1[:,i] = (1. - cope_comfort[:,i])*np.sum(cope_comfort[:,i:], axis = 1)
#            b2[:,i] = (1. - cope_comfort[:,i])*np.sum(cope_comfort[:,:i], axis = 1)
#        prob_brighter = b1/(2*self.Xt_landmark.shape[0])
#        prob_darker = b2/(2*self.Xt_landmark.shape[0])
#        return cope_comfort, prob_brighter, prob_darker
#             
#    