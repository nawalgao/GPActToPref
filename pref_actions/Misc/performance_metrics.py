# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:52:15 2017

@author: nawalgao
"""
import numpy as np
from scipy.spatial import distance

def hit_rate(p_mean, Y_test):
    """
    Hit rate accuracy
    """
    pred = p_mean > 0.5
    diff = pred.astype(int)-Y_test
    test_acc = 1. -np.sum(np.abs(diff)).astype(float)/diff.shape[0]
    #print 'Accuracy on test data:', test_acc
    return test_acc


def euclidean(p_pred, p_act):
    """
    Euclidean distance
    """
    dst = distance.euclidean(p_pred, p_act)
    return dst


def euc_matrix(p_pred_mat, p_act):
    
    """
    Euclidean distance between posterior samples and actual preference probabiliy.
    p_pred_mat: N x S matrix (N: total number of duels, S: number of posterior samples)
    Output: Expected euclidean distance (expectation taken over samples and number of datapoints)
    """
    N = p_pred_mat.shape[1]
    diff = p_pred_mat - p_act
    diff_sq = diff**2
    sum_row = np.sum(diff_sq, axis = 1)
    euc_mat = 1./N*np.sqrt(sum_row)
    euc_avg = np.mean(euc_mat)
    return euc_avg


def out_of_sample_deviance(p_pred_mat, p_act):
    """
    Calculates the out-of sample deviance (deviance of the testing set) when we know the actual preference probability
    Inputs:
    p_pred_mat: N x S matrix (N: total number of duels, S: number of posterior samples)
    p_act: actual preference probability
    
    Outputs:
    out-of sample deviance
    """
    p_diff = p_pred_mat - p_act
    p_diff_abs = np.abs(p_diff)
    log_ev = np.log(1 - p_diff_abs + 1e-10)
    dev = -2 * log_ev
    dev_sum = np.sum(dev, axis = 1)
    dev_mean = np.mean(dev_sum)
    return dev_mean

def euc_from_true_max(u_max, u_post_max):
    """
    Average Euclidean distance of posterior utility maximum samples
    from the maximum of utility (when known)
    u_max : true maximum of the utility
    u_post_max: posterior maximum of utility samples
    """
    S = u_post_max.shape[0]
    u_max_vec = np.repeat(u_max, S)
    diff = u_max_vec - u_post_max
    diff_sq = diff**2
    sq_sum = np.sum(diff_sq)
    unnorm_euc = np.sqrt(sq_sum)
    norm_euc = 1./S*unnorm_euc
    return norm_euc


def euc_from_true_max_2D(u_max, u_post_max):
    """
    Average Euclidean distance of posterior utility maximum samples
    from the maximum of utility (when known)
    u_max : true maximum of the utility
    u_post_max: posterior maximum of utility samples
    """
    S = u_post_max.shape[0]
    #u_max_vec = np.repeat(u_max, S)
    u_max_vec = np.tile(u_max, (S,1))
    diff = u_max_vec - u_post_max
    diff_sq = diff**2
    sq_sum = np.sum(diff_sq)
    unnorm_euc = np.sqrt(sq_sum)
    norm_euc = 1./S*unnorm_euc
    return norm_euc
    