#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:25 2017

@author: nawalgao
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
  
  
def read_pickle_object(pickle_file_name):
    """
    Reads GPflow object m saved as a pickle file
    """
    with open(pickle_file_name, 'rb') as input:
        m = pickle.load(input)
    return m


def read_samples(samples_file_name):
    samples = np.loadtxt(samples_file_name, delimiter = ',')
    return samples



def plot_hmc_chains(m, samples, save_fig_loc):
    """
    Inputs:
    m: gpflow model object
    samples: gpflow model object
    """
    kernel_samples = m.kern.get_samples_df(samples)
    #overall_variance = m.pref_var.get_samples_df(samples)
    plt.figure()
    plt.plot(kernel_samples)
    plt.savefig(save_fig_loc + '_kernel_samples.png' , dpi = 600)
    #plt.figure()
    #plt.plot(overall_variance)
    #plt.savefig(save_fig_loc + '_overall_var.png' , dpi = 600)
    plt.figure()
    plt.plot(samples)
    plt.savefig(save_fig_loc + '_samples_.png', dpi = 600)
    return

def plot_hmc_chains_files(model_pickle_loc, model_samples_loc, save_fig_loc):
    """
    Inputs:
    m : GPflow trained model object m pickle file location
    samples: GPFlow trained model samples txt file location
    
    Outputs:
    Posterior histograms of the concerned hyperparameters
    """
    m = read_pickle_object(model_pickle_loc)
    samples = read_samples(model_samples_loc)
    
    plot_hmc_chains(m, samples, save_fig_loc)
    return 

#def post_samples(m, samples):
#    """
#    Plotting histograms of posterior samples
#    """


if __name__ == '__main__':
    
#    from argparse import ArgumentParser
#    parser = ArgumentParser(description=__doc__)
#    # Add arguments
#    parser.add_argument('-m', '--model_pickle_loc', type=str, required=True,
#                        help='Pickle file containing GPflow trained model object')
#    parser.add_argument('-s', '--samples_file_loc', type=str, required=True,
#                        help='gpflow trained model samples')
#    
#    # Parse arguments
#    args = parser.parse_args()
#    trained_model_pickle_save_loc = args.model_pickle_loc
#    trained_model_samples_sav_loc = args.samples_file_loc
#    
    base_file_name = 'O4sunnyM4HMC_S10000T5B2000E0.1'
    trained_model_pickle_save_loc = ('../relevant_data/tmp_data/GPFlow_model_objects/' +
                                     base_file_name + '.pkl')
    trained_model_samples_sav_loc = ('../relevant_data/tmp_data/GPFlow_model_samples/' +
                                     base_file_name + '.txt')
    
    save_fig_loc = '../relevant_plots/hmc_chains/' + base_file_name
    plot_hmc_chains_files(trained_model_pickle_save_loc, trained_model_samples_sav_loc, save_fig_loc)
    