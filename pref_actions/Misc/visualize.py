#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:58:30 2018

@author: nimishawalgaonkar
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_context("talk", font_scale=1.4)

def contour(grid_levels, X_current, X_previous,
            p_mean_grid, p_var_grid, a, b, num_grid, num_data,unimodal = False):
    """
    Plotting the contour plots of :
    1. Posterior mean of preference probabilities
    2. Variance associated with posterior preference probabilities
    """

    var_levels =  np.linspace(0,0.15)
    mean_levels = np.linspace(0,1.)
    
    plt.figure(figsize=(12,10))
    var_reshape = p_var_grid.reshape((num_grid, num_grid), order='F')
    c = plt.contourf(grid_levels,grid_levels,var_reshape.T,
                     var_levels, cmap=plt.cm.Blues)
    plt.colorbar(c)
    plt.scatter(X_previous,X_current, marker = '.', c = 'r', s = 200, alpha = 1)
    plt.xlabel('Previous')
    plt.ylabel('Current')
    plt.title('Variance Contour Plot')
    if unimodal:
        plt.savefig('../data/syn_data/results/Unimodal/CV_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    else:    
        plt.savefig('../data/syn_data/results/CV_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    
    plt.figure(figsize=(12,10))
    mean_reshape = p_mean_grid.reshape((num_grid, num_grid), order='F')
    c = plt.contourf(grid_levels,grid_levels,mean_reshape.T,
                     mean_levels, cmap=plt.cm.Blues)
    plt.colorbar(c)
    plt.scatter(X_previous,X_current, marker = '.', c = 'r', s = 200, alpha = 1)
    plt.xlabel('Previous')
    plt.ylabel('Current')
    plt.title('Mean Predicted Preference Probability Contour Plot')
    if unimodal:
        plt.savefig('../data/syn_data/results/Unimodal/CM_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    else:
        plt.savefig('../data/syn_data/results/CM_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    
    return

def utility_1D(x, u, a, b, num_data,unimodal = False):
    """
    Plot the utility function
    """
    plt.figure()
    line, = plt.plot(x, np.mean(u, 0), lw=2)
    plt.fill_between(x, np.percentile(u, 5, axis=0), np.percentile(u, 95, axis=0),
                     color=line.get_color(), alpha = 0.2)
    plt.xlabel('feature values')
    plt.ylabel('y-values')
    if unimodal:
        plt.savefig('../data/syn_data/results/Unimodal/U_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    else:    
        plt.savefig('../data/syn_data/results/U_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    return

def utility_1D_samples(x, u, a, b, mini, maxi, num_data, unimodal = False):
    
    """
    Plot utility samples
    """
    plt.figure()
    for i in xrange(mini,maxi):
        plt.plot(x, u[i,:])
    if unimodal:
         plt.savefig('../data/syn_data/results/Unimodal/Usamp_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)
    else:    
        plt.savefig('../data/syn_data/results/Usamp_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)

    return

def latent_g(x, g, a, b, mini, maxi, num_data):
    """
    Plot latent g function
    """
    plt.figure()
    for i in xrange(mini, maxi):
        plt.plot(x, g[i,:])
    plt.savefig('../data/syn_data/results/Unimodal/latentg_' + 'O_a' + str(a) + '_b' + str(b) + '_' + str(num_data) + '_.png', dpi = 300)


def thompson_plot(copeland_mat, X, Tn):
    """
    Thompson sampling plot
    Inputs:
    X (size: L) : grid points
    p_mean_mat (size: L x N) : Matrix containing copeland scores corresponding to grid_points
    L : grid points points used for calculating copeland score
    N : MCMC samples
    Tn : number of thompson copeland score samples to display
    Output:
    Copeland Score Thompson sampling plots 
    (see Preferential Bayesian Optimization paper - Figure 3)
    """
    plt.figure(figsize=(12,10))
    # sampling from indexes
    indexes = np.arange(copeland_mat.shape[0])

    # index sample
    ind_samp = np.random.choice(indexes, size = Tn , replace = False)
    
    # Sampling from copeland mat
    
    for i in ind_samp:
        plt.plot(X, copeland_mat[i,:], c = 'gray')
        plt.xlabel('$x$')
        plt.ylabel('Copeland Score')
        plt.title('Copeland score at grid locations')
    
    cope_mean = np.mean(copeland_mat, axis = 0)
    plt.plot(X, cope_mean, c = 'black')
    return


def comfort_discomfort_probabilities(satisfied, brighter, darker, X, Tn):
    """
    Thompson sampling plot
    Inputs:
    X (size: L) : grid points
    p_mean_mat (size: L x N) : Matrix containing copeland scores corresponding to grid_points
    L : grid points points used for calculating copeland score
    N : MCMC samples
    Tn : number of thompson copeland score samples to display
    Output:
    Copeland Score Thompson sampling plots 
    (see Preferential Bayesian Optimization paper - Figure 3)
    """
    
    plt.figure(figsize=(12,10))
    # sampling from indexes
    indexes = np.arange(satisfied.shape[0])

    # index sample
    ind_samp = np.random.choice(indexes, size = Tn , replace = False)
    
    s = satisfied[ind_samp,:]
    b = brighter[ind_samp, :]
    d = darker[ind_samp, :]
    
    plt.figure(figsize=(12,10))
    line, = plt.plot(X, np.mean(s, 0), lw=2, color = 'g')
    plt.fill_between(X, np.percentile(s, 5, axis=0), np.percentile(s, 95, axis=0),
                     color=line.get_color(), alpha = 0.2)
    
    line, = plt.plot(X, np.mean(d, 0), lw=2, color = 'r')
    plt.fill_between(X, np.percentile(d, 5, axis=0), np.percentile(d, 95, axis=0),
                     color=line.get_color(), alpha = 0.2)
    
    line, = plt.plot(X, np.mean(b, 0), lw=2, color = 'b')
    plt.fill_between(X, np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0),
                     color=line.get_color(), alpha = 0.2)
    
    #plt.savefig('../data/results/C' + str(cluster) + '/F' + str(num_feat) + '/prob' + data_type + mean_func + '_.png', dpi = 600)
    
    return

def visualize_actions(action_data, title):
    """
    Pandas actions' dataframe
    """
    WI_prev = action_data.WI_prev
    WI_current = action_data.WI_current
    
    WI_prev_greater_ind = WI_prev > WI_current
    WI_prev_g = WI_prev[WI_prev_greater_ind]
    WI_current_g = WI_current[WI_prev_greater_ind]

    WI_prev_lesser_ind = WI_prev <= WI_current
    WI_prev_l = WI_prev[WI_prev_lesser_ind]
    WI_current_l = WI_current[WI_prev_lesser_ind]
    
    # Conditioning the data for plotting purposes
    x_l = WI_prev_l
    dx_l = WI_current_l - WI_prev_l

    x_g = WI_prev_g
    dx_g = WI_current_g - WI_prev_g

    x_l = np.array(x_l)
    dx_l = np.array(dx_l)
    x_g = np.array(x_g)
    dx_g = np.array(dx_g)
    
    plt.figure(figsize=(12,10))
    ax = plt.axes()
    for i in xrange(x_l.shape[0]):
        ax.arrow(0, x_l[i], 1, dx_l[i], head_width=0.05, head_length=0.05, fc='g', ec='g')
    for j in xrange(x_g.shape[0]):
        ax.arrow(0, x_g[j], 1, dx_g[j], head_width=0.05, head_length=0.05, fc='r', ec='r')

    ax.set_xlim(-0.25,1.25)
    ax.set_ylim(0,2400)
    plt.xticks([0.,1])
    plt.xlabel('Previous (0) vs. Current (1)')
    plt.ylabel('Workplane Illuminance')
    plt.title(title)
    
def visualize_prob_occ_to_cluster(p1, p2, p3, occupant_index):
    """
    Visualizing probability of occupant belong to a specific cluster
    """
    height = [p1, p2, p3]
    bars = ['1', '2', '3']
    y_pos = np.arange(len(bars))
 
    # Create bars
    plt.bar(y_pos, height)
 
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    
    plt.ylim(0.,1.)
 
    #plt.xlabel('Cluster index')
    #plt.ylabel('Expected Posterior Probability')
    plt.title(str(occupant_index))
    
    plt.savefig('../data/results/prob_occ_to_cluster_plots/O' + str(occupant_index) + '.png', dpi = 600)
    
    return


def prob_pred_boxplot(p1, p2, p3, occupant_index):
    """
    This function creates a boxplot for visualizing posterior predictive distribution
    Inputs:
    p : posterior samples
    Y : testing set actual value (the 0-1 response variables)
    """
    print "boxplot"
    l = [p1, p2, p3]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    bp = ax.boxplot(l, patch_artist=True)


    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel('Posterior Probability Distribution')
    ax.set_xlabel('Cluster index')
    ax.set_title('Posterior Probability of occupant' + str(occupant_index) + 'belonging to clusters')
    
    plt.savefig('../data/results/prob_occ_to_cluster_plots/O' + str(occupant_index) + '_post.png', dpi = 600)
  
    return
    
    

    
    
   