#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:14:40 2018

@author: nimishawalgaonkar
"""


import numpy as np
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import bernoulli
from matplotlib import pyplot as plt
import seaborn as sns


def modified_scale(x):
    """
    Normalized Copeland score ranges from 0 to 1
    We want to change the range from (0,1) to (-3,3)
    from [min, max] to [a,b]
    f(x) = (b - a)(x - min)/(max - min) + a
    """
    a = 0
    b = 3
    mod_x = (b - a)*x + a
    return mod_x

def probit_link_func(x):
    """
    Probit link function
    Output : CDF of x 
    """
    return norm.cdf(x)

class BetaDist(object):
    def __init__(self, a, b):
        """
        Beta distribution governing utility function
        Inputs:
            a, b : parameters governing the Beta distribution
        """
        self.a = a
        self.b = b
        self.dist = beta(self.a, self.b)
        
        self.xgrid = np.linspace(0, 1, 1002)[1:-1]
        self.pgrid = self.dist.pdf(self.xgrid)
        self.pgridmax = np.max(self.pgrid)
        self.pgridnorm = self.pgrid/self.pgridmax
    
    def beta_dist(self, x):
        """
        Beta distribution governed by parameters a and b
        Inputs:
            x : ranges from 0 to 1, input value at which we want to find PDF value
        Outputs:
            PDF at input x
        """
        p = self.dist.pdf(x)
        pnorm = p/self.pgridmax
        return modified_scale(pnorm)
    
    def visualize(self):
        """
        Plot Beta distribution
        """
        plt.figure()
     
        plt.plot(self.xgrid, modified_scale(self.pgridnorm), ls= '-',
                 c='black',
                 label=r'$\alpha=%.1f,\ \beta=%.1f$' % (self.a, self.b))
        plt.xlim(0, 1)
        
        plt.xlabel('$x$')
        plt.ylabel(r'$p(x|\alpha,\beta)$')
        plt.title('3* Normalized Beta Distribution')

        plt.legend(loc=0)
        #plt.show()
        
        return
    
class OccupantDuels(object):
    """
    Generate utility for different occupants
    """
    def __init__(self, a, b):
        """
        Based on parameters a and b, g
        generate pairwise comparison data
        save pairwise comparison data (../data/occupant_duels)
        save associated utility curve (../data/occupant_utility)
        """
        self.a = a
        self.b = b
        self.utility = BetaDist(self.a, self.b)
        
        self.dest_name = 'O_a'+ str(self.a) + '_b' + str(self.b)
        
    def generate_response(self, x1, x2):
        """
        Given two states of a duel [x1, x2],
        generate the artificial response of the occupant
        y = 1 : previous state is preferred
        y = 0 : current state is preferred
        """
        
        u1 = self.utility.beta_dist(x1)
        u2 = self.utility.beta_dist(x2)
        diff = u2 - u1
        p_probit = probit_link_func(diff)
        y_probit = bernoulli.rvs(p_probit)
        
        return p_probit, y_probit

    def generate_duels(self, num_duels, dataset = 'train'):
        """
        Generate pairwise comparisons
        Inputs:
            num_duels : number of duels to generate (less than 20 is preferred)
            dataset : whether we want training or testing data (seed is changed)
            
        """
        if dataset == 'train':
            np.random.seed(1)
        if dataset == 'test':
            np.random.seed(2)
        
        n = 50 # grid points
        x = np.linspace(0, 1, n)
        
        
        # indexes1 and indexes2 samples
        x1 = np.random.choice(x, size = num_duels)
        x2 = np.random.choice(x, size = num_duels)
        
        p, y = self.generate_response(x1, x2)
        
        return x1, x2, p, y
    
    def generate_actions(self, num_actions, dataset = 'train'):
        """
        Generate actions data
        """
        x1, x2, p, y = self.generate_duels(num_duels = num_actions, dataset = dataset)
        one_ind = y == 1
        prev = np.zeros(one_ind.shape[0])
        current = np.zeros(one_ind.shape[0])
        for i, boolean in enumerate(one_ind):
            if boolean:
                prev[i] = x1[i]
                current[i] = x2[i]
            else:
                prev[i] = x2[i]
                current[i] = x1[i]
        actions_array = np.vstack([prev, current]).T
        
        return actions_array
    
    def save_actions(self, num_actions, dataset = 'train'):
        """
        save actions data
        """
        actions_array = self.generate_actions(num_actions, dataset)
        np.savetxt('../data/syn_data/syn_occ/actions_data/' + dataset + '/' + self.dest_name + '_' + str(num_actions) + '_.csv',
                   actions_array, delimiter=",")
        
        return
        
        
    
    def save_duels(self, num_duels, dataset = 'train'):
        """
        Generate and save pairwise comparisons
        Inputs:
            num_duels : number of duels to generate (less than 20 is preferred)
            dataset : whether we want training or testing data (seed is changed)
        """
        x1, x2, p, y = self.generate_duels(num_duels, dataset)
        
        np.savez('../data/syn_data/syn_occ/occupant_duels/' + dataset + '/' + self.dest_name + '_' + str(num_duels) + '_.npz',
                 x1 = x1, x2 = x2, p = p,  y = y)
        
        return
        
    def visualize_utility(self):
        """
        Visualize utility function with parameters a and b
        """
        self.utility.visualize()
        
        return
        
    def save_utility_plot(self):
        """
        Save utility plots associated with the parameters a and b of the 
        occupant using those parameters
        """
        
        self.utility.visualize()
        plt.savefig('../data/syn_data/syn_occ/true_utility/' + self.dest_name + '.png', dpi = 300) 
        
        return
        
    def visualize_pref_prob(self):
        """
        Visualize contour plot of actual preference probabilities which are
        governed by utility function values as defined by parameters a and b
        
        Plot contour plot for preference probabilities
        Probability of grid points against grid points ex.
        probabilitity of grid points np.linspace(min, max, num) preferred over
        same grid points np.linspace(min, max, num)
        """
        num_grid = 100
        x = np.linspace(0,1, num_grid)
        Xtt1, Xtt2 = np.meshgrid(x, x)
        Xtt = np.zeros(shape =(np.ravel(Xtt1).shape[0],2))
        Xtt[:,0] = np.ravel(Xtt1) 
        Xtt[:,1] = np.ravel(Xtt2)
        
        y, p = self.generate_response(Xtt[:,0], Xtt[:,1])
        
        sns.set_context("talk", font_scale=1.6)
        
        plt.figure(figsize=(12,10))
        levels = np.linspace(0,1, 400)
        c = plt.contourf(x, x, 
                         p.reshape((num_grid, num_grid), order='F').T,
                         levels, cmap=plt.cm.Blues)
        plt.colorbar(c)   
        plt.ylabel('$x\'$')
        plt.xlabel('$x$')
        
        return
    
    def save_pref_prob(self):
        """
        Save actual preference probabilities
        """
        self.visualize_pref_prob()
        plt.savefig('../data/syn_data/syn_occ/true_pref_prob/' + self.dest_name + '.png',
                    dpi = 300)
        
        return
        
        
        
        
           
if __name__ == '__main__':
    a = 1
    b = 2
    
    Occupant = OccupantDuels(a, b)
    
#    Occupant.save_duels(2, 'train')
#    Occupant.save_duels(40, 'test')
#    Occupant.save_utility_plot()
#    Occupant.save_pref_prob()
    
    
    

        
        
