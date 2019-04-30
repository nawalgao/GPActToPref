# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collaborative preference learning from actions taken by the occupants
For more info, refer to:
http://papers.nips.cc/paper/4700-collaborative-gaussian-processes-for-preference-learning

Exploiting collaborative information from shared structure in occupants' behavior (actions taken by them)
Unsupervised learning of similarities in occupants' behavior is exploited without requiring access to specific occupants' features
"""


import numpy as np
import tensorflow as tf
from gpflow.param import Param

from gpflow.priors import Gaussian
from gpflow.conditionals import conditional
from gpflow._settings import settings
float_type = settings.dtypes.float_type
from .collab_model import GPCollabPrefLearn



def gen_req_action_data(action_data_matrix, X_grid):
    
    """
    Inputs:
    actions_data_matrix : action data matrix (X_prev, X_current)
    Outputs:
    Concatenation of [X_cur, X_prev]
    Index of action data in X_grid
    """
    prev = action_data_matrix[:,0][:,None]
    cur = action_data_matrix[:,1][:,None]
    concat = np.vstack([cur, prev])
    
    grid_sorted = np.argsort(X_grid)
    ypos = np.searchsorted(X_grid[grid_sorted], concat)
    indices = grid_sorted[ypos]
    return concat, indices


#def gen_rel_list_action_data(all_occ_action_data_list, X_grid):
#    
#    """
#    Generate relevant data from all occupant action data list
#    """
#    
#    concat_list = []
#    indices_list = []
#    for action_data in all_occ_action_data_list:
#        concat_actions, indices = gen_req_action_data(action_data, X_grid)
#        concat_list.append(concat_actions)
#        indices_list.append(indices)
#    return concat_list, indices_list
#
#
#def gen_rel_concat_action_data(all_occ_action_data_list, X_grid):
#    
#    """
#    Generate relevant data from all occupant action data list
#    """
#    
#    concat_list = []
#    indices_list = []
#    for i, action_data in enumerate(all_occ_action_data_list):
#        concat_actions, indices = gen_req_action_data(action_data, X_grid)
#        #print concat_actions.shape
#        concat_list.append(concat_actions[:,0])
#        indices_list.append(indices[:,0])
#        
#    return concat_list, indices_list
    
def occ_prev_cur_action_ind(action_data_matrix, X_grid):
    """
    Inputs:
    actions_data_matrix : action data matrix (X_prev, X_current)
    Outputs:
    Concatenation of [X_cur, X_prev]
    Index of action data in X_grid
    """
    prev = action_data_matrix[:,0][:,None]
    cur = action_data_matrix[:,1][:,None]
    
    grid_sorted = np.argsort(X_grid)
    
    ypos_prev = np.searchsorted(X_grid[grid_sorted], prev)
    prev_ind = grid_sorted[ypos_prev]
    
    ypos_cur = np.searchsorted(X_grid[grid_sorted], cur)
    cur_ind = grid_sorted[ypos_cur]
    
    return prev_ind, cur_ind

def all_occ_prev_cur_act_ind(action_data_matrix_list, X_grid):
    """
    Inputs:
    action_data_matrix_list : list of all actions' data matrix (num_list = number of occupants)
    Outputs: 
    prev_ind_list : previous feature indices list
    cur_ind_list : current feature indices list
    """
    prev_ind_list = []
    cur_ind_list = []
    for mat in action_data_matrix_list:
        prev_ind, cur_ind =  occ_prev_cur_action_ind(mat, X_grid)
        prev_ind_list.append(prev_ind[:,0])
        cur_ind_list.append(cur_ind[:,0])
    
    return prev_ind_list, cur_ind_list


def select_specific_column_indices(utility_matrix, ind_list):
    """
    Select specific column indices corresponding to specific row indices 
    Inputs:
    ind_list : list of features (previous or current) of all occupants
    Outputs:
    rel_tf_ind : relevant tensor : selecting specific columns from each rows of utility matrix.
    """
    total_ind_list = []
    for i in xrange(len(ind_list)):
        col = ind_list[i]
        row = np.repeat(i, col.shape[0])
        c_tf = tf.constant(col)
        r_tf = tf.constant(row)
        full_ind = tf.stack([r_tf, c_tf], axis=1)
        total_ind_list.append(full_ind)
    total_ind_tf = tf.concat(total_ind_list, 0)
    # retrieve values by indices
    S = tf.gather_nd(utility_matrix, total_ind_tf)
    return S


def gather_concerned_utilities(utility_matrix, concerned_ind_mat):
    """
    Gather concerned utility function values for each occupant
    """
    
    splits = tf.split(concerned_ind_mat, num_or_size_splits=concerned_ind_mat.shape[1],axis = 1)
    res_list = []
    for i,s in enumerate(splits):
        #print i
        idx = tf.stack([tf.reshape(tf.range(utility_matrix.shape[0]), (-1,1)), tf.to_int32(s)], axis=-1)
        res = tf.gather_nd(utility_matrix, idx)
        res_flatten = tf.reshape(res, [-1])
        res_list.append(res_flatten)
    concerned_mat = tf.transpose(tf.stack(res_list)) 
    return concerned_mat


class GPCollabPrefLearnGPMC(GPCollabPrefLearn):
    
    def __init__(self, actions_list, kerns_list, X_grid):
        """
        Collaboartive preference learning through occupants' actions/behavior
        Uses information : feature value before action and after action
        Inputs:
        actions_list : all actions data list (list of actions taken by each occupant) N_i X (2M) where,
        N_i : total number of actions taken by occupant i
        first half of 2M columns : feature values before action
        second half of 2M columns : feature values after action
        L : length of the list = total number of occupants
        kerns_list : kernels associated with all of the latent GPs 
        """
        
        # feature grid points
        self.X_grid = X_grid
        self.num_x_grid = self.X_grid.shape[0]
        
        # num of latent GPs
        self.num_latent_gps = len(kerns_list)
        
        # num of occupants
        self.num_occupants = len(actions_list)
        
        # generating indexes for actions : concerned matrix indices
        # U is calculated at all of the X_grid points
        # When it comes to actions taken by each occupant, we need to find the index associated with each actions
        # actions list is the input list, each element of which is the numpy matrix representing feature value before action and after action 
        prev_ind_list, cur_ind_list = all_occ_prev_cur_act_ind(actions_list, X_grid)
        
        
        GPCollabPrefLearn.__init__(self, prev_ind_list, cur_ind_list, X_grid, kerns_list)
        
        # Prior for H (latent GP matrix) setup
        
        # HMC sampling setup (standard normal distribution : whitening variables)
        self.V_h = Param(np.zeros((self.num_latent_gps, self.num_x_grid))) 
        self.V_h.prior =  Gaussian(0., 1.)
        
        
        self.W = Param(np.random.randn(self.num_occupants, self.num_latent_gps))
        
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        
        return super(GPCollabPrefLearnGPMC, self).compile(session = session,
                    graph = graph, optimizer = optimizer)
    
    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of collaborative GP
        \log p(Y| V_h, W, theta)
        """
        #self.W = tf.eye(num_rows = self.num_occupants,
        #                num_columns = self.num_latent_gps, dtype=float_type)
        
        V_h_splits = tf.split(self.V_h, num_or_size_splits = self.num_latent_gps)
        H_list = []
        for i in xrange(self.num_latent_gps):
            K_h_i = self.kerns_list[i].K(self.X_grid)
            L_h_i = tf.cholesky(K_h_i + tf.eye(tf.shape(self.X_grid)[0], dtype=float_type) * 1e-4)
            H_i = tf.matmul(L_h_i, tf.transpose(V_h_splits[i])) # ---> check this... transpose is this correct?
            H_list.append(H_i)
        self.H = tf.concat(H_list, 1) # Latent GPs (self.num_latent_gps x self.num_x_grid)
        
        self.U = tf.matmul(self.W, tf.transpose(self.H)) # utility function values for each occupant (self.num_occupants x self.num_x_grid)

        # Extract relevant element of utility function value at training points for each occupant
        U_cur = select_specific_column_indices(self.U, self.cur_ind_list)
        U_prev = select_specific_column_indices(self.U, self.prev_ind_list)
        
        U_diff = tf.subtract(U_cur,U_prev)
        
        flatten_U_diff = tf.reshape(U_diff, [-1, 1])
    
        return tf.reduce_sum(self.likelihood.logp(flatten_U_diff, self.Y))
    
    def build_predict_h(self, Xnew, full_cov = False):
        """
        Predict latent gps H values at new points ``Xnew''
        Xnew is a data matrix, point at which we want to predict
        This method computes p(H*|L = L_h_i*V_h_i)
        """
        mu_list = []
        var_list = []
        V_h_splits = tf.split(self.V_h, num_or_size_splits = self.num_latent_gps)
        for i in xrange(self.num_latent_gps):
            mu_i, var_i = conditional(Xnew, self.X_grid, self.kerns_list[i],
                                      tf.transpose(V_h_splits[i]),
                                      full_cov = full_cov, q_sqrt = None,
                                      whiten = True)
            mu_list.append(mu_i)
            var_list.append(var_i)
        
        return mu_list, var_list
            
            
        