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


import numpy as np
import tensorflow as tf
"""
Collaborative preference learning from actions taken by the occupants
For more info, refer to:
http://papers.nips.cc/paper/4700-collaborative-gaussian-processes-for-preference-learning

Exploiting collaborative information from shared structure in occupants' behavior (actions taken by them)
Unsupervised learning of similarities in occupants' behavior is exploited without requiring access to specific occupants' features
"""



import gpflow
from gpflow.model import GPModel
from gpflow.gpmc import GPMC
from gpflow.param import Param
from gpflow.param import DataHolder
from gpflow.conditionals import conditional
from gpflow.priors import Gaussian
from gpflow._settings import settings
float_type = settings.dtypes.float_type

#class GPPrefLearn(GPMC):
#    def __init__(self, X, Y, kern, likelihood,
#                 mean_function = None, num_latent = None):
#        
#        """
#        X is a data matrix, size : 2N x D
#        Y is a data matrix, size : N X 1
#        This is a vanilla implementation of a GP Preference learning model
#        with non-Gaussian likelihood.
#        {Chu, W., & Ghahramani, Z. (2005, August).
#        Preference learning with Gaussian processes.
#        In Proceedings of the 22nd international conference on Machine learning (pp. 137-144). ACM.}
#        The latent function values are represented by centered 
#        (whitened) variables, so
#             v ~ N(0, I)
#             f = Lv + m(x)
#         with
#         
#             L L^T = K
#             
#        """
#        GPMC.__init__(self, X, Y, kern, likelihood, mean_function, num_latent)
#    
#    
#    def compile(self, session=None, graph=None, optimizer=None):
#        """
#        Before calling the standard compile function, check to see if the size
#        of the data has changed and add parameters appropriately.
#
#        This is necessary because the shape of the parameters depends on the
#        shape of the data.
#        """
#        if not self.num_data == self.X.shape[0]:
#            self.num_data = self.X.shape[0]
#            self.V = Param(np.zeros((self.num_data, self.num_latent)))
#            self.V.prior = Gaussian(0., 1.)
#
#        return super(GPPrefLearn, self).compile(session=session,
#                                         graph=graph,
#                                         optimizer=optimizer)
#        
#    
#    def build_likelihood(self):
#        """
#        Construct a tf function to compute the likelihood of a general GP
#        model.
#
#            \log p(Y, V | theta).
#
#        """
#        K = self.kern.K(self.X)
#        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
#        F = tf.matmul(L, self.V) + self.mean_function(self.X)
#        
#        F1,F2 = tf.split(F, num_or_size_splits=2)
#        F_diff = tf.subtract(F2,F1)
#        
#        return tf.reduce_sum(self.likelihood.logp(F_diff, self.Y))

class GPCollabPrefLearn(GPMC):
    def __init__(self, X, kerns_list, num_latent_gps = 3):
        """
        Collaboartive preference learning through occupants' actions/behavior
        Uses information : feature value before action and after action
        Inputs:
        X : all actions data list (list of actions taken by each occupant) N_i X (2M) where,
        N_i : total number of actions taken by occupant i
        first half of 2M columns : feature values before action
        second half of 2M columns : feature values after action
        L : length of the list = total number of occupants
        kerns_list : kernels associated with all of the latent GPs 
        """
        
        # Sanity check
        # length of the kerns_list == num_latent_gps
        
    

