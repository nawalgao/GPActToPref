#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:46:18 2018

@author: nimishawalgaonkar
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import json
import os

class DataGen(object):
    def __init__(self, raw_data_file):
        """
        Clean up the action_data to get the required feature matrix
        Inputs:
            action_data : what Amir has provided.  (size : , as of now)
            Action data file is in xlsx form
            
        It has 3 clusters (prefer darker, prefer moderate, prefer brighter conditions)
        As of now, it has 11 different individual occupants
        """
        raw_data = pd.read_excel(open(raw_data_file, 'rb'), sheet_name = 0)
        self.concerned_data = raw_data.iloc[:,:26]
        self.previous_column_names = self.concerned_data.columns
        self.concerned_data.columns = ['Office_ID', 'Weather', 'Setup', 'Occupant_ID',
                                  'Cluster', 'Distance', 'Arrival_I', 'WI', 'LI',
                                  'ddd', 'I_B', 'I_D', 'I_G', 'I_P', 
                                  'I_V', 'I_SW', 'I_SC', 'SD', 'ED', 'LCAW', 
                                  'SQuery', 'CWP', 'TW', 'G_P', 'G_S', 'G_V']
        
    def get_specific_occ_setup(self, occupant_id, setup_id):
        
        """
        Get data for specific occupant and for specific control setup
        """
        rel_data = self.concerned_data
        rel_data = rel_data[rel_data['Occupant_ID'] == occupant_id]
        rel_data = rel_data[rel_data['Setup'] == setup_id]
            
        return rel_data
    
    def get_all_individual_occ_setup(self):
        setup1_data_list = []
        setup2_data_list = []
        for o in xrange(19):
            D1 = self.get_specific_occ_setup(o+1, 1)
            D2 = self.get_specific_occ_setup(o+1, 2)
            setup1_data_list.append(D1)
            setup2_data_list.append(D2)
        return setup1_data_list, setup2_data_list
        
            
            
            
        
    



if __name__ == '__main__':
    raw_data_file = '../data/raw_data/DATA.xlsx'
    raw_data = pd.read_excel(open(raw_data_file, 'rb'), sheetname = 0)
    concerned_data = raw_data.iloc[:,:26]
    previous_column_names = concerned_data.columns
    concerned_data.columns = ['Office_ID', 'Weather', 'Setup', 'Occupant_ID',
                              'Cluster', 'Distance', 'Arrival_I', 'WI', 'LI',
                              'ddd', 'I_B', 'I_D', 'I_G', 'I_P', 
                              'I_V', 'I_SW', 'I_SC', 'SD', 'ED', 'LCAW', 
                              'SQuery', 'CWP', 'TW', 'G_P', 'G_S', 'G_V']
    
