#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:30:39 2019

@author: spavko
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class PulsarModel(object):
    
    '''
    Class holding the model to classify or not a star as a pulsar        
    '''
    
    def __init__(self):
        
        self.model = []
            
    def predict(self, X):

        y_hat = self.model.predict(X)
        return y_hat
        
    def predict_proba(self, X):

        y_hat_proba = self.model.predict_proba(X)
        return y_hat_proba[:1]
        
                       