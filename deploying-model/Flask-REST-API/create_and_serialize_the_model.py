#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:53:00 2019

@author: spavko
"""

import pandas as pd
import os
import pickle

# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == '__main__':

    df = pd.read_csv('../../data/pulsar_stars.csv')
    df.info()
    
    # Data preparation
    X = df.drop(['target_class'],axis=1)
    y = df['target_class']
    
    # Scaling the data along side rows
    # X_scaled = MinMaxScaler().fit_transform(X.T).T
    
    # Here i decided not to scale the data as the input to the API is usually
    # single observation so scaling the data in such environemnt needs more
    # effort, and the benefit is not obvious
    
    X_scaled = X
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test  = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # setting the needed classifiers
    classifiers = {}
    classifiers['Logistic Regression'] = LogisticRegression(random_state=42)
    classifiers['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
    classifiers['KNN'] = KNeighborsClassifier()
    
    # setting the wanted hyperparameters
    hyperparams = {}
    hyperparams['Logistic Regression'] = {
                                            'penalty': ['l1', 'l2'],
                                            'C': [0.1, 0.25, 0.5],
                                            'solver': ['liblinear']
                                        }
    hyperparams['Gradient Boosting'] = {
                                    'loss': ['deviance', 'exponential'],
                                    'learning_rate': [0.05, 0.1, 0.3],
                                    'n_estimators':[10, 20, 50],
                                    'min_samples_split':[10, 50, 100]
                                   }
    hyperparams['KNN'] = {
                            'n_neighbors': [5, 10, 20],
                            'weights': ['uniform', 'distance']
                        }
    
    
    # Train the different models
    
    def train_predict_gscv(classifiers, hyperparams, X_train, y_train, X_test, y_test):
        '''
        Given a model train the model given the data
        '''
        
        best_params={}
        test_score={}
        models = {}
        cm = {}
        for model in classifiers:
            
    # use GridSearchCV class to create and object with certain parameters        
            gscv = GridSearchCV(estimator=classifiers[model],param_grid=hyperparams[model],scoring='roc_auc',cv=10,verbose=1)         
    
    # fit the GridSearchCV using the above provided params    
            gscv.fit(X_train, y_train)  
    
    # store the best parameters    
            best_params[model] = gscv.best_params_        
            models[model] = gscv
                    
    # predict with the best model found        
            y_hat = gscv.predict(X_test)
            
            test_score[model] = roc_auc_score(y_test, y_hat)
            cm[model] = confusion_matrix(y_test, y_hat)
            
        return best_params, test_score, cm, models
    
    
    # results
    best_params, test_score, cm ,models = train_predict_gscv(classifiers, hyperparams, X_train, y_train, X_test, y_test)
    
    print(best_params)
    
    print(test_score)
    
    print(cm['Gradient Boosting'])
    
    print('The best score had the Gradient Bosting, hence we will take that model')
    
    # save the model to use with the Web Server
    path = os.getcwd() + '/serialized_classifier.pkl'
    with open(path, 'wb') as f:
        pickle.dump(models['Gradient Boosting'], f)
        print("Pickled classifier at {}".format(path))
        