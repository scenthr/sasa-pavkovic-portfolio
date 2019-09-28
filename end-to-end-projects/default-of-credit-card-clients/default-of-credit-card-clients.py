#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:02:13 2019

@author: spavko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import itertools
import xgboost as xgb

from scipy.stats import gamma
from scipy.stats import randint as sp_randint

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# Load the dataset from excel file

df = pd.read_excel('data/default-of-credit-card-clients.xls', skiprows=1)

df.info()
 
# USEFUL FUNCTIONS

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plts the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#     plt.tight_layout()
    plt.show()

# CHECKING THE DATA

def df_stats(df):
    '''
    Given pandas dataset provides basic information
    '''
    
    print("----------Top-5- Record----------")
    print(df.head(5))
    print("-----------Information-----------")
    print(df.info())
    print("-----------Data Types-----------")
    print(df.dtypes)
    print("----------Missing value-----------")
    print(df.isnull().sum())
    print("----------Null value-----------")
    print(df.isna().sum())
    print("----------Shape of Data----------")
    print(df.shape)
    print("----------Potential Duplicates----------")
    print(df.duplicated().sum())
        
df_stats(df)    

# The dataset seems to be cleaned, but we can remove the ID as that keeps the
# lines different

df = df.drop('ID', axis=1)

df.info()

'''

Additional Data Inspection
---------------------------

Althought the dataset looks clean there are some questions about it. For 
instance there are 795 rows that Bill amt. and Pay amt. =0. Also in the PAY_0
there are many entries with value of 1 and -2 (i assumed this value means "null"). 

To me this suggests that here is also included the data for the clients that 
just started with their credit obligations. For these there is very little 
historical transactional data, hence making predictions would be based on just
client master data which is probably unsufficient to create a good model.

Hence i would exclude these entries during the model selection process to 
try to improve the model perfromance.

Example are IDs: 19, 20, 46, 80
'''

'''

Categorical variables
----------------------

There are several categorical variables and we will create dummy variables for
them so that they can be successfully used with the models that we will attempt.

I will also 
'''

df4model = pd.get_dummies(df,columns=['SEX','EDUCATION','MARRIAGE'])
df4model.info()


# EDA

# MODELLING


X = df4model.drop('default payment next month',axis=1)
y = df4model['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#########################
## Logistic Regression ##
#########################

model = LogisticRegression(random_state=42, class_weight='balanced')

###########################################
## 1 ## Univariate logistic regression 
###########################################

model.fit(np.array(X_train['PAY_0']).reshape(-1,1), y_train)

#np.array(X_train['PAY_0']).reshape(-1,1)
#np.array(X_train['PAY_0']).reshape(1,-1)

y_pred = model.predict(np.array(X_test['PAY_0']).reshape(-1,1))

conf_matrix = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_matrix, classes = [0,1])

y_test.hist()
pd.DataFrame(data=y_pred).hist()

acc_score = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

auc_score = str(round(auc(fpr, tpr),5))

print('Accuracy: {}, AUC: {}'.format(acc_score, auc_score))

############################################
## 2 ## Multivariate logistic regression  ##
############################################

## Attempting a randomized search to find a better model


model = LogisticRegression(random_state=42, 
                           class_weight='balanced', 
                           max_iter=150,
                           n_jobs=-1)

steps = [('scaler', StandardScaler()), ('LR', model)]
pipeline = Pipeline(steps) 


params = {'LR__C':gamma.rvs(2, size=10000),
          'LR__solver':['newton-cg', 'liblinear', 'saga']}

random_search = RandomizedSearchCV(pipeline, 
                                   param_distributions=params, 
                                   cv=3, 
                                   n_iter=21,
                                   scoring='roc_auc',
                                   iid=False, 
                                   verbose=100)

random_search.fit(X_train, y_train)

random_search.best_estimator_
random_search.best_score_


## Final model selected:

lr_model = LogisticRegression(random_state=42,                           
                           solver='newton-cg',
                           C=7,
                           class_weight='balanced',
                           max_iter=150,
                           n_jobs=1,
                           penalty='l2'
                           )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_matrix, classes = [0,1])

y_test.hist()
pd.DataFrame(data=y_pred).hist()

acc_score = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

auc_score = str(round(auc(fpr, tpr),5))

print('Accuracy: {}, AUC: {}'.format(acc_score, auc_score))

coefficients = pd.concat([pd.DataFrame(X_test.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)

coefficients

# plotLearningCurve(X_train, y_train, model)   # not relevant for this algorithm

'''
----------------------
Conclusions --
----------------------

1. It is important to set the parameter class_weight='balanced' during model 
training

2. There seems to be high correlation between PAY, BILL_AMT and PAY_AMT variables
as well as between the different months. This reduces the model quality, hence
either a high L2 regularization is needed or just remove the features that are 
highly correlated. Alternative build an LSTN that can handle the correlations 
better


Maybe some of the things to look into when i have some time:
    
    1. read about the logistic regression in the Book
    2. understand what C parameter stands for
    3. understand how the randomized grid search works
    4. check what other solvers are available
    5. understand what max_iter means as the hyper parameter

'''

#############################
## END Logistic Regression ##
#############################


####################
## Tree Ensembles ##
####################


def plot_roc_curve(fpr, tpr, label=None):
       
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.show()


def show_feature_importances(predictors, fitted_model):
    
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': fitted_model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   

def plotLearningCurve(_x_train, _y_train, learning_model_pipeline,  k_fold = 10, training_sample_sizes = np.linspace(0.1,1.0,10), jobsInParallel = -1):
    
    training_size, training_score, testing_score = learning_curve(estimator = learning_model_pipeline, \
                                                                X = _x_train, \
                                                                y = _y_train, \
                                                                train_sizes = training_sample_sizes, \
                                                                cv = k_fold, \
                                                                scoring = 'roc_auc', \
                                                                n_jobs = jobsInParallel) 


    training_mean = np.mean(training_score, axis = 1)
    training_std_deviation = np.std(training_score, axis = 1)
    testing_std_deviation = np.std(testing_score, axis = 1)
    testing_mean = np.mean(testing_score, axis = 1 )

    ## we have got the estimator in this case the perceptron running in 10 fold validation with 
    ## equal division of sizes betwwen .1 and 1. After execution, we get the number of training sizes used, 
    ## the training scores for those sizes and the test scores for those sizes. we will plot a scatter plot 
    ## to see the accuracy results and check for bias vs variance

    # training_size : essentially 10 sets of say a1, a2, a3,,...a10 sizes (this comes from train_size parameter, here we have given linespace for equal distribution betwwen 0.1 and 1 for 10 such values)
    # training_score : training score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    # testing_score : testing score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    ## the mean and std deviation for each are calculated simply to show ranges in the graph

    plt.plot(training_size, training_mean, label= "Training Data", marker= '+', color = 'blue', markersize = 8)
    plt.fill_between(training_size, training_mean+ training_std_deviation, training_mean-training_std_deviation, color='blue', alpha =0.12 )

    plt.plot(training_size, testing_mean, label= "Testing/Validation Data", marker= '*', color = 'green', markersize = 8)
    plt.fill_between(training_size, testing_mean+ testing_std_deviation, testing_mean-testing_std_deviation, color='green', alpha =0.14 )

    plt.title("Scoring of our training and testing data vs sample sizes")
    plt.xlabel("Number of Samples")
    plt.ylabel("Score")
    plt.legend(loc= 'best')
    plt.show()

## Attempting a randomized search to find a better model


model = GradientBoostingClassifier(random_state=42)

params = {'n_estimators':sp_randint(50, 200),
          'min_samples_leaf':sp_randint(20, 100),
          'max_depth':sp_randint(2, 3),
          'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}

random_search = RandomizedSearchCV(model, 
                                   param_distributions=params, 
                                   cv=3, 
                                   n_iter=21,
                                   scoring='roc_auc',
                                   iid=False, 
                                   verbose=100)


### Randomized Search CV with validation on the test set 
# scores = cross_val_score(random_search, X_train, y_train, 
#                         scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                      np.std(scores)))



random_search.fit(X_train, y_train)

random_search.best_estimator_
random_search.best_score_


## Final model selected:

gbc_model = random_search.best_estimator_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

conf_matrix = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_matrix, classes = [0,1])

y_test.hist()
pd.DataFrame(data=y_pred).hist()

acc_score = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

auc_score = str(round(auc(fpr, tpr),5))

print('Accuracy: {}, AUC: {}'.format(acc_score, auc_score))

show_feature_importances(X_train.columns, model)

# plotLearningCurve(X_train, y_train, random_search.best_estimator_)

plot_roc_curve(fpr, tpr)


'''

Based on the feature importances we will take into account only some of the
features. 

It did not prove to be any better, in fact slightly worse.

'''

####################
## XGBoost #####
################

'''

LINK: https://github.com/dmlc/xgboost/tree/master/demo/guide-python

'''


'''

Version 1:
    
SickitLearn wrapper arround XGB here used

'''

d_train = xgb.DMatrix(X_train, y_train)
d_test = xgb.DMatrix(X_test, y_test)

watchlist = [(d_train, 'train'), (d_test, 'test')]

model = XGBClassifier(random_state=42)

params = {'n_estimators':sp_randint(50, 200),
          'min_samples_leaf':sp_randint(20, 100),
          'max_depth':sp_randint(2, 6)
          }

random_search = RandomizedSearchCV(model, 
                                   param_distributions=params, 
                                   cv=3, 
                                   n_iter=3,
                                   scoring='roc_auc',
                                   iid=False, 
                                   verbose=100,
)

random_search.fit(X_train, y_train)


random_search.best_score_

xgb_model = random_search.best_estimator_

xgb_model.fit(X_train
              , y_train
              )

y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc_score = str(round(auc(fpr, tpr),5))
print('AUC score: {}'.format(auc_score))

plot_roc_curve(fpr, tpr)

'''

Version 2:
    
XGB used directly here - more details are available

'''

def XGB_plot_importance(model):

    fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
    xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="blue") 
    plt.show()

# as recomended at https://xgboost.readthedocs.io/en/latest/parameter.html

def runXGBoost(X_train, X_test, y_train, y_test, imbalanced_proportion): 

    d_train = xgb.DMatrix(X_train, y_train)
    d_test = xgb.DMatrix(X_test, y_test)
    
    watchlist = [(d_train, 'train'), (d_test, 'test')]
    
    params = {'objective': 'binary:logistic'
              , 'learning_rate': 0.2
              , 'n_estimators': 106
              , 'min_samples_leaf': 106
              , 'random_state': 42
              , 'scale_pos_weight': imbalanced_proportion
              , 'eval_metric': 'auc'          
            }
    
    max_rounds = 1000
    early_stop_rounds = 30
    
    xgbc_model = xgb.train(params
                      , d_train
                      , max_rounds
                      , watchlist
                      , early_stopping_rounds=early_stop_rounds
                      )
    
    XGB_plot_importance(xgbc_model)
    
    y_pred_proba = xgbc_model.predict(d_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    auc_score = str(round(auc(fpr, tpr),5))
    print('AUC score: {}'.format(auc_score))
    
    plot_roc_curve(fpr, tpr)
    
    

imbalanced_proportion = sum(y) / y[y==0].count()

runXGBoost(X_train, X_test, y_train, y_test, imbalanced_proportion)


'''

Dimensionality reduction does not help much

'''

from sklearn.decomposition import PCA

pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_[:6]))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

X_train_pca_5dim = pca.transform(X_train)[:,:6]
X_test_pca_5dim = pca.transform(X_test)[:,:6]


runXGBoost(X_train_pca_5dim, X_test_pca_5dim, y_train, y_test, imbalanced_proportion)


'''

Support Vector Machine

1. Using Principal Component Analysis to check how many dimensions would be needed
to represent a good proportion of the data as we already saw that some of the 
features are not so important

'''

model = SVC(kernel='rbf', class_weight='balanced')

steps = [('scaler', StandardScaler()), ('SVC', model)]
pipeline = Pipeline(steps) 


params = {'SVC__C':[1, 5, 10, 50],
          'SVC__gamma':[0.0001, 0.0005, 0.001, 0.005]}

grid_search = GridSearchCV(pipeline, 
                           param_grid=params, 
                           cv=3, 
                           scoring='roc_auc',
                           verbose=100)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# Second round of grid search

params = {'SVC__C':[10, 20],
          'SVC__gamma':[0.005, 0.01, 0.015]}

grid_search = GridSearchCV(pipeline, 
                           param_grid=params, 
                           cv=3, 
                           scoring='roc_auc',
                           verbose=100)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# Second round of grid search

params = {'SVC__C':[10, 12],
          'SVC__gamma':[0.015, 0.02]}

grid_search = GridSearchCV(pipeline, 
                           param_grid=params, 
                           cv=3, 
                           scoring='roc_auc',
                           verbose=100)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

## Setting up the SVC with the best parameters and 
## fit with the probability=True

svc_model = grid_search.best_estimator_

y_pred = svc_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc_score = str(round(auc(fpr, tpr),5))
print('AUC score: {}'.format(auc_score))

plot_roc_curve(fpr, tpr)

# model with the predicted probabilities so we can build the AUC curve

svc_model = SVC(kernel='rbf'
                , class_weight='balanced'
                , probability=True
                , C=12
                , degree=3
                , gamma=0.015
                , decision_function_shape='ovr')

svc_model.fit(X_train, y_train)

y_pred_proba = svc_model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc_score = str(round(auc(fpr, tpr),5))
print('AUC score: {}'.format(auc_score))

plot_roc_curve(fpr, tpr)


'''

voting classifier with soft voting attempt

'''

from sklearn.ensemble import VotingClassifier

voting_estimators = [('lr', lr_model), ('gbc', gbc_model), ('svc', svc_model)]

voting = VotingClassifier(voting_estimators, voting='soft')

voting.fit(X_train, y_train)

y_pred_proba = voting.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1], pos_label=1)
auc_score = str(round(auc(fpr, tpr),5))
print('AUC score: {}'.format(auc_score))

plot_roc_curve(fpr, tpr)


'''

Hard voting classifier

'''

    
voting_model = VotingClassifier( voting='hard'
                                , estimators=[
                                    ('xgb', xgb_model),
                                    ('logit', lr_model),
                                    ('svm', svc_model),
                                ]
                                )

voting_model.fit(X_train, y_train)

y_pred_proba = voting.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc_score = str(round(auc(fpr, tpr),5))
print('AUC score: {}'.format(auc_score))














