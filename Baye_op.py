# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:26:28 2020
A Python program to perform bayesian optimization in bagging-based PU learning with bayesian optimization
This code is released from the paper of Zhiqiang Zhang in computers and geosciences
authors: Zhiqiang Zhang, Gongwen Wang, Chong Liu,  Lizhen Cheng, Deming Sha
email: zq_zhang_geo@126.com, gwwang@cugb.edu.cn
@author: Zhiqiang Zhang
"""

from __future__ import division, print_function
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from hyperopt import fmin, tpe, Trials

def bestObj(space, n_iter_hopt, kFoldSplits, data_bootstrap, train_label, 
            seed = 42, metric = 'accuracy'):
    def objective(space):
        
        best_score=1.0 
        metric = 'f1'
        
        model =  RFC(**space)
        
        score = 1-cross_val_score(model, data_bootstrap,train_label
                                , cv=5
                                , scoring=metric
                                , verbose=False).mean() 
        
        if (score < best_score):
            best_score=score
            
        return score
    
    trials = Trials()
    
    best = fmin(objective, space = space
                , algo = tpe.suggest
                , max_evals = n_iter_hopt
                , trials = trials
                , rstate = np.random.RandomState(seed)
               )
    if best['n_estimators']==0:
        n_estimatorsObj = 1
    else:
        n_estimatorsObj = best['n_estimators']
    
    if best['max_depth']==0:
        max_depthObj = 1
    else:
        max_depthObj = best['max_depth']
    
    if best['min_samples_leaf']==0:
        min_samples_leafObj=1
    else:
        min_samples_leafObj=best['min_samples_leaf']
   
    return n_estimatorsObj, max_depthObj, min_samples_leafObj

