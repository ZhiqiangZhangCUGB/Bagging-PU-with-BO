# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:19:13 2020
A Python program to perform bagging-based PU learning with bayesian optimization
This code is released from the paper of Zhiqiang Zhang in computers and geosciences
authors: Zhiqiang Zhang, Gongwen Wang, Chong Liu,  Lizhen Cheng, Deming Sha
email: zq_zhang_geo@126.com, gwwang@cugb.edu.cn
@author: Zhiqiang Zhang
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from hyperopt import hp
from time import time
from Baye_op import bestObj
from data_pro import puData

from sklearn.metrics import recall_score

seed = 42 
metric = 'f1'
kFoldSplits = 5
n_iter_hopt = 50

space = { 'n_estimators'      :  hp.choice('n_estimators', range(1,200))
          ,'max_depth'        :  hp.choice('max_depth', range(1,20))
          ,'min_samples_leaf' :  hp.choice('min_samples_leaf',range(1,20))
          }



PX_train, PX_test,  data_U, n_oob, f_oob, K, TS, NU, train_label, test_label, t_m, t_test = puData(P_dir = r'your data path'
                                                                                            ,U_dir = r'your data path')
test_label=pd.Series(test_label)
test_label.to_csv(r'your data path',header=True)
begin_time = time()
feature_improtance = []

test_pro=[]

T = 10
for i in range(T):
    
    bootstrap_sample = np.random.choice( np.arange(NU)
                                       , replace=True
                                       , size = K
                                       )
    
    data_bootstrap = np.concatenate((  PX_train
                                     , data_U[bootstrap_sample, :]
                                     )
                                     , axis=0
                                    )    
   
    
    n_estimatorsObj, max_depthObj,min_samples_leafObj = bestObj(  space
                                                                 , n_iter_hopt
                                                                 , kFoldSplits
                                                                 , data_bootstrap
                                                                 , train_label
                                                                 , seed   = 42
                                                                 , metric = 'accuracy')
    
    
    model = RFC(     n_estimators  = n_estimatorsObj
                ,       max_depth  = max_depthObj
                , min_samples_leaf = min_samples_leafObj
                ,     class_weight = 'balanced'
                ,     random_state = 0
                ,           n_jobs = -1)
    
    model.fit(data_bootstrap, train_label)
   
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    
  
    bootstrap_test_sample = np.random.choice( idx_oob
                                            , replace=True
                                            , size = TS
                                            )
    data_test_bootstrap = np.concatenate((  PX_test
                                          ,data_U[bootstrap_test_sample, :]
                                          )
                                          ,axis=0
                                         )   
    
    t_idx=np.arange(TS)
   
    t_test[t_idx] += model.predict_proba(data_test_bootstrap[t_idx])              

    f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])
    feature_improtance.append(model.feature_importances_)
    n_oob[idx_oob] += 1
    t_m[t_idx] += 1
predict_proba = f_oob[:, 1]/n_oob.squeeze() 
fea_imp = sum(feature_improtance)/T 
predict_test=t_test[:, 1]/t_m.squeeze()
predict_proba=pd.DataFrame(predict_proba)
predict_proba.to_csv(r'your save path')
end_time = time()
run_time = end_time-begin_time
print ('paraming run time：',run_time)
predict_test=np.int64(predict_test>0.5)
recall= recall_score(test_label, predict_test)
Pr=(sum(predict_test))/(predict_test.shape[0])
score=(recall*recall)/Pr

print("Recall:", recall)
print("Pr:", Pr)
print("Score:", score)
