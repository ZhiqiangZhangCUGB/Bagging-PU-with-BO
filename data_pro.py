# -*- coding: utf-8 -*-

"""
Created on Mon Sep  7 09:48:03 2020
A Python program to perform data process in the bagging-based PU learning with bayesian optimization
This code is released from the paper of Zhiqiang Zhang in computers and geosciences
authors: Zhiqiang Zhang, Gongwen Wang, Chong Liu, Lizhen Cheng, Deming Sha
email: zq_zhang_geo@126.com, gwwang@cugb.edu.cn
@author: Zhiqiang Zhang
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def puData(P_dir,U_dir):
    '''
    P_dir: the path and name of positive data
    U_dir: the path and name of unlabeled data
    '''
    data_P = pd.read_csv(f'{P_dir}')
    data_U = pd.read_csv(f'{U_dir}')
    data_P=np.array(data_P)
    data_U=np.array(data_U)
    
    NP = data_P.shape[0]
    NU = data_U.shape[0]
    
                                                                   
    P_label= np.zeros(shape=(NP,))
    P_label[:]=1.0

    PX_train,PX_test,PY_train,PY_test=train_test_split(data_P,P_label,test_size=0.3,random_state=0)
   
    N_P_train= PX_train.shape[0]
    K = N_P_train
    train_label = np.zeros(shape=(N_P_train+K,))
    train_label[:N_P_train] = 1.0
   
 
    N_P_test=PX_test.shape[0]  
    TS=N_P_test
    test_label = np.zeros(shape=(N_P_test+TS,))
    test_label[:N_P_test] = 1.0
    
    n_oob = np.zeros(shape=(NU,1))
    f_oob = np.zeros(shape=(NU, 2))
    t_m   = np.zeros(shape=(2*TS,1))
    t_test = np.zeros(shape=(2*TS,2))
    
    return  PX_train, PX_test, data_U, n_oob, f_oob, K, TS, NU, train_label, test_label, t_m, t_test
