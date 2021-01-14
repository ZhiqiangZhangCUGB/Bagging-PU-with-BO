# -*- coding: utf-8 -*-
"""
Created on Sun Sep  22 16:26:28 2020
A Python program of P-V Plot
This code is released from the paper of Zhiqiang Zhang in computers and geosciences
authors: Zhiqiang Zhang, Gongwen Wang, Chong Liu,  Lizhen Cheng, Deming Sha
email: zq_zhang_geo@126.com, gwwang@cugb.edu.cn
@author: Zhiqiang Zhang
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from intersect import intersection

pv=[]#the percentage occupied 3D geological model volumes of the corresponding prospecting probabilities(for intersection point & P-V plot)
pr=[]#the percentage occupied known ore bodies volumes of the corresponding prospecting probabilities(for intersection point)
pr_p=[]#the percentage occupied known ore bodies volumes of the corresponding prospecting probabilities(for P-V plot)

def pvplot (P_dir,U_dir,prob_dir):
    '''
    P_dir: the path and name of positive data
    U_dir: the path and name of unlabeled data
    prob_dir: the path and name of the predict probability
    '''
    data_P=pd.read_csv(f'{P_dir}')
    data_U=pd.read_csv(f'{U_dir}')
    predict_data=pd.read_csv(f'{prob_dir}')
    
    true_labels = np.zeros(shape=(data_U.shape[0]))
    true_labels[:data_P.shape[0]] = 1.0

    predict_data=predict_data.iloc[:,1]
    low_p=min(predict_data)
    high_p=max(predict_data)
    predict_data=np.array(predict_data)
    
    for i in np.arange(low_p,high_p,0.01):
        p_label_v=np.int64(predict_data<i)
        p_v=(sum(p_label_v))/true_labels.shape[0]
        pv.append(p_v)
        p_r_p=recall_score(true_labels,p_label_v)
        pr_p.append(p_r_p)
        p_label_r=np.int64(predict_data>i)
        p_r=recall_score(true_labels,p_label_r)
        pr.append(p_r)
    # fit the curve   
    x=np.arange(low_p,high_p,0.01)
    fv = np.polyfit(x, pv, 18)
    fr = np.polyfit(x, pr, 18)
    fv = np.poly1d(fv)
    fr =  np.poly1d(fr)
    fv = fv(x)
    fr = fr(x)
    #calculate the intersection
    x_i,y_i=intersection(x, fv, x, fr)
    #P-V plot
    fig, ax = plt.subplots(figsize = (8,8))
    #the percentage occupied known ore bodies volumes of the corresponding prospecting probabilities
    ax. legend(loc='upper left',shadow=False, fontsize=12)
    ax.plot(x,pv,'g',label='Area')
    ax.set_xlabel('Predictive probability',fontsize=20)
    ax.set_ylabel('Precentage of the known orebodies',fontsize=20)
    plt.tick_params(labelsize=20)
    ax.set_ylim([-0.05,1.05])
    ax.set_xlim([-0.05,1.05]) 
    ax.legend(loc=3, fontsize=14)
    #the percentage occupied 3D geological model volumes of the corresponding prospecting probabilities
    ax1=ax.twinx()
    ax1.plot(x,pr_p,'r',label='Precition rate')
    ax1.plot(x_i,1-y_i,'bo',markersize=8, zorder=3)
    ax1.annotate(
        r'intersection',  
        xy = (x_i, 1-y_i),    
        xytext = (50,0),  
        textcoords = 'offset points',       
        fontsize = 16,
        arrowprops = dict(arrowstyle='->',         
                          connectionstyle='arc3')    
        )
    ax1.set_ylabel('Precentage of the study area',fontsize=20)
    plt.tick_params(labelsize=20)
    ax1.set_ylim([1.05,-0.05])
    ax1.legend(loc=4, fontsize=14)

    plt.show()
    # x_i, y_i is the intersection
    return x_i,y_i
   
    
    