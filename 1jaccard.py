# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:24:12 2019

@author: ghostymk(formerly mktesla :P)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('iris.csv')
x = data.iloc[:,[1,2,3,4]].values
#scale
scl = MinMaxScaler()
x = scl.fit_transform(x)

tx = x[:,[2,3]]

row,col= np.shape(x)

c = np.zeros((row,row))
#creating dissimilarity array
for i in range(row):
    for j in range(row):
        c[i,j] = euclidean([x[i,:]],[x[j,:]])
        
avg = np.zeros(row)
for i in range(row):
    avg[i]= np.average(c[i,:])
    
#appending datapoints of same cluster
cc = [] 

for i in range(row):
    l=[]
    for j in range(row):
        if(c[i,j]<avg[i]):
            l.append(j)
    cc.append(l)

#flag list to identify subsets
flag = [] 
for i in range(row):
    flag.append(0)
for i in range(row):
    for j in range(row):
        if(set(cc[j]).issubset(set(cc[i])) and i!=j):
            flag[j]=1
        

    
