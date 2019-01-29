# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:10:18 2019

@author: ghosty_mk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('iris.csv')
x = data.iloc[:,[1,2,3,4]].values
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10)
y=kmeans.fit_predict(x)

plt.scatter(x[y==0,0],x[y==0,1], c='blue', label = '1')
plt.scatter(x[y==1,0],x[y==1,1], c='red', label = '2')
plt.scatter(x[y==2,0],x[y==2,1], c='green', label = '3')

# Plot the training points

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(x[y==0, 2], x[y==0, 3], x[y==0, 1],c='blue')
ax.scatter(x[y==1, 2], x[y==1, 3], x[y==1, 1],c='red')
ax.scatter(x[y==2, 2], x[y==2, 3], x[y==2, 1],c='green')

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()