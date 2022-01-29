# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:05:24 2022

@author: User
"""
import pandas as pd
dataset = pd.read_csv(r'Social_Network_Ads.csv')
X = dataset[['User ID', 'Gender', 'Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

# In[]
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)
n = kf.get_n_splits(X)

# In[]
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index,"TEST:", test_index)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    print(X_test)
    y_train, y_test = y[train_index], y[test_index]




















