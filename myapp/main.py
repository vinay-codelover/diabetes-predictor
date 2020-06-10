# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:13:34 2020

@author: vijay
"""


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
print(np.ones(8))
dataset = pd.read_csv('diabetes.csv')
#data = dataset.drop(['Cement','Age'],axis = 0)
y = dataset.iloc[:,8].values
X= dataset.iloc[:,0:7].values
LogisticRegressor = LogisticRegression()
xTrain, xTest, yTrain, yTest = train_test_split(X, y,  random_state = 1000)
LogisticRegressor.fit(xTrain, yTrain)
model = pickle.dump(LogisticRegressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[10,125,70,26,115,31.1,0.205]]))
print(model.predict_proba([[10,125,70,26,115,31.1,0.205]]))

