# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 04:30:29 2020

@author: Arnob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
#num= int(input())
tem= input("Enter Temperature: ")
tem= float(tem)
month= input("Month(1-12): ")
month= float(month)
year= input("Year: ")
year= float(year)
d = {'tem': [tem], 'month': [month], 'year': [year]}

df = pd.DataFrame(data=d)

X_user_test = df.iloc[:,0:3]
#X_user_test = df.to_numpy()
#X_user_test= np.array([tem,month,year])
#X_user_test = X_user_test.reshape(-1,1)
dataset = pd.read_csv('Temp_and_rain.csv')
X= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

print("Correlation", dataset.corr())

sc= StandardScaler()
poly = PolynomialFeatures(degree=3)
#X= sc.fit_transform(X)
X= poly.fit_transform(X)
X_user_test = poly.fit_transform(X_user_test)
reg = LinearRegression()

linear_model = reg.fit(X, y)

y_user_test_predictions = linear_model.predict(X_user_test)

print('User input result %s => %d' % (X_user_test.tolist(), y_user_test_predictions))

print("Model score",linear_model.score(X,y)*100)