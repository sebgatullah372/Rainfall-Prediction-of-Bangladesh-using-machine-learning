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

dataset = pd.read_csv('Temp_and_rain.csv')
X= dataset.iloc[:,0:3]
y= dataset.iloc[:,3]
print("Correlation", dataset.corr())
#ax = plt.gca()
#dataset.plot(x ='Month', y='rain', kind = 'line',ax=ax)
#dataset.plot(x ='tem', y='rain', kind = 'line',ax=ax)
#plt.show()
sc= StandardScaler()
poly = PolynomialFeatures(degree=3)
#X= sc.fit_transform(X)
X= poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

reg = LinearRegression()

linear_model = reg.fit(X_train, y_train)

print("coef","intercept",linear_model.coef_, linear_model.intercept_)
print("Model score",linear_model.score(X_train,y_train)*100)
predictions = linear_model.predict(X_test)

y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)
#print(linear_model.score(X_test, y_test)*100)
for i in range(10):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))
var_score=explained_variance_score(y_test, predictions)
print("varriance score",var_score*100)
