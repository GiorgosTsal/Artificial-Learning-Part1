#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:34:00 2019

@author: gtsal
"""

import pandas as pd  # To read data
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt # To visualize
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Custom print function to show results of errors in 
def print_errors(y_true, y_pred, technique = ""): 
    if(technique!=""):
        print("For technique: " + str(technique))
    print("Mean absolute percentage error: " + str(mean_absolute_percentage_error(y_true, y_pred)))
    print("Mean absolute error: " + str( mean_absolute_error(y_true, y_pred)))
    print("Mean squared error: " + str(mean_squared_error(y_true, y_pred)))


#Read data and give header
df= pd.read_csv('Concrete_Data.csv')  # load data set
df.columns= ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength"]

for i in range(df.shape[1]-1):
    plt.plot(df[df.columns[i]],df[df.columns[df.shape[1]-1]],'x')
    plt.xlabel(df.columns[i])
    plt.ylabel("Feature " + str(i))
 
    
#for i in df.columns:
#    plt.figure()
#    plt.scatter(df[i])
#    
# Preprocessing Input data, split the dataset into train and test (test = 0.3)
#x = df.drop('Concrete compressive strength', axis=1)
#y = df.iloc[:,-1] # select last one dimension
X=df.iloc[:,0:8]
y=df.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3,shuffle=False)
#scale features X_train[0,1] because the algorithms computes distance(eg LDA)
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)
print(y_test)


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
Y_pred = linear_regressor.predict(X_test)  # make predictions

#Show errors for Ordinary Least Squares regression
#print_errors(y_test, Y_pred, "Ordinary Least Squares regression")
print_errors(y_test, Y_pred, "Ordinary Least Squares regression")