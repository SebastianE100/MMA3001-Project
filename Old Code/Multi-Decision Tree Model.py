# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:08:36 2026

@author: Octob
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

Data=pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')
OccupData=Data.resample('60min',on='collecteddate').max().reset_index()
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60


DayData=OccupData.loc[OccupData["Day"].isin(["Monday"])]

ExamData=DayData[DayData["Class"].isin([1])]

ExamData['Time']
X = ExamData[["Time"]]
Y = ExamData[["headcount"]]

TSS=TimeSeriesSplit()#Creates 5 splits

DepthTrainError=[]
DepthValError=[]

TrainError=[]
ValError=[]
for trainIndex, valIndex in TSS.split(X):

    
    X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
    Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
    
    
    
    Day_tree = DecisionTreeRegressor(
        max_depth=9,
        min_samples_leaf=3,
        random_state=98,
    )
    Day_tree.fit(X_train, Y_train)
    
        
    TrainPredic=Day_tree.predict(X_train)
    
    TrainPredic = Day_tree.predict(X_train)
    ValPredic = Day_tree.predict(X_val)
    
    
    TrainError.append(root_mean_squared_error(Y_train,TrainPredic))
    ValError.append(root_mean_squared_error(Y_val,ValPredic))
    
plt.figure(1)
plt.plot([1,2,3,4,5],TrainError,label="Training")
plt.plot([1,2,3,4,5],ValError,label="Validation")
plt.xlabel("Fold")
plt.ylabel('root_mean_squared_error')
plt.title("Decison Tree Model on Mondays")
plt.legend()

plt.figure(2)
plt.plot(np.array(X_train).T.flatten(),TrainPredic)
plt.plot(np.array(X_train).T.flatten(),Y_train,".")
plt.title("T3")