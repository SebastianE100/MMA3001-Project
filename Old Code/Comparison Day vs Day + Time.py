# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:24 2026

@author: Octob
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

TSS=TimeSeriesSplit()#Creates 5 splits

#Import Data
OccupData = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')

#Set sample interval to be 15 mins
interval=15
OccupData=OccupData.resample(str(interval)+'min',on='collecteddate').max().reset_index().fillna(0)

#Adds day name as a collumn
OccupData['Day'] = OccupData['collecteddate'].dt.day_name()

#Give each day of the week a number
dayMap = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
OccupData["dayNum"]=OccupData["Day"].map(dayMap)

#Adds the time of the day in hours
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60

#Adds 1 week lag into dataet
DeltaIntervals=int((7*24*60/interval))
OccupData["Lag"]=OccupData["headcount"].shift(int(DeltaIntervals))
OccupData=OccupData.dropna(subset=['Lag'])#removes first week of data

Test=[1,2,3,4,5,6]#Selects depths to be tested

X = OccupData[["dayNum","Time",'Class']]
Y = OccupData[["headcount"]]
Base=OccupData[["Lag"]]

DepthTrainError=[]
DepthBaseError=[]
DepthValError=[]
DepthDayTError=[]
DepthDayVError=[]

for depth in Test:
    TrainError=[]
    BaseError=[]
    
    DayTError=[]
    DayVError=[]
    
    dayTError=[]
    dayVError=[]
    
    ValError=[]
    #SplitData in 5 groups
    for trainIndex, valIndex in TSS.split(X):
        
        X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
        Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
        Base_train, Base_val = Base.iloc[trainIndex] , Base.iloc[valIndex]
        
        #Test for different depths
        DayTime_tree = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=3,
            random_state=98,
        )
        
        Day_tree = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=3,
            random_state=98,
        )
        Day_tree.fit(X_train[["Time"]], Y_train)
        
        
        
        DayTime_tree.fit(X_train, Y_train)
            
        TrainPredic=DayTime_tree.predict(X_train)  
        ValPredic=DayTime_tree.predict(X_val)  
        
        DayTPredic=Day_tree.predict(X_train[["Time"]])  
        DayVPredic=Day_tree.predict(X_val[["Time"]])  
        
        TrainError.append(r2_score(Y_train,TrainPredic))
        ValError.append(r2_score(Y_val,ValPredic))
        BaseError.append(r2_score(Y_val,Base_val))
        
        dayTError.append(r2_score(Y_train,DayTPredic))
        dayVError.append(r2_score(Y_val,DayVPredic))
    DepthTrainError.append(np.mean(TrainError))
    DepthBaseError.append(np.mean(BaseError))
    DepthValError.append(np.mean(ValError))
    
    DepthDayVError.append(np.mean(dayVError))
    DepthDayTError.append(np.mean(dayTError))

ErrorData=pd.DataFrame({
    'Baseline Model': DepthBaseError,
    'Training Error ': DepthTrainError,
    'Validation Error': DepthValError,
    "Time Only T error":DepthDayTError,
    'Time only Valid error':DepthDayVError
    },index=Test)
ErrorData.plot(title="Day + Time + Classes Running Model - Depth Testing",xlabel="Depth",ylabel="R^2 Score")
plt.show()
