# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:24 2026

@author: Octob
"""
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

#Import Data
Data = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')

depth=2
Test=[10,15,20,30,60]
TrainError=[]
BaseError=[]
ValError=[]
dayTimeError=[]
timeError=[]
for interval in Test:
    OccupData=Data
    OccupData=OccupData.resample(str(interval)+'min',on='collecteddate').max().reset_index().fillna(0)
    OccupData['Day'] = OccupData['collecteddate'].dt.day_name()
    
    #Give each day of the week a number
    dayMap = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    OccupData["dayNum"]=OccupData["Day"].map(dayMap)
    OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60
    OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60
    DeltaIntervals=int((7*24*60/interval))
    OccupData["Lag"]=OccupData["headcount"].shift(int(DeltaIntervals))
    OccupData=OccupData.dropna(subset=['Lag'])
    
    
    
    
    #SplitData
    X = OccupData[["dayNum","Time","Lag"]]
    Y = OccupData[["headcount","Lag"]]
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=98,
        shuffle=False
    )
    
    #Test for different depths
    occup_tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=3,
        random_state=98,
    )
    
    nolag_tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=3,
        random_state=98,
    )
    
    time_tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=3,
        random_state=98,
    )
    occup_tree.fit(X_train, Y_train["headcount"])
    nolag_tree.fit(X_train[['dayNum','Time']], Y_train["headcount"])
    time_tree.fit(X_train[["Time"]], Y_train["headcount"])
    
    trainPredi = occup_tree.predict(X_train)
    valPredi = occup_tree.predict(X_val)  
    dayTimePredi=nolag_tree.predict(X_val[['dayNum','Time']])  
    timePredi=time_tree.predict(X_val[["Time"]])  
    
    TrainError.append(mean_squared_error(Y_train["headcount"],trainPredi))
    ValError.append(mean_squared_error(Y_val["headcount"],valPredi))
    BaseError.append(mean_squared_error(Y_val["headcount"],Y_val["Lag"]))
    dayTimeError.append(mean_squared_error(Y_val["headcount"],dayTimePredi))
    timeError.append(mean_squared_error(Y_val["headcount"],timePredi))

ErrorData=pd.DataFrame({
    'Baseline Model': BaseError,
    'Time + Day Model': dayTimeError,
    'Time Only Model': timeError,
    'Time + Lag + Day Model' : ValError
    },index=Test)
ErrorData.plot(kind="bar",title="Decision Tree Model - Interval Testing")
