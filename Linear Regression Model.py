# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:06:02 2026

@author: Octob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

Data=pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')

OccupData=Data.resample('60min',on='collecteddate').max().reset_index()
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60
DayList=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

PeriodList=["Class","Exam","Neither"]

Model={}

TSS=TimeSeriesSplit()#Creates 5 splits

for day in DayList:
    DayData=Data[Data["Day"].isin([day])]
    Model[day]={}
    for Condition in PeriodList:
        if Condition=="Neither":
            PeriodData=DayData[DayData['Class'].isin([0])|DayData['Exam'].isin([0])]
        else:
            PeriodData=DayData[DayData[Condition].isin([1])]
        X=PeriodData["Time"]
        Y=PeriodData["headcount"]
        for trainIndex, valIndex in TSS.split(X):
            
            X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
            Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
            
            p_fitted = Polynomial.fit(X_train, Y_train, deg=3)
        Model[day][Condition]=p_fitted.coef
        
OccupData["Condition"]=np.where(OccupData["Class"]==1,"Class",np.where(OccupData["Exam"]==1,"Exam","Neither"))
(OccupData["Day"].map(Model)).json_normalize
