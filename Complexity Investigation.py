# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:24 2026

@author: Octob
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor 


#Import Data
OccupData = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')
OccupData=OccupData.resample('60min',on='collecteddate').max().reset_index().dropna()

TSS=TimeSeriesSplit()#Creates 5 chronological splits in data

#Give each day of the week a number
dayMap = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
OccupData["dayNum"]=OccupData["Day"].map(dayMap)

    
Models=["Time","Time + Day","Time + Day + Context"]

for i in range(0,len(Models)):
    title=Models[i]
    
    if i==0:
        X = OccupData[["Time"]]
        
    elif i==1:
        X = OccupData[["Time","dayNum"]]
    elif i==2:
        X = OccupData[["Time","dayNum","Class","Exam"]]
    Y = OccupData[["headcount"]]
    
    #SplitData in 5 groups
    TrainError=[]
    ValError=[]
    ValR2=[]
    TrainR2=[]
    
    BaseR2=[]
    BaseError=[]
    for trainIndex, valIndex in TSS.split(X):
        
        
        
        X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
        Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
        
        Baseline={}
        for hour in X_train["Time"].unique():
            Baseline[str(hour)]=np.mean(Y_train[X_train["Time"].isin([hour])]["headcount"])
        
        #Test for different depths
        Decision_tree = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=3,
            random_state=90,
        )

        Decision_tree.fit(X_train, Y_train)

        TrainPredic = Decision_tree.predict(X_train)
        ValPredic = Decision_tree.predict(X_val)
        Base_val=X_val["Time"].astype(str).map(Baseline)
        

        
        TrainError.append(root_mean_squared_error(Y_train,TrainPredic))
        ValError.append(root_mean_squared_error(Y_val,ValPredic))
        TrainR2.append(r2_score(Y_train,TrainPredic))
        ValR2.append(r2_score(Y_val,ValPredic))
        
        BaseR2.append(r2_score(Y_val,Base_val))
        BaseError.append(root_mean_squared_error(Y_val,Base_val))
        
            
    plt.figure()
    plt.plot([1,2,3,4,5],TrainError,label="Training")
    plt.plot([1,2,3,4,5],ValError,label="Validation")
    plt.plot([1,2,3,4,5],BaseError,label="Baseline")
    plt.xlabel("Fold")
    plt.ylabel('Root Mean Square Error')
    plt.title("RMSE Variation for "+title+" Model")
    plt.xticks(range(1,6))
    plt.legend()
    loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/RMSE '+title+'.png'
    plt.savefig(loc)
    
    
    plt.figure()
    plt.plot([1,2,3,4,5],TrainR2,label="Training")
    plt.plot([1,2,3,4,5],ValR2,label="Validation")
    plt.plot([1,2,3,4,5],BaseR2,label="Baseline")
    plt.xlabel("Fold")
    plt.ylabel('Coefficient Of Determination (r²)')
    plt.title("r² Variation for "+title+" Model")
    plt.xticks(range(1,6))
    plt.legend()
    loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/r2 '+title+'.png'
    plt.savefig(loc)
    plt.show()




