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

ZeroThres=2

TSS=TimeSeriesSplit()#Creates 5 chronological splits in data

#Give each day of the week a number
dayMap = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
OccupData["dayNum"]=OccupData["Day"].map(dayMap)

FalsePosVError=[]
FalsePosTError=[]

FalseNegVError=[]
FalseNegTError=[]

FalsePosBError=[]
FalseNegBError=[]
    
weights=range(1,100,5)

for wValue in weights:
    title="Weight of "+str(wValue)
    
    X = OccupData[["Time","dayNum"]]
    Y = OccupData[["headcount"]]
    
    #SplitData in 5 groups
    
    ValError=[]
    ValR2=[]
    
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
        w=np.where(Y_train["headcount"] == 0, wValue, 1)

        Decision_tree.fit(X_train, Y_train,sample_weight=w)

        TrainPredic = Decision_tree.predict(X_train)
        ValPredic = Decision_tree.predict(X_val)
        Base_val=X_val["Time"].astype(str).map(Baseline)
        

        

        ValError.append(root_mean_squared_error(Y_val,ValPredic))

        ValR2.append(r2_score(Y_val,ValPredic))
        
    FalsePosVError.append(len(Y_val[(Y_val["headcount"]<=ZeroThres)&(ValPredic>ZeroThres)])/len(Y_val[(Y_val["headcount"]<=ZeroThres)]))
    FalsePosTError.append(len(Y_train[(Y_train["headcount"]<=ZeroThres)&(TrainPredic>ZeroThres)])/len(Y_train[(Y_train["headcount"]<=ZeroThres)]))
    
    FalseNegVError.append(len(Y_val[(Y_val["headcount"]>ZeroThres)&(ValPredic<=ZeroThres)])/len(Y_val[(Y_val["headcount"]>ZeroThres)]))
    FalseNegTError.append(len(Y_train[(Y_train["headcount"]>ZeroThres)&(TrainPredic<=ZeroThres)])/len(Y_train[(Y_train["headcount"]>ZeroThres)]))
    
    FalsePosBError.append(len(Y_train[(Y_train["headcount"]<=ZeroThres)&(Base_val>ZeroThres)])/len(Y_train[(Y_train["headcount"]<=ZeroThres)]))
    FalseNegBError.append(len(Y_train[(Y_train["headcount"]>ZeroThres)&(Base_val<=ZeroThres)])/len(Y_train[(Y_train["headcount"]>ZeroThres)]))
        

    print(title+" Final R2: "+str(ValR2[-1]))
    print(title+" Final RMSE: "+str(ValError[-1])+"\n")

    
plt.figure()
plt.plot(weights,FalsePosVError,label="False Positive")
plt.plot(weights,FalseNegVError,label="False Negative")
plt.xlabel("Weight Value")
plt.ylabel('Percentage Error')
plt.title("Valdiation Weight Sweep")
plt.legend()
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/ Val Weight Sweep.png'
plt.savefig(loc)
plt.show()

plt.figure()
plt.plot(weights,FalsePosTError,label="False Positive")
plt.plot(weights,FalseNegTError,label="False Negative")
plt.xlabel("Weight Value")
plt.ylabel('Percentage Error')
plt.title("Training Weight Sweep")
plt.legend()
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/ Train Weight Sweep.png'
plt.savefig(loc)
plt.show()


