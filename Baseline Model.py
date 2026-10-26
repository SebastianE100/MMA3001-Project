# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:32:32 2026

@author: Octob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error,r2_score

#Import Data
OccupData = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')

#Resample Data and remove Nan values
OccupData=OccupData.resample('60min',on='collecteddate').max().reset_index().dropna()

#Creates 5 chronological splits in data
TSS=TimeSeriesSplit()

#Test both baseline models
for meanVar in ["Time","Day"]:
    #Select title of graph
    if meanVar=="Time":
        title="Mean Time"
    else:
        title="Mean Day"
    
    #Set data the model has access to
    X = OccupData[[meanVar]]
    Y = OccupData[["headcount"]]
    
    #Generate list that store performance criteria
    TrainError=[]
    ValError=[]
    ValR2=[]
    TrainR2=[]
    
    #Train model and evalutate on 5 folds
    for trainIndex, valIndex in TSS.split(X):
        
        #Create training and validation data
        X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
        Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
        
        #Create baseline model
        Baseline={}
        for param in X_train[meanVar].unique():
            Baseline[str(param)]=np.mean(Y_train[X_train[meanVar].isin([param])]["headcount"])
        
        #Get model outputs
        TrainPredic=X_train[meanVar].astype(str).map(Baseline)
        ValPredic=X_val[meanVar].astype(str).map(Baseline)
        

        #Evaluaute model performance
        TrainError.append(root_mean_squared_error(Y_train,TrainPredic))
        ValError.append(root_mean_squared_error(Y_val,ValPredic))
        TrainR2.append(r2_score(Y_train,TrainPredic))
        ValR2.append(r2_score(Y_val,ValPredic))

        
    #Create and save RMSE graph
    plt.figure()
    plt.plot([1,2,3,4,5],TrainError,label="Training")
    plt.plot([1,2,3,4,5],ValError,label="Validation")
    plt.xlabel("Fold")
    plt.ylabel('Root Mean Square Error')
    plt.title("RMSE Variation for "+title+" Model")
    plt.xticks(range(1,6))
    plt.legend()
    loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/RMSE '+title+'.png'
    plt.savefig(loc)
    
    #Create and save r2 graph
    plt.figure()
    plt.plot([1,2,3,4,5],TrainR2,label="Training")
    plt.plot([1,2,3,4,5],ValR2,label="Validation")
    plt.xlabel("Fold")
    plt.ylabel('Coefficient Of Determination (r²)')
    plt.title("r² Variation for "+title+" Model")
    plt.xticks(range(1,6))
    plt.legend()
    loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/r2 '+title+'.png'
    plt.savefig(loc)
    plt.show()