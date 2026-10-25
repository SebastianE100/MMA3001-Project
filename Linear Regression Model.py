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


    
def ModelEval(Eval):
    return np.polynomial.polynomial.polyval(Eval["Time"],Eval["Coeff"])
Data=pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')

OccupData=Data.resample('60min',on='collecteddate').max().reset_index().dropna()
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60
DayList=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

PeriodList=["Class","Exam","Neither"]

Model={}

TSS=TimeSeriesSplit()#Creates 5 splits
X=OccupData[["Time","Day"]]
Y=OccupData["headcount"]

R2Train=[]
TrainError=[]

R2Val=[]
ValError=[]

BaseError=[]
BaseR2=[]
for trainIndex, valIndex in TSS.split(X):
    Model={}
    X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
    Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
    
    Baseline={}
    for hour in X_train["Time"].unique():
        Baseline[str(hour)]=np.mean(Y_train[X_train["Time"].isin([hour])])
        
    for day in DayList:
        PeriodData=X_train[X_train["Day"].isin([day])]
        YPeriodData=Y_train[X_train["Day"].isin([day])]
        if len(PeriodData)==0:
            continue
        p_fitted = Polynomial.fit(PeriodData["Time"], YPeriodData, deg=3)
        Model[str(day)]=p_fitted.convert().coef
    Base_val=X_val["Time"].astype(str).map(Baseline)
    
    TrainCoeff=(X_train["Day"]).map(Model)
    TrainEval = pd.concat([X_train["Time"], TrainCoeff.rename("Coeff")], axis=1)
    TrainPredict=TrainEval.apply(ModelEval, axis=1)
    
    ValCoeff=(X_val["Day"]).map(Model)
    ValEval = pd.concat([X_val["Time"], ValCoeff.rename("Coeff")], axis=1)
    ValPredict=ValEval.apply(ModelEval, axis=1)
    ValNoNan = pd.concat([Y_val.rename("True"), ValPredict.rename("Prediction")], axis=1).dropna()
    
    
    R2Train.append(r2_score(Y_train,TrainPredict))
    R2Val.append(r2_score(ValNoNan["True"],ValNoNan["Prediction"]))
    
    BaseR2.append(r2_score(Y_val,Base_val))
    BaseError.append(root_mean_squared_error(Y_val,Base_val))
    
    TrainError.append(root_mean_squared_error(Y_train,TrainPredict))
    ValError.append(root_mean_squared_error(ValNoNan["True"],ValNoNan["Prediction"]))
    
    
        
plt.figure()
plt.plot([1,2,3,4,5],TrainError,label="Training")
plt.plot([1,2,3,4,5],ValError,label="Validation")
plt.plot([1,2,3,4,5],BaseError,label="Baseline")
plt.xlabel("Fold")
plt.ylabel('Root Mean Square Error')
plt.title("RMSE Variation for Regression Model")
plt.xticks(range(1,6))
plt.legend()
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/RMSE Regression.png'
plt.savefig(loc)


plt.figure()
plt.plot([1,2,3,4,5],R2Train,label="Training")
plt.plot([1,2,3,4,5],R2Val,label="Validation")
plt.plot([1,2,3,4,5],BaseR2,label="Baseline")
plt.xlabel("Fold")
plt.ylabel('Coefficient Of Determination (r²)')
plt.title("r² Variation for Regression Model")
plt.xticks(range(1,6))
plt.legend()
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/r2 Regression.png'
plt.savefig(loc)
plt.show()
    