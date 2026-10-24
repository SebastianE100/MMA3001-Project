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
OccupData["Condition"]=np.where(OccupData["Class"]==1,"Class",np.where(OccupData["Exam"]==1,"Exam","Neither"))

PeriodList=["Class","Exam","Neither"]

Model={}

TSS=TimeSeriesSplit()#Creates 5 splits
X=OccupData[["Time","Day","Condition"]]
Y=OccupData["headcount"]

R2Train=[]
R2Val=[]
for trainIndex, valIndex in TSS.split(X):
    Model={}
    X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
    Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
    for day in DayList:
        DayData=X_train#[X_train["Day"].isin([day])]
        YDayData=Y_train#[X_train["Day"].isin([day])]
        for Condition in PeriodList:
            PeriodData=DayData#[DayData['Condition'].isin([Condition])]
            YPeriodData=YDayData#[DayData['Condition'].isin([Condition])]
            if len(PeriodData)==0:
                continue
            p_fitted = Polynomial.fit(PeriodData["Time"], YPeriodData, deg=3)
            Model[str(day)+str(Condition)]=p_fitted.convert().coef
    
    TrainCoeff=(X_train["Day"]+X_train["Condition"]).map(Model)
    TrainEval = pd.concat([X_train["Time"], TrainCoeff.rename("Coeff")], axis=1)
    TrainPredict=TrainEval.apply(ModelEval, axis=1)
    
    ValCoeff=(X_val["Day"]+X_val["Condition"]).map(Model)
    ValEval = pd.concat([X_val["Time"], ValCoeff.rename("Coeff")], axis=1)
    ValPredict=ValEval.apply(ModelEval, axis=1)
    ValNoNan = pd.concat([Y_val.rename("True"), ValPredict.rename("Prediction")], axis=1).dropna()
    R2Train.append(r2_score(Y_train,TrainPredict))
    R2Val.append(r2_score(ValNoNan["True"],ValNoNan["Prediction"]))
    
plt.figure(1)
plt.plot([1,2,3,4,5],R2Val,label="Val")
plt.plot([1,2,3,4,5],R2Train,label="Train")
plt.legend()
    