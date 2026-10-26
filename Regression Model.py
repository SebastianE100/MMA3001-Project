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


#Define function used to evaluate the model
def ModelEval(Eval):
    return np.polynomial.polynomial.polyval(Eval["Time"],Eval["Coeff"])

#Import data
Data=pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')
#Resample
OccupData=Data.resample('60min',on='collecteddate').max().reset_index().dropna()

#Define time in hours
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60

#Define days of the week
DayList=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


#Creates 5 splits
TSS=TimeSeriesSplit()

#Define data model has acess to
X=OccupData[["Time","Day"]]
Y=OccupData["headcount"]

#Define lists to store values
R2Train=[]
TrainError=[]

R2Val=[]
ValError=[]

BaseError=[]
BaseR2=[]

#Creating folds
for trainIndex, valIndex in TSS.split(X):
    
    #Define directionary that will contain model
    Model={}
    
    #Define valdiation data and training data
    X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
    Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
        
    #Define time mean baseline
    Baseline={}
    for hour in X_train["Time"].unique():
        Baseline[str(hour)]=np.mean(Y_train[X_train["Time"].isin([hour])])
    
    #Create a quadratic for each day of the week
    for day in DayList:
        PeriodData=X_train[X_train["Day"].isin([day])]
        YPeriodData=Y_train[X_train["Day"].isin([day])]
        if len(PeriodData)==0:#If there is no data for a day skip that day
            continue
        p_fitted = Polynomial.fit(PeriodData["Time"], YPeriodData, deg=2)
        #Add coefficients for the day to the directionary
        Model[str(day)]=p_fitted.convert().coef
    #Predict baseline model
    Base_val=X_val["Time"].astype(str).map(Baseline)
    
    #Get collums of coefficicents for given inputs
    TrainCoeff=(X_train["Day"]).map(Model)
    #Merge collums of times and model to predict time
    TrainEval = pd.concat([X_train["Time"], TrainCoeff.rename("Coeff")], axis=1)
    #Evalute the model
    TrainPredict=TrainEval.apply(ModelEval, axis=1)
    
    #Get collums of coefficicents for given inputs
    ValCoeff=(X_val["Day"]).map(Model)
    #Merge collums of times and model to predict time
    ValEval = pd.concat([X_val["Time"], ValCoeff.rename("Coeff")], axis=1)
    #Evalute the model
    ValPredict=ValEval.apply(ModelEval, axis=1)
    ValNoNan = pd.concat([Y_val.rename("True"), ValPredict.rename("Prediction")], axis=1).dropna()
    
    #Save perfomance
    R2Train.append(r2_score(Y_train,TrainPredict))
    R2Val.append(r2_score(ValNoNan["True"],ValNoNan["Prediction"]))
    
    BaseR2.append(r2_score(Y_val,Base_val))
    BaseError.append(root_mean_squared_error(Y_val,Base_val))
    
    TrainError.append(root_mean_squared_error(Y_train,TrainPredict))
    ValError.append(root_mean_squared_error(ValNoNan["True"],ValNoNan["Prediction"]))
    
    
#Create and save figures   
plt.figure()
plt.plot([1,2,3,4,5],TrainError,label="Training")
plt.plot([1,2,3,4,5],ValError,label="Validation")
plt.plot([1,2,3,4,5],BaseError,label="Baseline")
plt.xlabel("Fold")
plt.ylabel('Root Mean Square Error')
plt.title("RMSE Variation for Polynomial Model")
plt.xticks(range(1,6))
plt.yticks([2,2.5,3,3.5,4,4.5,5])
plt.legend()
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/RMSE Regression.png'
plt.savefig(loc)


plt.figure()
plt.plot([1,2,3,4,5],R2Train,label="Training")
plt.plot([1,2,3,4,5],R2Val,label="Validation")
plt.plot([1,2,3,4,5],BaseR2,label="Baseline")
plt.xlabel("Fold")
plt.ylabel('Coefficient Of Determination (r²)')
plt.title("r² Variation for Polynomial Model")
plt.xticks(range(1,6))
plt.legend()
plt.yticks([0.1,0.2,0.3,0.4,0.5])
loc='C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/r2 Regression.png'
plt.savefig(loc)
plt.show()
    