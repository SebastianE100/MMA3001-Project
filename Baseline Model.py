# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:32:32 2026

@author: Octob
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

#Create first model
OccupData = pd.read_parquet('PeopleData.parquet')

#Creating Baseline model prediction is same as 1 week ago
BaseModel=pd.DataFrame()
models=[]
error=[]
for DeltaT in [1,6,24,24*7]:
    DeltaIntervals=int((DeltaT*60/15)) #How many 15 mins intervals ago
    
    Predic=OccupData["headcount"].shift(DeltaIntervals).rename("Predict")#Prediction
    Model=pd.concat([OccupData["headcount"], Predic], axis=1)
    Model=Model.dropna(subset=['Predict'])
    RMSE=root_mean_squared_error(Model["headcount"],Model['Predict'])#Error
    print(str(DeltaT)+" Day Error: "+str(RMSE))
    models.append(str(DeltaT))
    error.append(RMSE)
    
plt.figure(1)
plt.bar(models, error)
plt.title("Baseline Model Comparison")
plt.xlabel("Hours Delay")
plt.ylabel("RMSE Error")