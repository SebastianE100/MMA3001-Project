# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:32:32 2026

@author: Octob
"""

import pandas as pd
import matplotlib.pyplot as plt

#Create first model
OccupData = pd.read_parquet('PeopleData.parquet')

#Creating Baseline model prediction is same as 1 week ago

MaxDeltaT=11 #preiction will be the same value as 7 days ago

models=[]
error=[]
for DeltaT in range(1,MaxDeltaT):
    DeltaIntervals=int((DeltaT*24*60/15)) #How many 15 mins intervals ago
    
    Predic=OccupData["headcount"].shift(DeltaIntervals)#Prediction
    RMSE=((OccupData["headcount"]-Predic)**2).sum()/len(Predic)#Error
    print(str(DeltaT)+" Day Error: "+str(RMSE))
    models.append(str(DeltaT))
    error.append(RMSE)
    
plt.figure(1)
plt.bar(models, error)
plt.title("Baseline model Comparison")
plt.xlabel("Days")
plt.ylabel("RMS Error")