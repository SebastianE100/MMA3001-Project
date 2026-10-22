# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:41:51 2026

@author: Octob
"""
import pandas as pd
import numpy as np
import io

#Lets create a model that predicts the rate which C02 and temp increase due to people in the room
#Extra scope is to make a model to predict occupacy in a week
#Thus building can plan when is best to bring air in the save heating it up.

EnvData=pd.read_csv("Documents/Uni/2026/MMA3001 Assignment/5EnvSensor_MayToDec2024_180kRows.csv")
JsonData=EnvData["jsondata"]
TempData=np.ones((len(JsonData),3))
CO2Data=np.ones((len(JsonData),3))
i=0
for time in JsonData:
    Time0=pd.read_json(io.StringIO(time))
    TempData[i,1]=Time0["values"][5][0]["value"]
    TempData[i,0]=Time0["values"][5][0]["time"]
    TempData[i,2]=EnvData["sensorid"][i]
    CO2Data[i,1]=Time0["values"][1][0]["value"]
    CO2Data[i,0]=Time0["values"][1][0]["time"]
    CO2Data[i,2]=EnvData["sensorid"][i]
    i+=1

OccupData=pd.read_csv("Documents/Uni/2026/MMA3001 Assignment/5occupancySensor_MayToDec2024_9MRows.csv")
OccupData=OccupData[0:10000]
PeopleData=np.ones((len(OccupData),3),dtype="<U36")
for u in range(0,len(OccupData)):
    PeopleData[u,0]=OccupData.loc[u]["deviceid"]
    PeopleData[u,1]=OccupData.loc[u]["headcount"]
    PeopleData[u,2]=OccupData.loc[u]["collecteddate"]
       
np.save("Documents/Uni/2026/MMA3001 Assignment/TempData.npy", TempData)
np.save("Documents/Uni/2026/MMA3001 Assignment/CO2Data.npy", CO2Data)
np.save("Documents/Uni/2026/MMA3001 Assignment/PeopleData.npy", PeopleData)
        
