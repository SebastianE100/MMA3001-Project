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

def DuplicateDelete(array):
    """
Deletes duplicate data entries

Parameters
----------
array : array of floats
 the first collum contains the Unix Timecode of sample.

Returns
-------
return : array of floats
 the orginal array without duplicates.
"""
    SortData=np.sort(array, axis=0) #Sorts data to be in ascending order of Unix Timecode
    rowDelete=[] #Array to stores deplicate rows
    for i in range(1,len(SortData)):
        #Loop checks if deplicate data from the same sensor
        if SortData[i][0]==SortData[i-1][0] and SortData[i][2]==SortData[i-1][2]:
            rowDelete.append(i)
    #Deletes depliucate rows        
    FilData=np.delete(SortData, rowDelete, axis=0)
    return FilData

 #Loading Data
EnvData=pd.read_csv("Documents/Uni/2026/5EnvSensor_MayToDec2024_180kRows.csv")
JsonData=EnvData["jsondata"]
TempData=np.ones((len(JsonData),3))
CO2Data=np.ones((len(JsonData),3))

#Convert JSON format to numpy array
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
    
#Removing duplicates    
TempData=DuplicateDelete(TempData)
CO2Data=DuplicateDelete(CO2Data)

#Loading Data
OccupData=pd.read_csv("Documents/Uni/2026/5occupancySensor_MayToDec2024_9MRows.csv")
#Limit Scope Test
OccupData=OccupData[0:10000]

#Convert data into numpy array 
PeopleData=np.ones((len(OccupData),3),dtype="<U36")
for u in range(0,len(OccupData)):
    PeopleData[u,0]=OccupData.loc[u]["deviceid"]
    PeopleData[u,1]=OccupData.loc[u]["headcount"]
    PeopleData[u,2]=OccupData.loc[u]["collecteddate"]
       
#Saving Data
np.save("Documents/Uni/2026/MMA3001 Assignment/TempData.npy", TempData)
np.save("Documents/Uni/2026/MMA3001 Assignment/CO2Data.npy", CO2Data)
np.save("Documents/Uni/2026/MMA3001 Assignment/PeopleData.npy", PeopleData)

