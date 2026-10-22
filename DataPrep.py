# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:41:51 2026

@author: Octob
"""
import pandas as pd
import numpy as np
from datetime import datetime as dt
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

date=np.ones((len(TempData),1)).astype("str")
time=np.ones((len(TempData),1)).astype("str")
for i in range(0,len(TempData)):
    timestamp=dt.fromtimestamp(TempData[i][0])
    date[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    time[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[11:20]
    
TempData=np.column_stack((TempData, date))
TempData=np.column_stack((TempData, time))


date=np.ones((len(CO2Data),1)).astype("str")
time=np.ones((len(CO2Data),1)).astype("str")
for i in range(0,len(CO2Data)):
    timestamp=dt.fromtimestamp(CO2Data[i][0])
    date[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    time[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[11:20]
    
CO2Data=np.column_stack((CO2Data, date))
CO2Data=np.column_stack((CO2Data, time))


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

#Loading Weather data from https://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_nccObsCode=122&p_display_type=dailyDataFile&p_startYear=&p_c=&p_stn_num=86077    
WeatherData=np.array(pd.read_csv("Documents/Uni/2026/IDCJAC0010_86077_1800_Data.csv"))
years=[2023,2024,2025,2026] #Years of interest

#Deleting years that are not in range of interest
delList=[]
for k in range(0,len(WeatherData)):
    if WeatherData[k][2] not in years:
        delList.append(k)
WeatherData=np.delete(WeatherData, delList, axis=0)

#Putting Date into string format and storing max temp of that day
date=np.ones((len(WeatherData),1)).astype("str")
TValue=np.ones((len(WeatherData),1)).astype("float")
for j in range(0,len(WeatherData)):
    year=str(WeatherData[j][2])
    if len(str(WeatherData[j][3]))==1:
        month="0"+str(WeatherData[j][3])
    else:
        month=str(WeatherData[j][3])
    if len(str(WeatherData[j][4]))==1:
        day="0"+str(WeatherData[j][4])
    else:
        day=str(WeatherData[j][4])

    date[j]=year+"-"+month+"-"+day
    TValue[j]=WeatherData[j][5]

WeatherExport=np.column_stack((date,TValue))
       
#Saving Data
np.save("Documents/Uni/2026/MMA3001 Assignment/TempData.npy", TempData)
np.save("Documents/Uni/2026/MMA3001 Assignment/CO2Data.npy", CO2Data)
np.save("Documents/Uni/2026/MMA3001 Assignment/PeopleData.npy", PeopleData)
np.save("Documents/Uni/2026/MMA3001 Assignment/WeatherExport.npy", WeatherExport)

