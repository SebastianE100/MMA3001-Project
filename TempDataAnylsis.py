# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:30:24 2026

@author: Octob
"""
import pandas as pd
import numpy as np
from datetime import datetime as dt
Tempdata=np.load("C:/Users/Octob/Documents/Uni/2026/MMA3001 Assignment/TempData.npy")
Data=Tempdata[0:100]


date=np.ones((len(Data),1)).astype("str")
time=np.ones((len(Data),1)).astype("str")
for i in range(0,len(Data)):
    timestamp=dt.fromtimestamp(Data[i][0])
    date[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    time[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[11:20]
    
Data=np.column_stack((Data, date))
Data=np.column_stack((Data, time))


WeatherData=np.array(pd.read_csv("Documents/Uni/2026/IDCJAC0010_86077_1800_Data.csv"))
years=[2023,2024,2025,2026]
delList=[]
for k in range(0,len(WeatherData)):
    if WeatherData[k][2] not in years:
        delList.append(k)
WeatherData=np.delete(WeatherData, delList, axis=0)


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

    

