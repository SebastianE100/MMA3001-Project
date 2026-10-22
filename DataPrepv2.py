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

#Loading Data
OccupData=pd.read_csv("Documents/Uni/2026/5occupancySensor_MayToDec2024_9MRows.csv",nrows=100,usecols=["headcount","collecteddate"])
print(len(OccupData))
OccupData.drop_duplicates(subset=['collecteddate'], inplace=True)
print(len(OccupData))
print("Any NaN:"+str(OccupData["headcount"].isna().any()))
print("Max Headcount: "+str(OccupData["headcount"].max()))
print("Min Headcount: "+str(OccupData["headcount"].min()))
OccupData[['Date', 'Time']] = OccupData['collecteddate'].str.split(' ', expand=True)

OccupData['Date'] = pd.to_datetime(OccupData['Date'])
OccupData['Day'] = OccupData['Date'].dt.day_name()

OccupData.to_parquet('OccupData.parquet')

      


