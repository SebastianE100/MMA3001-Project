# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:41:51 2026

@author: Octob
"""
import pandas as pd

#Lets create a model that predicts the rate which C02 and temp increase due to people in the room
#Extra scope is to make a model to predict occupacy in a week
#Thus building can plan when is best to bring air in the save heating it up.

#Loading Data
OccupData=pd.read_csv("Documents/Uni/2026/5occupancySensor_MayToDec2024_9MRows.csv",usecols=["headcount","collecteddate"])
print("Original Length: "+str(len(OccupData)))

#Remove duplicates
OccupData.drop_duplicates(subset=['collecteddate'], inplace=True)
print("Filtered Length: "+str(len(OccupData)))
#Checks if any invalid data
print("Any NaN:"+str(OccupData["headcount"].isna().any()))

#Checks if range of data is reasonable
print("Max Headcount: "+str(OccupData["headcount"].max()))
print("Min Headcount: "+str(OccupData["headcount"].min()))

#Converts string to time format
OccupData['collecteddate'] = pd.to_datetime(OccupData['collecteddate'],format="ISO8601", utc=True)
#Converts time back into Melbourne time
OccupData['collecteddate'] =pd.to_datetime(OccupData['collecteddate'] , utc=True).dt.tz_convert("Australia/Melbourne")
CleanData=OccupData.resample('15min',on='collecteddate').max().reset_index().fillna(0)


#Finds day of the week
CleanData['Day'] = CleanData['collecteddate'].dt.day_name()


CleanData.to_parquet('PeopleData.parquet')

      


