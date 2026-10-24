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


#Needs to filter data

#Finds day of the week
OccupData['Day'] = OccupData['collecteddate'].dt.day_name()
ListofDates = OccupData['collecteddate'].dt.date.drop_duplicates().tolist()
OccupData=OccupData.resample('60min',on='collecteddate').max().reset_index().fillna(0)
print("ReSampled Len: "+str(len(OccupData)))
OccupData['Date'] = OccupData['collecteddate'].dt.date
OccupData = OccupData[OccupData['Date'].isin(ListofDates)]
print("Removed Len: "+str(len(OccupData)))
OccupData['Day'] = OccupData['collecteddate'].dt.day_name()


Sem1Y24=pd.to_datetime(['2024-02-26', '2024-05-27']).tz_localize("Australia/Melbourne")
Sem1ExamY24=pd.to_datetime(['2024-05-27', '2024-06-21']).tz_localize("Australia/Melbourne")

Sem2Y24=pd.to_datetime(['2024-07-22', '2024-10-21']).tz_localize("Australia/Melbourne")
Sem2ExamY24=pd.to_datetime(['2024-10-21', '2024-11-15']).tz_localize("Australia/Melbourne")

Sem1Y25=pd.to_datetime(['2025-03-03', '2025-06-02']).tz_localize("Australia/Melbourne")
Sem1ExamY25=pd.to_datetime(['2025-06-02', '2025-06-27']).tz_localize("Australia/Melbourne")

Sem2Y25=pd.to_datetime(['2025-07-28', '2025-10-27']).tz_localize("Australia/Melbourne")
Sem2ExamY25=pd.to_datetime(['2025-10-27', '2025-11-19']).tz_localize("Australia/Melbourne")

Sem1Y26=pd.to_datetime(['2026-03-02', '2026-06-01']).tz_localize("Australia/Melbourne")
Sem1ExamY26=pd.to_datetime(['2026-06-01', '2026-06-26']).tz_localize("Australia/Melbourne")

Sem2Y26=pd.to_datetime(['2026-07-27', '2026-10-26']).tz_localize("Australia/Melbourne")
Sem2ExamY26=pd.to_datetime(['2026-10-26', '2026-11-18']).tz_localize("Australia/Melbourne")

Class2024=OccupData['collecteddate'].between(Sem1Y24[0],Sem1Y24[1]) | OccupData['collecteddate'].between(Sem2Y24[0],Sem2Y24[1]) 
Exam2024=OccupData['collecteddate'].between(Sem1ExamY24[0],Sem1ExamY24[1]) | OccupData['collecteddate'].between(Sem2ExamY24[0],Sem2ExamY24[1]) 

Class2025=OccupData['collecteddate'].between(Sem1Y25[0],Sem1Y25[1]) | OccupData['collecteddate'].between(Sem2Y25[0],Sem2Y25[1]) 
Exam2025=OccupData['collecteddate'].between(Sem1ExamY25[0],Sem1ExamY25[1]) | OccupData['collecteddate'].between(Sem2ExamY25[0],Sem2ExamY25[1]) 

Class2026=OccupData['collecteddate'].between(Sem1Y26[0],Sem1Y26[1]) | OccupData['collecteddate'].between(Sem2Y26[0],Sem2Y26[1]) 
Exam2026=OccupData['collecteddate'].between(Sem1ExamY26[0],Sem1ExamY26[1]) | OccupData['collecteddate'].between(Sem2ExamY26[0],Sem2ExamY26[1]) 

OccupData['Class']=Class2024|Class2025|Class2026
OccupData['Exam']=Exam2024|Exam2025|Exam2026

#Adds the time of the day in hours
OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60

OccupData.to_parquet('PeopleData.parquet')

      


