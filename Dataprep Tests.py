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

#EnvData=pd.read_csv("Documents/Uni/2026/MMA3001 Assignment/5EnvSensor_MayToDec2024_180kRows.csv")
#JsonData=EnvData["jsondata"]
#TempData=np.ones((len(JsonData),2))
#CO2Data=np.ones((len(JsonData),2))
#i=0
#for time in JsonData:
#    Time0=pd.read_json(io.StringIO(time))
#    TempData[i,1]=Time0["values"][5][0]["value"]
#    TempData[i,0]=Time0["values"][5][0]["time"]
#    CO2Data[i,1]=Time0["values"][1][0]["value"]
#    CO2Data[i,0]=Time0["values"][1][0]["time"]
#Need ID as well
#    i+=1

#np.save("Documents/Uni/2026/MMA3001 Assignment/TempData.npy", TempData)
#np.save("Documents/Uni/2026/MMA3001 Assignment/CO2Data.npy", CO2Data)

OccupData=pd.read_csv("Documents/Uni/2026/MMA3001 Assignment/5occupancySensor_MayToDec2024_9MRows.csv")
OccupData=OccupData[0:1000]
PeopleData=np.ones((len(OccupData),3),dtype="<U36")

import time

# Start the timer

type=["<U1","<U10","<U25","<U50","<U75","<U100","<U125","<U150"]
# --- Put the code you want to test below ---
total = 100

for t in type:
    start_time = time.perf_counter()
    for i in range(total):
        PeopleData=np.ones((len(OccupData),3),dtype=t)
        for u in range(0,len(OccupData)):
            PeopleData[u,0]=OccupData.loc[u]["deviceid"]
            PeopleData[u,1]=OccupData.loc[u]["headcount"]
            PeopleData[u,2]=OccupData.loc[u]["collecteddate"]
    # -------------------------------------------
    
    # Stop the timer
    end_time = time.perf_counter()

    # Calculate and print execution time
    execution_time = (end_time - start_time)/total
    print(f"Type: {t} Average time: {execution_time:.6f} seconds")
