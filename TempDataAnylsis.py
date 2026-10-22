# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:30:24 2026

@author: Octob
"""

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





