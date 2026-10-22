<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:30:24 2026

@author: Octob
"""

import numpy as np
from datetime import datetime as dt
#Tempdata=np.load("C:/Users/Octob/Documents/Uni/2026/MMA3001 Assignment/TempData.npy")
TestData=np.load("C:/Users/Octob/Documents/Uni/2026/MMA3001 Assignment/TestData.npy")


date=np.ones((len(TestData),1)).astype("str")
time=np.ones((len(TestData),1)).astype("str")
for i in range(0,len(TestData)):
    timestamp=dt.fromtimestamp(TestData[i][0])
    TestData[i][0]=TestData[i][0].astype('str')
    date[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    time[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[11:20]
    
TestData=np.column_stack((TestData, date))
TestData=np.column_stack((TestData, time))
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:30:24 2026

@author: Octob
"""

import numpy as np
from datetime import datetime as dt
#Tempdata=np.load("C:/Users/Octob/Documents/Uni/2026/MMA3001 Assignment/TempData.npy")
TestData=np.load("C:/Users/Octob/Documents/Uni/2026/MMA3001 Assignment/TestData.npy")


date=np.ones((len(TestData),1)).astype("str")
time=np.ones((len(TestData),1)).astype("str")
for i in range(0,len(TestData)):
    timestamp=dt.fromtimestamp(TestData[i][0])
    TestData[i][0]=TestData[i][0].astype('str')
    date[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    time[i]=timestamp.strftime('%Y-%m-%d %H:%M:%S')[11:20]
    
TestData=np.column_stack((TestData, date))
TestData=np.column_stack((TestData, time))
>>>>>>> 924485b77c091f678ebad864ae39601a8c374282
