# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:24 2026

@author: Octob
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor 
timelist=[]
TSS=TimeSeriesSplit()#Creates 5 splits
for i in range(0,1):
        #Import Data
        OccupData = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')
        OCCUPIED_THRESHOLD=2
        
        #Set sample interval to be 15 mins
        interval=15
        OccupData=OccupData.resample('60min',on='collecteddate').max().reset_index().dropna()
        
        #Adds day name as a collumn
        OccupData['Day'] = OccupData['collecteddate'].dt.day_name()
        
        #Give each day of the week a number
        dayMap = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        OccupData["dayNum"]=OccupData["Day"].map(dayMap)
        
        #Adds the time of the day in hours
        OccupData["Time"]=OccupData["collecteddate"].dt.hour+OccupData["collecteddate"].dt.minute/60
        
        #Adds 1 week lag into dataet
        DeltaIntervals=int((7*24*60/interval))
        OccupData["Lag"]=OccupData["headcount"].shift(int(DeltaIntervals))
        OccupData=OccupData.dropna(subset=['Lag'])#removes first week of data
        
        Test=[1,2,3,4,5,6]#Selects depths to be tested
        
        X = OccupData[["dayNum","Time",'Class',"Exam"]]
        Y = OccupData[["headcount"]]
        Base=OccupData[["Lag"]]
        
        #SplitData in 5 groups
        TrainError=[]
        ValError=[]
        for trainIndex, valIndex in TSS.split(X):
            
            X_train, X_val, = X.iloc[trainIndex] , X.iloc[valIndex]
            Y_train, Y_val = Y.iloc[trainIndex] , Y.iloc[valIndex]
            Base_train, Base_val = Base.iloc[trainIndex] , Base.iloc[valIndex]
            
            #Test for different depths
            Decision_tree = DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=3,
                random_state=98,
            )
            w=np.where(Y_train["headcount"] == 0, 5, 1)
            Decision_tree.fit(X_train, Y_train,sample_weight=w)
            
            timeEnd=time.perf_counter()
            
            timeStrt=time.perf_counter()
            TrainPredic = Decision_tree.predict(X_train)
            ValPredic = Decision_tree.predict(X_val)
            timeEnd=time.perf_counter()
            
            TrainError.append(root_mean_squared_error(Y_train,TrainPredic))
            ValError.append(root_mean_squared_error(Y_val,ValPredic))
            
        timelist.append(timeEnd-timeStrt)
        
    

plt.figure(1)
plt.plot([1,2,3,4,5],TrainError,label="Training")
plt.plot([1,2,3,4,5],ValError,label="Validation")
plt.xlabel("Fold")
plt.ylabel('RMSE')
plt.title("RMSE Variation for Decision Tree Model")
plt.xticks(range(1,6))
plt.legend()

'''
def Decision_Tree_Model(dayNum, time, class_flag, exam_flag):
    """Predict hourly building headcount from calendar inputs.

    Uses a fitted decision tree regressor to forecast headcount. All inputs
    are known in advance (day, time and teaching or exam period), so the
    function supports the one-week forecast horizon. No occupancy readings
    from the forecast period are needed.

    Args:
        dayNum (int or array-like of int): Day of the week, 0 (Monday) to
            6 (Sunday).
        time (float or array-like of float): Time of day in hours, from 0 up
            to but not including 24 (for example 14.25 is 2:15 pm).
        class_flag (bool or array-like of bool): True if a teaching period
            is running.
        exam_flag (bool or array-like of bool): True if an exam period is
            running.

    Returns:
        pandas.DataFrame: One row per input row, with columns:
            - "headcount_pred" (float): predicted headcount in people,
              0 or greater.
            - "occupied" (bool): True if the prediction is greater than
              OCCUPIED_THRESHOLD (2 people).

    Raises:
        ValueError: If the inputs differ in length, if dayNum is not an
            integer from 0 to 6, if time is outside [0, 24), or if any value
            is missing (NaN).

    Notes:
        A decision tree always returns a value, even for combinations it
        never saw in training, so predictions for such combinations (for
        example an exam period earlier than any in the training data) are
        unreliable. A tree predicts the mean of the training samples in a
        leaf, so the output takes only as many distinct values as the tree
        has leaves.

    Example:
        >>> Decision_Tree_Model(tree, dayNum=0, time=14.0,
        ...                     class_flag=True, exam_flag=False)
           headcount_pred  occupied
        0             9.2      True
    """
    tree=Decision_tree
    inputs = pd.DataFrame({
        "dayNum": np.atleast_1d(dayNum),
        "Time": np.atleast_1d(time),
        "Class": np.atleast_1d(class_flag),
        "Exam": np.atleast_1d(exam_flag),
    })

    if inputs.isna().any().any():
        raise ValueError("Inputs must not contain missing values.")
    if not inputs["dayNum"].between(0, 6).all() or (inputs["dayNum"] % 1 != 0).any():
        raise ValueError("dayNum must be an integer from 0 (Monday) to 6 (Sunday).")
    if not ((inputs["Time"] >= 0) & (inputs["Time"] < 24)).all():
        raise ValueError("time must be in hours from 0 up to, but not including, 24.")

    inputs = inputs.astype({"dayNum": int, "Time": float, "Class": bool, "Exam": bool})

    pred = np.clip(tree.predict(inputs[["dayNum", "Time", "Class", "Exam"]]), 0, None)
    return pred


'''



