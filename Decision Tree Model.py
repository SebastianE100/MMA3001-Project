# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:24 2026

@author: Octob
"""
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree


pd.set_option("display.precision", 4)

OccupData = pd.read_parquet('PeopleData.parquet').sort_values(by='collecteddate')
dayMap = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
OccupData["dayNum"]=OccupData["Day"].map(dayMap)

X = OccupData[["collecteddate"]]
Y = OccupData[["headcount"]]

X_train, X_val, Y_train, Y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=98,
)
TrainError=[]
ValError=[]
DepthTest=range(1,31)
for depth in DepthTest:
    occup_tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=1,
        random_state=98,
    )
    occup_tree.fit(X_train, Y_train)
    
    tree_summary = pd.DataFrame({
        "property": ["depth", "number of leaves", "training samples"],
        "value": [
            occup_tree.get_depth(),
            occup_tree.get_n_leaves(),
            len(X_train),
        ],
    })
    
    trainPredi = occup_tree.predict(X_train)
    valPredi = occup_tree.predict(X_val)
    TrainError.append(mean_squared_error(Y_train,trainPredi))
    ValError.append(mean_squared_error(Y_val,valPredi))

ErrorData=pd.DataFrame({
    'Training Error' : TrainError,
    'Validation Error' : ValError,
    },index=DepthTest)
ErrorData.plot(kind="bar",title="Decision Tree Model - Random Split")
    
