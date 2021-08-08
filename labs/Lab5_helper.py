import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import LeavePOut

def exercise_1(data):
    df = None
    return df

def exercise_2(data):
    df = None
    return df

def exercise_3(df):
    df_scaled = df.copy()
    return df_scaled

def exercise_4(df_scaled):
    X = df_scaled.drop('Types.Lung.Total',axis=1)
    y = df_scaled['Types.Lung.Total']
    
    median_r2_knn = None
    median_r2_linear = None
    
    # Your code here
    # Try n_neighbors from 1 to 4
    # Use scoring r2
    # Set the outermost cross-validation folds to LeavePOut(p=2) and the inner most folds to 2
    
    return median_r2_knn,median_r2_linear

def exercise_5(top_names,sampledata13,resultsdata13,pest_codes):
    df = None
    return df

def exercise_6(df_train):
    X = df_train.drop('Types.Lung.Total',axis=1)
    y = df_train['Types.Lung.Total']
    
    median_r2_knn = None
    median_r2_linear = None
    
    # Your code here
    # Try n_neighbors from 1 to 4
    # Use scoring r2
    # Set the outermost cross-validation folds to LeavePOut(p=2) and the inner most folds to 2
    
    return median_r2_knn,median_r2_linear