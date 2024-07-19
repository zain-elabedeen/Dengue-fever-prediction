# Imports: 
# from __future__ import print_function
# from __future__ import division

import pandas as pd
import numpy as np

# from matplotlib import pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import max_error, mean_absolute_error, r2_score

# import statsmodels.api as sm

# from typing import Dict, Tuple

#def preprocessing(df: pd.DataFrame, parameters: Dict) -> Tuple:

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for doing the preprocessing steps: 
        1) Handling the date column
        2) Imputation of missing values
        3) One-Hot encoding of the city parameter
    
    Input: Big DF beast from Arpad

    Output: Big DF that has only numerical values.
                - TODO: Save DF within kedro folders..?
    """

    # 1) Handling the Date:
    df = changeDate(df) # TODO: best way to pass a dataframe?

    # 2) Imputation: 
    df = impute(df)

    # 3) One-hot encoding: 
    df = encode(df)

    return df # TODO: How to return something saveable?


def changeDate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For now, just remove it. (TODO)
    """
    
    df.drop('week_start_date', axis=1, inplace=True)

    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Following the benchmark example: Filling in whatever
    """
    # TODO: Gotta think about the \hat{y} predicted values here? 
    df.fillna(method='ffill', inplace=True)

    return df 


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Using one-hot encoding for the city column. 

    New columns to df: "sj", "iq" = 0 or 1
    """

    return pd.get_dummies(data=df, columns=['city'])


def ps(df):
    print("df.shape:", df.shape)
