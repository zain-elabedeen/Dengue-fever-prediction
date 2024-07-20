from typing import Dict, Tuple

import pandas as pd
import numpy as np
# from pyspark.sql import Column
# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import regexp_replace
# from pyspark.sql.types import DoubleType
# from preprocessing_node import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def merge_data(dengue_features_test: pd.DataFrame, dengue_features_train: pd.DataFrame, dengue_labels_train: pd.DataFrame) -> pd.DataFrame:
    dengue_features_test.loc[:, "type"] = "test"
    dengue_features_train.loc[:, "type"] = "train"

    x_total = pd.concat([dengue_features_test, dengue_features_train], axis=0)

    return pd.merge(
        left=x_total,
        right=dengue_labels_train,
        on=["city", "year", "weekofyear"],
        how="left",
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for doing the preprocessing steps:
        1) Handling the date column
        2) Imputation of missing values
        3) One-Hot encoding of the city parameter

    Input: Big DF beast from Arpad

    Output: Big DF that has only numerical values.
    """

    # 1) Handling the Date:
    df = changeDate(df) # TODO: best way to pass a dataframe?

    # 2) Imputation:
    df = impute(df)

    # 3) One-hot encoding:
    df = encode(df)

    return df


def engineer_data(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: 
        - Cyclical encoding of weekofyear column
        - Moving averages for (...) columns
    
    Args:
        preprocessed_data (pd.DataFrame): 

    Returns:
        pd.DataFrame: new df "engineered data:
    """
    engineered_data = encode_weeks(preprocessed_data)
    engineered_data = calculate_moving_averages(engineered_data)
    
    return engineered_data

def train_model(preprocessed_data: pd.DataFrame, parameters: Dict) -> RandomForestRegressor:
    training_data = preprocessed_data[preprocessed_data['type'] == "train"]

    X = training_data[parameters["features"]]
    y = training_data["total_cases"]


    regressor = RandomForestRegressor()

    regressor.fit(X, y)
    return regressor

def prediction(model: pd.DataFrame,
               preprocerssed_data: pd.DataFrame,
               parameters: Dict) -> pd.DataFrame:
    prediction = model.predict(preprocerssed_data[parameters["features"]])
    prediction = np.round(prediction).astype(int)
    predicton_data = pd.Series(prediction, name='predicted_total_cases')

    prediction_results = pd.concat([preprocerssed_data, predicton_data], axis=1)
    return prediction_results

def submission(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame in submission format.

    Args:
        df (pd.DataFrame): input dataframe containing predicted case numbers

    Returns:
        pd.DataFrame: output DF
    """

    # Select only the test rows:
    test_mask = df.loc[:, "type"] == "test"

    # Reverse one-hot encoding of "city" column: 
    df.loc[:, "city"] = df.loc[:, "city_sj"].replace(
        {
            1: "sj",
            0: "iq"
        }
    )

    # Round the 'year' and 'weekofyear' columns to the nearest integer
    df['year'] = df['year'].round().astype(int)
    df['weekofyear'] = df['weekofyear'].round().astype(int)

    # Columns for submission format:
    desired_columns = ["city", "year", "weekofyear", "predicted_total_cases"] 

    # output df
    out = df.loc[test_mask, desired_columns]

    # Return output df with correct total_cases name: 
    return out.rename(columns={'predicted_total_cases': 'total_cases'})

# Helper functions:

def calculate_moving_averages(df: pd.DataFrame, window=3) -> pd.DataFrame:
    columns_to_average = [
        'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
        'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
        'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
        'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
        'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c',
        'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm'
    ]
    
    # Create a new DataFrame to store the moving averages
    moving_averages_df = df.copy()
    
    # Calculate the moving averages for each column and add to the new DataFrame
    for column in columns_to_average:
        # Calculate moving averages
        moving_averages_df[f'moving_average_{column}'] = df[column].rolling(window=window).mean()
        
        # Ensure the first two rows have the same values as the original columns
        moving_averages_df[f'moving_average_{column}'].iloc[0] = df[column].iloc[0]
        moving_averages_df[f'moving_average_{column}'].iloc[1] = df[column].iloc[1]
    
    return moving_averages_df

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

    return pd.get_dummies(data=df, columns=['city'], dtype=int)

# From TimeSeries class, NB 6:
def week_sin(t, k=1, m=52):
    return np.sin(2*t*np.pi*k/m)

def week_cos(t, k=1, m=52):
    return np.cos(2*t*np.pi*k/m)

def encode_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding of  weekofyear 

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: replace weekofyear column with two columns for sine and cosine
    """

    # add one column for the sine: 
    df['woy_sin'] = week_sin(df['weekofyear'])

    # and another for the cosine:
    df['woy_cos'] = week_cos(df['weekofyear'])

    # drop original weekofyear from df:
    # df.drop('weekofyear', axis=1, inplace=True)

    return df