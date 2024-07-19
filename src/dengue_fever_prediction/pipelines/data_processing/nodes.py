from typing import Dict, Tuple

import pandas as pd
# from pyspark.sql import Column
# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import regexp_replace
# from pyspark.sql.types import DoubleType
# from preprocessing_node import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def merge_data(dengue_features_test: pd.DataFrame, dengue_features_train: pd.DataFrame, dengue_labels_train: pd.DataFrame) -> pd.DataFrame:
    dengue_features_test.insert(0, 'type', 'test')
    dengue_features_train.insert(0, 'type', 'train')
    column_to_merge = dengue_labels_train[['total_cases']]
    train_dataset = pd.concat([dengue_features_train, column_to_merge], axis=1)
    merged_data = pd.concat([train_dataset, dengue_features_test], axis=0)
    #writing it out
    #dataset.to_csv('dataset.csv', index=False)
    #return {}
    return merged_data


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


def train_model(preprocessed_data: pd.DataFrame, parameters: Dict) -> RandomForestRegressor:
    training_data = preprocessed_data[preprocessed_data['type_train'] == 1]

    X = training_data[parameters["features"]]
    y = training_data["total_cases"]


    regressor = RandomForestRegressor()
    regressor.fit(X, y)
    return regressor

def prediction(model: pd.DataFrame,
               preprocerssed_data: pd.DataFrame,
               parameters: Dict) -> pd.DataFrame:
    prediction = model.predict(preprocerssed_data[parameters["features"]])
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
    test_mask = df.loc[:, "type_test"] == 1

    # Reverse one-hot encoding of "city" column: 
    df.loc[:, "city"] = df.loc[:, "city_sj"].replace(
        {
            1: "sj",
            0: "iq"
        }
    )

    # Columns for submission format:
    desired_columns = ["city", "year", "weekofyear", "predicted_total_cases"] 

    # output df
    out = df.loc[test_mask, desired_columns]

    # Return output df with correct total_cases name: 
    return out.rename(columns={'predicted_total_cases': 'total_cases'})



# Helper functions:

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

    return pd.get_dummies(data=df, columns=['city', 'type'], dtype=int)