from typing import Dict, Tuple

import pandas as pd
# from pyspark.sql import Column
# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import regexp_replace
# from pyspark.sql.types import DoubleType
#from preprocessing_node import preprocessing


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


def fit_model():
    return {}

def prediction():
    return {}

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
    desired_columns = ["city", "year", "weekofyear", "total_cases"] 

    # Return output df: 
    return df.loc[test_mask, desired_columns]



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