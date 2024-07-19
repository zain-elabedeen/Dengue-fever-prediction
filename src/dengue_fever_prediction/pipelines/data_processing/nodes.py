from typing import Dict, Tuple

import pandas as pd
# from pyspark.sql import Column
# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import regexp_replace
# from pyspark.sql.types import DoubleType
from preprocessing_node import preprocessing


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

def preprocess_data(df: pd.DataFrame):
    return preprocessing(df)

def fit_model():
    return {}

def prediction():
    return {}

def submittion():
    return {}