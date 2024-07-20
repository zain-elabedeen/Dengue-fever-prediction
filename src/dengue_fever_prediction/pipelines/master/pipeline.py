from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_data,
    preprocess_data,
    engineer_data,
    train_model,
    prediction,
    submission,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_data,
                inputs={
                    "dengue_features_test": "dengue_features_test",
                    "dengue_features_train": "dengue_features_train",
                    "dengue_labels_train": "dengue_labels_train",
                },
                outputs="merged_data",
                name="merge_data_node",
            ),
            node(
                func=preprocess_data,
                inputs="merged_data",
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
            node(
                func=engineer_data,
                inputs="preprocessed_data",
                outputs="engineered_data",
                name="engineer_data_node",
            ),
            node(
                func=train_model,
                inputs=["engineered_data", "params:model_options"],
                outputs="model",
                name="train_model_node"
            ),
            node(
                func=prediction,
                inputs=["model", "engineered_data", "params:model_options"],
                outputs="prediction_data",
                name="prediction_node"
            ),
            node(
                func=submission,
                inputs="prediction_data",
                outputs="submission_data",
                name="submission_node"
            ),
        ]
    )
