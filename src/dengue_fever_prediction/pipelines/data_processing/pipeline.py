from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_data,
    preprocess_data,
    train_model,
    prediction,
    submission,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_data,
                inputs=["dengue_features_train", "dengue_features_test", "dengue_labels_train"],
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
                func=train_model,
                inputs=["preprocessed_data", "params:model_options"],
                outputs="model",
                name="train_model_node"
            ),
            node(
                func=prediction,
                inputs=["model", "preprocessed_data", "params:model_options"],
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
