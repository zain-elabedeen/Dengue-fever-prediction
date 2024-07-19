from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_data,
    preprocess_data,
    fit_model,
    prediction,
    submittion,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_data,
                inputs="dengue_labels_train, dengue_features_test, dengue_label_train",
                outputs=["merged_data"],
                name="merge_data_node",
            ),
            node(
                func=preprocess_data,
                inputs="merged_data",
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
            node(
                func=fit_model,
                inputs=["preprocessed_data"],
                outputs="fitted_model_data",
                name="fit_model_node"
            ),
            node(
                func=prediction,
                inputs=["fitted_model_data"],
                outputs="prediction_results",
                name="prediction_node"
            ),
            node(
                func=submittion,
                inputs=[],
                outputs="",
                name="submittion_node"
            ),
        ]
    )
