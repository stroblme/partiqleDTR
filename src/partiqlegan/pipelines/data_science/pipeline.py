"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_qgnn


def create_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=train_qgnn,
                inputs=["torch_dataset_lca_and_leaves", "params:n_hid", "params:n_momenta", "params:dropout_rate", "params:learning_rate", "params:learning_rate_decay", "params:gamma", "params:batch_size", "params:epochs"],
                outputs={
                    "model":"model"
                },
                name="train_qgnn"
        )
    ])
