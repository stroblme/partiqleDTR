"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes_bak import train_qgnn


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=train_qgnn,
                inputs=["model_parameters", "all_leaves_shuffled", "all_lca_shuffled"],
                outputs={
                    "model":"model"
                },
                name="train_qgnn"
        )
    ])
