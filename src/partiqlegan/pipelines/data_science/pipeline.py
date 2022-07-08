"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=log_git_repo,
                name="log_git_repo"
        ),
        node(
                func=calculate_n_fsps,
                inputs={
                        "torch_dataset_lca_and_leaves":"torch_dataset_lca_and_leaves",
                    },
                outputs={
                        "n_fsps":"n_fsps"
                },
                name="calculate_n_fsps"
        ),
        node(
                func=create_model,
                inputs={
                        "n_fsps":"n_fsps",
                        "n_momenta":"params:n_momenta",
                        "n_blocks":"params:n_blocks",
                        "dim_feedforward":"params:dim_feedforward",
                        "n_layers_mlp":"params:n_layers_mlp",
                        "n_additional_mlp_layers":"params:n_additional_mlp_layers",
                        "n_final_mlp_layers":"params:n_final_mlp_layers",
                        "dropout_rate":"params:dropout_rate",
                        "learning_rate":"params:learning_rate",
                        "learning_rate_decay":"params:learning_rate_decay",
                        "gamma":"params:gamma",
                    },
                outputs={
                        "nri_model":"nri_model"
                },
                name="create_model"
        ),
        node(
                func=create_instructor,
                inputs={
                        "torch_dataset_lca_and_leaves":"torch_dataset_lca_and_leaves",
                        "model":"nri_model",
                        "learning_rate":"params:learning_rate",
                        "learning_rate_decay":"params:learning_rate_decay",
                        "gamma":"params:gamma",
                        "batch_size":"params:batch_size",
                        "epochs":"params:epochs"
                    },
                outputs={
                        "instructor":"instructor"
                },
                name="create_instructor"
        ),
        node(
                func=train_qgnn,
                inputs={
                        "instructor":"instructor"
                    },
                outputs={
                        "model_qgnn":"trained_model"
                },
                name="train_qgnn"
        )
    ])
