"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_metadata_nodes(**kwargs) -> Pipeline:
    return [
        node(
            func=log_git_repo,
            inputs={"git_hash_identifier": "params:git_hash_identifier"},
            outputs={},
            name="log_git_repo",
        ),
        node(
            func=calculate_n_classes,
            inputs={
                "dataset_lca_and_leaves": "dataset_lca_and_leaves",
            },
            outputs={"n_classes": "n_classes"},
            name="calculate_n_classes",
        ),
        node(
            func=calculate_n_fsps,
            inputs={
                "dataset_lca_and_leaves": "dataset_lca_and_leaves",
            },
            outputs={"n_fsps": "n_fsps"},
            name="calculate_n_fsps",
            tags="split_run",
        ),
    ]


def create_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            *create_metadata_nodes(**kwargs),
            node(
                func=create_model,
                inputs={
                    "n_classes": "n_classes",
                    "n_momenta": "params:n_momenta",
                    "model_sel": "params:model_sel",
                    "n_blocks": "params:n_blocks",
                    "dim_feedforward": "params:dim_feedforward",
                    "n_layers_mlp": "params:n_layers_mlp",
                    "n_additional_mlp_layers": "params:n_additional_mlp_layers",
                    "n_final_mlp_layers": "params:n_final_mlp_layers",
                    "dropout_rate": "params:dropout_rate",
                    "factor": "params:factor",
                    "tokenize": "params:tokenize",
                    "embedding_dims": "params:embedding_dims",
                    "batchnorm": "params:batchnorm",
                    "symmetrize": "params:symmetrize",
                    "data_reupload": "params:data_reupload",
                    "add_rot_gates": "params:add_rot_gates",
                    "n_layers_vqc": "params:n_layers_vqc",
                    "n_fsps": "n_fsps",
                    "device": "params:device",
                },
                outputs={"nri_model": "nri_model"},
                name="create_model",
            ),
            node(
                func=create_instructor,
                inputs={
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "model": "nri_model",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "detectAnomaly": "params:detect_anomaly",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train,
                inputs={"instructor": "instructor"},
                outputs={
                    "trained_model": "trained_quantum_model",
                    "checkpoint": "checkpoint",
                    "gradients": "gradients",
                },
                name="train_qgnn",
            ),
        ]
    )


def create_training_optuna_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            *create_metadata_nodes(**kwargs),
            node(
                func=create_hyperparam_optimizer,
                inputs={
                    "n_classes": "n_classes",
                    "n_momenta": "params:n_momenta",
                    "model_sel": "params:model_sel",
                    "n_blocks_range": "params:n_blocks_range",
                    "dim_feedforward_range": "params:dim_feedforward_range",
                    "n_layers_mlp_range": "params:n_layers_mlp_range",
                    "n_additional_mlp_layers_range": "params:n_additional_mlp_layers_range",
                    "n_final_mlp_layers_range": "params:n_final_mlp_layers_range",
                    "dropout_rate_range": "params:dropout_rate_range",
                    "factor": "params:factor",
                    "tokenize": "params:tokenize",
                    "embedding_dims": "params:embedding_dims",
                    "batchnorm": "params:batchnorm",
                    "symmetrize": "params:symmetrize",
                    "data_reupload_range": "params:data_reupload_range",
                    "n_fsps": "n_fsps",
                    "device": "params:device",
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "learning_rate_range": "params:learning_rate_range",
                    "learning_rate_decay_range": "params:learning_rate_decay_range",
                    "gamma": "params:gamma",
                    "batch_size_range": "params:batch_size_range",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "detectAnomaly": "params:detect_anomaly",
                    "redis_host": "params:redis_host",
                    "redis_port": "params:redis_port",
                    "redis_path": "params:redis_path",
                    "redis_password": "params:redis_password",
                },
                outputs={"hyperparam_optimizer": "hyperparam_optimizer"},
                name="create_instructor",
            ),
            node(
                func=train_optuna,
                inputs={"hyperparam_optimizer": "hyperparam_optimizer"},
                outputs={
                    "trained_model": "trained_quantum_model",
                    "checkpoint": "checkpoint",
                    "gradients": "gradients",
                },
                name="train_optuna",
            ),
        ]
    )


def create_resume_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            *create_metadata_nodes(**kwargs),
            node(
                func=unpack_checkpoint,
                inputs={
                    "checkpoint": "checkpoint",
                },
                outputs={
                    "model_state_dict": "model_state_dict",
                    "optimizer_state_dict": "optimizer_state_dict",
                    "start_epoch": "start_epoch",
                },
                name="unpack_checkpoint",
                tags="split_run",
            ),
            node(
                func=create_model,
                inputs={
                    "n_classes": "n_classes",
                    "n_momenta": "params:n_momenta",
                    "model_sel": "params:model_sel",
                    "n_blocks": "params:n_blocks",
                    "dim_feedforward": "params:dim_feedforward",
                    "n_layers_mlp": "params:n_layers_mlp",
                    "n_additional_mlp_layers": "params:n_additional_mlp_layers",
                    "n_final_mlp_layers": "params:n_final_mlp_layers",
                    "dropout_rate": "params:dropout_rate",
                    "factor": "params:factor",
                    "tokenize": "params:tokenize",
                    "embedding_dims": "params:embedding_dims",
                    "batchnorm": "params:batchnorm",
                    "symmetrize": "params:symmetrize",
                    "data_reupload": "params:data_reupload",
                    "add_rot_gates": "params:add_rot_gates",
                    "n_layers_vqc": "params:n_layers_vqc",
                    "n_fsps": "n_fsps",
                    "device": "params:device",
                },
                outputs={"nri_model": "nri_model"},
                name="create_model",
            ),
            node(
                func=create_instructor,
                inputs={
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "model": "nri_model",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "detectAnomaly": "params:detect_anomaly",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                    "model_state_dict": "model_state_dict",
                    "optimizer_state_dict": "optimizer_state_dict",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train,
                inputs={"instructor": "instructor", "start_epoch": "start_epoch"},
                outputs={
                    "trained_model": "trained_quantum_model",
                    "checkpoint": "checkpoint_out",
                    "gradients": "gradients",
                },
                name="train_qgnn",
            ),
        ]
    )


# additional pipeline just to add it as an exception in kedro
def create_debug_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return create_training_qgnn_pipeline(**kwargs)


def create_debug_training_optuna_pipeline(**kwargs) -> Pipeline:
    return create_training_optuna_pipeline(**kwargs)


# def create_split_training_qgnn_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                 func=log_git_repo,
#                 inputs={"git_hash_identifier": "params:git_hash_identifier"},
#                 outputs={},
#                 name="log_git_repo",
#                 tags="split_run",
#             ),
#             node(
#                 func=calculate_n_classes,
#                 inputs={
#                     "dataset_lca_and_leaves": "dataset_lca_and_leaves",
#                 },
#                 outputs={"n_classes": "n_classes"},
#                 name="calculate_n_classes",
#                 tags="split_run",
#             ),
#             node(
#                 func=calculate_n_fsps,
#                 inputs={
#                     "dataset_lca_and_leaves": "dataset_lca_and_leaves",
#                 },
#                 outputs={"n_fsps": "n_fsps"},
#                 name="calculate_n_fsps",
#                 tags="split_run",
#             ),
#             node(
#                 func=create_model,
#                 inputs={
#                     "n_classes": "n_classes",
#                     "n_momenta": "params:n_momenta",
#                     "model_sel": "params:post_model_sel",
#                     "n_blocks": "params:n_blocks",
#                     "dim_feedforward": "params:dim_feedforward",
#                     "n_layers_mlp": "params:n_layers_mlp",
#                     "n_additional_mlp_layers": "params:n_additional_mlp_layers",
#                     "n_final_mlp_layers": "params:n_final_mlp_layers",
#                     "dropout_rate": "params:dropout_rate",
#                     "factor": "params:factor",
#                     "tokenize": "params:tokenize",
#                     "embedding_dims": "params:embedding_dims",
#                     "batchnorm": "params:batchnorm",
#                     "symmetrize": "params:symmetrize",
#                     "pre_trained_model": "trained_model",
#                     "data_reupload": "params:data_reupload",
#                     "add_rot_gates": "params:add_rot_gates",
#                     "n_layers_vqc": "params:n_layers_vqc",
#                     "n_fsps": "n_fsps",
#                     "device": "params:device",
#                 },
#                 outputs={"nri_model": "quantum_model"},
#                 name="create_quantum_model",
#                 tags="split_run",
#             ),
#             node(
#                 func=create_instructor,
#                 inputs={
#                     "dataset_lca_and_leaves": "dataset_lca_and_leaves",
#                     "model": "quantum_model",
#                     "learning_rate": "params:learning_rate",
#                     "learning_rate_decay": "params:learning_rate_decay",
#                     "gamma": "params:gamma",
#                     "batch_size": "params:batch_size",
#                     "epochs": "params:epochs",
#                     "normalize": "params:normalize",
#                     "plot_mode": "params:plot_mode",
#                     "detectAnomaly": "params:detect_anomaly",
#                     "device": "params:device",
#                     "n_fsps": "n_fsps",
#                 },
#                 outputs={"instructor": "quantum_instructor"},
#                 name="create_quantum_instructor",
#                 tags="split_run",
#             ),
#             node(
#                 func=train,
#                 inputs={
#                     "instructor": "quantum_instructor",
#                 },
#                 outputs={
#                     "trained_model": "trained_quantum_model",
#                     "checkpoint": "checkpoint",
#                     "gradients": "gradients",
#                 },
#                 name="train_qgnn",
#                 tags=["split_run", "training"],
#             ),
#         ]
#     )
