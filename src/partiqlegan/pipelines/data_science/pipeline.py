"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


# def create_training_qgnn_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#                 func=log_decay_parameter,
#                 inputs={
#                     "masses":"params:masses",
#                     "fsp_masses":"params:fsp_masses",
#                     "n_topologies":"params:n_topologies",
#                     "max_depth":"params:max_depth",
#                     "max_children":"params:max_children",
#                     "min_children":"params:min_children",
#                     "isp_weight":"params:isp_weight",
#                     "iso_retries":"params:iso_retries",
#                     "generate_unknown":"params:generate_unknown",
#                     "modes_names":"params:modes_names",
#                     "train_events_per_top":"params:train_events_per_top",
#                     "val_events_per_top":"params:val_events_per_top",
#                     "test_events_per_top":"params:test_events_per_top",
#                     "seed":"params:seed"
#                 },
#                 outputs={},
#                 name="log_git_repo"
#         ),
#         create_training_qgnn_pipeline_no_param_log(**kwargs)
#     ])


def create_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #         func=log_decay_parameter,
            #         inputs={
            #             "masses":"params:masses",
            #             "fsp_masses":"params:fsp_masses",
            #             "n_topologies":"params:n_topologies",
            #             "max_depth":"params:max_depth",
            #             "max_children":"params:max_children",
            #             "min_children":"params:min_children",
            #             "isp_weight":"params:isp_weight",
            #             "iso_retries":"params:iso_retries",
            #             "generate_unknown":"params:generate_unknown",
            #             "modes_names":"params:modes_names",
            #             "train_events_per_top":"params:train_events_per_top",
            #             "val_events_per_top":"params:val_events_per_top",
            #             "test_events_per_top":"params:test_events_per_top",
            #             "seed":"params:seed"
            #         },
            #         outputs={},
            #         name="log_git_repo"
            # ),
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
                    "detectAnomaly": "params:detect_anomaly",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train_qgnn,
                inputs={"instructor": "instructor"},
                outputs={"trained_model": "trained_model"},
                name="train_qgnn",
            ),
        ]
    )


def create_split_training_qgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #         func=log_decay_parameter,
            #         inputs={
            #             "masses":"params:masses",
            #             "fsp_masses":"params:fsp_masses",
            #             "n_topologies":"params:n_topologies",
            #             "max_depth":"params:max_depth",
            #             "max_children":"params:max_children",
            #             "min_children":"params:min_children",
            #             "isp_weight":"params:isp_weight",
            #             "iso_retries":"params:iso_retries",
            #             "generate_unknown":"params:generate_unknown",
            #             "modes_names":"params:modes_names",
            #             "train_events_per_top":"params:train_events_per_top",
            #             "val_events_per_top":"params:val_events_per_top",
            #             "test_events_per_top":"params:test_events_per_top",
            #             "seed":"params:seed"
            #         },
            #         outputs={},
            #         name="log_git_repo"
            # ),
            node(
                func=log_git_repo,
                inputs={"git_hash_identifier": "params:git_hash_identifier"},
                outputs={},
                name="log_git_repo",
                tags="split_run",
            ),
            node(
                func=calculate_n_classes,
                inputs={
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                },
                outputs={"n_classes": "n_classes"},
                name="calculate_n_classes",
                tags="split_run",
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
            # node(
            #         func=create_model,
            #         inputs={
            #             "n_classes":"n_classes",
            #             "n_momenta":"params:n_momenta",
            #             "model_sel":"params:model_sel",
            #             "n_blocks":"params:n_blocks",
            #             "dim_feedforward":"params:dim_feedforward",
            #             "n_layers_mlp":"params:n_layers_mlp",
            #             "n_additional_mlp_layers":"params:n_additional_mlp_layers",
            #             "n_final_mlp_layers":"params:n_final_mlp_layers",
            #             "dropout_rate":"params:dropout_rate",
            #             "factor":"params:factor",
            #             "tokenize":"params:tokenize",
            #             "embedding_dims":"params:embedding_dims",
            #             "batchnorm":"params:batchnorm",
            #             "symmetrize":"params:symmetrize",
            #         },
            #         outputs={
            #                 "nri_model":"classical_model"
            #         },
            #         name="create_model"
            # ),
            # node(
            #         func=create_instructor,
            #         inputs={
            #             "dataset_lca_and_leaves":"dataset_lca_and_leaves",
            #             "model":"classical_model",
            #             "learning_rate":"params:learning_rate",
            #             "learning_rate_decay":"params:learning_rate_decay",
            #             "gamma":"params:gamma",
            #             "batch_size":"params:batch_size",
            #             "epochs":"params:epochs",
            #             "normalize":"params:normalize"
            #         },
            #         outputs={
            #             "instructor":"classical_instructor"
            #         },
            #         name="create_instructor"
            # ),
            # node(
            #         func=train_qgnn,
            #         inputs={
            #             "instructor":"classical_instructor"
            #             },
            #         outputs={
            #             "trained_model":"trained_classical_model"
            #         },
            #         name="train_split_model"
            # ),
            node(
                func=create_model,
                inputs={
                    "n_classes": "n_classes",
                    "n_momenta": "params:n_momenta",
                    "model_sel": "params:post_model_sel",
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
                    "pre_trained_model": "trained_model",
                    "n_fsps": "n_fsps",
                    "device": "params:device",
                },
                outputs={"nri_model": "quantum_model"},
                name="create_quantum_model",
                tags="split_run",
            ),
            node(
                func=create_instructor,
                inputs={
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "model": "quantum_model",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "plot_mode": "params:plot_mode",
                    "detectAnomaly": "params:detect_anomaly",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                },
                outputs={"instructor": "quantum_instructor"},
                name="create_quantum_instructor",
                tags="split_run",
            ),
            node(
                func=train_qgnn,
                inputs={
                    "instructor": "quantum_instructor",
                },
                outputs={"trained_model": "trained_quantum_model"},
                name="train_qgnn",
                tags=["split_run", "training"],
            ),
        ]
    )
