"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


nd_create_model = node(
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
        "skip_block": "params:skip_block",
        "skip_global": "params:skip_global",
        "dropout_rate": "params:dropout_rate",
        "batchnorm": "params:batchnorm",
        "symmetrize": "params:symmetrize",
        "data_reupload": "params:data_reupload",
        "add_rot_gates": "params:add_rot_gates",
        "n_layers_vqc": "params:n_layers_vqc",
        "padding_dropout": "params:padding_dropout",
        "predefined_vqc": "params:predefined_vqc",
        "predefined_iec": "params:predefined_iec",
        "measurement": "params:measurement",
        "backend": "params:backend",
        "n_shots": "params:n_shots",
        "n_fsps": "n_fsps",
        "device": "params:device",
        "initialization_constant": "params:initialization_constant",
        "initialization_offset": "params:initialization_offset",
        "parameter_seed": "params:parameter_seed"
    },
    outputs={"model": "model"},
    name="create_model",
)


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


def create_training_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            *create_metadata_nodes(**kwargs),
            nd_create_model,
            node(
                func=create_instructor,
                inputs={
                    "model": "model",
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "normalize_individually": "params:normalize_individually",
                    "zero_mean": "params:zero_mean",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "log_gradients": "params:log_gradients",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                    "n_classes": "n_classes",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "torch_seed": "params:torch_seed",
                    "gradient_curvature_threshold": "params:gradient_curvature_threshold",
                    "gradient_curvature_history": "params:gradient_curvature_history",
                    "quantum_optimizer": "params:quantum_optimizer",
                    "classical_optimizer": "params:classical_optimizer",
                    "detectAnomaly": "params:detect_anomaly",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train,
                inputs={
                    "instructor": "instructor",
                    "enabled_modes": "params:default_modes",
                },
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
                    "skip_block": "params:skip_block",
                    "skip_global": "params:skip_global",
                    "dropout_rate_range": "params:dropout_rate_range",
                    "batchnorm": "params:batchnorm",
                    "symmetrize": "params:symmetrize",
                    "data_reupload_range_quant": "params:data_reupload_range_quant",
                    "add_rot_gates": "params:add_rot_gates",
                    "n_layers_vqc_range_quant": "params:n_layers_vqc_range_quant",
                    "padding_dropout": "params:padding_dropout",
                    "predefined_vqc_range_quant": "params:predefined_vqc_range_quant",
                    "predefined_iec": "params:predefined_iec",
                    "measurement": "params:measurement",
                    "backend": "params:backend",
                    "n_shots_range_quant": "params:n_shots_range_quant",
                    "n_fsps": "n_fsps",

                    "device": "params:device",
                    "initialization_constant_range_quant": "params:initialization_constant_range_quant",
                    "initialization_offset_range_quant": "params:initialization_offset_range_quant",
                    "parameter_seed": "params:parameter_seed",
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "learning_rate_range": "params:learning_rate_range",
                    "learning_rate_decay_range": "params:learning_rate_decay_range",
                    "gamma": "params:gamma",
                    "batch_size_range": "params:batch_size_range",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "normalize_individually": "params:normalize_individually",
                    "zero_mean": "params:zero_mean",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "log_gradients": "params:log_gradients",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "torch_seed": "params:torch_seed",
                    "gradient_curvature_threshold_range_quant": "params:gradient_curvature_threshold_range_quant",
                    "gradient_curvature_history_range_quant": "params:gradient_curvature_history_range_quant",
                    "quantum_optimizer": "params:quantum_optimizer",
                    "classical_optimizer": "params:classical_optimizer",
                    "detectAnomaly": "params:detect_anomaly",
                    "n_trials": "params:n_trials",
                    "timeout": "params:timeout",
                    "optuna_path": "params:optuna_path",
                    "optuna_sampler_seed": "params:optuna_sampler_seed",
                    "selective_optimization": "params:selective_optimization"
                    "resume_study": "params:resume_study"
                },
                outputs={"hyperparam_optimizer": "hyperparam_optimizer"},
                name="create_instructor",
            ),
            node(
                func=train_optuna,
                inputs={"hyperparam_optimizer": "hyperparam_optimizer"},
                outputs={
                },
                name="train_optuna",
            ),
        ]
    )


def create_resume_training_pipeline(**kwargs) -> Pipeline:
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
            nd_create_model,
            node(
                func=create_instructor,
                inputs={
                    "model": "model",
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "normalize_individually": "params:normalize_individually",
                    "zero_mean": "params:zero_mean",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "log_gradients": "params:log_gradients",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                    "n_classes": "n_classes",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "torch_seed": "params:torch_seed",
                    "gradient_curvature_threshold": "params:gradient_curvature_threshold",
                    "gradient_curvature_history": "params:gradient_curvature_history",
                    "quantum_optimizer": "params:quantum_optimizer",
                    "classical_optimizer": "params:classical_optimizer",
                    "detectAnomaly": "params:detect_anomaly",
                    "model_state_dict": "model_state_dict",
                    "optimizer_state_dict": "optimizer_state_dict",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train,
                inputs={
                    "instructor": "instructor",
                    "start_epoch": "start_epoch",
                    "enabled_modes": "params:default_modes",
                },
                outputs={
                    "trained_model": "trained_quantum_model",
                    "checkpoint": "checkpoint_out",
                    "gradients": "gradients",
                    "metrics": "metrics"
                },
                name="train_qgnn",
            ),
        ]
    )


def create_validation_qgnn_pipeline(**kwargs) -> Pipeline:
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
                func=create_instructor,
                inputs={
                    "model": "model",
                    "dataset_lca_and_leaves": "dataset_lca_and_leaves",
                    "learning_rate": "params:learning_rate",
                    "learning_rate_decay": "params:learning_rate_decay",
                    "gamma": "params:gamma",
                    "batch_size": "params:batch_size",
                    "epochs": "params:epochs",
                    "normalize": "params:normalize",
                    "normalize_individually": "params:normalize_individually",
                    "zero_mean": "params:zero_mean",
                    "plot_mode": "params:plot_mode",
                    "plotting_rows": "params:plotting_rows",
                    "log_gradients": "params:log_gradients",
                    "device": "params:device",
                    "n_fsps": "n_fsps",
                    "n_classes": "n_classes",
                    "gradients_clamp": "params:gradients_clamp",
                    "gradients_spreader": "params:gradients_spreader",
                    "torch_seed": "params:torch_seed",
                    "gradient_curvature_threshold": "params:gradient_curvature_threshold",
                    "gradient_curvature_history": "params:gradient_curvature_history",
                    "quantum_optimizer": "params:quantum_optimizer",
                    "classical_optimizer": "params:classical_optimizer",
                    "detectAnomaly": "params:detect_anomaly",
                    "model_state_dict": "model_state_dict",
                    "optimizer_state_dict": "optimizer_state_dict",
                },
                outputs={"instructor": "instructor"},
                name="create_instructor",
            ),
            node(
                func=train,
                inputs={
                    "instructor": "instructor",
                    "enabled_modes": "params:validation_mode",
                },
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
def create_debug_training_pipeline(**kwargs) -> Pipeline:
    return create_training_pipeline(**kwargs)


# def create_training_optuna_pipeline(**kwargs) -> Pipeline:
#     return create_training_optuna_pipeline(**kwargs)
