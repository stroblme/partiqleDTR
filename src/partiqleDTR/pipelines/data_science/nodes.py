import git
import os

from typing import List

import torch as t
from torch.nn.parallel import DataParallel

import redis
import mlflow

from .hyperparam_optimizer import Hyperparam_Optimizer
from .instructor import Instructor
from .qgnn import qgnn
from .gnn import gnn


# from .dqgnn import dqgnn
models = {
    "gnn": gnn,
    "qgnn": qgnn,
}

from typing import Dict

import logging

log = logging.getLogger(__name__)


def log_git_repo(git_hash_identifier: str):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if repo.is_dirty(untracked_files=True):
        log.warning(
            "Uncommited and/or untracked files found. Please cleanup before running experiments"
        )
    else:
        log.info(f"Repository was found to be clean with sha {sha}")
    mlflow.set_tag(git_hash_identifier, sha)

    return {}


def calculate_n_classes(dataset_lca_and_leaves: Dict) -> int:
    n_classes = 0
    for _, subset in dataset_lca_and_leaves.items():
        for lca in subset.y:
            n_classes = int(lca.max() if lca.max() > n_classes else n_classes)
    # n_fsps = int(max([len(subset[0]) for _, subset in dataset_lca_and_leaves.items()]))+1

    log.info(f"Number of Classes calculated to {n_classes}")
    return {
        "n_classes": n_classes + 1  # +1 for starting counting from zero (len(0..5)=5+1)
    }


def calculate_n_fsps(dataset_lca_and_leaves: Dict) -> int:
    n_fsps = 0
    for _, subset in dataset_lca_and_leaves.items():
        for lca in subset.y:
            n_fsps = lca.shape[0] if lca.shape[0] > n_fsps else n_fsps
    # n_fsps = int(max([len(subset[0]) for _, subset in dataset_lca_and_leaves.items()]))+1
    log.info(f"Number of FSPS calculated to {n_fsps}")
    return {"n_fsps": n_fsps}


def unpack_checkpoint(checkpoint: Dict) -> Dict:
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    start_epoch = checkpoint["start_epoch"]

    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "start_epoch": start_epoch,
    }


def create_redis_service(host: str, port: int):
    r = redis.Redis(host=host, port=port)


def create_hyperparam_optimizer(
    n_classes,
    n_momenta,
    model_sel,
    n_blocks_range: List,
    dim_feedforward_range: List,
    n_layers_mlp_range: List,
    n_additional_mlp_layers_range: List,
    n_final_mlp_layers_range: List,
    dropout_rate_range: List,
    batchnorm: bool,
    symmetrize: bool,
    data_reupload_range: bool,
    n_fsps: int,
    device: str,
    dataset_lca_and_leaves: Dict,
    learning_rate_range: List,
    learning_rate_decay_range: List,
    gamma: float,
    batch_size_range: List,
    epochs: int,
    normalize: str,
    plot_mode: str,
    plotting_rows: int,
    gradients_clamp: int,
    gradients_spreader: float,
    detectAnomaly: bool,
    redis_host: str,
    redis_port: int,
    redis_path: str,
    redis_password: str,
) -> Hyperparam_Optimizer:

    hyperparam_optimizer = Hyperparam_Optimizer(
        name="hyperparam_optimizer",
        id=0,
        host=redis_host,
        port=redis_port,
        path=redis_path,
        password=redis_password,
    )

    hyperparam_optimizer.set_variable_parameters(
        {
            "n_blocks_range": n_blocks_range,
            "dim_feedforward_range": dim_feedforward_range,
            "n_layers_mlp_range": n_layers_mlp_range,
            "n_additional_mlp_layers_range": n_additional_mlp_layers_range,
            "n_final_mlp_layers_range": n_final_mlp_layers_range,
            "dropout_rate_range": dropout_rate_range,
            "data_reupload_range": data_reupload_range,
        },
        {
            "learning_rate_range": learning_rate_range,
            "learning_rate_decay_range": learning_rate_decay_range,
            "batch_size_range": batch_size_range,
        },
    )

    hyperparam_optimizer.set_fixed_parameters(
        {
            "n_classes": n_classes,
            "n_momenta": n_momenta,
            "model_sel": model_sel,
            "batchnorm": batchnorm,
            "symmetrize": symmetrize,
            "n_fsps": n_fsps,
            "device": device,
        },
        {
            "dataset_lca_and_leaves": dataset_lca_and_leaves,
            "model": None,  # this must be overwritten later in the optimization step and just indicates the difference in implementation here
            "gamma": gamma,
            "epochs": epochs,
            "normalize": normalize,
            "plot_mode": plot_mode,
            "plotting_rows": plotting_rows,
            "gradients_clamp": gradients_clamp,
            "gradients_spreader": gradients_spreader,
            "detectAnomaly": detectAnomaly,
        },
    )

    hyperparam_optimizer.create_model = create_model
    hyperparam_optimizer.create_instructor = create_instructor
    hyperparam_optimizer.objective = train

    return {"hyperparam_optimizer": hyperparam_optimizer}


def train_optuna(hyperparam_optimizer: Hyperparam_Optimizer):

    hyperparam_optimizer.minimize()

    trained_model, gradients = hyperparam_optimizer.get_artifacts()

    return {"trained_model": trained_model, "gradients": gradients}


def create_model(
    n_classes,
    n_momenta,
    model_sel,
    n_blocks: int,
    dim_feedforward: int,
    n_layers_mlp: int,
    n_additional_mlp_layers: int,
    n_final_mlp_layers: int,
    skip_block: bool,
    skip_global: bool,
    dropout_rate: float,
    batchnorm: bool,
    symmetrize: bool,
    data_reupload: bool,
    add_rot_gates: bool,
    n_layers_vqc: bool,
    padding_dropout: bool,
    predefined_vqc: str,
    predefined_iec: str,
    measurement: str,
    backend: str,
    n_shots: int,
    n_fsps: int,
    device: str,
    initialization_constant: str,
    parameter_seed:int,
    pre_trained_model: DataParallel = None,
    **kwargs,
) -> DataParallel:

    model = models[model_sel](
        n_momenta=n_momenta,
        n_classes=n_classes,
        n_blocks=n_blocks,
        dim_feedforward=dim_feedforward,
        n_layers_mlp=n_layers_mlp,
        n_additional_mlp_layers=n_additional_mlp_layers,
        n_final_mlp_layers=n_final_mlp_layers,
        skip_block=skip_block,
        skip_global=skip_global,
        dropout_rate=dropout_rate,
        batchnorm=batchnorm,
        symmetrize=symmetrize,
        data_reupload=data_reupload,
        add_rot_gates=add_rot_gates,
        n_layers_vqc=n_layers_vqc,
        padding_dropout=padding_dropout,
        predefined_vqc=predefined_vqc,
        predefined_iec=predefined_iec,
        measurement=measurement,
        backend=backend,
        n_shots=n_shots,
        n_fsps=n_fsps,
        device=device,
        pre_trained_model=pre_trained_model,
        initialization_constant=initialization_constant,
        parameter_seed=parameter_seed,
        **kwargs,
    )

    if device == "cpu":
        model = model.to(t.device(device))
        # os.environ["CUDA_VISIBLE_DEVICES"]=""
    elif t.cuda.is_available():
        model = DataParallel(model)
    else:
        raise ValueError(f"Specified device {device} although no cuda support detected")

    return {"model": model}


def create_instructor(
    model: DataParallel,
    dataset_lca_and_leaves: Dict,
    learning_rate: float,
    learning_rate_decay: int,
    gamma: float,
    batch_size: int,
    epochs: int,
    normalize: str,
    normalize_individually: bool,
    zero_mean: bool,
    plot_mode: str,
    plotting_rows: int,
    log_gradients: bool,
    detectAnomaly: bool,
    device: str,
    n_fsps: int,
    n_classes: int,
    gradients_clamp: int,
    gradients_spreader: float,
    torch_seed: int,
    gradient_curvature_threshold: float,
    gradient_curvature_history: float,
    quantum_optimizer: str,
    classical_optimizer: str,
    **kwargs: Dict,
) -> Instructor:
    instructor = Instructor(
        model=model,
        data=dataset_lca_and_leaves,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        gamma=gamma,
        batch_size=batch_size,
        epochs=epochs,
        normalize=normalize,
        normalize_individually=normalize_individually,
        zero_mean=zero_mean,
        plot_mode=plot_mode,
        plotting_rows=plotting_rows,
        log_gradients=log_gradients,
        detectAnomaly=detectAnomaly,
        device=device,
        n_fsps=n_fsps,
        n_classes=n_classes,
        gradients_clamp=gradients_clamp,
        gradients_spreader=gradients_spreader,
        torch_seed=torch_seed,
        gradient_curvature_threshold=gradient_curvature_threshold,
        gradient_curvature_history=gradient_curvature_history,
        quantum_optimizer=quantum_optimizer,
        classical_optimizer=classical_optimizer,
        **kwargs,
    )

    return {"instructor": instructor}


def train(instructor: Instructor, start_epoch=1, enabled_modes=["train", "val"]):

    result = instructor.train(
        start_epoch=start_epoch, enabled_modes=enabled_modes
    )  # returns a dict of e.g. the model, checkpoints and the gradients

    return result
