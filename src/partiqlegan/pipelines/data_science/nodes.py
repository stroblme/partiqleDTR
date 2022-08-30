import git
import os

import torch as t
from torch.nn.parallel import DataParallel


import mlflow

from .hyperparam_optimizer import Hyperparam_Optimizer
from .instructor import Instructor
from .gnn import gnn
from .qftgnn import qftgnn
from .qgnn import qgnn
from .dqgnn import dqgnn
from .dgnn import dgnn
from .pqgnn import pqgnn
from .sgnn import sgnn
from .sqgnn import sqgnn

# from .dqgnn import dqgnn
models = {
    "gnn": gnn,
    "sgnn": sgnn,
    "qgnn": qgnn,
    "dqgnn": dqgnn,
    "qftgnn": qftgnn,
    "sqgnn": sqgnn,
    "dgnn": dgnn,
    "pqgnn": pqgnn,
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

def create_hyperparam_optimizer(
    n_classes,
    n_momenta,
    model_sel,
    n_blocks_range:list,
    dim_feedforward_range:list,
    n_layers_mlp_range:list,
    n_additional_mlp_layers_range:list,
    n_final_mlp_layers_range:list,
    dropout_rate_range:list,
    factor:bool,
    tokenize:bool,
    embedding_dims:int,
    batchnorm:bool,
    symmetrize:bool,
    data_reupload_range:bool,
    pre_trained_model:DataParallel,
    n_fsps:int,
    device:str,

    dataset_lca_and_leaves: Dict,
    model: DataParallel,
    learning_rate_range: list,
    learning_rate_decay_range: list,
    gamma: float,
    batch_size_range: list,
    epochs: int,
    normalize: bool,
    plot_mode: str,
    detectAnomaly: bool,
) -> Hyperparam_Optimizer:

    hyperparam_optimizer = Hyperparam_Optimizer()

    hyperparam_optimizer.set_variable_parmeters(
        [
            n_blocks_range,
            dim_feedforward_range,
            n_layers_mlp_range,
            n_additional_mlp_layers_range,
            n_final_mlp_layers_range,
            dropout_rate_range,
            data_reupload_range
        ],
        [
            learning_rate_decay_range,
            batch_size_range
        ]
    )

    hyperparam_optimizer.set_fixed_parameters(
        [
            n_classes,
            n_momenta,
            model_sel,
            factor,
            tokenize,
            embedding_dims,
            batchnorm,
            symmetrize,
            pre_trained_model,
            n_fsps,
            device,
        ],
        [
            dataset_lca_and_leaves,
            model,
            gamma,
            epochs,
            normalize,
            plot_mode,
            detectAnomaly,
        ]
    )

    hyperparam_optimizer.create_model = create_model
    hyperparam_optimizer.create_instructor = create_instructor
    hyperparam_optimizer.objective = train

    return hyperparam_optimizer

def train_optuna(hyperparam_optimizer: Hyperparam_Optimizer):
    
    hyperparam_optimizer.minimize()

    trained_model, gradients = hyperparam_optimizer.get_artifacts()

    return {"trained_model": trained_model, "gradients": gradients}


def create_model(
    n_classes,
    n_momenta,
    model_sel,
    n_blocks:int,
    dim_feedforward:int,
    n_layers_mlp:int,
    n_additional_mlp_layers:int,
    n_final_mlp_layers:int,
    dropout_rate:float,
    factor:bool,
    tokenize:bool,
    embedding_dims:bool,
    batchnorm:bool,
    symmetrize:bool,
    pre_trained_model:DataParallel,
    n_fsps:int,
    device:str,
) -> DataParallel:

    model = models[model_sel](
        n_momenta=n_momenta,
        n_classes=n_classes,
        n_blocks=n_blocks,
        dim_feedforward=dim_feedforward,
        n_layers_mlp=n_layers_mlp,
        n_additional_mlp_layers=n_additional_mlp_layers,
        n_final_mlp_layers=n_final_mlp_layers,
        dropout_rate=dropout_rate,
        factor=factor,
        tokenize=tokenize,
        embedding_dims=embedding_dims,
        batchnorm=batchnorm,
        symmetrize=symmetrize,
        pre_trained_model=pre_trained_model,
        n_fsps=n_fsps,
        device=device,
    )

    # if pre_trained_model: #TODO: check if this case decision is necessary
    #     model = models[model_sel](n_momenta=n_momenta,
    #                     n_classes=n_classes,
    #                     n_blocks=n_blocks,
    #                     dim_feedforward=dim_feedforward,
    #                     n_layers_mlp=n_layers_mlp,
    #                     n_additional_mlp_layers=n_additional_mlp_layers,
    #                     n_final_mlp_layers=n_final_mlp_layers,
    #                     dropout_rate=dropout_rate,
    #                     factor=factor,
    #                     tokenize=tokenize,
    #                     embedding_dims=embedding_dims,
    #                     batchnorm=batchnorm,
    #                     symmetrize=symmetrize,
    #                     pre_trained_model=pre_trained_model,
    #                     n_fsps=n_fsps)
    # else:
    #     model = models[model_sel](n_momenta=n_momenta,
    #                         n_classes=n_classes,
    #                         n_blocks=n_blocks,
    #                         dim_feedforward=dim_feedforward,
    #                         n_layers_mlp=n_layers_mlp,
    #                         n_additional_mlp_layers=n_additional_mlp_layers,
    #                         n_final_mlp_layers=n_final_mlp_layers,
    #                         dropout_rate=dropout_rate,
    #                         factor=factor,
    #                         tokenize=tokenize,
    #                         embedding_dims=embedding_dims,
    #                         batchnorm=batchnorm,
    #                         symmetrize=symmetrize)

    if device == "cpu":
        nri_model = model.to(t.device(device))
        # os.environ["CUDA_VISIBLE_DEVICES"]=""
    elif t.cuda.is_available():
        nri_model = DataParallel(model)
    else:
        raise ValueError(f"Specified device {device} although no cuda support detected")

    return {"nri_model": nri_model}




def create_instructor(
    dataset_lca_and_leaves: Dict,
    model: DataParallel,
    learning_rate: float,
    learning_rate_decay: int,
    gamma: float,
    batch_size: int,
    epochs: int,
    normalize: bool,
    plot_mode: str,
    detectAnomaly: bool,
    device: str,
    n_fsps: int,
) -> Instructor:
    instructor = Instructor(
        model,
        dataset_lca_and_leaves,
        learning_rate,
        learning_rate_decay,
        gamma,
        batch_size,
        epochs,
        normalize,
        plot_mode,
        detectAnomaly,
        device,
        n_fsps,
    )

    return {"instructor": instructor}


def train(instructor: Instructor):

    trained_model, gradients = instructor.train()

    return {"trained_model": trained_model, "gradients": gradients}

