import git
import os

from typing import List

import torch as t
from torch.nn.parallel import DataParallel

import redis
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
from .qmlp import qmlp

# from .dqgnn import dqgnn
models = {
    "gnn": gnn,
    "sgnn": sgnn,
    "qgnn": qgnn,
    "dqgnn": dqgnn,
    "qftgnn": qftgnn,
    "sqgnn": sqgnn,
    "qmlp": qmlp,
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

def create_redis_service(host:str, port:int):
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
    factor: bool,
    tokenize: bool,
    embedding_dims: int,
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
    normalize: bool,
    plot_mode: str,
    detectAnomaly: bool,
    redis_host: str,
    redis_port: int,
    redis_path: str,
    redis_password: str
) -> Hyperparam_Optimizer:

    hyperparam_optimizer = Hyperparam_Optimizer(
                                name="hyperparam_optimizer",
                                id=0,
                                host=redis_host,
                                port=redis_port,
                                path=redis_path,
                                password=redis_password                        
                            )

    hyperparam_optimizer.set_variable_parameters(
        {
            "n_blocks_range":n_blocks_range,
            "dim_feedforward_range":dim_feedforward_range,
            "n_layers_mlp_range":n_layers_mlp_range,
            "n_additional_mlp_layers_range":n_additional_mlp_layers_range,
            "n_final_mlp_layers_range":n_final_mlp_layers_range,
            "dropout_rate_range":dropout_rate_range,
            "data_reupload_range":data_reupload_range,
        },
        {
            "learning_rate_range":learning_rate_range,
            "learning_rate_decay_range":learning_rate_decay_range,
            "batch_size_range":batch_size_range
        }
    )

    hyperparam_optimizer.set_fixed_parameters(
        {
            "n_classes":n_classes,
            "n_momenta":n_momenta,
            "model_sel":model_sel,
            "factor":factor,
            "tokenize":tokenize,
            "embedding_dims":embedding_dims,
            "batchnorm":batchnorm,
            "symmetrize":symmetrize,
            "n_fsps":n_fsps,
            "device":device,
        },
        {
            "dataset_lca_and_leaves":dataset_lca_and_leaves,
            "model":None, # this must be overwritten later in the optimization step and just indicates the difference in implementation here
            "gamma":gamma,
            "epochs":epochs,
            "normalize":normalize,
            "plot_mode":plot_mode,
            "detectAnomaly":detectAnomaly,
        }
    )

    hyperparam_optimizer.create_model = create_model
    hyperparam_optimizer.create_instructor = create_instructor
    hyperparam_optimizer.objective = train

    return {
        "hyperparam_optimizer":hyperparam_optimizer
    }


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
    dropout_rate: float,
    factor: bool,
    tokenize: bool,
    embedding_dims: bool,
    batchnorm: bool,
    symmetrize: bool,
    n_fsps: int,
    device: str,
    data_reupload: bool,
    add_rot_gates: bool,
    n_layers_vqc: bool,
    pre_trained_model: DataParallel = None,
    **kwargs
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
        n_fsps=n_fsps,
        device=device,
        data_reupload=data_reupload,
        add_rot_gates=add_rot_gates,
        n_layers_vqc=n_layers_vqc,
        pre_trained_model=pre_trained_model,
        **kwargs
    )

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
    **kwargs: Dict
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
        **kwargs
    )

    return {"instructor": instructor}


def train(instructor: Instructor):

    result = instructor.train() # returns a dict of e.g. the model, checkpoints and the gradients

    return result
