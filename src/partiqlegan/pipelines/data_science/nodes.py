import git

import torch as t
from torch.nn.parallel import DataParallel


import mlflow

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
models = {"gnn":gnn, "sgnn":sgnn, "qgnn":qgnn, "dqgnn":dqgnn, "qftgnn":qftgnn, "sqgnn":sqgnn, "dgnn":dgnn, "pqgnn":pqgnn}

from typing import Dict

import logging
log = logging.getLogger(__name__)

def log_git_repo(git_hash_identifier:str):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_tag(git_hash_identifier, sha)

    return {}

def calculate_n_classes(dataset_lca_and_leaves:Dict) -> int:
    n_classes = 0
    for _, subset in dataset_lca_and_leaves.items():
        for lca in subset.y:
            n_classes = int(lca.max() if lca.max() > n_classes else n_classes)
    # n_fsps = int(max([len(subset[0]) for _, subset in dataset_lca_and_leaves.items()]))+1

    return{
        "n_classes": n_classes+1 # +1 for starting counting from zero (len(0..5)=5+1)
    }

def calculate_n_fsps(dataset_lca_and_leaves:Dict) -> int:
    n_fsps = 0
    for _, subset in dataset_lca_and_leaves.items():
        for lca in subset.y:
            n_fsps = lca.shape[0] if lca.shape[0] > n_fsps else n_fsps
    # n_fsps = int(max([len(subset[0]) for _, subset in dataset_lca_and_leaves.items()]))+1

    return{
        "n_fsps": n_fsps
    }


def create_model(   n_classes,
                    n_momenta,
                    model_sel,
                    n_blocks=3,
                    dim_feedforward=128,
                    n_layers_mlp=2,
                    n_additional_mlp_layers=2,
                    n_final_mlp_layers=2,
                    dropout_rate=0.3,
                    factor=True,
                    tokenize=None,
                    embedding_dims=None,
                    batchnorm=True,
                    symmetrize=True,
                    pre_trained_model:DataParallel=None,
                    n_fsps:int=-1,
                    device="cpu"
                ) -> DataParallel:

    model = models[model_sel](  n_momenta=n_momenta,
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
                                n_fsps=n_fsps)

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

    if device == 'cpu':
        nri_model = model
    else:
        nri_model = DataParallel(model)

    return{
        "nri_model":nri_model
    }

def create_instructor(  dataset_lca_and_leaves:Dict,
                        model: DataParallel,
                        learning_rate: float, learning_rate_decay: int, gamma: float,
                        batch_size:int, epochs:int, normalize:bool, plot_mode:str, detectAnomaly:bool, device:str, n_fsps:int) -> Instructor:
    instructor = Instructor(model, dataset_lca_and_leaves, 
                            learning_rate, learning_rate_decay, gamma, 
                            batch_size, epochs, normalize, plot_mode, detectAnomaly, device, n_fsps)

    return{
        "instructor":instructor
    }

def train_qgnn(instructor:Instructor):

    trained_model = instructor.train()

    return{
        "trained_model":trained_model
    }
