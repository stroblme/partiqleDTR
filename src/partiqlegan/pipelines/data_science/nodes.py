import git

import torch as t
from torch.nn.parallel import DataParallel


import mlflow

from .instructor import Instructor
from .nri_gnn import bb_NRIModel

from typing import Dict

import logging
log = logging.getLogger(__name__)

def log_git_repo():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_tag("git_hash", str(sha))

def calculate_n_fsps(torch_dataset_lca_and_leaves:Dict) -> int:
    n_fsps = int(max([len(subset[0]) for _, subset in torch_dataset_lca_and_leaves.items()]))+1

    return{
        "n_fsps": n_fsps
    }

def create_model(   n_momenta,
                    n_fsps,
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
                    symmetrize=True
                ) -> DataParallel:

    model = bb_NRIModel(n_momenta=n_momenta,
                        n_fsps=n_fsps,
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
                        symmetrize=symmetrize)

    model = DataParallel(model)

    return{
        "nri_model":model
    }

def generate_instructor(torch_dataset_lca_and_leaves:Dict,
                        model: DataParallel, data: Dict, 
                        learning_rate: float, learning_rate_decay: int, gamma: float,
                        batch_size:int, epochs:int) -> Instructor:
    ins = Instructor(model, torch_dataset_lca_and_leaves, learning_rate, learning_rate_decay, gamma, batch_size, epochs)

    return{
        "instructor":ins
    }

def train_qgnn(instructor:Instructor):

    model = instructor.train()

    return{
        "trained_model":model
    }



