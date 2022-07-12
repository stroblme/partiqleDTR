import git

import torch as t
from torch.nn.parallel import DataParallel


import mlflow

from .instructor import Instructor
from .gnn import gnn
from .qftgnn import qftgnn
from .qgnn import qgnn
from .dgnn import dgnn
# from .dqgnn import dqgnn
models = {"gnn":gnn, "qgnn":qgnn, "qftgnn":qftgnn, "dgnn":dgnn}

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
                    symmetrize=True
                ) -> DataParallel:

    model = models[model_sel](n_momenta=n_momenta,
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
                        symmetrize=symmetrize)

    nri_model = DataParallel(model)

    return{
        "nri_model":nri_model
    }

def create_instructor(  dataset_lca_and_leaves:Dict,
                        model: DataParallel,
                        learning_rate: float, learning_rate_decay: int, gamma: float,
                        batch_size:int, epochs:int) -> Instructor:
    instructor = Instructor(model, dataset_lca_and_leaves, 
                            learning_rate, learning_rate_decay, gamma, 
                            batch_size, epochs)

    return{
        "instructor":instructor
    }

def train_qgnn(instructor:Instructor):

    trained_model = instructor.train()

    return{
        "trained_model":trained_model
    }



