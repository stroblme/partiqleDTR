import git

import torch as t
from torch.nn.parallel import DataParallel


import mlflow

from .instructor import Instructor
from .nri_gnn import bb_NRIModel

from typing import Dict, List

import logging
log = logging.getLogger(__name__)

def log_git_repo(git_hash_identifier:str):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_tag(git_hash_identifier, str(sha))

def log_decay_parameter(
                        masses:List[int],
                        fsp_masses:List[int],
                        n_topologies:int,
                        max_depth:int,
                        max_children:int,
                        min_children:int,
                        isp_weight:int,
                        iso_retries:int,
                        generate_unknown: bool,
                        modes_names: List[str],
                        train_events_per_top: int,
                        val_events_per_top: int,
                        test_events_per_top: int,
                        seed: int):
    pass # just calling is enough for auto logging
    # mlflow.log_param("masses", masses)
    # mlflow.log_param("fsp_masses", fsp_masses)
    # mlflow.log_param("n_topologies", n_topologies)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("max_children", max_children)
    # mlflow.log_param("min_children", min_children)
    # mlflow.log_param("isp_weight", isp_weight)
    # mlflow.log_param("iso_retries", iso_retries)
    # mlflow.log_param("generate_unknown", generate_unknown)
    # mlflow.log_param("modes_names", modes_names)
    # mlflow.log_param("train_events_per_top", train_events_per_top)
    # mlflow.log_param("val_events_per_top", val_events_per_top)
    # mlflow.log_param("test_events_per_top", test_events_per_top)
    # mlflow.log_param("seed", seed)


def calculate_n_fsps(torch_dataset_lca_and_leaves:Dict) -> int:
    n_fsps = int(max([len(subset[0]) for _, subset in torch_dataset_lca_and_leaves.items()]))+1

    return{
        "n_fsps": n_fsps
    }

def create_model(   n_fsps,
                    n_momenta,
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

    nri_model = DataParallel(model)

    return{
        "nri_model":nri_model
    }

def create_instructor(  torch_dataset_lca_and_leaves:Dict,
                        model: DataParallel,
                        learning_rate: float, learning_rate_decay: int, gamma: float,
                        batch_size:int, epochs:int) -> Instructor:
    instructor = Instructor(model, torch_dataset_lca_and_leaves, 
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



