"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, List

import torch
import torchvision
import torchdata as td

data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# class DecayTreeDataset(torch.utils.data.dataset):
#     def __init__(self, dataset:Dict[str, np.ndarray]):

#         self.decayTreeDataset = dataset
        
#         raise NotImplementedError("Sorry, not yet..")

#     def __len__(self):
#         return len(self.decayTreeDataset)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         raise NotImplementedError("Sorry, not yet..")
        
allowed_fsps = ["e+", "e-", "pi+", "pi-", "gamma"]

def tree_data_to_discriminator(
    decay_tree_structure: Tuple[List, List]
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")
    nodes = list()
    probabilities = list()

    def add_node(name):
        if name in allowed_fsps:
            nodes.append(name)


    def get_nodes(tree):
        if type(tree) == str:
            add_node(tree)

        elif type(tree) == list:
            for sub_tree in tree:
                get_nodes(sub_tree)
                

        elif type(tree) == dict:
            for key, value in tree.items():
                # fsps
                if key == "fs":
                    get_nodes(value)
                # probabilities
                elif key == "bf":
                    probabilities.append(value)
                # 1st gen particles
                else:
                    add_node(key)
                    get_nodes(value)

    get_nodes(decay_tree_structure)
    return decay_tree_structure

def tree_data_to_generator(
    decay_tree_events: Dict
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")

    return decay_tree_events


def normalize(
    data: np.ndarray
) -> np.ndarray:

    return data