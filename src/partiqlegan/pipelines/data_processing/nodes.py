"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, List

import torch
import torchvision
import torchdata as td

import copy

import re

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
    fsps = list()
    probabilities = list()

    def add_node(name):
        if name in allowed_fsps:
            fsps.append(name)


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
    # print(fsps)
    return decay_tree_structure

def tree_data_to_generator(
    decay_tree_events: Dict
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")

    num_removed_particles = 0

    def combine_weights_and_particles(weights, particles):
        assert len(weights) == len(particles)

        num_of_sets = len(particles)
        empty_sets = list()

        for set_i in range(num_of_sets):
            particles_in_set = list(particles[set_i].keys())

            for particle in particles_in_set:
                try:
                    idx = [f in particle for f in allowed_fsps].index(True)
                except ValueError:
                    idx = None
                    particles[set_i].pop(particle)  #remove the particle from the set
                    num_removed_particles += 1

                if particles[set_i] == {}:
                    empty_sets.append(set_i)

        shifter = 0
        for set_i in empty_sets:
            particles.pop(set_i-shifter)
            weights.pop(set_i-shifter)
            shifter += 1

        print(f"Removed {num_removed_particles} particles from the events as they are not in the list of allowed fsps: {allowed_fsps}")

    combine_weights_and_particles(*decay_tree_events)

    return decay_tree_events


def normalize(
    data: np.ndarray
) -> np.ndarray:

    return data