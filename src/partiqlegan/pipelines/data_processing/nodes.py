"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, List
from sqlalchemy import desc

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
allowed_nodes = ["e+", "e-", "pi+", "pi-", "pi0", "gamma", "omega", "mu+", "mu-", "eta"]


def tree_data_to_discriminator(
    decay_tree_structure: Tuple[List, List]
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")
    particles = list()
    probabilities = list()

    def add_node(name, cur_adj_list):
        # if name in allowed_fsps:
            # fsps.append(name)
        if name is not "":
            particles.append(name)

            cur_adj_list.append(name)


    adjacency_list = list()

    # def get_nodes(tree, cur_adj_list):
    #     if type(tree) == str:
    #         add_node(tree, cur_adj_list)
    #     elif type(tree) == list:
    #         for sub_tree in tree:
    #             get_nodes(sub_tree)
                

    #     elif type(tree) == dict:
    #         for key, value in tree.items():
    #             # fsps
    #             if key == "fs":
    #                 get_nodes(value)
    #             # probabilities
    #             elif key == "bf":
    #                 probabilities.append(value)
    #             # 1st gen particles
    #             else:
    #                 add_node(key)
    #                 get_nodes(value)

    # get_nodes(decay_tree_structure)

    def append_node(list_to_append, node_name):
        if node_name in allowed_nodes:
            list_to_append.append(node_name)
        else:
            print(f"{node_name} not in allowed_nodes")

    def append_list(list_to_append, other_list):
        list_to_append.append(other_list)
        return list_to_append[-1]
    
    def recursive_add_dep(current_tree, current_adjacency_list, level=0):
        if type(current_tree) == dict:
            for key, value in current_tree.items():
                # fsps
                if key == "fs":
                    # First, add a new list to which we will append any subsequent leaves
                    current_adjacency_list[-1].append(list())
                    for particle in value:
                        # A string? add it directly to the list
                        if type(particle) == str:
                            append_node(current_adjacency_list[-1][-1], particle)
                            # current_adjacency_list[-1].append(particle)
                        # Special treating for the dicts
                        else:
                            # for key in list(particle.keys()):
                                # append_node(current_adjacency_list[-1][-1], key)
                            # these are only 1 dim dicts, thus it's fair enough to just get the first key
                            append_node(current_adjacency_list[-1][-1], list(particle.keys())[0])

                    # This appears to be redundant, but is much easier than keeping track of indices in multi dimensional recursive calls
                    for particle in value:
                        # This time we only go for the dicts
                        if type(particle) == dict:
                            # add a new entry to the overall adjacency list
                            current_adjacency_list.append(list())
                            # remember, there is only one key in the dict
                            recursive_add(particle, current_adjacency_list)

                # key is an node
                elif key in allowed_nodes:
                    if type(value) == list:
                        for entry in value:
                            new_list = append_list(current_adjacency_list, list())
                            append_node(current_adjacency_list[-1], key)
                            # current_adjacency_list[-1].append(key)
                            recursive_add(entry, new_list)
                    else:
                        pass
        if type(current_tree) == list:
            for entry in current_tree:
                current_adjacency_list.append(list())

                recursive_add(entry, current_adjacency_list)
            # for particle in current_tree['fs']:
            #     if type(particle) == str:
            #         current_adjacency_list[-1].append(particle)
            #     else:
            #         current_adjacency_list.append(current_adjacency_list[-1])
            #         recursive_add(value, current_adjacency_list)

    def recursive_add(current_tree, current_adjacency_list):
        for key, value in current_tree.items():
            # fsps
            if key == "fs":
                # First, add a new list to which we will append any subsequent leaves
                current_adjacency_list[-1].append(list())
                for particle in value:
                    # A string? add it directly to the list
                    if type(particle) == str:
                        append_node(current_adjacency_list[-1][-1], particle)
                        # current_adjacency_list[-1].append(particle)
                    # Special treating for the dicts
                    else:
                        # for key in list(particle.keys()):
                            # append_node(current_adjacency_list[-1][-1], key)
                        # these are only 1 dim dicts, thus it's fair enough to just get the first key
                        append_node(current_adjacency_list[-1][-1], list(particle.keys())[0])

                # This appears to be redundant, but is much easier than keeping track of indices in multi dimensional recursive calls
                for particle in value:
                    # This time we only go for the dicts
                    if type(particle) == dict:
                        # add a new entry to the overall adjacency list
                        # current_adjacency_list.append(list())
                        # remember, there is only one key in the dict
                        recursive_add(particle, current_adjacency_list)

            # key is an node
            elif key in allowed_nodes:
                if type(value) == list:
                    for entry in value:
                        new_list = append_list(current_adjacency_list, list())
                        append_node(new_list, key)
                        # current_adjacency_list[-1].append(key)
                        recursive_add(entry, current_adjacency_list)
                else:
                    raise NotImplementedError()
        # if type(current_tree) == list:
        #     for entry in current_tree:
        #         current_adjacency_list.append(list())

        #         recursive_add(entry, current_adjacency_list)



    def create_adj_list_from_tree(tree):
        nonlocal adjacency_list
        # adjacency_list.append(list())
        recursive_add(tree, adjacency_list)


    def create_adj_mat_from_adj_list(adj_list):
        adjacency_matrix = np.ndarray((len(particles), len(particles)))

        for part_r in particles:
            for part_c in particles: 

                adj_list[part_r]

    create_adj_list_from_tree(decay_tree_structure)
    create_adj_mat_from_adj_list(adjacency_list)
    # print(fsps)
    return decay_tree_structure

def tree_data_to_generator(
    decay_tree_events: Dict
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")

    num_removed_particles = 0

    def combine_weights_and_particles(weights, particles):
        assert len(weights) == len(particles)

        nonlocal num_removed_particles
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