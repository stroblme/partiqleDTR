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

class particle_node():
    def __init__(self, name, parent_node):
        self.name = name
        self.parent_node = parent_node

    def __eq__(self, __o: object) -> bool:
        try:
            return (self.name == __o.name) and (self.parent_node == __o.parent_node)
        except:
            return False

            

def tree_data_to_discriminator(
    decay_tree_structure: Tuple[List, List]
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")
    particles = list()
    probabilities = list()

    # def add_node(name, cur_adj_list):
    #     # if name in allowed_fsps:
    #         # fsps.append(name)
    #     if name is not "":
    #         particles.append(name)

    #         cur_adj_list.append(name)


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

    def append_node(list_to_append, node_name, parent_node_instance):
        if node_name in allowed_nodes:
            particle_instance = particle_node(node_name, parent_node_instance)
            list_to_append.append(particle_instance)
            particles.append(particle_instance)
            return particle_instance
        else:
            print(f"{node_name} not in allowed_nodes")

    def append_list(list_to_append, other_list):
        list_to_append.append(other_list)
        return list_to_append[-1]
    

    def create_adj_list_from_tree(tree, adj_list, parent=None):

        first_key = list(tree.keys())[0]

        # fsps
        if "fs" and "bf" in tree.keys():
            # First, add a new list to which we will append any subsequent leaves
            adj_list[-1].append(list())

            # strings are moved to the very beginning. This is important to not mixup with the [-1] calls
            type_sorted_values=sorted(tree["fs"], key=lambda x: not isinstance(x,str))
            for particle in type_sorted_values:
                # A string? add it directly to the list
                if type(particle) == str:
                    append_node(adj_list[-1][-1], particle, parent)
                # Special treating for the dicts
                else:
                    # these are only 1 dim dicts, thus it's fair enough to just get the first key
                    particle_instance = append_node(adj_list[-1][-1], list(particle.keys())[0], parent)
                    # a recursive call at this point is ok, since there should be no strings anymore due to the sorting prior to the iteration
                    create_adj_list_from_tree(particle, adj_list, particle_instance)

        # key is a node
        elif first_key in allowed_nodes:
            for entry in tree[first_key]:
                new_list = append_list(adj_list, list())
                particle_instance = append_node(new_list, first_key, parent)
                create_adj_list_from_tree(entry, adj_list, particle_instance)

    def create_adj_mat_from_adj_list(adj_list):
        adjacency_matrix = np.zeros((len(particles), len(particles)))

        for edge in adj_list:
            pass

        for part_r in range(len(particles)):
            for part_c in range(len(particles)-part_r): 

                adj_list[part_r]

    adjacency_list = list()
    create_adj_list_from_tree(decay_tree_structure, adjacency_list)
    adjacency_matrix = create_adj_mat_from_adj_list(adjacency_list)
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